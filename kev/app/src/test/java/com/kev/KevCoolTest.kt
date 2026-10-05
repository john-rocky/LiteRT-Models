package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The wait before a timed set ([KevCooler]) with a stand-in GPU state and clock, and the start time
 * every timed call records next to its ms ([KevTimingCore]).
 */
class KevCoolTest {
  /**
   * A clock that only [sleep] moves, and the GPU states [read] returns in turn (the last repeats).
   */
  private class Stand(vararg states: KevGpuState?) {
    var now = 0L
    var reads = 0
    val sleeps = ArrayList<Long>()
    private val queue = states.toList()

    fun read(): KevGpuState? = queue[minOf(reads++, queue.size - 1)]

    fun sleep(ms: Long) {
      sleeps.add(ms)
      now += ms
    }

    fun cooler(coolMs: Long) = KevCooler(coolMs, ::read, { now }, ::sleep)
  }

  @Test
  fun theWaitEndsWhenTheCeilingAndTheTemperatureAreBack() {
    val base = KevGpuState(1300, 41_000)
    // Right after a compile: ceiling 826 MHz and 60 °C, then 1,000 MHz and 47 °C, then back.
    val stand =
      Stand(base, KevGpuState(826, 60_000), KevGpuState(1000, 47_000), KevGpuState(1300, 45_900))
    val cooler = stand.cooler(60_000)
    assertEquals(base, cooler.readBase())
    assertEquals(base, cooler.readBase())
    assertEquals(1, stand.reads)
    val record = cooler.waitForBase()
    assertEquals("kgsl", record["mode"])
    assertEquals(1_000L, record["waited_ms"])
    assertEquals(true, record["recovered"])
    assertEquals(base.toJson(), record["base"])
    assertEquals(KevGpuState(1300, 45_900).toJson(), record["end"])
    assertEquals(listOf(500L, 500L), stand.sleeps)
    // Already back: no wait.
    val again = cooler.waitForBase()
    assertEquals(0L, again["waited_ms"])
    assertEquals(true, again["recovered"])
    assertTrue(KevCooler.recovered(base, KevGpuState(1300, 46_000)))
    assertFalse(KevCooler.recovered(base, KevGpuState(1300, 46_001)))
    assertFalse(KevCooler.recovered(base, KevGpuState(1200, 41_000)))
  }

  @Test
  fun theWaitStopsAtItsLimitOrFallsBackToAFixedSleep() {
    // The ceiling never comes back: the wait ends at the limit, not recovered.
    val hot = Stand(KevGpuState(1300, 41_000), KevGpuState(578, 67_800))
    val capped = hot.cooler(2_000)
    capped.readBase()
    val record = capped.waitForBase()
    assertEquals("kgsl", record["mode"])
    assertEquals(2_000L, record["waited_ms"])
    assertEquals(false, record["recovered"])
    // The state cannot be read: a fixed wait of the limit.
    val blind = Stand(null)
    val fixed = blind.cooler(60_000)
    assertEquals(null, fixed.readBase())
    val sleep = fixed.waitForBase()
    assertEquals("fixed", sleep["mode"])
    assertEquals(60_000L, sleep["waited_ms"])
    assertEquals(listOf(60_000L), blind.sleeps)
    // The state stops being readable during the wait.
    val lost = Stand(KevGpuState(1300, 41_000), KevGpuState(826, 60_000), null)
    val partial = lost.cooler(60_000)
    partial.readBase()
    val gone = partial.waitForBase()
    assertEquals(false, gone["recovered"])
    assertEquals(linkedMapOf("readable" to false), gone["end"])
    // cool_ms 0: no read, no wait.
    val off = Stand(KevGpuState(1300, 41_000))
    assertEquals(linkedMapOf("cool_ms" to 0L, "mode" to "none"), off.cooler(0).waitForBase())
    assertEquals(0, off.reads)
  }

  @Test
  fun everyTimedCallRecordsItsStart() {
    val rows =
      KevTimingRows.parse(
        ("""{"pad_id": 248044, "sets": [""" +
            """{"name": "two", "kind": "request", "L": 8, "rows": [""" +
            """{"key": "a", "ids": [1, 2, ${KevEncoder.QUESTION_ID}, 5]},""" +
            """{"key": "b", "ids": [1, 2, ${KevEncoder.QUESTION_ID}, 6, 7]}]}]}""")
          .toByteArray()
      )
    val row =
      object : RowRunner {
        override val length = 8

        override fun run(ids: IntArray, valid: FloatArray) =
          FloatArray(length * KevPointerHead.HIDDEN_SIZE)
      }
    val pair =
      object : PairRunner {
        override val stateLength = 4
        override val questionLength = 4

        override fun runState(ids: IntArray, valid: FloatArray) = Unit

        override fun runQuestion(ids: IntArray, valid: FloatArray) =
          FloatArray(questionLength * KevPointerHead.HIDDEN_SIZE)
      }
    var tick = 1_000L
    val clock = { tick++ }
    val pipeline =
      KevPipeline(
        ExternalTestData.tokenizer(),
        KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)),
      )
    // The GPU ceiling falls from 1,300 to 578 MHz after the first 15 reads.
    var reads = 0
    val ceiling = { if (reads++ < 15) 1300 else 578 }
    val set = KevTimingCore(pipeline, row, clock, ceiling) { false }.timeSet(rows.sets.single())
    val calls = set["calls_ms"] as List<*>
    val starts = set["call_starts_ms"] as List<*>
    val ceilings = set["call_max_clock_mhz"] as List<*>
    assertEquals(40, calls.size)
    assertEquals(40, starts.size)
    assertEquals(5, (set["warmup_starts_ms"] as List<*>).size)
    assertEquals(starts.sortedBy { it as Long }, starts)
    assertEquals(List(5) { 1300 }, set["warmup_max_clock_mhz"])
    assertEquals(List(10) { 1300 } + List(30) { 578 }, ceilings)
    reads = 0
    val onPair =
      KevTimingCore(pipeline, null, clock, ceiling) { false }.timePairSet(rows.sets.single(), pair)
    assertEquals(20, (onPair["request_starts_ms"] as List<*>).size)
    assertEquals(20, (onPair["request_calls_ms"] as List<*>).size)
    assertEquals(40, (onPair["question_starts_ms"] as List<*>).size)
    assertEquals(40, (onPair["question_calls_ms"] as List<*>).size)
    assertEquals(5, (onPair["warmup_starts_ms"] as List<*>).size)
    // Two reads per request (before and after): the warm-ups take reads 1-10, so the third timed
    // request reads 1,300 before it and 578 after it, the fourth 578 before it.
    val before = onPair["request_max_clock_mhz"] as List<*>
    val after = onPair["request_max_clock_mhz_after"] as List<*>
    assertEquals(listOf(1300, 1300, 1300, 578), before.take(4))
    assertEquals(listOf(1300, 1300, 578, 578), after.take(4))
    assertEquals(20, before.size)
    // Without a reader, the ceilings are null.
    val blind = KevTimingCore(pipeline, row, clock) { false }.timeSet(rows.sets.single())
    assertEquals(List(40) { null }, blind["call_max_clock_mhz"])
  }
}
