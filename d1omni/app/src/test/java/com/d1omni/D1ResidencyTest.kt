package com.d1omni

import java.io.Closeable
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The resident decision graphs: at most two, a request the resident graphs hold compiles nothing,
 * the largest bucket first, the memory read before a second compile and the second graph given up
 * under 2,500,000,000 bytes (every row then on the largest bucket).
 */
class D1ResidencyTest {
  private class FakeGraph(val bucket: Int, val log: MutableList<String>) : Closeable {
    override fun close() {
      log.add("close $bucket")
    }
  }

  private val installed = listOf(128, 256, 512)
  private val plenty = 6_000_000_000L
  private val short = 2_000_000_000L

  private fun residency(@Suppress("UNUSED_PARAMETER") log: MutableList<String>) = D1Residency<FakeGraph>()

  private fun opener(log: MutableList<String>): (Int) -> FakeGraph = { bucket ->
    log.add("open $bucket")
    FakeGraph(bucket, log)
  }

  @Test
  fun largestFirstThenTheSecondWithMemory() {
    val log = ArrayList<String>()
    val graphs = residency(log)
    val readings = ArrayList<String>()
    val prepared =
      graphs.prepare(listOf(100, 200, 90), installed, { readings.add("read"); plenty }, opener(log))
    assertEquals(listOf("open 256", "open 128"), log)
    assertEquals(listOf(128, 256), graphs.resident)
    assertEquals(listOf(128, 256, 128), prepared.rowBuckets)
    assertEquals(listOf(plenty), prepared.availableBeforeSecond)
    assertFalse(prepared.secondRefused)
    assertEquals(1, readings.size)
    // The same buckets again: nothing compiles, no memory read.
    log.clear()
    val again = graphs.prepare(listOf(50, 250), installed, { fail("read"); 0 }, opener(log))
    assertTrue(log.isEmpty())
    assertEquals(listOf(128, 256), again.rowBuckets)
  }

  @Test
  fun shortMemoryKeepsOnlyTheLargest() {
    val log = ArrayList<String>()
    val graphs = residency(log)
    val prepared = graphs.prepare(listOf(100, 200), installed, { short }, opener(log))
    assertEquals(listOf("open 256"), log)
    assertEquals(listOf(256, 256), prepared.rowBuckets)
    assertTrue(prepared.secondRefused)
  }

  @Test
  fun aLargerGraphNextToAResidentOne() {
    val log = ArrayList<String>()
    val graphs = residency(log)
    graphs.prepare(listOf(100), installed, { fail("read"); 0 }, opener(log))
    assertEquals(listOf("open 128"), log)
    // L256 compiles next to L128 when there is memory (the round's memory leg).
    val both = graphs.prepare(listOf(200), installed, { plenty }, opener(log))
    assertEquals(listOf("open 128", "open 256"), log)
    assertEquals(listOf(128, 256), graphs.resident)
    assertEquals(listOf(256), both.rowBuckets)
    // A third bucket: the request's own buckets stay, the unused one is closed for the slot.
    log.clear()
    val third = graphs.prepare(listOf(400, 200), installed, { plenty }, opener(log))
    assertEquals(listOf("close 128", "open 512"), log)
    assertEquals(listOf(256, 512), graphs.resident)
    assertEquals(listOf(512, 256), third.rowBuckets)
  }

  @Test
  fun aLargerGraphWithoutMemoryRunsAlone() {
    val log = ArrayList<String>()
    val graphs = residency(log)
    graphs.prepare(listOf(100), installed, { 0 }, opener(log))
    val alone = graphs.prepare(listOf(200, 100), installed, { short }, opener(log))
    assertEquals(listOf("open 128", "close 128", "open 256"), log)
    assertEquals(listOf(256), graphs.resident)
    assertEquals(listOf(256, 256), alone.rowBuckets)
    assertTrue(alone.secondRefused)
    assertEquals(listOf(128), alone.closed)
  }

  @Test
  fun ensureNamesTheGraphs() {
    val log = ArrayList<String>()
    val graphs = residency(log)
    graphs.ensure(listOf(128), emptyList(), { fail("read"); 0 }, opener(log))
    val second = graphs.ensure(listOf(256), listOf(10, 200), { plenty }, opener(log))
    assertEquals(listOf("open 128", "open 256"), log)
    assertEquals(listOf(128, 256), second.rowBuckets)
    graphs.close()
    assertEquals(listOf("open 128", "open 256", "close 128", "close 256"), log)
    assertTrue(graphs.resident.isEmpty())
  }

  @Test
  fun aRowLongerThanEveryInstalledGraphIsRejected() {
    val log = ArrayList<String>()
    try {
      residency(log).prepare(listOf(600), installed, { plenty }, opener(log))
      fail("a 600-position row was accepted with L512 the largest")
    } catch (expected: IllegalArgumentException) {
      assertTrue(expected.message!!, expected.message!!.contains("600"))
    }
    assertTrue(log.isEmpty())
  }
}
