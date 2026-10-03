package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The counting of the on-device gate and timing runs, with stand-in graphs: the gate core over the
 * bundled asset with a graph that returns the oracle's hidden states must pass with every row
 * identical, and its limit and stop file must cut the run; the timing core must make the protocol's
 * numbers of calls.
 */
class KevDeviceRunsTest {
  private val asset by lazy {
    KevGateChecks.parseAsset(ExternalTestData.moduleFile("app/src/debug/assets/gate_fixtures.json").readBytes())
  }

  /** Returns the oracle's hidden states at the readout positions of whichever asset row it gets. */
  private class OracleGraph(override val length: Int, private val rows: Map<String, Pair<IntArray, FloatArray>>) : RowRunner {
    var calls = 0

    override fun run(ids: IntArray, valid: FloatArray): FloatArray {
      calls++
      val real = valid.count { it == 1f }
      val (positions, values) = requireNotNull(rows[KevPipeline.idsSha256(ids.copyOf(real))]) { "unknown row" }
      val hidden = FloatArray(length * HIDDEN)
      for ((row, position) in positions.withIndex()) System.arraycopy(values, row * HIDDEN, hidden, position * HIDDEN, HIDDEN)
      return hidden
    }
  }

  private fun oracleGraph(window: Int): OracleGraph {
    val rows = HashMap<String, Pair<IntArray, FloatArray>>()
    OracleFixtures.Npz(ExternalTestData.file(ExternalTestData.HIDDEN)).use { npz ->
      for (item in asset) {
        for (question in item.questions) {
          rows[KevPipeline.idsSha256(question.rowIds)] =
            (intArrayOf(question.decideIndex) + question.optionIndices) to npz.floats("${item.id}/${question.qid}").second
        }
      }
    }
    return OracleGraph(window, rows)
  }

  private fun pipeline() = KevPipeline(ExternalTestData.tokenizer(), KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)))

  private fun probes() = KevGateChecks.parseProbes(ExternalTestData.moduleFile("app/src/debug/assets/tokenizer_probes.json").readBytes())

  @Test
  fun gatePassesWithTheOracleGraph() {
    val graph = oracleGraph(512)
    val gate = KevGateCore(pipeline(), graph, 0, { false }, {})
    val probes = gate.probes(probes())
    gate.run(asset) {}
    val summary = gate.summary()
    ExternalTestData.writeReport("gate_core.json", summary + ("tokenizer_probes" to probes))
    println("KEV_GATE_CORE ${KevJson.write(summary.filterKeys { it !in setOf("skipped_rows", "infer_ms", "head_ms") })}")
    assertEquals(54, probes["raw_equal"])
    assertEquals(54, probes["user_equal"])
    assertEquals(156, summary["input_tokens_identical"])
    assertEquals(181, summary["ids_identical"])
    assertEquals(181, summary["indices_identical"])
    assertEquals(172, summary["rows_run"])
    assertEquals(9, summary["rows_skipped"])
    assertTrue((summary["skipped_rows"] as List<*>).all { (it as Map<*, *>)["reason"] == "needs L2048" })
    assertEquals(172, summary["argmax_equal"])
    assertEquals(172, summary["answers_equal_oracle"])
    assertEquals(0, summary["nonfinite_rows"])
    assertTrue(summary["max_abs_dp"] as Double <= 1e-5)
    assertEquals(172, graph.calls)
    assertTrue(gate.passed())
    // Every asset question has an entry; run rows carry the probabilities and times.
    assertEquals(181, gate.rows.size)
    val run = gate.rows.map { it as Map<*, *> }.filter { it["status"] == "run" }
    assertTrue(run.all { it.containsKey("probs") && it.containsKey("infer_ms") && it.containsKey("ids_sha256") })
  }

  @Test
  fun gateLimitAndStopCutTheRun() {
    val limited = KevGateCore(pipeline(), oracleGraph(512), 20, { false }, {})
    limited.run(asset) {}
    assertEquals(20, limited.summary()["rows_run"])
    assertEquals(161, limited.summary()["rows_skipped"])
    assertFalse(limited.stoppedEarly)
    val graph = oracleGraph(512)
    val stopped = KevGateCore(pipeline(), graph, 0, { graph.calls >= 10 }, {})
    stopped.run(asset) {}
    assertTrue(stopped.stoppedEarly)
    assertEquals(10, stopped.summary()["rows_run"])
    assertFalse(stopped.passed())
    // A NaN graph fails the gate without throwing.
    val nan =
      object : RowRunner {
        override val length = 512

        override fun run(ids: IntArray, valid: FloatArray) = FloatArray(length * HIDDEN) { Float.NaN }
      }
    val broken = KevGateCore(pipeline(), nan, 3, { false }, {})
    broken.run(asset) {}
    assertEquals(3, broken.summary()["nonfinite_rows"])
    assertFalse(broken.passed())
  }

  @Test
  fun timingSelectsTheRequestedSets() {
    val rows = KevTimingRows.parse(ExternalTestData.file("device/timing_rows.json").readBytes())
    assertEquals(listOf("fiveq", "T300", "T1000"), rows.sets.map { it.name })
    assertEquals(listOf("fiveq", "T300"), rows.select(null, 512).run.map { it.name })
    assertEquals(listOf("T1000" to "the resident graph is L512"), rows.select(null, 512).skipped.map { it.name to it.reason })
    val t300 = rows.select(KevLaunch.setNames("T300"), 512)
    assertEquals(listOf("T300"), t300.run.map { it.name })
    assertEquals(listOf("fiveq" to "not requested", "T1000" to "not requested"), t300.skipped.map { it.name to it.reason })
    assertEquals(listOf("T1000"), rows.select(KevLaunch.setNames("T1000"), 1024).run.map { it.name })
    assertTrue(rows.select(KevLaunch.setNames("none"), 512).run.isEmpty())
    assertEquals(listOf("fiveq", "T300"), rows.select(KevLaunch.setNames(" fiveq,T300 ,"), 512).run.map { it.name })
    val unknown = rows.select(KevLaunch.setNames("T300,T999"), 512)
    assertEquals(listOf("T300"), unknown.run.map { it.name })
    assertEquals(KevTimingSkip("T999", null, "not in the rows file").reason, unknown.skipped.last().reason)
    assertNull(unknown.skipped.last().window)
  }

  @Test
  fun timingMakesTheProtocolsCalls() {
    var calls = 0
    val zeros =
      object : RowRunner {
        override val length = 512

        override fun run(ids: IntArray, valid: FloatArray): FloatArray {
          calls++
          return FloatArray(length * HIDDEN)
        }
      }
    val rows = KevTimingRows.parse(ExternalTestData.file("device/timing_rows.json").readBytes())
    val timing = KevTimingCore(pipeline(), zeros) { false }
    val fiveq = timing.timeSet(rows.sets[0])
    assertEquals(5, (fiveq["warmup_ms"] as List<*>).size)
    assertEquals(100, (fiveq["per_call_ms"] as Map<*, *>)["n"])
    assertEquals(20, (fiveq["per_request_ms"] as Map<*, *>)["n"])
    assertEquals(true, fiveq["finite"])
    val single = timing.timeSet(rows.sets[1])
    assertEquals(20, (single["per_call_ms"] as Map<*, *>)["n"])
    assertNull(single["per_request_ms"])
    assertEquals(105 + 25, calls)
    assertTrue(runCatching { timing.timeSet(rows.sets[2]) }.isFailure)
    val item = asset.first { it.id == KevTimingCore.REQUEST_PATH_RECORD }
    val path = timing.timeRequestPath(item.id, KevRequest.fromJson(item.request))
    assertEquals(5, (path["warmup_ms"] as List<*>).size)
    assertEquals(20, (path["request_ms"] as Map<*, *>)["n"])
    assertEquals(100, (path["infer_ms_per_call"] as Map<*, *>)["n"])
    assertEquals(listOf(132, 142, 132, 128, 128), path["row_lens"])
    assertEquals(130 + 125, calls)
    // The stop file ends the set after the call that saw it.
    var stopAfter = 0
    val stopping = KevTimingCore(pipeline(), zeros) { ++stopAfter >= 7 }
    val cut = stopping.timeSet(rows.sets[0])
    assertTrue(stopping.stoppedEarly)
    assertEquals(5, (cut["warmup_ms"] as List<*>).size)
    assertEquals(2, (cut["per_call_ms"] as Map<*, *>)["n"])
    assertNull((cut["per_request_ms"] as Map<*, *>?))
  }

  private companion object {
    const val HIDDEN = KevPointerHead.HIDDEN_SIZE
  }
}
