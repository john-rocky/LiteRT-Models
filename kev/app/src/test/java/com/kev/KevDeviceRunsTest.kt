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
    KevGateChecks.parseAsset(
      ExternalTestData.moduleFile("app/src/debug/assets/gate_fixtures.json").readBytes()
    )
  }

  /** Returns the oracle's hidden states at the readout positions of whichever asset row it gets. */
  private class OracleGraph(
    override val length: Int,
    private val rows: Map<String, Pair<IntArray, FloatArray>>,
  ) : RowRunner {
    var calls = 0

    override fun run(ids: IntArray, valid: FloatArray): FloatArray {
      calls++
      val real = valid.count { it == 1f }
      val (positions, values) =
        requireNotNull(rows[KevPipeline.idsSha256(ids.copyOf(real))]) { "unknown row" }
      val hidden = FloatArray(length * HIDDEN)
      for ((row, position) in positions.withIndex()) System.arraycopy(
        values,
        row * HIDDEN,
        hidden,
        position * HIDDEN,
        HIDDEN,
      )
      return hidden
    }
  }

  /**
   * A pair stand-in that returns the oracle's hidden states at the branch's readout positions of
   * whichever asset request (state + branch) it gets; counts the state and question calls.
   */
  private class OraclePair(private val rows: Map<String, Pair<IntArray, FloatArray>>) : PairRunner {
    override val stateLength = 128
    override val questionLength = 64
    var stateCalls = 0
    var questionCalls = 0
    private var state = IntArray(0)

    override fun runState(ids: IntArray, valid: FloatArray) {
      stateCalls++
      state = ids.copyOf(valid.count { it == 1f })
    }

    override fun runQuestion(ids: IntArray, valid: FloatArray): FloatArray {
      questionCalls++
      val branch = ids.copyOf(valid.count { it == 1f })
      val (positions, values) =
        requireNotNull(rows[KevPipeline.idsSha256(state + branch)]) { "unknown row" }
      val hidden = FloatArray(questionLength * HIDDEN)
      for ((row, position) in positions.withIndex()) {
        System.arraycopy(values, row * HIDDEN, hidden, (position - state.size) * HIDDEN, HIDDEN)
      }
      return hidden
    }
  }

  private fun oracleRows(): Map<String, Pair<IntArray, FloatArray>> {
    val rows = HashMap<String, Pair<IntArray, FloatArray>>()
    OracleFixtures.Npz(ExternalTestData.file(ExternalTestData.HIDDEN)).use { npz ->
      for (item in asset) {
        for (question in item.questions) {
          rows[KevPipeline.idsSha256(question.rowIds)] =
            (intArrayOf(question.decideIndex) + question.optionIndices) to
              npz.floats("${item.id}/${question.qid}").second
        }
      }
    }
    return rows
  }

  private fun oracleGraph(window: Int): OracleGraph {
    val rows = HashMap<String, Pair<IntArray, FloatArray>>()
    OracleFixtures.Npz(ExternalTestData.file(ExternalTestData.HIDDEN)).use { npz ->
      for (item in asset) {
        for (question in item.questions) {
          rows[KevPipeline.idsSha256(question.rowIds)] =
            (intArrayOf(question.decideIndex) + question.optionIndices) to
              npz.floats("${item.id}/${question.qid}").second
        }
      }
    }
    return OracleGraph(window, rows)
  }

  private fun pipeline() =
    KevPipeline(
      ExternalTestData.tokenizer(),
      KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)),
    )

  private fun probes() =
    KevGateChecks.parseProbes(
      ExternalTestData.moduleFile("app/src/debug/assets/tokenizer_probes.json").readBytes()
    )

  @Test
  fun gatePassesWithTheOracleGraph() {
    val graph = oracleGraph(512)
    val gate = KevGateCore(pipeline(), graph, 0, { false }, {})
    val probes = gate.probes(probes())
    gate.run(asset) {}
    val summary = gate.summary()
    ExternalTestData.writeReport("gate_core.json", summary + ("tokenizer_probes" to probes))
    println(
      "KEV_GATE_CORE ${KevJson.write(summary.filterKeys { it !in setOf("skipped_rows", "infer_ms", "head_ms") })}"
    )
    assertEquals(54, probes["raw_equal"])
    assertEquals(54, probes["user_equal"])
    assertEquals(156, summary["input_tokens_identical"])
    assertEquals(181, summary["ids_identical"])
    assertEquals(181, summary["indices_identical"])
    assertEquals(172, summary["rows_run"])
    assertEquals(9, summary["rows_skipped"])
    assertTrue(
      (summary["skipped_rows"] as List<*>).all { (it as Map<*, *>)["reason"] == "needs L2048" }
    )
    assertEquals(172, summary["argmax_equal"])
    assertEquals(172, summary["answers_equal_oracle"])
    assertEquals(0, summary["nonfinite_rows"])
    assertTrue(summary["max_abs_dp"] as Double <= 1e-5)
    assertEquals(172, graph.calls)
    assertTrue(gate.passed())
    // Every asset question has an entry; run rows carry the probabilities and times.
    assertEquals(181, gate.rows.size)
    val run = gate.rows.map { it as Map<*, *> }.filter { it["status"] == "run" }
    assertTrue(
      run.all {
        it.containsKey("probs") && it.containsKey("infer_ms") && it.containsKey("ids_sha256")
      }
    )
  }

  @Test
  fun gateRunsTheRowsTheWindowHolds() {
    // L256 holds every row L512 holds (the longest of them is 142 tokens); L128 the 147 rows of
    // 128 tokens or fewer. Longer rows are skipped with the window that would hold them.
    for ((window, run, skipped) in
      listOf(
        Triple(256, 172, mapOf("needs L2048" to 9)),
        Triple(128, 147, mapOf("needs L256" to 25, "needs L2048" to 9)),
      )) {
      val graph = oracleGraph(window)
      val gate = KevGateCore(pipeline(), graph, 0, { false }, {})
      gate.probes(probes())
      gate.run(asset) {}
      val summary = gate.summary()
      assertEquals("L$window", run, summary["rows_run"])
      assertEquals("L$window", run, graph.calls)
      assertEquals(
        "L$window",
        skipped,
        (summary["skipped_rows"] as List<*>).groupingBy { (it as Map<*, *>)["reason"] }.eachCount(),
      )
      assertEquals(run, summary["argmax_equal"])
      assertTrue("L$window", gate.passed())
    }
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

        override fun run(ids: IntArray, valid: FloatArray) =
          FloatArray(length * HIDDEN) { Float.NaN }
      }
    val broken = KevGateCore(pipeline(), nan, 3, { false }, {})
    broken.run(asset) {}
    assertEquals(3, broken.summary()["nonfinite_rows"])
    assertFalse(broken.passed())
  }

  @Test
  fun pairGateRunsTheRequestsThePairTakes() {
    // The pair takes 124 of the asset's 156 requests (132 questions): 24 have a branch over 64
    // tokens, 8 a state over 128. Each request runs its state once.
    val pair = OraclePair(oracleRows())
    val gate = KevGateCore(pipeline(), KevGateGraph.Pair(pair), 0, { false }, {})
    val probes = gate.probes(probes())
    gate.run(asset) {}
    val summary = gate.summary()
    ExternalTestData.writeReport("gate_core_pair.json", summary + ("tokenizer_probes" to probes))
    println(
      "KEV_GATE_CORE_PAIR ${KevJson.write(summary.filterKeys { it !in setOf("skipped_rows", "not_fitting", "infer_ms", "head_ms", "state_ms") })}"
    )
    assertEquals("pair", summary["form"])
    assertEquals(181, summary["ids_identical"])
    assertEquals(132, summary["rows_run"])
    assertEquals(49, summary["rows_skipped"])
    assertEquals(124, summary["requests_run"])
    assertEquals(32, summary["requests_not_fitting"])
    val reasons =
      (summary["not_fitting"] as List<*>)
        .map { ((it as Map<*, *>)["reason"] as String).substringBefore(" ") }
        .groupingBy { it }
        .eachCount()
    assertEquals(mapOf("branch" to 24, "state" to 8), reasons)
    assertEquals(124, pair.stateCalls)
    assertEquals(132, pair.questionCalls)
    assertEquals(132, summary["argmax_equal"])
    assertEquals(132, summary["answers_equal_oracle"])
    assertTrue(summary["max_abs_dp"] as Double <= 1e-5)
    assertTrue(gate.passed())
    val run = gate.rows.map { it as Map<*, *> }.filter { it["status"] == "run" }
    assertTrue(run.all { it["form"] == "pair" && it["window"] == 64 && it.containsKey("state_ms") })
    // A limit counts questions; the request it ends in keeps its state call.
    val limitedPair = OraclePair(oracleRows())
    val limited = KevGateCore(pipeline(), KevGateGraph.Pair(limitedPair), 40, { false }, {})
    limited.probes(probes())
    limited.run(asset) {}
    assertEquals(40, limited.summary()["rows_run"])
    assertEquals(40, limitedPair.questionCalls)
    assertTrue(limited.passed())
  }

  @Test
  fun pairTimingMakesTheProtocolsCalls() {
    var states = 0
    var questions = 0
    val zeros =
      object : PairRunner {
        override val stateLength = 128
        override val questionLength = 64

        override fun runState(ids: IntArray, valid: FloatArray) {
          assertEquals(stateLength, ids.size)
          states++
        }

        override fun runQuestion(ids: IntArray, valid: FloatArray): FloatArray {
          assertEquals(questionLength, ids.size)
          questions++
          return FloatArray(questionLength * HIDDEN)
        }
      }
    val rows = KevTimingRows.parse(ExternalTestData.file("device/timing_rows.json").readBytes())
    val shape = KevPairShape(128, 64)
    // The cut T300 / T1000 rows have no question token; fiveq is one request of five branches.
    val selection = rows.selectPair(null, shape)
    assertEquals(listOf("fiveq"), selection.run.map { it.name })
    assertEquals(
      listOf("T300", "T1000"),
      selection.skipped.map { it.name },
    )
    assertEquals(
      listOf("fiveq"),
      rows.selectPair(KevLaunch.setNames("fiveq"), shape).run.map { it.name },
    )
    val timing = KevTimingCore(pipeline(), null) { false }
    val fiveq = timing.timePairSet(rows.sets[0], zeros)
    assertEquals(5, (fiveq["warmup_request_ms"] as List<*>).size)
    assertEquals(20, (fiveq["request_ms"] as Map<*, *>)["n"])
    assertEquals(20, (fiveq["state_ms"] as Map<*, *>)["n"])
    assertEquals(100, (fiveq["question_ms"] as Map<*, *>)["n"])
    assertEquals(
      listOf(linkedMapOf("state_len" to 99, "branch_lens" to listOf(33, 43, 33, 29, 29))),
      fiveq["requests"],
    )
    assertEquals(true, fiveq["finite"])
    assertEquals(25, states)
    assertEquals(125, questions)
    // The request path on the pair: the state once per request.
    val item = asset.first { it.id == KevTimingCore.REQUEST_PATH_RECORD }
    val path =
      timing.timeRequestPath(item.id, KevRequest.fromJson(item.request), KevRunners.Pair(zeros))
    assertEquals("pair", path["form"])
    assertEquals(20, (path["request_ms"] as Map<*, *>)["n"])
    assertEquals(20, (path["state_ms"] as Map<*, *>)["n"])
    assertEquals(100, (path["infer_ms_per_call"] as Map<*, *>)["n"])
    assertEquals(99, path["state_len"])
    assertEquals(50, states)
    assertEquals(250, questions)
  }

  @Test
  fun timingSelectsTheRequestedSets() {
    val rows = KevTimingRows.parse(ExternalTestData.file("device/timing_rows.json").readBytes())
    assertEquals(listOf("fiveq", "T300", "T1000"), rows.sets.map { it.name })
    assertEquals(listOf("fiveq", "T300"), rows.select(null, 512).run.map { it.name })
    assertEquals(
      listOf("T1000" to "the resident graph is L512"),
      rows.select(null, 512).skipped.map { it.name to it.reason },
    )
    val t300 = rows.select(KevLaunch.setNames("T300"), 512)
    assertEquals(listOf("T300"), t300.run.map { it.name })
    assertEquals(
      listOf("fiveq" to "not requested", "T1000" to "not requested"),
      t300.skipped.map { it.name to it.reason },
    )
    assertEquals(
      listOf("T1000"),
      rows.select(KevLaunch.setNames("T1000"), 1024).run.map { it.name },
    )
    assertTrue(rows.select(KevLaunch.setNames("none"), 512).run.isEmpty())
    assertEquals(
      listOf("fiveq", "T300"),
      rows.select(KevLaunch.setNames(" fiveq,T300 ,"), 512).run.map { it.name },
    )
    val unknown = rows.select(KevLaunch.setNames("T300,T999"), 512)
    assertEquals(listOf("T300"), unknown.run.map { it.name })
    assertEquals(
      KevTimingSkip("T999", null, "not in the rows file").reason,
      unknown.skipped.last().reason,
    )
    assertNull(unknown.skipped.last().window)
  }

  @Test
  fun timingSelectsTheSetOfTheWindow() {
    // One rows file can carry a set name for several windows; a launch times the one of its window.
    val rows =
      KevTimingRows.parse(
        ("""{"pad_id": 248044, "sets": [""" +
            """{"name": "fiveq", "kind": "request", "L": 256, "rows": [{"key": "a", "ids": [1, 2]}]},""" +
            """{"name": "fiveq", "kind": "request", "L": 512, "rows": [{"key": "a", "ids": [1, 2]}]},""" +
            """{"name": "S80", "kind": "single", "L": 128, "rows": [{"key": "b", "ids": [3]}]}]}""")
          .toByteArray()
      )
    val l256 = rows.select(KevLaunch.setNames("fiveq"), 256)
    assertEquals(listOf(256), l256.run.map { it.window })
    assertEquals(
      listOf("fiveq" to "the resident graph is L256", "S80" to "not requested"),
      l256.skipped.map { it.name to it.reason },
    )
    assertEquals(listOf("S80"), rows.select(null, 128).run.map { it.name })
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
