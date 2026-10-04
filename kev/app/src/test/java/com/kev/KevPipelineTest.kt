package com.kev

import java.io.Closeable
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The app's decision path end to end with fake graphs: every oracle request goes through
 * [KevPipeline] (request → rows → the graphs [KevResidentGraphs] plans for the installed windows →
 * padded inputs → [RowRunner] → readout rows → head → `to_answers`), where each fake graph checks
 * the padded inputs and places the oracle's hidden states (`hidden_0.8b.npz`) at the decide and
 * option positions. All 402 answers must equal the oracle's, and all 377 `usage.input_tokens`,
 * whichever windows are installed and whether the memory allows a second graph or not: the window
 * changes the padding, not the answer. The shared-state pair path ([PairRunner]: the state once,
 * then each branch) must give the row path's answers on every request the pair takes.
 */
class KevPipelineTest {
  /**
   * A graph stand-in of window [length] that knows the oracle rows by the sha256 of their IDs and
   * returns each row's hidden states at its readout positions in an otherwise zero `hidden`.
   */
  private class OracleGraph(
    override val length: Int,
    private val rows: Map<String, Pair<IntArray, FloatArray>>,
  ) : RowRunner, Closeable {
    var inputsChecked = 0

    override fun run(ids: IntArray, valid: FloatArray): FloatArray {
      assertEquals(length, ids.size)
      assertEquals(length, valid.size)
      val real = valid.count { it == 1f }
      for (index in 0 until length) {
        val isReal = index < real
        assertEquals(if (isReal) 1f else 0f, valid[index])
        if (!isReal) assertEquals(KevEncoder.PAD_ID, ids[index])
      }
      val (positions, values) =
        requireNotNull(rows[KevPipeline.idsSha256(ids.copyOf(real))]) { "unknown row" }
      inputsChecked++
      val hidden = FloatArray(length * HIDDEN)
      for ((row, position) in positions.withIndex()) {
        System.arraycopy(values, row * HIDDEN, hidden, position * HIDDEN, HIDDEN)
      }
      return hidden
    }

    override fun close() = Unit
  }

  /**
   * A pair stand-in: [runState] keeps the state's real IDs; [runQuestion] finds the oracle row
   * state + branch by its sha256 and returns its hidden states at the branch's readout positions.
   * It checks the padding of both calls and records every state and branch it gets.
   */
  private class OraclePair(
    override val stateLength: Int,
    override val questionLength: Int,
    private val rows: Map<String, Pair<IntArray, FloatArray>>,
  ) : PairRunner, Closeable {
    private var state: IntArray? = null
    val states = ArrayList<IntArray>()
    val branches = ArrayList<IntArray>()

    override fun runState(ids: IntArray, valid: FloatArray) {
      state = padded(ids, valid, stateLength).also { states.add(it) }
    }

    override fun runQuestion(ids: IntArray, valid: FloatArray): FloatArray {
      val branch = padded(ids, valid, questionLength).also { branches.add(it) }
      val prefix = requireNotNull(state) { "a question before the state" }
      val (positions, values) =
        requireNotNull(rows[KevPipeline.idsSha256(prefix + branch)]) { "unknown row" }
      val hidden = FloatArray(questionLength * HIDDEN)
      for ((row, position) in positions.withIndex()) {
        System.arraycopy(values, row * HIDDEN, hidden, (position - prefix.size) * HIDDEN, HIDDEN)
      }
      return hidden
    }

    /** The real IDs of a call padded to [length], after checking the pads and the mask. */
    private fun padded(ids: IntArray, valid: FloatArray, length: Int): IntArray {
      assertEquals(length, ids.size)
      assertEquals(length, valid.size)
      val real = valid.count { it == 1f }
      for (index in 0 until length) {
        assertEquals(if (index < real) 1f else 0f, valid[index])
        if (index >= real) assertEquals(KevEncoder.PAD_ID, ids[index])
      }
      return ids.copyOf(real)
    }

    override fun close() = Unit
  }

  /** What one pass over the oracle requests with [installed] windows gave. */
  private class OracleRun(
    val answersEqual: Int,
    val inputsChecked: Int,
    val usageEqual: Int,
    val windows: Map<Int, Int>,
    val smallestWindow: Int,
    val compiles: List<Int>,
    val failures: List<String>,
    val ours: Map<String, Any?>,
  )

  private fun runOracle(installed: List<Int>, availableBytes: Long): OracleRun {
    val pipeline =
      KevPipeline(
        ExternalTestData.tokenizer(),
        KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)),
      )
    val oracle = OracleFixtures.loadOracle()
    val byKey = oracle.questions.associateBy { it.key }
    val usage = oracle.requests.associateBy { it.id }
    val rows = HashMap<String, Pair<IntArray, FloatArray>>()
    OracleFixtures.Npz(ExternalTestData.file(ExternalTestData.HIDDEN)).use { npz ->
      for (question in oracle.questions) {
        rows[KevPipeline.idsSha256(question.rowIds)] =
          (intArrayOf(question.decideIndex) + question.optionIndices) to
            npz.floats(question.key).second
      }
    }
    val opened = ArrayList<OracleGraph>()
    val compiles = ArrayList<Int>()
    fun open(window: Int) = OracleGraph(window, rows).also { opened.add(it) }
    val graphs = KevResidentGraphs<OracleGraph, OraclePair>()
    var answersEqual = 0
    var usageEqual = 0
    var smallestWindow = 0
    val windows = linkedMapOf(*KevFiles.WINDOWS.map { it to 0 }.toTypedArray())
    val failures = ArrayList<String>()
    // This path's answers and rows per question, for recounting independently.
    val ours = LinkedHashMap<String, Any?>()
    for (record in OracleFixtures.loadRecords()) {
      val prepared = pipeline.prepare(KevRequest.fromJson(record.request))
      if (prepared.inputTokens == usage.getValue(record.id).inputTokens) usageEqual++
      assertEquals(prepared.tokenizeMs, prepared.stateMs + prepared.branchMs.sum(), 1e-9)
      val plan = graphs.plan(prepared.rowLengths, installed) as KevWindowPlan.Ready
      val run = graphs.prepare(plan, { availableBytes }, ::open)
      compiles.addAll(run.compiled)
      assertTrue(graphs.windows.size <= KevResidentGraphs.MAX_RESIDENT)
      val results = ArrayList<KevQuestionResult>()
      for ((index, meta) in prepared.meta.withIndex()) {
        val expected = byKey.getValue("${record.id}/${meta.id}")
        val row = prepared.rows[index]
        assertArrayEquals(expected.key, expected.rowIds, row.ids)
        val result = pipeline.run(prepared, index, run.graphs[index])
        assertEquals(run.windows[index], result.window)
        if (result.window == row.window(installed)) smallestWindow++
        windows[result.window] = windows.getValue(result.window) + 1
        assertTrue(result.inferMs >= 0 && result.headMs >= 0)
        results.add(result)
      }
      val answers = pipeline.answers(prepared, results)
      for ((index, meta) in prepared.meta.withIndex()) {
        val key = "${record.id}/${meta.id}"
        ours[key] =
          linkedMapOf(
            "row_len" to prepared.rows[index].length,
            "window" to results[index].window,
            "ids_sha256" to KevPipeline.idsSha256(prepared.rows[index].ids),
            "answer" to answers[meta.id],
          )
        val difference = OracleFixtures.jsonDifference(byKey.getValue(key).answer, answers[meta.id])
        if (difference == null) answersEqual++
        else if (failures.size < 3) failures.add("$key $difference")
      }
      val response = pipeline.response(prepared, answers)
      assertEquals(listOf("model", "answers", "usage"), response.keys.toList())
      assertEquals(prepared.inputTokens, (response["usage"] as Map<*, *>)["input_tokens"])
    }
    graphs.close()
    return OracleRun(
      answersEqual,
      opened.sumOf { it.inputsChecked },
      usageEqual,
      windows,
      smallestWindow,
      compiles,
      failures,
      ours,
    )
  }

  private fun report(name: String, installed: List<Int>, availableBytes: Long, run: OracleRun) {
    ExternalTestData.writeReport("${name}_answers_ours.json", run.ours)
    ExternalTestData.writeReport(
      "$name.json",
      linkedMapOf(
        "test" to "KevPipelineTest",
        "installed_windows" to installed,
        "available_bytes" to availableBytes,
        "questions" to run.ours.size,
        "answers_equal" to run.answersEqual,
        "padded_inputs_checked" to run.inputsChecked,
        "input_tokens_equal" to run.usageEqual,
        "windows" to run.windows.mapKeys { it.key.toString() },
        "on_smallest_installed_window" to run.smallestWindow,
        "graph_compiles" to run.compiles,
        "failures" to run.failures,
        "execution" to
          "Desktop JVM ${System.getProperty("java.version")}, fake graphs with the oracle's h_sel",
      ),
    )
    println(
      "KEV_PIPELINE installed=$installed available_bytes=$availableBytes answers_equal=${run.answersEqual}/402 inputs_checked=${run.inputsChecked} usage=${run.usageEqual}/377 windows=${run.windows} smallest=${run.smallestWindow} compiles=${run.compiles.size}"
    )
  }

  @Test
  fun oracleRequestsGiveTheOracleAnswers() {
    val installed = listOf(512, 1024, 2048)
    val run = runOracle(installed, PLENTY)
    report("pipeline", installed, PLENTY, run)
    assertEquals(run.failures.joinToString("\n"), 402, run.answersEqual)
    assertEquals(402, run.inputsChecked)
    assertEquals(377, run.usageEqual)
    assertEquals(mapOf(128 to 0, 256 to 0, 512 to 393, 1024 to 0, 2048 to 9), run.windows)
    assertEquals(402, run.smallestWindow)
  }

  @Test
  fun everyPublishedWindowGivesTheSameAnswers() {
    val installed = KevFiles.WINDOWS
    // Enough memory for a second graph: every question runs on its smallest installed window (no
    // oracle request mixes rows of L256 or less with longer ones).
    val two = runOracle(installed, PLENTY)
    report("pipeline_5w", installed, PLENTY, two)
    assertEquals(two.failures.joinToString("\n"), 402, two.answersEqual)
    assertEquals(402, two.inputsChecked)
    assertEquals(377, two.usageEqual)
    assertEquals(mapOf(128 to 322, 256 to 64, 512 to 7, 1024 to 0, 2048 to 9), two.windows)
    assertEquals(402, two.smallestWindow)
    // Too little memory for a second graph: in the two requests with rows for both L128 and L256,
    // the rows of 128 tokens or fewer run on L256 too.
    val low = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1
    val one = runOracle(installed, low)
    report("pipeline_5w_low_memory", installed, low, one)
    assertEquals(one.failures.joinToString("\n"), 402, one.answersEqual)
    assertEquals(402, one.inputsChecked)
    assertEquals(377, one.usageEqual)
    assertEquals(mapOf(128 to 319, 256 to 67, 512 to 7, 1024 to 0, 2048 to 9), one.windows)
    assertEquals(399, one.smallestWindow)
  }

  @Test
  fun thePairGivesTheRowAnswersOnEveryRequestItTakes() {
    val pipeline =
      KevPipeline(
        ExternalTestData.tokenizer(),
        KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)),
      )
    val oracle = OracleFixtures.loadOracle()
    val byKey = oracle.questions.associateBy { it.key }
    val rows = HashMap<String, Pair<IntArray, FloatArray>>()
    OracleFixtures.Npz(ExternalTestData.file(ExternalTestData.HIDDEN)).use { npz ->
      for (question in oracle.questions) {
        rows[KevPipeline.idsSha256(question.rowIds)] =
          (intArrayOf(question.decideIndex) + question.optionIndices) to
            npz.floats(question.key).second
      }
    }
    val shape = KevFiles.PAIRS.single()
    val pair = OraclePair(shape.stateLength, shape.questionLength, rows)
    val row = OracleGraph(KevFiles.WINDOWS.last(), rows)
    var requestsTaken = 0
    var questionsTaken = 0
    var sameAsRows = 0
    var sameAsOracle = 0
    val notTaken = linkedMapOf("state" to 0, "branch" to 0, "both" to 0)
    val failures = ArrayList<String>()
    val ours = LinkedHashMap<String, Any?>()
    for (record in OracleFixtures.loadRecords()) {
      val prepared = pipeline.prepare(KevRequest.fromJson(record.request))
      val state = prepared.encoded.stateIds
      val branches = prepared.encoded.branches
      val stateOver = state.size > shape.stateLength
      val branchOver = branches.any { it.ids.size > shape.questionLength }
      val plan =
        KevPlanner.plan(
          state.size,
          branches.map { it.ids.size },
          emptyList(),
          listOf(shape),
          emptyList(),
          PLENTY,
          KevGraphMode.PAIR,
        )
      if (stateOver || branchOver) {
        assertTrue(record.id, plan is KevPlan.NoPair)
        val reason = if (stateOver && branchOver) "both" else if (stateOver) "state" else "branch"
        notTaken[reason] = notTaken.getValue(reason) + 1
        continue
      }
      assertTrue(record.id, plan is KevPlan.Pair)
      requestsTaken++
      val statesBefore = pair.states.size
      val result = pipeline.runRequest(prepared, KevRunners.Pair(pair))
      // One state call per request: [state] + state tokens, padded to Ls.
      assertEquals(statesBefore + 1, pair.states.size)
      assertArrayEquals(record.id, state, pair.states.last())
      assertEquals(KevEncoder.STATE_ID, state[0])
      assertEquals(state.size, result.state?.tokens)
      assertEquals(shape.stateLength, result.state?.window)
      val rowsResult = pipeline.runRequest(prepared, KevRunners.Rows(prepared.rows.map { row }))
      val pairAnswers = pipeline.answers(prepared, result.questions)
      val rowAnswers = pipeline.answers(prepared, rowsResult.questions)
      for ((index, meta) in prepared.meta.withIndex()) {
        questionsTaken++
        val key = "${record.id}/${meta.id}"
        val question = result.questions[index]
        val branch = pair.branches[pair.branches.size - prepared.meta.size + index]
        // The branch call gets the question's branch; state + branch is the oracle's row.
        assertArrayEquals(key, branches[index].ids, branch)
        assertArrayEquals(key, byKey.getValue(key).rowIds, state + branch)
        assertEquals(KevForm.PAIR, question.form)
        assertEquals(shape.questionLength, question.window)
        assertEquals(branch.size, question.branchLength)
        // The result keeps the row's coordinates (the run JSON's row_ids, decide_idx, opt_idx).
        assertArrayEquals(key, byKey.getValue(key).rowIds, question.row.ids)
        assertEquals(byKey.getValue(key).decideIndex, question.row.decideIndex)
        if (OracleFixtures.jsonDifference(rowAnswers[meta.id], pairAnswers[meta.id]) == null) {
          sameAsRows++
        } else if (failures.size < 3) {
          failures.add("$key differs from the row path")
        }
        if (OracleFixtures.jsonDifference(byKey.getValue(key).answer, pairAnswers[meta.id]) == null)
          sameAsOracle++
        ours[key] =
          linkedMapOf(
            "state_len" to state.size,
            "branch_len" to branch.size,
            "answer" to pairAnswers[meta.id],
          )
      }
    }
    ExternalTestData.writeReport("pipeline_pair_answers_ours.json", ours)
    ExternalTestData.writeReport(
      "pipeline_pair.json",
      linkedMapOf(
        "test" to "KevPipelineTest.thePairGivesTheRowAnswersOnEveryRequestItTakes",
        "pair" to linkedMapOf("Ls" to shape.stateLength, "Lq" to shape.questionLength),
        "requests_taken" to requestsTaken,
        "questions_taken" to questionsTaken,
        "answers_equal_row_path" to sameAsRows,
        "answers_equal_oracle" to sameAsOracle,
        "requests_not_taken" to notTaken,
        "failures" to failures,
      ),
    )
    println(
      "KEV_PIPELINE_PAIR requests=$requestsTaken questions=$questionsTaken same_as_rows=$sameAsRows same_as_oracle=$sameAsOracle not_taken=$notTaken"
    )
    assertEquals(failures.joinToString("\n"), questionsTaken, sameAsRows)
    assertEquals(questionsTaken, sameAsOracle)
    assertEquals(304, requestsTaken)
    assertEquals(312, questionsTaken)
    assertEquals(mapOf("state" to 24, "branch" to 47, "both" to 2), notTaken)
  }

  @Test
  fun rowsOverTheWindowAndNonFiniteOutputsAreRejected() {
    val pipeline =
      KevPipeline(
        ExternalTestData.tokenizer(),
        KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)),
      )
    val request =
      KevRequest.parse("""{"state":"${"word ".repeat(600)}","questions":{"q":{"type":"noul"}}}""")
    val prepared = pipeline.prepare(request)
    assertEquals(1024, prepared.window(KevFiles.WINDOWS))
    assertNull(prepared.window(listOf(256, 512)))
    val small =
      object : RowRunner {
        override val length = 512

        override fun run(ids: IntArray, valid: FloatArray) = FloatArray(length * HIDDEN)
      }
    val tooLong = runCatching { pipeline.run(prepared, 0, small) }.exceptionOrNull()
    assertTrue(tooLong is KevWindowException)
    assertEquals(prepared.rows[0].length, (tooLong as KevWindowException).rowTokens)
    val nan =
      object : RowRunner {
        override val length = 1024

        override fun run(ids: IntArray, valid: FloatArray) =
          FloatArray(length * HIDDEN) { if (it == 5) Float.NaN else 0f }
      }
    val nonFinite = runCatching { pipeline.run(prepared, 0, nan) }.exceptionOrNull()
    assertTrue(nonFinite is KevNonFiniteException)
    assertEquals(1, (nonFinite as KevNonFiniteException).count)
    // Pads may hold anything: only real positions are checked.
    val padNan =
      object : RowRunner {
        override val length = 1024

        override fun run(ids: IntArray, valid: FloatArray) =
          FloatArray(length * HIDDEN) { if (it == length * HIDDEN - 1) Float.NaN else 0f }
      }
    assertNull(runCatching { pipeline.run(prepared, 0, padNan) }.exceptionOrNull())
  }

  @Test
  fun thePairRejectsLongStatesAndBranchesAndChecksOnlyTheBranchsTokens() {
    val pipeline =
      KevPipeline(
        ExternalTestData.tokenizer(),
        KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)),
      )
    fun pair(nanAt: Int?) =
      object : PairRunner {
        override val stateLength = 128
        override val questionLength = 64

        override fun runState(ids: IntArray, valid: FloatArray) = Unit

        override fun runQuestion(ids: IntArray, valid: FloatArray) =
          FloatArray(questionLength * HIDDEN) { if (it == nanAt) Float.NaN else 0f }
      }
    val long =
      pipeline.prepare(
        KevRequest.parse("""{"state":"${"word ".repeat(200)}","questions":{"q":{"type":"noul"}}}""")
      )
    val state = runCatching { pipeline.runState(long, pair(null)) }.exceptionOrNull()
    assertTrue(state is KevWindowException)
    assertEquals(KevPipeline.STATE_ID, (state as KevWindowException).questionId)
    val wide =
      pipeline.prepare(
        KevRequest.parse(
          """{"state":"s","questions":{"q":{"type":"noul","instructions":"${"word ".repeat(80)}"}}}"""
        )
      )
    pipeline.runState(wide, pair(null))
    val branch = runCatching { pipeline.runBranch(wide, 0, pair(null)) }.exceptionOrNull()
    assertTrue(branch is KevWindowException)
    assertEquals(64, (branch as KevWindowException).window)
    val short =
      pipeline.prepare(KevRequest.parse("""{"state":"s","questions":{"q":{"type":"noul"}}}"""))
    pipeline.runState(short, pair(null))
    val nan = runCatching { pipeline.runBranch(short, 0, pair(3)) }.exceptionOrNull()
    assertTrue(nan is KevNonFiniteException)
    // Pads of the branch may hold anything: only the branch's real positions are checked.
    val padNan = pair(64 * HIDDEN - 1)
    assertNull(runCatching { pipeline.runBranch(short, 0, padNan) }.exceptionOrNull())
  }

  @Test
  fun idsSha256IsTheHashOfLittleEndianInt32() {
    // Python: data = b"".join(x.to_bytes(4, "little", signed=True) for x in ids)
    //         hashlib.sha256(data).hexdigest()
    assertEquals(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      KevPipeline.idsSha256(IntArray(0)),
    )
    assertEquals(
      "5d2e1ab69b0c65f59ece86510333246011d017a170ec82c09ac16c21a8633364",
      KevPipeline.idsSha256(intArrayOf(248060, 1, -1)),
    )
  }

  private companion object {
    const val HIDDEN = KevPointerHead.HIDDEN_SIZE

    /** More available memory than the limit for a second graph. */
    const val PLENTY = 8_000_000L * 1024
  }
}
