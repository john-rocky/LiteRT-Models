package com.kev

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The app's decision path end to end with a fake graph: every oracle request goes through
 * [KevPipeline] (request → rows → padded inputs → [RowRunner] → readout rows → head →
 * `to_answers`), where the fake runner checks the padded inputs and places the oracle's hidden
 * states (`hidden_0.8b.npz`) at the decide and option positions. All 402 answers must equal the
 * oracle's, and all 377 `usage.input_tokens`.
 */
class KevPipelineTest {
  /** A graph stand-in that returns [rows] at [positions] of an otherwise zero `hidden`. */
  private class OracleRunner(
    override val length: Int,
    private val expectedIds: IntArray,
    private val positions: IntArray,
    private val rows: FloatArray,
  ) : RowRunner {
    var inputsChecked = false

    override fun run(ids: IntArray, valid: FloatArray): FloatArray {
      assertEquals(length, ids.size)
      assertEquals(length, valid.size)
      for (index in 0 until length) {
        val real = index < expectedIds.size
        assertEquals(if (real) expectedIds[index] else KevEncoder.PAD_ID, ids[index])
        assertEquals(if (real) 1f else 0f, valid[index])
      }
      inputsChecked = true
      val hidden = FloatArray(length * HIDDEN)
      for ((row, position) in positions.withIndex()) {
        System.arraycopy(rows, row * HIDDEN, hidden, position * HIDDEN, HIDDEN)
      }
      return hidden
    }
  }

  @Test
  fun oracleRequestsGiveTheOracleAnswers() {
    val pipeline = KevPipeline(ExternalTestData.tokenizer(), KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)))
    val oracle = OracleFixtures.loadOracle()
    val byKey = oracle.questions.associateBy { it.key }
    val usage = oracle.requests.associateBy { it.id }
    var answersEqual = 0
    var inputsChecked = 0
    var usageEqual = 0
    val windows = linkedMapOf(512 to 0, 1024 to 0, 2048 to 0)
    val failures = ArrayList<String>()
    // This path's answers and rows per question, for recounting independently.
    val ours = LinkedHashMap<String, Any?>()
    var maxInferMs = 0.0
    OracleFixtures.Npz(ExternalTestData.file(ExternalTestData.HIDDEN)).use { npz ->
      for (record in OracleFixtures.loadRecords()) {
        val prepared = pipeline.prepare(KevRequest.fromJson(record.request))
        if (prepared.inputTokens == usage.getValue(record.id).inputTokens) usageEqual++
        assertEquals(prepared.tokenizeMs, prepared.stateMs + prepared.branchMs.sum(), 1e-9)
        val results = ArrayList<KevQuestionResult>()
        for ((index, meta) in prepared.meta.withIndex()) {
          val expected = byKey.getValue("${record.id}/${meta.id}")
          val row = prepared.rows[index]
          assertArrayEquals(expected.key, expected.rowIds, row.ids)
          val window = row.requireWindow()
          windows[window] = windows.getValue(window) + 1
          val (_, values) = npz.floats(expected.key)
          val runner = OracleRunner(window, row.ids, intArrayOf(row.decideIndex) + row.optionIndices, values)
          val result = pipeline.run(prepared, index, runner)
          if (runner.inputsChecked) inputsChecked++
          assertEquals(window, result.window)
          assertTrue(result.inferMs >= 0 && result.headMs >= 0)
          maxInferMs = maxOf(maxInferMs, result.inferMs)
          results.add(result)
        }
        val answers = pipeline.answers(prepared, results)
        for ((index, meta) in prepared.meta.withIndex()) {
          val key = "${record.id}/${meta.id}"
          ours[key] = linkedMapOf("row_len" to prepared.rows[index].length, "ids_sha256" to KevPipeline.idsSha256(prepared.rows[index].ids), "answer" to answers[meta.id])
          val difference = OracleFixtures.jsonDifference(byKey.getValue(key).answer, answers[meta.id])
          if (difference == null) answersEqual++ else if (failures.size < 3) failures.add("$key $difference")
        }
        val response = pipeline.response(prepared, answers)
        assertEquals(listOf("model", "answers", "usage"), response.keys.toList())
        assertEquals(prepared.inputTokens, (response["usage"] as Map<*, *>)["input_tokens"])
      }
    }
    ExternalTestData.writeReport("pipeline_answers_ours.json", ours)
    ExternalTestData.writeReport(
      "pipeline.json",
      linkedMapOf(
        "test" to "KevPipelineTest",
        "questions" to oracle.questions.size,
        "answers_equal" to answersEqual,
        "padded_inputs_checked" to inputsChecked,
        "requests" to oracle.requests.size,
        "input_tokens_equal" to usageEqual,
        "windows" to windows.mapKeys { it.key.toString() },
        "failures" to failures,
        "execution" to "Desktop JVM ${System.getProperty("java.version")}, fake RowRunner with the oracle's h_sel",
      ),
    )
    println("KEV_PIPELINE answers_equal=$answersEqual/${oracle.questions.size} inputs_checked=$inputsChecked usage=$usageEqual/${oracle.requests.size} windows=$windows")
    assertEquals(failures.joinToString("\n"), 402, answersEqual)
    assertEquals(402, inputsChecked)
    assertEquals(377, usageEqual)
    assertEquals(mapOf(512 to 393, 1024 to 0, 2048 to 9), windows)
  }

  @Test
  fun rowsOverTheWindowAndNonFiniteOutputsAreRejected() {
    val pipeline = KevPipeline(ExternalTestData.tokenizer(), KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD)))
    val request = KevRequest.parse("""{"state":"${"word ".repeat(600)}","questions":{"q":{"type":"noul"}}}""")
    val prepared = pipeline.prepare(request)
    assertEquals(1024, prepared.window)
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

        override fun run(ids: IntArray, valid: FloatArray) = FloatArray(length * HIDDEN) { if (it == 5) Float.NaN else 0f }
      }
    val nonFinite = runCatching { pipeline.run(prepared, 0, nan) }.exceptionOrNull()
    assertTrue(nonFinite is KevNonFiniteException)
    assertEquals(1, (nonFinite as KevNonFiniteException).count)
    // Pads may hold anything: only real positions are checked.
    val padNan =
      object : RowRunner {
        override val length = 1024

        override fun run(ids: IntArray, valid: FloatArray) = FloatArray(length * HIDDEN) { if (it == length * HIDDEN - 1) Float.NaN else 0f }
      }
    assertNull(runCatching { pipeline.run(prepared, 0, padNan) }.exceptionOrNull())
  }

  @Test
  fun idsSha256IsTheHashOfLittleEndianInt32() {
    // Python: hashlib.sha256(b"".join(x.to_bytes(4, "little", signed=True) for x in ids)).hexdigest()
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
  }
}
