package com.kev

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * Request → rows against the author's oracle: every fixture request goes through
 * [KevRequest.fromJson], [KevRecords.toRecord] and [KevEncoder.encode]; each question's row IDs,
 * decide index and option indices must equal the oracle's `row_ids` / `decide_idx` / `opt_idx`
 * (402/402, red arm included), and each request's `input_tokens` its `usage` (377/377).
 */
class KevEncoderTest {
  @Test
  fun rowsAndUsageMatchTheOracle() {
    val oracle = OracleFixtures.loadOracle()
    val records = OracleFixtures.loadRecords()
    val encoder = KevEncoder(ExternalTestData.tokenizer())
    assertEquals(377, records.size)
    assertEquals(402, oracle.questions.size)
    assertEquals(KevEncoder.PAD_ID, oracle.padTokenId)
    assertEquals(KevPointerHead.TEMPERATURE, oracle.temperature, 0.0)
    val expectedQuestions = oracle.questions.associateBy { it.key }
    val expectedUsage = oracle.requests.associateBy { it.id }
    val failures = ArrayList<Map<String, Any?>>()
    val usageFailures = ArrayList<Map<String, Any?>>()
    val windows = linkedMapOf(512 to 0, 1024 to 0, 2048 to 0)
    // Every row and usage this port builds, for counting matches independently of this test.
    val ourRows = LinkedHashMap<String, Any?>()
    val ourUsage = LinkedHashMap<String, Any?>()
    var questions = 0
    var matched = 0
    val started = System.nanoTime()
    for (record in records) {
      val request = KevRequest.fromJson(record.request)
      val (kevRecord, meta) = KevRecords.toRecord(request)
      val encoded = encoder.encode(kevRecord)
      val usage = requireNotNull(expectedUsage[record.id])
      ourUsage[record.id] = encoded.inputTokens
      if (encoded.inputTokens != usage.inputTokens) {
        usageFailures.add(
          linkedMapOf(
            "id" to record.id,
            "expected" to usage.inputTokens,
            "actual" to encoded.inputTokens,
          )
        )
      }
      assertEquals(record.id, usage.questions, meta.size)
      for ((index, question) in meta.withIndex()) {
        questions++
        val expected =
          requireNotNull(expectedQuestions["${record.id}/${question.id}"]) {
            "${record.id}/${question.id}"
          }
        assertEquals(expected.keys, question.keys)
        assertEquals(expected.type, question.type.wireName)
        val row = encoded.row(index)
        val window = row.requireWindow()
        windows[window] = windows.getValue(window) + 1
        ourRows["${record.id}/${question.id}"] =
          linkedMapOf(
            "row_ids" to row.ids,
            "decide_idx" to row.decideIndex,
            "opt_idx" to row.optionIndices,
            "window" to window,
          )
        val same =
          row.ids.contentEquals(expected.rowIds) &&
            row.decideIndex == expected.decideIndex &&
            row.optionIndices.contentEquals(expected.optionIndices)
        if (same) {
          matched++
        } else {
          failures.add(difference(expected, row))
        }
      }
    }
    val seconds = (System.nanoTime() - started) / 1e9
    ExternalTestData.writeReport(
      "encoder_rows.json",
      linkedMapOf("rows" to ourRows, "input_tokens" to ourUsage),
    )
    ExternalTestData.writeReport(
      "encoder.json",
      linkedMapOf(
        "test" to "KevEncoderTest",
        "requests" to records.size,
        "questions" to questions,
        "rows_identical" to matched,
        "input_tokens_identical" to records.size - usageFailures.size,
        "windows" to windows.mapKeys { it.key.toString() },
        "seconds" to seconds,
        "tokenizer_load_ms" to ExternalTestData.tokenizerLoadMillis(),
        "first_failures" to failures.take(3),
        "usage_failures" to usageFailures.take(3),
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println(
      "KEV_ENCODER rows=$matched/$questions input_tokens=${records.size - usageFailures.size}/${records.size} windows=$windows"
    )
    assertEquals("rows with differences: ${KevJson.write(failures.take(3))}", 402, matched)
    assertEquals(402, questions)
    assertEquals(
      "usage differences: ${KevJson.write(usageFailures.take(3))}",
      0,
      usageFailures.size,
    )
    assertEquals(linkedMapOf(512 to 393, 1024 to 0, 2048 to 9), windows)
  }

  @Test
  fun paddingAndWindowRules() {
    val row = KevRow(intArrayOf(248060, 11, 248061, 248049, 12, 248050, 248062), 6, intArrayOf(5))
    assertEquals(512, row.window)
    val padded = row.padded(512)
    assertEquals(512, padded.ids.size)
    assertArrayEquals(row.ids, padded.ids.copyOf(row.length))
    assertTrue(padded.ids.drop(row.length).all { it == KevEncoder.PAD_ID })
    assertTrue(padded.valid.take(row.length).all { it == 1f })
    assertTrue(padded.valid.drop(row.length).all { it == 0f })
    assertEquals(1024, KevRow(IntArray(513), 512, intArrayOf()).window)
    assertEquals(2048, KevRow(IntArray(2048), 2047, intArrayOf()).window)
    val tooLong = KevRow(IntArray(2049), 2048, intArrayOf())
    assertNull(tooLong.window)
    try {
      tooLong.requireWindow()
      fail("a 2,049-token row was accepted")
    } catch (expected: IllegalArgumentException) {
      assertTrue(
        expected.message!!,
        expected.message!!.contains("2049") && expected.message!!.contains("2048"),
      )
    }
  }

  @Test
  fun emptyInstructionsStartTheBranchWithTheQuestionToken() {
    val encoder = KevEncoder(ExternalTestData.tokenizer())
    val request = KevRequest.parse("""{"state": "s", "questions": {"q": {"type": "noul"}}}""")
    val (record, _) = KevRecords.toRecord(request)
    val row = encoder.encode(record).row(0)
    val no = encoder.userTokens("no")
    val yes = encoder.userTokens("yes")
    val expected =
      intArrayOf(KevEncoder.STATE_ID) +
        encoder.userTokens("s") +
        KevEncoder.QUESTION_ID +
        KevEncoder.OPTION_START_ID +
        no +
        KevEncoder.OPTION_END_ID +
        KevEncoder.OPTION_START_ID +
        yes +
        KevEncoder.OPTION_END_ID +
        KevEncoder.DECIDE_ID
    assertArrayEquals(expected, row.ids)
    assertEquals(expected.size - 1, row.decideIndex)
  }

  private fun difference(expected: OracleFixtures.Question, row: KevRow): Map<String, Any?> {
    val first = OracleFixtures.firstDifference(expected.rowIds, row.ids)
    val window = if (first == null) null else maxOf(0, first - CONTEXT)..first + CONTEXT
    return linkedMapOf(
      "id" to expected.key,
      "first_difference" to first,
      "oracle_around" to window?.mapNotNull { expected.rowIds.getOrNull(it) },
      "ours_around" to window?.mapNotNull { row.ids.getOrNull(it) },
      "oracle_length" to expected.rowIds.size,
      "ours_length" to row.ids.size,
      "oracle_decide" to expected.decideIndex,
      "ours_decide" to row.decideIndex,
      "oracle_options" to expected.optionIndices,
      "ours_options" to row.optionIndices,
    )
  }

  private companion object {
    const val CONTEXT = 8
  }
}
