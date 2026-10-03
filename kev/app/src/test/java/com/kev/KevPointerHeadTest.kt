package com.kev

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The host pointer head against the author's fp32 oracle: the oracle's selected hidden states
 * (`hidden_0.8b.npz`, [decide, options…] per question) through [KevPointerHead] must give the
 * oracle's probabilities within 1e-5 and temperature-scaled logits within 1e-4, for all 402
 * questions and for the 12 bundled ones (`head_fixture.*`; the head weights stay external).
 */
class KevPointerHeadTest {
  @Test
  fun oracleQuestionsMatch() {
    val head = KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD))
    val oracle = OracleFixtures.loadOracle()
    var maxProbability = 0.0
    var maxLogit = 0.0
    var maxRawLogit = 0.0
    var argmaxSame = 0
    val failures = ArrayList<Map<String, Any?>>()
    // This port's outputs per question, for recomputing the differences independently.
    val ours = LinkedHashMap<String, Any?>()
    val started = System.nanoTime()
    OracleFixtures.Npz(ExternalTestData.file(ExternalTestData.HIDDEN)).use { npz ->
      for (question in oracle.questions) {
        val (shape, values) = npz.floats(question.key)
        assertEquals(
          question.key,
          listOf(1 + question.keys.size, KevPointerHead.HIDDEN_SIZE),
          shape.toList(),
        )
        val scores =
          head.score(
            OracleFixtures.row(values, 0, KevPointerHead.HIDDEN_SIZE),
            (1..question.keys.size).map {
              OracleFixtures.row(values, it, KevPointerHead.HIDDEN_SIZE)
            },
          )
        ours[question.key] =
          linkedMapOf(
            "z_pre" to scores.zPre,
            "z_post" to scores.zPost,
            "probs" to scores.probabilities,
          )
        val dp = maxDifference(question.probabilities, scores.probabilities)
        val dz = maxDifference(question.zPost, scores.zPost)
        maxProbability = maxOf(maxProbability, dp)
        maxLogit = maxOf(maxLogit, dz)
        maxRawLogit = maxOf(maxRawLogit, maxDifference(question.zPre, scores.zPre))
        if (KevAnswers.firstArgmax(question.probabilities) == firstArgmax(scores.probabilities))
          argmaxSame++
        if (dp > PROBABILITY_TOLERANCE || dz > LOGIT_TOLERANCE) {
          failures.add(
            linkedMapOf("id" to question.key, "max_abs_dp" to dp, "max_abs_dz_post" to dz)
          )
        }
      }
    }
    val seconds = (System.nanoTime() - started) / 1e9
    ExternalTestData.writeReport("head_outputs.json", ours)
    ExternalTestData.writeReport(
      "head.json",
      linkedMapOf(
        "test" to "KevPointerHeadTest",
        "questions" to oracle.questions.size,
        "within_tolerance" to oracle.questions.size - failures.size,
        "max_abs_dp" to maxProbability,
        "max_abs_dz_post" to maxLogit,
        "max_abs_dz_pre" to maxRawLogit,
        "argmax_same" to argmaxSame,
        "tolerance" to linkedMapOf("probs" to PROBABILITY_TOLERANCE, "z_post" to LOGIT_TOLERANCE),
        "seconds" to seconds,
        "failures" to failures.take(5),
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println(
      "KEV_HEAD within=${oracle.questions.size - failures.size}/${oracle.questions.size} max|dp|=$maxProbability max|dz_post|=$maxLogit"
    )
    assertEquals(
      "questions outside tolerance: ${KevJson.write(failures.take(5))}",
      0,
      failures.size,
    )
    assertEquals(402, oracle.questions.size)
  }

  @Test
  fun bundledFixtureMatches() {
    val head = KevPointerHead(ExternalTestData.file(ExternalTestData.HEAD))
    val fixture =
      KevJson.parse(ExternalTestData.resource("head_fixture.json").readBytes()) as Map<*, *>
    val bytes = ExternalTestData.resource("head_fixture.f32").readBytes()
    val floats =
      FloatArray(bytes.size / 4).also {
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it)
      }
    assertEquals((fixture["floats"] as JsonNumber).toInt(), floats.size)
    val items = fixture["items"] as List<*>
    var maxProbability = 0.0
    for (item in items) {
      val entry = item as Map<*, *>
      val rows = (entry["rows"] as JsonNumber).toInt()
      val offset = (entry["offset_floats"] as JsonNumber).toInt()
      val hidden = { row: Int ->
        floats.copyOfRange(
          offset + row * KevPointerHead.HIDDEN_SIZE,
          offset + (row + 1) * KevPointerHead.HIDDEN_SIZE,
        )
      }
      val scores = head.score(hidden(0), (1 until rows).map(hidden))
      val dp = maxDifference(OracleFixtures.doubles(entry["probs"]), scores.probabilities)
      assertTrue("${entry["id"]} max|dp| $dp", dp <= PROBABILITY_TOLERANCE)
      assertTrue(
        maxDifference(OracleFixtures.doubles(entry["z_post"]), scores.zPost) <= LOGIT_TOLERANCE
      )
      maxProbability = maxOf(maxProbability, dp)
    }
    assertEquals(12, items.size)
    ExternalTestData.writeReport(
      "head_fixture.json",
      linkedMapOf(
        "test" to "KevPointerHeadTest.bundledFixtureMatches",
        "questions" to items.size,
        "max_abs_dp" to maxProbability,
      ),
    )
    println("KEV_HEAD_FIXTURE questions=${items.size} max|dp|=$maxProbability")
  }

  @Test
  fun constantsMatchTheHeadDescription() {
    val description =
      KevJson.parse(ExternalTestData.file(ExternalTestData.HEAD_JSON).readBytes()) as Map<*, *>
    assertEquals(
      KevPointerHead.TEMPERATURE,
      (description["temperature"] as JsonNumber).toDouble(),
      0.0,
    )
    val head = description["head"] as Map<*, *>
    assertEquals(KevPointerHead.HEAD_DIM, (head["head_dim"] as JsonNumber).toInt())
    assertEquals(KevPointerHead.SCALE.toDouble(), (head["scale"] as JsonNumber).toDouble(), 0.0)
    assertEquals(
      KevPointerHead.HIDDEN_SIZE,
      ((description["base"] as Map<*, *>)["hidden_size"] as JsonNumber).toInt(),
    )
    val ids = description["delimiter_token_ids"] as Map<*, *>
    assertEquals(KevEncoder.STATE_ID, (ids["state"] as JsonNumber).toInt())
    assertEquals(KevEncoder.QUESTION_ID, (ids["question"] as JsonNumber).toInt())
    assertEquals(KevEncoder.OPTION_START_ID, (ids["option_start"] as JsonNumber).toInt())
    assertEquals(KevEncoder.OPTION_END_ID, (ids["option_end"] as JsonNumber).toInt())
    assertEquals(KevEncoder.DECIDE_ID, (ids["decide"] as JsonNumber).toInt())
    assertEquals(KevEncoder.PAD_ID, (description["pad_token_id"] as JsonNumber).toInt())
  }

  @Test
  fun softmaxIsStableAndNormalized() {
    val probabilities = KevPointerHead.softmax(floatArrayOf(1000f, 1000f, -1000f))
    assertEquals(0.5f, probabilities[0], 0f)
    assertEquals(0.5f, probabilities[1], 0f)
    assertEquals(0f, probabilities[2], 0f)
  }

  private fun maxDifference(expected: DoubleArray, actual: FloatArray): Double {
    assertEquals(expected.size, actual.size)
    return expected.indices.maxOf { abs(expected[it] - actual[it].toDouble()) }
  }

  private fun firstArgmax(values: FloatArray): Int =
    KevAnswers.firstArgmax(DoubleArray(values.size) { values[it].toDouble() })

  private companion object {
    const val PROBABILITY_TOLERANCE = 1e-5
    const val LOGIT_TOLERANCE = 1e-4
  }
}
