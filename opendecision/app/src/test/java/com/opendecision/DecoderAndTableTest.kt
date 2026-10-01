package com.opendecision

import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Resource-only tests: the read-out against the official answers, the float16 widening, and the question editor. */
class DecoderAndTableTest {
  @Test
  fun oracleLogitsDecodeToTheOfficialAnswers() {
    val cases = JSONObject(ExternalTestData.resource("decoder_cases.json")).getJSONArray("cases")
    for (c in 0 until cases.length()) {
      val case = cases.getJSONObject(c)
      val questions = parse(case.getJSONArray("questions"))
      val logits = ArrayList<Float>()
      val perQuestion = case.getJSONArray("logits")
      for (q in 0 until perQuestion.length()) {
        val a = perQuestion.getJSONArray(q)
        for (k in 0 until a.length()) logits.add(a.getDouble(k).toFloat())
      }
      while (logits.size < DecisionInputs.OPTION_SLOTS) logits.add(0f)
      val answers = DecisionDecoder.decode(logits.toFloatArray(), questions)
      val api = case.getJSONArray("api")
      for ((q, answer) in answers.withIndex()) {
        val expected = api.getJSONObject(q)
        when (answer.question.kind) {
          Question.Kind.CHOICE -> {
            assertEquals(case.getString("id"), expected.getString("choice"), answer.label)
            assertTrue(abs(expected.getDouble("confidence") - answer.confidence) < 1e-6)
          }
          Question.Kind.SCORE -> assertTrue(case.getString("id"), abs(expected.getDouble("score") - answer.expectedLevel) < 1e-5)
          Question.Kind.NOUL -> assertTrue(case.getString("id"), abs(expected.getDouble("noul") - answer.yes) < 1e-6)
        }
      }
    }
  }

  @Test
  fun halfToFloatMatchesNumpy() {
    val json = JSONObject(ExternalTestData.resource("embedding_rows_fp16.json"))
    val samples = json.getJSONArray("samples")
    var checked = 0
    for (r in 0 until samples.length()) {
      val sample = samples.getJSONObject(r)
      val got = DecisionInputs.halfToFloat(sample.getInt("bits").toShort())
      assertEquals(sample.getLong("float_bits").toInt(), java.lang.Float.floatToRawIntBits(got))
      checked++
    }
    val specials = json.getJSONArray("special_patterns")
    for (s in 0 until specials.length()) {
      val p = specials.getJSONObject(s)
      val got = DecisionInputs.halfToFloat(p.getInt("bits").toShort())
      if (p.getBoolean("is_nan")) assertTrue(got.isNaN()) else assertEquals(p.getLong("float_bits").toInt(), java.lang.Float.floatToRawIntBits(got))
    }
    println("HALF_TO_FLOAT checked=$checked values + ${specials.length()} special patterns")
  }

  @Test
  fun editorParsesAndValidates() {
    val questions = Question.parseLines("choice: Which team? | billing, shipping\nscore: How angry? | calm, angry\nnoul: Asks for a refund.")
    Question.validate(questions)
    assertEquals(listOf(Question.Kind.CHOICE, Question.Kind.SCORE, Question.Kind.NOUL), questions.map { it.kind })
    assertEquals(Question.NOUL_OPTIONS, questions[2].options)
    assertTrue(runCatching { Question.validate(Question.parseLines("choice: one option? | only")) }.isFailure)
    assertTrue(runCatching { Question.parseLines("noul: statement | no, yes") }.isFailure)
  }

  private fun parse(array: JSONArray): List<Question> =
    List(array.length()) { i ->
      val q = array.getJSONObject(i)
      val options = q.optJSONArray("options")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList()
      when (Question.Kind.fromKey(q.getString("type"))) {
        Question.Kind.NOUL -> Question.noul(q.getString("instructions"))
        Question.Kind.CHOICE -> Question.choice(q.getString("instructions"), options)
        Question.Kind.SCORE -> Question.score(q.getString("instructions"), options)
      }
    }
}
