package com.kev

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * `to_answers` against the author: the oracle's float32 probabilities through [KevAnswers.toAnswers]
 * must give the oracle's `answer` object exactly (keys, key order, strings, every double; 402/402),
 * and CPython 3.12's `sum`, `round` and the confidence formulas must match on fixed and random
 * inputs (`python_numbers.json`).
 */
class KevAnswersTest {
  @Test
  fun oracleAnswersMatchExactly() {
    val oracle = OracleFixtures.loadOracle()
    // Keys and score legends come from this port's to_record on the fixture requests.
    val metaByRecord =
      OracleFixtures.loadRecords().associate { record ->
        record.id to KevRecords.toRecord(KevRequest.fromJson(record.request)).second
      }
    val failures = ArrayList<Map<String, Any?>>()
    // This port's answer per question, for comparing independently of this test.
    val ours = LinkedHashMap<String, Any?>()
    for (question in oracle.questions) {
      val meta = requireNotNull(metaByRecord[question.id]).single { it.id == question.qid }
      val answers = KevAnswers.toAnswers(listOf(question.probabilities), listOf(meta))
      ours[question.key] = answers[question.qid]
      OracleFixtures.jsonDifference(question.answer, answers[question.qid])?.let {
        failures.add(linkedMapOf("id" to question.key, "difference" to it))
      }
    }
    ExternalTestData.writeReport("answers_ours.json", ours)
    // Each request's answers object holds the same answers in question order.
    val byKey = oracle.questions.associateBy { it.key }
    var requestsIdentical = 0
    for (request in oracle.requests) {
      val meta = requireNotNull(metaByRecord[request.id])
      val probabilities = meta.map { requireNotNull(byKey["${request.id}/${it.id}"]).probabilities }
      val answers = KevAnswers.toAnswers(probabilities, meta)
      if (OracleFixtures.jsonDifference(request.answers, answers) == null) requestsIdentical++
    }
    ExternalTestData.writeReport(
      "answers.json",
      linkedMapOf(
        "test" to "KevAnswersTest",
        "questions" to oracle.questions.size,
        "answers_identical" to oracle.questions.size - failures.size,
        "requests" to oracle.requests.size,
        "request_answers_identical" to requestsIdentical,
        "failures" to failures.take(5),
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println("KEV_ANSWERS identical=${oracle.questions.size - failures.size}/${oracle.questions.size} requests=$requestsIdentical/${oracle.requests.size}")
    assertEquals("answers with differences: ${KevJson.write(failures.take(5))}", 0, failures.size)
    assertEquals(402, oracle.questions.size)
    assertEquals(oracle.requests.size, requestsIdentical)
  }

  @Test
  fun pythonSumRoundAndReprMatchCPython() {
    val numbers = KevJson.parse(ExternalTestData.resource("python_numbers.json").readBytes()) as Map<*, *>
    for (case in numbers["sum"] as List<*>) {
      val entry = case as Map<*, *>
      val values = OracleFixtures.doubles(entry["values"])
      val expected = (entry["sum"] as JsonNumber).toDouble()
      assertEquals("sum(${entry["values"]})", expected.toRawBits(), KevAnswers.pythonSum(values).toRawBits())
    }
    for (case in numbers["round"] as List<*>) {
      val entry = case as Map<*, *>
      val x = (entry["x"] as JsonNumber).toDouble()
      val expected = (entry["round_prob"] as JsonNumber).toDouble()
      assertEquals("round(${entry["x"]}, 4)", expected.toRawBits(), KevAnswers.roundProb(x).toRawBits())
    }
    // The exact binary values decide: 0.12345 is just above and 0.12355 just below their halfway
    // points (Python: round(0.12345, 4) == round(0.12355, 4) == 0.1235); 0.03125 is an exact tie.
    assertEquals(0.1235, KevAnswers.roundProb(0.12345), 0.0)
    assertEquals(0.1235, KevAnswers.roundProb(0.12355), 0.0)
    assertEquals(0.0312, KevAnswers.roundProb(0.03125), 0.0)
  }

  @Test
  fun answerMathMatchesTheAuthorOnSyntheticDistributions() {
    val numbers = KevJson.parse(ExternalTestData.resource("python_numbers.json").readBytes()) as Map<*, *>
    val sets = numbers["answers"] as List<*>
    for (set in sets) {
      val entry = set as Map<*, *>
      val p = OracleFixtures.doubles(entry["p"])
      assertArrayEquals(OracleFixtures.doubles(entry["normalize"]), KevAnswers.normalize(p), 0.0)
      assertEquals((entry["choice_confidence"] as JsonNumber).toDouble().toRawBits(), KevAnswers.choiceConfidence(p).toRawBits())
      assertEquals((entry["score_confidence"] as JsonNumber).toDouble().toRawBits(), KevAnswers.scoreConfidence(p).toRawBits())
      val meta =
        (entry["meta"] as List<*>).map {
          val question = it as Map<*, *>
          QuestionMeta(
            question["id"] as String,
            QuestionType.fromWireName(question["type"]),
            (question["keys"] as List<*>).map { key -> key as String },
            (question["legend"] as Map<*, *>?)?.entries?.associate { (k, v) -> k as String to v as String },
          )
        }
      val answers = KevAnswers.toAnswers(List(meta.size) { p }, meta)
      assertNull(OracleFixtures.jsonDifference(entry["answers"], answers))
    }
    assertEquals(46, sets.size)
  }

  @Test
  fun ruleEdges() {
    // All-zero probabilities count as uniform; one option is fully confident.
    assertArrayEquals(doubleArrayOf(0.25, 0.25, 0.25, 0.25), KevAnswers.normalize(DoubleArray(4)), 0.0)
    assertEquals(1.0, KevAnswers.choiceConfidence(doubleArrayOf(0.7)), 0.0)
    assertEquals(1.0, KevAnswers.scoreConfidence(doubleArrayOf(0.7)), 0.0)
    // The mode and the choice are the first maximum.
    assertEquals(1, KevAnswers.firstArgmax(doubleArrayOf(0.2, 0.4, 0.4)))
    val meta = QuestionMeta("q", QuestionType.CHOICE, listOf("a", "b", "c"), null)
    assertEquals("b", (KevAnswers.toAnswers(listOf(doubleArrayOf(0.2, 0.4, 0.4)), listOf(meta))["q"] as Map<*, *>)["choice"])
    val response = KevAnswers.response(linkedMapOf("q" to linkedMapOf("type" to "noul", "noul" to 0.5)), 42)
    assertEquals("{\"model\":\"kev-latest\",\"answers\":{\"q\":{\"type\":\"noul\",\"noul\":0.5}},\"usage\":{\"input_tokens\":42}}", KevJson.write(response))
  }
}
