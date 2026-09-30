// SPDX-License-Identifier: Apache-2.0
package com.julia1

import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class JuliaDecoderParityTest {
  @Test
  fun answersMatchThePythonArithmeticOnEveryOracleRow() {
    val requests = JuliaTestData.requests().associateBy { it["id"] as String }
    val references =
      JuliaJson.asArray(
          JuliaJson.asObject(JuliaTestData.read("fixtures/decoder_reference.json"))["rows"]
        )
        .map(JuliaJson::asObject)
    assertEquals(2100, references.size)
    var worst = 0.0
    val failures = mutableListOf<Map<String, Any?>>()
    for (reference in references) {
      val id = reference["id"] as String
      val row = requests.getValue(id)
      val question = JuliaTestData.question(row)
      val answer = JuliaDecoder.decode(JuliaTestData.doubles(row["logits"]), question)
      val expected = JuliaTestData.doubles(reference["probabilities"])
      val delta =
        answer.probabilities.indices.maxOf { abs(answer.probabilities[it] - expected[it]) }
      worst = maxOf(worst, delta)
      val problems = mutableListOf<String>()
      if (delta > 1e-12) {
        problems += "probabilities differ by $delta"
      }
      when (question.type) {
        "choice" -> {
          val index = (reference["choice_index"] as Number).toInt()
          if (answer.choice != question.keys[index]) {
            problems += "choice ${answer.choice} != ${question.keys[index]}"
          }
        }
        "score" -> {
          val score = (reference["score"] as Number).toDouble()
          if (abs(checkNotNull(answer.score) - score) > 1e-12) {
            problems += "score ${answer.score} != $score"
          }
        }
        else -> {
          val noul = (reference["noul"] as Number).toDouble()
          if (abs(checkNotNull(answer.noul) - noul) > 1e-12) {
            problems += "noul ${answer.noul} != $noul"
          }
        }
      }
      reference["max_probability"]?.let {
        if (abs(checkNotNull(answer.maxProbability) - (it as Number).toDouble()) > 1e-12) {
          problems += "max_probability differs"
        }
      }
      if (problems.isNotEmpty()) {
        failures += linkedMapOf("id" to id, "problems" to problems)
      }
    }
    JuliaTestData.report(
      "jvm_decoder_parity.json",
      linkedMapOf(
        "status" to if (failures.isEmpty()) "PASS" else "FAIL",
        "rows" to references.size,
        "max_abs_probability_difference" to worst,
        "failures" to failures,
      ),
    )
    println("JULIA1_JVM_DECODER rows=${references.size} max_abs_dp=$worst")
    assertTrue("Decoder differs from the Python arithmetic: $worst", worst <= 1e-12)
    JuliaTestData.assertNoFailures("Decoder parity", failures)
  }
}
