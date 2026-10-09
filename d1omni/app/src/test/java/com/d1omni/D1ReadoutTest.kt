package com.d1omni

import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Kotlin read-out against the Python host's (`d1_host.readout_f64`, `d1_prompt.answer`) on 20
 * public rows whose scores came from the repository's graphs on a desktop CPU
 * (`fixtures/scores_probe.json`: text 14, image 3, audio 3; noul, choice and score; near-ties): the
 * probabilities from the whole scores vector (P + markers, the temperature for text only, softmax,
 * a noul reversed), the answer object, and the three-decimal strings the screen shows.
 */
class D1ReadoutTest {
  @Test
  fun readoutAnswersAndShownStringsMatchPython() {
    val contract = ExternalTestData.contract()
    val data = ExternalTestData.json(ExternalTestData.demoFile(ExternalTestData.SCORES))
    val rows = (data["rows"] as List<*>).map { it as Map<*, *> }
    var worst = 0.0
    var answersEqual = 0
    var shownEqual = 0
    val kinds = LinkedHashMap<String, Int>()
    val types = LinkedHashMap<String, Int>()
    for (row in rows) {
      val key = row["key"] as String
      val question = D1Prompt.asQuestion(row["question"])
      val scores = ExternalTestData.doubles(row["scores"]).map { it.toFloat() }.toFloatArray()
      val prefixRows = (row["P"] as JsonNumber).toInt()
      val markers = ExternalTestData.ints(row["markers"])
      val calibrate = row["calibrate"] as Boolean
      val expectedMarkerScores = ExternalTestData.doubles(row["scores_at_markers"])
      val markerScores = D1Readout.markerScores(scores, prefixRows, markers, question.options)
      for ((index, score) in markerScores.withIndex()) {
        assertEquals("$key score $index", expectedMarkerScores[index], score.toDouble(), 0.0)
      }
      if (calibrate) {
        assertEquals(key, (row["temperature"] as JsonNumber).toDouble(), contract.temperature(question), 0.0)
        assertEquals(key, row["temperature_key"], D1Prompt.temperatureKey(question))
      }
      val probs = D1Readout.probabilities(scores, prefixRows, markers, question, calibrate, contract)
      val expected = ExternalTestData.doubles(row["probs"])
      assertEquals(key, expected.size, probs.size)
      for (index in probs.indices) worst = maxOf(worst, abs(probs[index] - expected[index]))
      val difference = jsonDifference(row["answer"], D1Prompt.answer(question, probs))
      if (difference == null) answersEqual++ else println("D1_READOUT answer $key: $difference")
      val shown = probs.map { D1Readout.shown(it) }
      if (shown == (row["shown3"] as List<*>)) shownEqual++ else println("D1_READOUT shown $key: $shown vs ${row["shown3"]}")
      kinds[row["kind"] as String] = (kinds[row["kind"] as String] ?: 0) + 1
      types[question.type.wireName] = (types[question.type.wireName] ?: 0) + 1
    }
    ExternalTestData.writeReport(
      "readout.json",
      linkedMapOf(
        "test" to "D1ReadoutTest",
        "rows" to rows.size,
        "kinds" to kinds,
        "types" to types,
        "max_abs_dp_vs_python" to worst,
        "answers_equal" to answersEqual,
        "shown3_equal" to shownEqual,
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println("D1_READOUT rows=${rows.size} $kinds $types max|dp|=$worst answers=$answersEqual shown3=$shownEqual")
    assertEquals(20, rows.size)
    assertTrue("max |dp| $worst", worst <= 1e-12)
    assertEquals(rows.size, answersEqual)
    assertEquals(rows.size, shownEqual)
  }

  @Test
  fun numpySumOrderAndShownRounding() {
    // Fewer than 8 values: one by one from 0.0.
    assertEquals(0.6000000000000001, D1Readout.numpySum(doubleArrayOf(0.1, 0.2, 0.3)), 0.0)
    // Ten values: 8 running sums combined pairwise, then the rest one by one.
    val ten = DoubleArray(10) { 0.1 }
    val lanes = DoubleArray(8) { 0.1 }
    var expected = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    expected += 0.1
    expected += 0.1
    assertEquals(expected, D1Readout.numpySum(ten), 0.0)
    // Half to even on the exact binary value: 0.0625 is exact, 0.8125 too.
    assertEquals("0.062", D1Readout.shown(0.0625))
    assertEquals("0.812", D1Readout.shown(0.8125))
    assertEquals("1.000", D1Readout.shown(0.99999))
    assertEquals("0.000", D1Readout.shown(1.2e-9))
  }

  /** The path of the first difference of two answer objects (numbers within 1e-12), or null. */
  private fun jsonDifference(expected: Any?, actual: Any?, path: String = "$"): String? {
    fun number(value: Any?): Double? =
      when (value) {
        is JsonNumber -> value.toDouble()
        is Double -> value
        is Int -> value.toDouble()
        else -> null
      }
    val e = number(expected)
    val a = number(actual)
    if (e != null || a != null) {
      return if (e != null && a != null && abs(e - a) <= 1e-12) null else "$path: $expected vs $actual"
    }
    return when (expected) {
      is Map<*, *> -> {
        if (actual !is Map<*, *> || expected.keys.toList() != actual.keys.toList()) {
          "$path: keys ${expected.keys} vs ${(actual as? Map<*, *>)?.keys}"
        } else {
          expected.keys.firstNotNullOfOrNull { jsonDifference(expected[it], actual[it], "$path.$it") }
        }
      }
      is List<*> -> {
        if (actual !is List<*> || actual.size != expected.size) "$path: $expected vs $actual"
        else expected.indices.firstNotNullOfOrNull { jsonDifference(expected[it], actual[it], "$path[$it]") }
      }
      else -> if (expected == actual) null else "$path: $expected vs $actual"
    }
  }
}
