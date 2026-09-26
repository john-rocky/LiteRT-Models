package com.gliformer

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Gates all 80 original fixture paths against Python and the official fp32 entity oracle. */
class GliformerDecoderParityTest {
  @Test
  fun publishedCpuLogitsMatchPythonAndOfficialOracle() {
    val fixtures = ExternalTestData.resolve()
    val corpus = JSONObject(File(fixtures, "corpus.json").readText()).getJSONArray("rows")
    val manifest = JSONObject(File(fixtures, "logits/manifest.json").readText())
    val logitRows = manifest.getJSONArray("rows")
    val logitById =
      (0 until logitRows.length()).associate { index ->
        logitRows.getJSONObject(index).let { it.getString("id") to it }
      }
    assertEquals("Original fixture count", 80, corpus.length())
    assertEquals("Logit manifest must contain every fixture exactly once", 80, logitRows.length())
    assertEquals("Logit IDs must be unique", 80, logitById.size)
    val failures = mutableListOf<String>()
    val rows = JSONArray()
    val windows = sortedMapOf<Int, Int>()
    var pythonIdentical = 0
    var oracleIdentical = 0
    var pythonOrderIdentical = 0
    var maxPythonDifference = 0.0
    var maxOracleDifference = 0.0
    var maxPythonId = ""
    var maxOracleId = ""
    var minimumLogit = Float.POSITIVE_INFINITY
    var maximumLogit = Float.NEGATIVE_INFINITY
    var allFinite = true

    for (index in 0 until corpus.length()) {
      val fixture = corpus.getJSONObject(index)
      val id = fixture.getString("id")
      val window = fixture.getInt("window")
      val entry = requireNotNull(logitById[id]) { "Missing logits for $id" }
      assertEquals("$id selected window", window, entry.getInt("window"))
      windows[window] = (windows[window] ?: 0) + 1
      val capacity =
        when (window) {
          128 -> 48
          256 -> 256
          512 -> 512
          else -> error("Unexpected window $window")
        }
      assertEquals("$id text capacity", capacity, entry.getInt("text_capacity"))
      val bytes = File(fixtures, "logits/${entry.getString("file")}").readBytes()
      assertEquals("$id raw float32 bytes", capacity * 15 * 4, bytes.size)
      if (entry.has("sha256")) {
        assertEquals("$id logits hash", entry.getString("sha256"), sha256(bytes))
      }
      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      val logits = FloatArray(buffer.remaining()).also { buffer.get(it) }
      val finite = logits.all { it.isFinite() }
      allFinite = allFinite && finite
      assertTrue("$id logits contain nonfinite values", finite)
      minimumLogit = minOf(minimumLogit, logits.minOrNull()!!)
      maximumLogit = maxOf(maximumLogit, logits.maxOrNull()!!)
      val captured = fixture.getJSONObject("captured")
      val text = fixture.getString("text")
      val starts = captured.getJSONArray("start_map").ints()
      val ends = captured.getJSONArray("end_map").ints()
      val labels = fixture.getJSONArray("labels").strings()
      val actual = GliformerDecoder.decode(logits, text, starts, ends, labels)
      val actualBySpan = actual.associateBy { Triple(it.label, it.start, it.end) }
      assertEquals("$id duplicate span keys", actual.size, actualBySpan.size)
      val pythonEntities = entry.getJSONArray("python_entities")
      val python = expectedSpans(pythonEntities)
      val oracle = expectedSpans(fixture.getJSONArray("oracle_entities"))
      val pythonMatch = actualBySpan.keys == python.keys
      val oracleMatch = actualBySpan.keys == oracle.keys
      val orderMatch = actual.map { Triple(it.label, it.start, it.end) } == python.keys.toList()
      if (pythonMatch) pythonIdentical++
      else
        failures +=
          "$id Python spans missing=${python.keys - actualBySpan.keys}; extra=${actualBySpan.keys - python.keys}"
      if (oracleMatch) oracleIdentical++
      else
        failures +=
          "$id oracle spans missing=${oracle.keys - actualBySpan.keys}; extra=${actualBySpan.keys - oracle.keys}"
      if (orderMatch) pythonOrderIdentical++
      else failures += "$id Python output order mismatch (inspect candidate score ties)"
      var pythonDifference = 0.0
      var oracleDifference = 0.0
      for ((key, entity) in actualBySpan) {
        assertTrue("$id finite entity score", entity.score.isFinite())
        assertEquals(
          "$id original entity surface",
          text.substring(
            text.offsetByCodePoints(0, entity.start),
            text.offsetByCodePoints(0, entity.end),
          ),
          entity.text,
        )
        python[key]?.let {
          pythonDifference = maxOf(pythonDifference, abs(entity.score.toDouble() - it))
        }
        oracle[key]?.let {
          oracleDifference = maxOf(oracleDifference, abs(entity.score.toDouble() - it))
        }
      }
      if (pythonDifference > maxPythonDifference) {
        maxPythonDifference = pythonDifference
        maxPythonId = id
      }
      if (oracleDifference > maxOracleDifference) {
        maxOracleDifference = oracleDifference
        maxOracleId = id
      }
      if (pythonDifference > 1e-5)
        failures += "$id Python score difference $pythonDifference > 1e-5"
      if (oracleDifference > 1e-3)
        failures += "$id oracle score difference $oracleDifference > 1e-3"
      rows.put(
        JSONObject()
          .put("id", id)
          .put("window", window)
          .put("all_finite", finite)
          .put("python_span_sets_identical", pythonMatch)
          .put("oracle_span_sets_identical", oracleMatch)
          .put("python_output_order_identical", orderMatch)
          .put("max_score_difference_python", pythonDifference)
          .put("max_score_difference_oracle", oracleDifference)
          .put(
            "entities",
            JSONArray(
              actual.map { entity ->
                JSONObject()
                  .put("label", entity.label)
                  .put("text", entity.text)
                  .put("start", entity.start)
                  .put("end", entity.end)
                  .put("score", entity.score.toDouble())
              }
            ),
          )
      )
    }
    val report =
      JSONObject()
        .put("status", if (failures.isEmpty()) "PASS" else "FAIL")
        .put("fixtures", corpus.length())
        .put("python_span_sets_identical", pythonIdentical)
        .put("oracle_span_sets_identical", oracleIdentical)
        .put("python_output_order_identical", pythonOrderIdentical)
        .put("max_score_difference_python", maxPythonDifference)
        .put("max_score_difference_python_id", maxPythonId)
        .put("max_score_difference_oracle", maxOracleDifference)
        .put("max_score_difference_oracle_id", maxOracleId)
        .put("all_finite", allFinite)
        .put("logit_min", minimumLogit.toDouble())
        .put("logit_max", maximumLogit.toDouble())
        .put("counts_by_window", JSONObject(windows.mapKeys { "s${it.key}" }))
        .put("failures", JSONArray(failures))
        .put("rows", rows)
    val reportFile = ExternalTestData.reportFile("decoder_parity.json")
    reportFile.writeText(report.toString(2) + "\n")
    println(
      "Decoder: Python $pythonIdentical/80, oracle $oracleIdentical/80, order $pythonOrderIdentical/80; max score error Python=$maxPythonDifference, oracle=$maxOracleDifference"
    )
    assertTrue(failures.joinToString("\n"), failures.isEmpty())
  }

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

  private fun JSONArray.strings() = (0 until length()).map { getString(it) }

  private fun expectedSpans(entities: JSONArray): Map<Triple<String, Int, Int>, Double> =
    (0 until entities.length()).associate { index ->
      entities.getJSONObject(index).let {
        Triple(it.getString("label"), it.getInt("start"), it.getInt("end")) to it.getDouble("score")
      }
    }

  private fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
