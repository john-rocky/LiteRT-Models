package com.gliner25

import java.io.File
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/** Uses external CPU references described in scripts/TEST_DATA.md. */
class GlinerDecoderParityTest {
  private lateinit var root: File

  @Before
  fun locateExternalData() {
    root = ExternalTestData.resolve()
  }

  @Test
  fun publishedPackedOutputsMatchPythonAndOfficialOracle() {
    ExternalTestData.requireFiles(
      root,
      "fixtures/packed/reference.json",
      "fixtures/packed/decoder_traces.json",
      "host_assets/sparse_decoder_fp32.safetensors",
    )
    val referenceFile = File(root, "fixtures/packed/reference.json")
    assertTrue("Generate the published CPU outputs first: $referenceFile", referenceFile.isFile)
    val reference = JSONObject(referenceFile.readText())
    val entries = reference.getJSONArray("entries")
    val decoder = GlinerDecoder(File(root, "host_assets"))
    val pythonTraces =
      JSONObject(File(root, "fixtures/packed/decoder_traces.json").readText())
        .getJSONArray("traces")
    val traceByKey =
      (0 until pythonTraces.length()).associate { i ->
        val trace = pythonTraces.getJSONObject(i)
        "s${trace.getInt("window")}/${trace.getString("input_id")}" to trace
      }
    val pairs = JSONArray()
    var poolOrderIdentical = 0
    var maxLogitDifference = 0.0
    var maxCompatibilityDifference = 0.0
    var oracleIdentical = 0
    var pythonIdentical = 0
    var maxPythonDifference = 0.0
    var maxOracleDifference = 0.0
    var maxPythonFixture = ""
    var maxOracleFixture = ""
    val uniqueInputs = mutableSetOf<String>()
    val byWindow = mutableMapOf<Int, Int>()
    val failures = mutableListOf<String>()
    for (i in 0 until entries.length()) {
      val entry = entries.getJSONObject(i)
      val window = entry.getInt("window")
      val id = entry.getString("input_id")
      val key = "s$window/$id"
      uniqueInputs += id
      byWindow[window] = (byWindow[window] ?: 0) + 1
      val captured = JSONObject(File(root, entry.getString("captured")).readText())
      val text = entry.getString("text")
      val starts = captured.getJSONArray("start_mappings").getJSONArray(0)
      val ends = captured.getJSONArray("end_mappings").getJSONArray(0)
      val words =
        (0 until starts.length()).map { word ->
          val start = starts.getInt(word)
          val end = ends.getInt(word)
          GlinerInputs.Word(
            text.substring(text.offsetByCodePoints(0, start), text.offsetByCodePoints(0, end)),
            start,
            end,
          )
        }
      val packed =
        GlinerDecoder.readPacked(File(root, "fixtures/packed/${entry.getString("packed")}"))
      assertEquals(
        "$key packed size",
        1108 *
          when (window) {
            128 -> 48
            256 -> 192
            512 -> 384
            else -> error(window)
          } + 4574,
        packed.size,
      )
      val actual = decoder.decode(packed, text, words)
      val trace = decoder.trace(packed, words.size)
      val expectedTrace = traceByKey.getValue(key).getJSONArray("candidates")
      val expectedPool =
        (0 until expectedTrace.length()).map { c ->
          val item = expectedTrace.getJSONObject(c)
          item.getInt("start") to item.getInt("end")
        }
      val actualPool = trace.candidates.map { it.start to it.end }
      val poolMatches = actualPool == expectedPool
      if (poolMatches) {
        poolOrderIdentical++
      } else {
        failures += "$key candidate pool/order mismatch"
      }
      val expectedTraceBySpan =
        (0 until expectedTrace.length()).associate { c ->
          val item = expectedTrace.getJSONObject(c)
          (item.getInt("start") to item.getInt("end")) to item
        }
      for ((c, candidate) in trace.candidates.withIndex()) {
        expectedTraceBySpan[candidate.start to candidate.end]?.let { expected ->
          maxCompatibilityDifference =
            maxOf(
              maxCompatibilityDifference,
              abs(candidate.compatibility.toDouble() - expected.getDouble("compatibility")),
            )
          val expectedLogits = expected.getJSONArray("logits")
          for (q in 0 until expectedLogits.length()) {
            maxLogitDifference =
              maxOf(
                maxLogitDifference,
                abs(trace.logits[c][q].toDouble() - expectedLogits.getDouble(q)),
              )
          }
        }
      }
      val actualByKey = actual.associateBy { Triple(it.label, it.start, it.end) }
      assertEquals("$key contains duplicate spans", actual.size, actualByKey.size)
      val python = expectedSpans(entry.getJSONArray("python_spans"))
      val oracle = expectedSpans(entry.getJSONArray("oracle_spans"))
      val pythonSetsMatch = actualByKey.keys == python.keys
      val oracleSetsMatch = actualByKey.keys == oracle.keys
      if (pythonSetsMatch) {
        pythonIdentical++
      } else {
        failures +=
          "$key Python sets: missing=${python.keys - actualByKey.keys}, " +
            "extra=${actualByKey.keys - python.keys}"
      }
      if (oracleSetsMatch) {
        oracleIdentical++
      } else {
        failures +=
          "$key oracle sets: missing=${oracle.keys - actualByKey.keys}, " +
            "extra=${actualByKey.keys - oracle.keys}"
      }
      var pythonDifference = 0.0
      var oracleDifference = 0.0
      for ((spanKey, span) in actualByKey) {
        assertTrue("$key nonfinite confidence", span.confidence.isFinite())
        assertEquals(
          "$key original surface",
          text.substring(
            text.offsetByCodePoints(0, span.start),
            text.offsetByCodePoints(0, span.end),
          ),
          span.text,
        )
        python[spanKey]?.let {
          pythonDifference = maxOf(pythonDifference, abs(span.confidence.toDouble() - it))
        }
        oracle[spanKey]?.let {
          oracleDifference = maxOf(oracleDifference, abs(span.confidence.toDouble() - it))
        }
      }
      if (pythonDifference > maxPythonDifference) {
        maxPythonDifference = pythonDifference
        maxPythonFixture = key
      }
      if (oracleDifference > maxOracleDifference) {
        maxOracleDifference = oracleDifference
        maxOracleFixture = key
      }
      if (pythonDifference > 1e-5) {
        failures += "$key confidence vs Python: $pythonDifference > 1e-5"
      }
      if (oracleDifference > 5e-3) {
        failures += "$key confidence vs oracle: $oracleDifference > 5e-3"
      }
      val row =
        JSONObject()
          .put("window", window)
          .put("input_id", id)
          .put("pool_order_identical", poolMatches)
          .put("oracle_span_sets_identical", oracleSetsMatch)
          .put("python_span_sets_identical", pythonSetsMatch)
          .put("max_confidence_difference_python", pythonDifference)
          .put("max_confidence_difference_oracle", oracleDifference)
          .put(
            "spans",
            JSONArray(
              actual.map { span ->
                JSONObject()
                  .put("label", span.label)
                  .put("text", span.text)
                  .put("start", span.start)
                  .put("end", span.end)
                  .put("confidence", span.confidence.toDouble())
              }
            ),
          )
      pairs.put(row)
      if (!pythonSetsMatch || pythonDifference > 1e-5 || !poolMatches) {
        val debugFile = ExternalTestData.reportFile("decoder_debug/${key.replace('/', '_')}.json")
        debugFile.parentFile.mkdirs()
        debugFile.writeText(
          JSONObject()
            .put(
              "candidates",
              JSONArray(
                trace.candidates.mapIndexed { c, candidate ->
                  JSONObject()
                    .put("start", candidate.start)
                    .put("end", candidate.end)
                    .put("compatibility", candidate.compatibility.toDouble())
                    .put("logits", JSONArray(trace.logits[c].map { it.toDouble() }))
                }
              ),
            )
            .toString(2)
        )
      }
    }
    val aliases = reference.getJSONArray("aliases")
    val fixturePaths =
      (0 until entries.length())
        .map { entries.getJSONObject(it).getString("captured") }
        .toMutableSet()
    for (i in 0 until aliases.length()) {
      val alias = aliases.getJSONObject(i)
      fixturePaths += alias.getString("captured")
      val canonical =
        File(root, "fixtures/packed/s${alias.getInt("window")}/${alias.getString("input_id")}.bin")
      val duplicate = File(root, "fixtures/packed/${alias.getString("packed")}")
      assertTrue(
        "Duplicate short output differs: $duplicate",
        canonical.readBytes().contentEquals(duplicate.readBytes()),
      )
    }
    assertEquals("all 80 fixture paths covered", 80, fixturePaths.size)
    assertEquals("all 30 duplicate window pairs covered", 30, aliases.length())
    val report =
      JSONObject()
        .put(
          "status",
          if (failures.isEmpty()) {
            "PASS"
          } else {
            "FAIL"
          },
        )
        .put("fixture_files", fixturePaths.size)
        .put("file_window_pairs", entries.length() + aliases.length())
        .put("duplicate_pairs_byte_identical", aliases.length())
        .put("unique_inputs", uniqueInputs.size)
        .put("unique_window_input_pairs", entries.length())
        .put("pool_order_identical", poolOrderIdentical)
        .put("max_pair_logit_difference_python", maxLogitDifference)
        .put("max_pool_compatibility_difference_python", maxCompatibilityDifference)
        .put("oracle_span_sets_identical", oracleIdentical)
        .put("python_span_sets_identical", pythonIdentical)
        .put("max_confidence_difference_python", maxPythonDifference)
        .put("max_confidence_difference_python_fixture", maxPythonFixture)
        .put("max_confidence_difference_oracle", maxOracleDifference)
        .put("max_confidence_difference_oracle_fixture", maxOracleFixture)
        .put("counts_by_window", JSONObject(byWindow.mapKeys { "s${it.key}" }))
        .put("failures", JSONArray(failures))
        .put("pairs", pairs)
    ExternalTestData.reportFile("decoder_parity.json").writeText(report.toString(2))
    println(
      "Decoder: ${entries.length()} pairs, oracle $oracleIdentical, Python $pythonIdentical; " +
        "max confidence differences Python=$maxPythonDifference ($maxPythonFixture), " +
        "oracle=$maxOracleDifference ($maxOracleFixture)"
    )
    assertEquals("fixture pair count", 195, entries.length())
    assertEquals("unique input count", 70, uniqueInputs.size)
    assertEquals(mapOf(128 to 60, 256 to 65, 512 to 70), byWindow)
    assertTrue(failures.joinToString("\n"), failures.isEmpty())
  }

  private fun expectedSpans(spans: JSONArray): Map<Triple<String, Int, Int>, Double> =
    (0 until spans.length()).associate { i ->
      val span = spans.getJSONObject(i)
      Triple(span.getString("label"), span.getInt("start"), span.getInt("end")) to
        span.getDouble("confidence")
    }
}
