package com.gliner25

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.util.Log
import java.io.File
import java.util.Locale
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject

/** Debug-asset gate, called only on the ViewModel's model dispatcher. */
class GlinerGateRunner(
  private val context: Context,
  private val extractor: () -> GlinerExtractor,
) {
  data class Summary(val backend: String, val passed: Boolean, val path: String, val error: String?)

  fun run(requestedAccelerator: String?): List<Summary> {
    check(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
      "Gate fixtures are available only in the debug APK."
    }
    val backends =
      if (requestedAccelerator == null) {
        listOf(GlinerExtractor.Backend.GPU, GlinerExtractor.Backend.CPU)
      } else {
        listOf(GlinerExtractor.Backend.valueOf(requestedAccelerator.uppercase(Locale.ROOT)))
      }
    val fixtureRoot =
      context.assets.open("gate_fixtures.json").bufferedReader().use {
        JSONObject(it.readText())
      }
    val fixtures = fixtureRoot.getJSONArray("fixtures")
    check(fixtures.length() == 10)
    return backends.map { runBackend(it, fixtures) }
  }

  private fun runBackend(backend: GlinerExtractor.Backend, fixtures: JSONArray): Summary {
    val destination = File(context.filesDir, "gate/gate_${backend.name}.json")
    destination.parentFile?.mkdirs()
    val rows = JSONArray()
    val header =
      JSONObject()
        .put("litert_version", GlinerExtractor.LITERT_VERSION)
        .put("accelerator", backend.name)
        .put("precision", backend.precision)
        .put("device_model", Build.MODEL)
        .put("manufacturer", Build.MANUFACTURER)
        .put("android_sdk", Build.VERSION.SDK_INT)
        .put("build_fingerprint", Build.FINGERPRINT)
        .put("fixture_count", fixtures.length())
        .put("warmup_runs_per_fixture", 1)
        .put("timed_runs_per_fixture", 5)
        .put(
          "cpu_threads",
          if (backend == GlinerExtractor.Backend.CPU) {
            4
          } else {
            JSONObject.NULL
          },
        )
        .put("confidence_tolerance", 5e-3)
        .put("status", "RUNNING")
        .put("compile_status", "NOT_RUN")
        .put("compile_message", "Not attempted")
        .put("graph_timing", "First input write through synchronized output readback")
        .put("fixtures", rows)
    write(destination, header)
    var failureText: String? = null
    var matches = 0
    var maxDifference = 0.0
    var completedRuns = 0
    try {
      val helper = extractor()
      helper.initialize(backend)
      header.put("compile_status", "PASS").put("compile_message", "s128 compiled successfully")
      write(destination, header)
      for (index in 0 until fixtures.length()) {
        val fixture = fixtures.getJSONObject(index)
        val text = fixture.getString("text")
        val row = JSONObject().put("id", fixture.getInt("index")).put("text", text)
        rows.put(row)
        try {
          helper.warmUp(text, backend)
          val runs = (0 until 5).map { helper.extract(text, backend) }
          completedRuns += runs.size
          val comparisons = runs.map { compare(it.spans, fixture.getJSONArray("spans")) }
          val identical = comparisons.all { it.first }
          val difference = comparisons.maxOf { it.second }
          val passed = identical && difference <= 5e-3
          if (passed) {
            matches++
          }
          maxDifference = maxOf(maxDifference, difference)
          val final = runs.last()
          row
            .put(
              "status",
              if (passed) {
                "PASS"
              } else {
                "FAIL"
              },
            )
            .put("window", final.window)
            .put("encoded_tokens", final.encodedTokens)
            .put("text_words", final.textWords)
            .put("oracle_identical", identical)
            .put("max_confidence_diff", difference)
            .put("spans", spans(final.spans))
            .put(
              "ms",
              JSONObject()
                .put("tokenize_embed_median", median(runs.map { it.timing.tokenizeEmbedMs }))
                .put("graph_median", median(runs.map { it.timing.graphMs }))
                .put("decode_median", median(runs.map { it.timing.decodeMs }))
                .put("write_median", median(runs.map { it.timing.writeMs }))
                .put("enqueue_median", median(runs.map { it.timing.enqueueMs }))
                .put("readback_median", median(runs.map { it.timing.readbackMs })),
            )
            .put(
              "runs",
              JSONArray(
                runs.mapIndexed { run, result ->
                  JSONObject()
                    .put("index", run)
                    .put("oracle_identical", comparisons[run].first)
                    .put("max_confidence_diff", comparisons[run].second)
                    .put("spans", spans(result.spans))
                    .put("tokenize_embed_ms", result.timing.tokenizeEmbedMs)
                    .put("graph_ms", result.timing.graphMs)
                    .put("decode_ms", result.timing.decodeMs)
                    .put("write_ms", result.timing.writeMs)
                    .put("enqueue_ms", result.timing.enqueueMs)
                    .put("readback_ms", result.timing.readbackMs)
                }
              ),
            )
        } catch (failure: Exception) {
          failureText = describe(failure)
          row
            .put("status", "FAIL")
            .put("error", failureText)
            .put("oracle_identical", false)
            .put("max_confidence_diff", JSONObject.NULL)
            .put("spans", JSONArray())
        }
        write(destination, header)
      }
    } catch (failure: Exception) {
      failureText = describe(failure)
      header
        .put("compile_status", "FAIL")
        .put("compile_message", "Initialization/compile failed: $failureText")
    } catch (failure: LinkageError) {
      failureText = describe(failure)
      header
        .put("compile_status", "FAIL")
        .put("compile_message", "Native runtime load failed: $failureText")
    }
    val passed = matches == 10 && completedRuns == 50 && failureText == null
    header
      .put(
        "status",
        if (passed) {
          "PASS"
        } else {
          "FAIL"
        },
      )
      .put("oracle_matching_fixtures", matches)
      .put("completed_timed_runs", completedRuns)
      .put("max_confidence_diff", maxDifference)
      .put("error", failureText ?: JSONObject.NULL)
    write(destination, header)
    Log.i(
      "GLINER_GATE",
      JSONObject()
        .put("accelerator", backend.name)
        .put("status", header.getString("status"))
        .put("oracle_matching_fixtures", matches)
        .put("fixture_count", 10)
        .put("max_confidence_diff", maxDifference)
        .put("path", destination.absolutePath)
        .put("error", failureText ?: JSONObject.NULL)
        .toString(),
    )
    return Summary(backend.name, passed, destination.absolutePath, failureText)
  }

  private fun compare(
    actual: List<GlinerDecoder.Span>,
    expected: JSONArray,
  ): Pair<Boolean, Double> {
    val observed = actual.associateBy { Triple(it.label, it.start, it.end) }
    val oracle =
      (0 until expected.length()).associate { index ->
        val span = expected.getJSONObject(index)
        Triple(span.getString("label"), span.getInt("start"), span.getInt("end")) to
          span.getDouble("confidence")
      }
    val identical = observed.size == actual.size && observed.keys == oracle.keys
    val difference =
      observed.keys.intersect(oracle.keys).maxOfOrNull {
        abs(observed.getValue(it).confidence.toDouble() - oracle.getValue(it))
      } ?: 0.0
    return identical to difference
  }

  private fun spans(values: List<GlinerDecoder.Span>) =
    JSONArray(
      values.map {
        JSONObject()
          .put("label", it.label)
          .put("text", it.text)
          .put("start", it.start)
          .put("end", it.end)
          .put("confidence", it.confidence.toDouble())
      }
    )

  private fun median(values: List<Double>): Double = values.sorted()[values.size / 2]

  private fun write(file: File, content: JSONObject) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(content.toString(2) + "\n")
    check(temporary.renameTo(file)) { "Could not save gate result ${file.absolutePath}" }
  }

  private fun describe(failure: Throwable) =
    "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}"
}
