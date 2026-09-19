package com.gliner25

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.util.Log
import java.io.File
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/** Extended debug validation, including Android host-input parity and lazy window loading. */
internal class GlinerF1GateRunner(
  private val context: Context,
  private val extractor: () -> GlinerExtractor,
  private val profile: Boolean = false,
) {
  fun run(requestedAccelerator: String?): List<GlinerGateRunner.Summary> {
    check(
      context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0 ||
        (BuildConfig.BUILD_TYPE == "benchmark" && profile)
    ) {
      "Gate fixtures are available only in the debug APK."
    }
    val backends =
      if (requestedAccelerator == null) {
        listOf(GlinerExtractor.Backend.GPU, GlinerExtractor.Backend.CPU)
      } else {
        listOf(GlinerExtractor.Backend.valueOf(requestedAccelerator.uppercase(Locale.ROOT)))
      }
    val fixtures =
      context.assets.open("gate_f1_fixtures.json").bufferedReader().use {
        JSONObject(it.readText()).getJSONArray("fixtures")
      }
    check(fixtures.length() == 70)
    return backends.map { runBackend(it, fixtures) }
  }

  private fun runBackend(
    backend: GlinerExtractor.Backend,
    fixtures: JSONArray,
  ): GlinerGateRunner.Summary {
    val timedRuns =
      if (profile) {
        5
      } else {
        3
      }
    val stageTimings =
      GlinerInputs.WINDOWS.associate { it.sequenceLength to mutableListOf<Map<String, Double>>() }
    val destination = File(context.filesDir, "gate/gate_f1_${backend.name}.json")
    destination.parentFile?.mkdirs()
    val rows = JSONArray()
    val windows = JSONObject()
    val timings =
      GlinerInputs.WINDOWS.associate {
        it.sequenceLength to mutableListOf<GlinerExtractor.Timing>()
      }
    val loaded = mutableSetOf<Int>()
    GlinerInputs.WINDOWS.forEach { window ->
      windows.put(
        "s${window.sequenceLength}",
        JSONObject()
          .put("window", window.sequenceLength)
          .put("text_capacity", window.textCapacity)
          .put(
            "expected_fixtures",
            if (window.sequenceLength == 128) {
              60
            } else {
              5
            },
          )
          .put("fixture_count", 0)
          .put("completed_warmup_runs", 0)
          .put("completed_timed_runs", 0)
          .put("compile_status", "NOT_RUN")
          .put("load_or_warmup_error", JSONObject.NULL),
      )
    }
    val header =
      JSONObject()
        .put("set", "f1")
        .put("litert_version", GlinerExtractor.LITERT_VERSION)
        .put("accelerator", backend.name)
        .put("precision", backend.precision)
        .put("device_model", Build.MODEL)
        .put("manufacturer", Build.MANUFACTURER)
        .put("android_sdk", Build.VERSION.SDK_INT)
        .put("build_fingerprint", Build.FINGERPRINT)
        .put("fixture_count", fixtures.length())
        .put("warmup_runs_per_fixture", 1)
        .put("timed_runs_per_fixture", timedRuns)
        .put("debuggable", context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0)
        .put("decoder_profile", profile)
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
        .put(
          "input_comparison",
          "Same extractor input builder, outside timed runs; padding checked",
        )
        .put(
          "finite_check",
          "Every warm-up and timed packed output is checked before sparse decoding",
        )
        .put(
          "window_median_method",
          "All timed runs in each window, midpoint average for even counts",
        )
        .put("windows", windows)
        .put("fixtures", rows)
    write(destination, header)
    var failureText: String? = null
    var passedFixtures = 0
    var inputMatches = 0
    var spanMatches = 0
    var confidenceMatches = 0
    var maxDifference = 0.0
    var completedRuns = 0
    var finiteFixtures = 0
    try {
      val helper = extractor()
      helper.initialize(backend)
      loaded.add(128)
      windows.getJSONObject("s128").put("compile_status", "PASS")
      header.put("compile_message", "s128 compiled; s256/s512 load on first fitting input")
      for (index in 0 until fixtures.length()) {
        val fixture = fixtures.getJSONObject(index)
        val text = GlinerGateFixtures.text(fixture)
        val row =
          JSONObject()
            .put("id", fixture.getInt("index"))
            .put("text", text)
            .put("expected_window", fixture.getInt("window"))
            .put("inputs_identical", false)
            .put("oracle_identical", false)
            .put("all_outputs_finite", false)
        rows.put(row)
        var currentWindow: JSONObject? = null
        var warmupComplete = false
        try {
          val prepared = helper.inspectInputs(text)
          val window = prepared.window.sequenceLength
          currentWindow = windows.getJSONObject("s$window")
          currentWindow.put("fixture_count", currentWindow.getInt("fixture_count") + 1)
          val comparison = GlinerGateFixtures.compareInputs(fixture, prepared)
          val inputsIdentical = comparison.getBoolean("identical")
          if (inputsIdentical) {
            inputMatches++
          }
          row
            .put("window", window)
            .put("encoded_tokens", prepared.encodedLength)
            .put("text_words", prepared.words.size)
            .put("inputs_identical", inputsIdentical)
            .put("inputs", comparison)
          helper.warmUp(text, backend)
          warmupComplete = true
          loaded.add(window)
          currentWindow
            .put("compile_status", "PASS")
            .put("completed_warmup_runs", currentWindow.getInt("completed_warmup_runs") + 1)
          val runs = (0 until timedRuns).map { helper.extract(text, backend) }
          completedRuns += runs.size
          timings.getValue(window).addAll(runs.map { it.timing })
          stageTimings.getValue(window).addAll(runs.map { it.decoderStagesMs })
          currentWindow.put("completed_timed_runs", timings.getValue(window).size)
          val comparisons = runs.map {
            GlinerGateFixtures.compareSpans(it.spans, fixture.getJSONArray("spans"))
          }
          val identical = comparisons.all { it.first }
          val difference = comparisons.maxOf { it.second }
          val confidencePass = identical && difference <= 5e-3
          if (identical) {
            spanMatches++
          }
          if (confidencePass) {
            confidenceMatches++
          }
          val passed = inputsIdentical && confidencePass
          if (passed) {
            passedFixtures++
          }
          finiteFixtures++
          maxDifference = maxOf(maxDifference, difference)
          row
            .put(
              "status",
              if (passed) {
                "PASS"
              } else {
                "FAIL"
              },
            )
            .put("oracle_identical", identical)
            .put("max_confidence_diff", difference)
            .put("all_outputs_finite", true)
            .put("spans", GlinerGateFixtures.spans(runs.last().spans))
            .put("ms", medianTimes(runs.map { it.timing }))
            .put(
              "runs",
              JSONArray(
                runs.mapIndexed { run, result ->
                  JSONObject()
                    .put("index", run)
                    .put("oracle_identical", comparisons[run].first)
                    .put("max_confidence_diff", comparisons[run].second)
                    .put("spans", GlinerGateFixtures.spans(result.spans))
                    .put("tokenize_embed_ms", result.timing.tokenizeEmbedMs)
                    .put("graph_ms", result.timing.graphMs)
                    .put("decode_ms", result.timing.decodeMs)
                    .put("decoder_stages_ms", JSONObject(result.decoderStagesMs))
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
            .put("max_confidence_diff", JSONObject.NULL)
          if (!warmupComplete) {
            currentWindow?.put("load_or_warmup_error", failureText)
          }
        }
        write(destination, header)
      }
    } catch (failure: Exception) {
      failureText = describe(failure)
    } catch (failure: LinkageError) {
      failureText = describe(failure)
    }
    timings.forEach { (window, values) ->
      windows.getJSONObject("s$window").put("ms", medianTimes(values))
      if (profile) {
        val samples = stageTimings.getValue(window)
        val stages = JSONObject()
        samples.firstOrNull()?.keys?.forEach { key ->
          stages.put(key, median(samples.map { it.getValue(key) }))
        }
        windows.getJSONObject("s$window").put("decoder_stages_ms", stages)
        Log.i(
          "GLINER_GATE",
          JSONObject()
            .put("profile_window", window)
            .put("accelerator", backend.name)
            .put("debuggable", header.getBoolean("debuggable"))
            .put("ms", medianTimes(values))
            .put("decoder_stages_ms", stages)
            .toString(),
        )
      }
    }
    val compiled = loaded == setOf(128, 256, 512)
    val passed =
      passedFixtures == 70 && completedRuns == 70 * timedRuns && compiled && failureText == null
    header
      .put(
        "status",
        if (passed) {
          "PASS"
        } else {
          "FAIL"
        },
      )
      .put(
        "compile_status",
        if (compiled) {
          "PASS"
        } else {
          "INCOMPLETE"
        },
      )
      .put(
        "compile_message",
        "Compiled windows: ${loaded.sorted()}; ${failureText ?: "no exception"}",
      )
      .put("passed_fixtures", passedFixtures)
      .put("inputs_matching_fixtures", inputMatches)
      .put("oracle_matching_fixtures", spanMatches)
      .put("oracle_confidence_matching_fixtures", confidenceMatches)
      .put("completed_timed_runs", completedRuns)
      .put("all_outputs_finite", finiteFixtures == 70)
      .put("max_confidence_diff", maxDifference)
      .put("error", failureText ?: JSONObject.NULL)
    write(destination, header)
    Log.i(
      "GLINER_GATE",
      JSONObject()
        .put("set", "f1")
        .put("accelerator", backend.name)
        .put("status", header.getString("status"))
        .put("fixture_count", 70)
        .put("inputs_matching_fixtures", inputMatches)
        .put("oracle_matching_fixtures", spanMatches)
        .put("max_confidence_diff", maxDifference)
        .put("path", destination.absolutePath)
        .put("error", failureText ?: JSONObject.NULL)
        .toString(),
    )
    return GlinerGateRunner.Summary(backend.name, passed, destination.absolutePath, failureText)
  }

  private fun medianTimes(values: List<GlinerExtractor.Timing>): JSONObject =
    JSONObject()
      .put("tokenize_embed_median", median(values.map { it.tokenizeEmbedMs }))
      .put("graph_median", median(values.map { it.graphMs }))
      .put("decode_median", median(values.map { it.decodeMs }))
      .put("write_median", median(values.map { it.writeMs }))
      .put("enqueue_median", median(values.map { it.enqueueMs }))
      .put("readback_median", median(values.map { it.readbackMs }))

  private fun median(values: List<Double>): Any {
    if (values.isEmpty()) {
      return JSONObject.NULL
    }
    val sorted = values.sorted()
    return (sorted[(sorted.size - 1) / 2] + sorted[sorted.size / 2]) / 2.0
  }

  private fun write(file: File, content: JSONObject) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(content.toString(2) + "\n")
    check(temporary.renameTo(file)) { "Could not save gate result ${file.absolutePath}" }
  }

  private fun describe(failure: Throwable): String =
    generateSequence(failure) { it.cause }
      .joinToString("; caused by ") { "${it.javaClass.simpleName}: ${it.message.orEmpty()}" }
}
