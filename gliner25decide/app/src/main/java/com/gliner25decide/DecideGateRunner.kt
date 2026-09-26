package com.gliner25decide

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.util.Log
import java.io.File
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-asset gate, called only on the ViewModel's model dispatcher. For every (fixture, window)
 * pair of `gate_fixtures.json` (the Galaxy S26 device-gate set, 42 per window): (1) the on-device
 * tokenizer/schema/padding reproduce the captured Python inputs byte for byte; (2) the graph on the
 * chosen backend gives the official decisions, with max |Δlogit| / |Δprob| against the fp32 oracle
 * and max |Δlogit| against Mac LiteRT CPU on the same graph and table; (3) timing medians.
 */
class DecideGateRunner(
  private val context: Context,
  private val classifier: () -> DecideClassifier,
) {
  /** Outcome of one backend's gate and the path of its JSON report. */
  data class Summary(val backend: String, val passed: Boolean, val path: String, val error: String?)

  /** Passed and total (fixture, window) pairs of one window. */
  private class WindowCount(var passed: Int = 0, var total: Int = 0)

  /**
   * Runs the gate on [requestedAccelerator] ("GPU" or "CPU"), or on GPU then CPU when it is null,
   * and writes `files/gate/gate_<BACKEND>.json` for each backend.
   */
  fun run(requestedAccelerator: String?): List<Summary> {
    check(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
      "Gate fixtures are available only in the debug APK."
    }
    val backends =
      if (requestedAccelerator == null) {
        listOf(DecideClassifier.Backend.GPU, DecideClassifier.Backend.CPU)
      } else {
        listOf(DecideClassifier.Backend.valueOf(requestedAccelerator.uppercase(Locale.ROOT)))
      }
    val root =
      context.assets.open("gate_fixtures.json").bufferedReader().use { JSONObject(it.readText()) }
    val fixtures = DecideGateFixtures.parse(root)
    check(fixtures.sumOf { it.windows.size } == EXPECTED_PAIRS) { "Unexpected gate fixture set" }
    return backends.map { runBackend(it, fixtures) }
  }

  private fun runBackend(
    backend: DecideClassifier.Backend,
    fixtures: List<DecideGateFixtures.Fixture>,
  ): Summary {
    val destination = File(context.filesDir, "gate/gate_${backend.name}.json")
    destination.parentFile?.mkdirs()
    val rows = JSONArray()
    val header =
      JSONObject()
        .put("litert_version", DecideClassifier.LITERT_VERSION)
        .put("accelerator", backend.name)
        .put("precision", backend.precision)
        .put("device_model", Build.MODEL)
        .put("manufacturer", Build.MANUFACTURER)
        .put("android_sdk", Build.VERSION.SDK_INT)
        .put("build_fingerprint", Build.FINGERPRINT)
        .put("fixture_count", fixtures.size)
        .put("window_pairs", EXPECTED_PAIRS)
        .put("warmup_runs_per_pair", WARMUP_RUNS)
        .put("timed_runs_per_pair", TIMED_RUNS)
        .put(
          "cpu_threads",
          if (backend == DecideClassifier.Backend.CPU) {
            DecideClassifier.CPU_THREADS
          } else {
            JSONObject.NULL
          },
        )
        .put("status", "RUNNING")
        .put("compile_status", "NOT_RUN")
        .put("compile_message", "Not attempted")
        .put("graph_timing", "First input write through synchronized output readback")
        .put("pairs", rows)
    write(destination, header)
    var failureText: String? = null
    var inputMatches = 0
    var decisionMatches = 0
    var passedPairs = 0
    var completedRuns = 0
    var maxLogit = 0.0
    var maxProbability = 0.0
    var maxMacLogit = 0.0
    val perWindow = linkedMapOf<Int, WindowCount>()
    try {
      val helper = classifier()
      helper.initialize(backend)
      header.put("compile_status", "PASS").put("compile_message", "s128 compiled successfully")
      write(destination, header)
      for (fixture in fixtures) {
        for (window in fixture.windows) {
          val row = JSONObject().put("id", fixture.id).put("window", window)
          rows.put(row)
          val counts = perWindow.getOrPut(window) { WindowCount() }
          counts.total++
          try {
            val inputs =
              DecideGateFixtures.compareInputs(
                fixture,
                helper.inspectInputs(fixture.text, fixture.tasks, window),
              )
            val inputsIdentical = inputs.getBoolean("identical")
            if (inputsIdentical) {
              inputMatches++
            }
            repeat(WARMUP_RUNS) { helper.warmUp(fixture.text, fixture.tasks, backend, window) }
            val runs =
              (0 until TIMED_RUNS).map {
                helper.classify(fixture.text, fixture.tasks, backend, window)
              }
            completedRuns += runs.size
            val labelCount = fixture.oracleLogits.size
            val finite = runs.all { run -> run.logits.all { it.isFinite() } }
            val decisionsEqual = runs.all {
              DecideGateFixtures.decisionsEqual(it.decisions, fixture.official)
            }
            if (decisionsEqual) {
              decisionMatches++
            }
            val logitDifference = runs.maxOf {
              DecideGateFixtures.maxAbsDifference(it.logits, fixture.oracleLogits, labelCount)
            }
            val probabilityDifference = runs.maxOf {
              DecideGateFixtures.maxAbsDifference(
                DecideGateFixtures.probabilities(it.decisions),
                fixture.oracleProbabilities,
              )
            }
            val macDifference =
              fixture.macCpuLogits[window]?.let { reference ->
                runs.maxOf { DecideGateFixtures.maxAbsDifference(it.logits, reference, labelCount) }
              }
            maxLogit = maxOf(maxLogit, logitDifference)
            maxProbability = maxOf(maxProbability, probabilityDifference)
            macDifference?.let { maxMacLogit = maxOf(maxMacLogit, it) }
            val passed = inputsIdentical && decisionsEqual && finite
            if (passed) {
              passedPairs++
              counts.passed++
            }
            val final = runs.last()
            row
              .put("status", if (passed) "PASS" else "FAIL")
              .put("inputs", inputs)
              .put("encoded_tokens", final.encodedTokens)
              .put("label_count", final.labelCount)
              .put("decisions_equal_official", decisionsEqual)
              .put("all_logits_finite", finite)
              .put("max_abs_dlogit_vs_oracle", logitDifference)
              .put("max_abs_dprob_vs_oracle", probabilityDifference)
              .put("max_abs_dlogit_vs_mac_cpu", macDifference ?: JSONObject.NULL)
              .put("decisions", DecideGateFixtures.decisionsJson(final.decisions))
              .put("logits", JSONArray(final.logits.map { it.toDouble() }))
              .put(
                "ms",
                JSONObject()
                  .put("tokenize_embed_median", median(runs.map { it.timing.tokenizeEmbedMs }))
                  .put("graph_median", median(runs.map { it.timing.graphMs }))
                  .put("decode_median", median(runs.map { it.timing.decodeMs }))
                  .put("write_median", median(runs.map { it.timing.writeMs }))
                  .put("enqueue_median", median(runs.map { it.timing.enqueueMs }))
                  .put("readback_median", median(runs.map { it.timing.readbackMs }))
                  .put("graph_runs", JSONArray(runs.map { it.timing.graphMs })),
              )
          } catch (failure: Exception) {
            failureText = describe(failure)
            row.put("status", "FAIL").put("error", failureText)
          }
          write(destination, header)
        }
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
    val passed =
      passedPairs == EXPECTED_PAIRS &&
        completedRuns == EXPECTED_PAIRS * TIMED_RUNS &&
        failureText == null
    header
      .put("status", if (passed) "PASS" else "FAIL")
      .put("passed_pairs", passedPairs)
      .put("input_identical_pairs", inputMatches)
      .put("decision_identical_pairs", decisionMatches)
      .put("completed_timed_runs", completedRuns)
      .put("max_abs_dlogit_vs_oracle", maxLogit)
      .put("max_abs_dprob_vs_oracle", maxProbability)
      .put("max_abs_dlogit_vs_mac_cpu", maxMacLogit)
      .put(
        "passed_by_window",
        JSONObject().apply {
          perWindow.forEach { (window, count) ->
            put("s$window", "${count.passed}/${count.total}")
          }
        },
      )
      .put("error", failureText ?: JSONObject.NULL)
    write(destination, header)
    Log.i(
      DecideGateFixtures.GATE_LOG_TAG,
      JSONObject()
        .put("accelerator", backend.name)
        .put("status", header.getString("status"))
        .put("passed_pairs", passedPairs)
        .put("pairs", EXPECTED_PAIRS)
        .put("max_abs_dlogit_vs_oracle", maxLogit)
        .put("path", destination.absolutePath)
        .put("error", failureText ?: JSONObject.NULL)
        .toString(),
    )
    return Summary(backend.name, passed, destination.absolutePath, failureText)
  }

  private fun median(values: List<Double>): Double = values.sorted()[values.size / 2]

  private fun write(file: File, content: JSONObject) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(content.toString(DecideGateFixtures.REPORT_JSON_INDENT) + "\n")
    check(temporary.renameTo(file)) { "Could not save gate result ${file.absolutePath}" }
  }

  private fun describe(failure: Throwable) =
    "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}"

  companion object {
    /** 42 fixtures per window × 3 windows (the device-gate selection in the debug asset). */
    const val EXPECTED_PAIRS = 126

    /** Untimed full passes per (fixture, window) pair before the timed runs. */
    const val WARMUP_RUNS = 1

    /** Timed classifications per (fixture, window) pair. */
    const val TIMED_RUNS = 3
  }
}
