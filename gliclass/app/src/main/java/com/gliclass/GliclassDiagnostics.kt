package com.gliclass

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Diagnostic launches of the debug and benchmark builds (never the release build): the first
 * classification after a normal startup, and paced requests. Every report goes to `files/` and, as
 * compact JSON lines, to logcat under [GliclassGateFixtures.GATE_LOG_TAG], so the non-debuggable
 * benchmark build (no `run-as`) can be read from logcat.
 */
internal object GliclassDiagnostics {
  /** Reported instead of a thermal status below API 29, which has no thermal status. */
  const val THERMAL_STATUS_UNAVAILABLE = -1

  /** True in the debug and benchmark builds. */
  val enabled: Boolean
    get() = BuildConfig.DEBUG || BuildConfig.BUILD_TYPE == "benchmark"

  /** Startup phases of one launch, in milliseconds. */
  data class Startup(
    val processStartToReadyMs: Double,
    val activityCreateToReadyMs: Double,
    val initializeMs: Double,
    val warmUpMs: Double,
  )

  /** Records the startup and the first classification after Ready (`files/first_tap.json`). */
  fun firstTap(
    context: Context,
    startup: Startup,
    endToEndMs: Double,
    result: GliclassClassifier.Result,
  ) {
    val report =
      header(context, "firsttap", "PASS")
        .put("process_start_to_ready_ms", startup.processStartToReadyMs)
        .put("activity_create_to_ready_ms", startup.activityCreateToReadyMs)
        .put("initialize_ms", startup.initializeMs)
        .put("startup_warmup_ms", startup.warmUpMs)
        .put("startup_warmup_iterations", GliclassClassifier.STARTUP_WARMUP_ITERATIONS)
        .put("e2e_ms", endToEndMs)
        .put("e2e_timing", E2E_TIMING)
        .putResult(result)
    write(context, "first_tap.json", report)
  }

  /** Records a failed diagnostic launch. */
  fun failure(context: Context, set: String, failure: Throwable) {
    write(
      context,
      "${set}_failure.json",
      header(context, set, "FAIL").put("error", failure.message ?: failure.javaClass.simpleName),
    )
  }

  /** Paced requests: the normal startup, then the bundled example once per interval. */
  class Paced(
    private val context: Context,
    backend: GliclassClassifier.Backend,
    count: Int,
    intervalMs: Long,
  ) {
    private val requests = JSONArray()
    private val fileName = "paced_${backend.name}.json"
    private val report =
      header(context, "paced", "RUNNING")
        .put("count", count)
        .put("interval_ms", intervalMs)
        .put("schedule", "Request i starts (i + 1) × interval_ms after Ready, or at once if late")
        .put("startup_warmup_iterations", GliclassClassifier.STARTUP_WARMUP_ITERATIONS)
        .put("e2e_timing", E2E_TIMING)
        .put("requests", requests)

    /** Records model initialization (compile included) and the startup warm-up time. */
    fun startup(startup: Startup) {
      report
        .put("process_start_to_ready_ms", startup.processStartToReadyMs)
        .put("activity_create_to_ready_ms", startup.activityCreateToReadyMs)
        .put("initialize_ms", startup.initializeMs)
        .put("startup_warmup_ms", startup.warmUpMs)
    }

    /** Records request [index], its start offset after Ready and its end-to-end time. */
    fun request(
      index: Int,
      startOffsetMs: Double,
      endToEndMs: Double,
      result: GliclassClassifier.Result,
    ) {
      val line =
        JSONObject()
          .put("index", index)
          .put("start_offset_ms", startOffsetMs)
          .put("e2e_ms", endToEndMs)
          .put("thermal_status", thermalStatus(context))
          .put("process_importance", processImportance())
          .putResult(result)
      requests.put(line)
      Log.i(GliclassGateFixtures.GATE_LOG_TAG, "paced_request " + line.toString())
    }

    /** Writes the final report after the last request. */
    fun complete() {
      report.put("status", "PASS")
      write(context, fileName, report, logLine = false)
      Log.i(
        GliclassGateFixtures.GATE_LOG_TAG,
        "paced_done " + JSONObject(report.toString()).apply { remove("requests") }.toString(),
      )
    }

    /** Records a failure. */
    fun failure(failure: Throwable) {
      report.put("status", "FAIL").put("error", failure.message ?: failure.javaClass.simpleName)
      write(context, fileName, report)
    }
  }

  private const val E2E_TIMING =
    "Wall time around GliclassClassifier.classify from the model dispatcher: tokenize+embed, " +
      "graph (first input write through output readback), decode and the LiteRT worker hand-off"

  private fun JSONObject.putResult(result: GliclassClassifier.Result): JSONObject =
    put("text", result.text)
      .put("window", result.window)
      .put("encoded_tokens", result.encodedTokens)
      .put("label_count", result.decision.labels.size)
      .put("accelerator", result.backend.name)
      .put("precision", result.backend.precision)
      .put("mode", result.decision.mode.key)
      .put("tokenize_embed_ms", result.timing.tokenizeEmbedMs)
      .put("graph_ms", result.timing.graphMs)
      .put("decode_ms", result.timing.decodeMs)
      .put("write_ms", result.timing.writeMs)
      .put("enqueue_ms", result.timing.enqueueMs)
      .put("readback_ms", result.timing.readbackMs)
      .put("all_logits_finite", result.decision.logits.all { it.isFinite() })
      .put("logits", JSONArray(result.decision.logits.map { it.toDouble() }))
      .put("predictions", GliclassGateFixtures.predictionsJson(result.decision))

  private fun header(context: Context, set: String, status: String): JSONObject =
    JSONObject()
      .put("set", set)
      .put("status", status)
      .put("litert_version", GliclassClassifier.LITERT_VERSION)
      .put("device_model", Build.MODEL)
      .put("build_type", BuildConfig.BUILD_TYPE)
      .put("debuggable", BuildConfig.DEBUG)
      .put("thermal_status", thermalStatus(context))

  private fun thermalStatus(context: Context): Int =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      context.getSystemService(PowerManager::class.java).currentThermalStatus
    } else {
      THERMAL_STATUS_UNAVAILABLE
    }

  private fun processImportance(): Int =
    ActivityManager.RunningAppProcessInfo().also { ActivityManager.getMyMemoryState(it) }.importance

  private fun write(context: Context, name: String, report: JSONObject, logLine: Boolean = true) {
    check(enabled) { "Diagnostics are not available in the release build." }
    val destination = File(context.filesDir, name)
    val temporary = File(context.filesDir, "$name.tmp")
    temporary.writeText(report.toString(GliclassGateFixtures.REPORT_JSON_INDENT) + "\n")
    check(temporary.renameTo(destination)) { "Could not save ${destination.absolutePath}" }
    if (logLine) {
      Log.i(GliclassGateFixtures.GATE_LOG_TAG, "${report.getString("set")} $report")
    }
  }
}
