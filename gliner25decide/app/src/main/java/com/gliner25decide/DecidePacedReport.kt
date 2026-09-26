package com.gliner25decide

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-only record of paced requests: the normal startup, then the bundled example classified once
 * per interval, the way a user taps Classify. Written to `files/gate/paced_<BACKEND>.json` after
 * every request.
 */
internal class DecidePacedReport(
  private val context: Context,
  backend: DecideClassifier.Backend,
  count: Int,
  intervalMs: Long,
) {
  private val destination = File(context.filesDir, "gate/paced_${backend.name}.json")
  private val requests = JSONArray()
  private val report =
    JSONObject()
      .put("set", "paced")
      .put("status", "RUNNING")
      .put("litert_version", DecideClassifier.LITERT_VERSION)
      .put("device_model", Build.MODEL)
      .put("debuggable", BuildConfig.DEBUG)
      .put("accelerator", backend.name)
      .put("precision", backend.precision)
      .put("count", count)
      .put("interval_ms", intervalMs)
      .put(
        "schedule",
        "Request i starts (i + 1) × interval_ms after Ready, " +
          "or at once if the previous one ran late",
      )
      .put("startup_warmup_iterations", DecideClassifier.STARTUP_WARMUP_ITERATIONS)
      .put(
        "e2e_timing",
        "Wall time around DecideClassifier.classify on the model dispatcher: tokenize+embed, " +
          "graph (first input write through output readback), decode and the LiteRT worker " +
          "hand-off",
      )
      .put("requests", requests)

  /** Writes the header before the model loads. */
  fun begin() = write()

  /** Records model initialization (compile included) and the startup warm-up time. */
  fun startup(initializeMs: Double, warmUpMs: Double) {
    report.put("initialize_ms", initializeMs).put("startup_warmup_ms", warmUpMs)
    write()
  }

  /** Records request [index], its start offset after Ready and its end-to-end time. */
  fun request(
    index: Int,
    startOffsetMs: Double,
    endToEndMs: Double,
    result: DecideClassifier.Result,
  ) {
    requests.put(
      JSONObject()
        .put("index", index)
        .put("start_offset_ms", startOffsetMs)
        .put("e2e_ms", endToEndMs)
        .put("tokenize_embed_ms", result.timing.tokenizeEmbedMs)
        .put("graph_ms", result.timing.graphMs)
        .put("write_ms", result.timing.writeMs)
        .put("enqueue_ms", result.timing.enqueueMs)
        .put("readback_ms", result.timing.readbackMs)
        .put("decode_ms", result.timing.decodeMs)
        .put("window", result.window)
        .put("encoded_tokens", result.encodedTokens)
        .put("label_count", result.labelCount)
        .put("all_logits_finite", result.logits.all { it.isFinite() })
        .put("thermal_status", thermalStatus())
        .put("process_importance", processImportance())
        .put("logits", JSONArray(result.logits.map { it.toDouble() }))
        .put("decisions", DecideGateFixtures.decisionsJson(result.decisions))
    )
    write()
  }

  /** Marks the run as passed after the last request. */
  fun complete() {
    report.put("status", "PASS").put("text", context.getString(R.string.example_text))
    write()
    Log.i(
      DecideGateFixtures.GATE_LOG_TAG,
      "paced ${report.getString("accelerator")} PASS ${requests.length()} requests",
    )
  }

  /** Records a failed startup or request. */
  fun failure(failure: Throwable) {
    report.put("status", "FAIL").put("error", failure.message ?: failure.javaClass.simpleName)
    write()
    Log.e(
      DecideGateFixtures.GATE_LOG_TAG,
      "paced ${report.getString("accelerator")} FAIL: ${failure.message}",
    )
  }

  private fun thermalStatus(): Int =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      context.getSystemService(PowerManager::class.java).currentThermalStatus
    } else {
      UNKNOWN_THERMAL_STATUS
    }

  private fun processImportance(): Int =
    ActivityManager.RunningAppProcessInfo().also { ActivityManager.getMyMemoryState(it) }.importance

  private fun write() {
    check(BuildConfig.DEBUG) { "Paced diagnostics require a debug APK." }
    destination.parentFile?.mkdirs()
    val temporary = File(destination.parentFile, "${destination.name}.tmp")
    temporary.writeText(report.toString(DecideGateFixtures.REPORT_JSON_INDENT) + "\n")
    check(temporary.renameTo(destination)) { "Could not save ${destination.absolutePath}" }
  }

  private companion object {
    /** Reported below API 29, where `PowerManager.getCurrentThermalStatus` does not exist. */
    const val UNKNOWN_THERMAL_STATUS = -1
  }
}
