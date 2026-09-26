package com.gliner25decide

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/** Debug-only observation of the normal startup and the same classification the button runs. */
internal object DecideFirstTapReport {
  /** Marks the run as started before the model loads. */
  fun begin(context: Context) {
    write(context, header("RUNNING"))
  }

  /** Records the startup warm-up time and the first classification after Ready. */
  fun complete(context: Context, startupWarmUpMs: Double, result: DecideClassifier.Result) {
    write(
      context,
      header("PASS")
        .put("startup_warmup_ms", startupWarmUpMs)
        .put("text", result.text)
        .put("window", result.window)
        .put("encoded_tokens", result.encodedTokens)
        .put("label_count", result.labelCount)
        .put("accelerator", result.backend.name)
        .put("precision", result.backend.precision)
        .put("tokenize_embed_ms", result.timing.tokenizeEmbedMs)
        .put("graph_ms", result.timing.graphMs)
        .put("decode_ms", result.timing.decodeMs)
        .put("write_ms", result.timing.writeMs)
        .put("enqueue_ms", result.timing.enqueueMs)
        .put("readback_ms", result.timing.readbackMs)
        .put("all_logits_finite", result.logits.all { it.isFinite() })
        .put("logits", JSONArray(result.logits.map { it.toDouble() }))
        .put("decisions", DecideGateFixtures.decisionsJson(result.decisions)),
    )
  }

  /** Records a failed startup or first classification. */
  fun failure(context: Context, startupWarmUpMs: Double?, failure: Throwable) {
    write(
      context,
      header("FAIL")
        .put("startup_warmup_ms", startupWarmUpMs ?: JSONObject.NULL)
        .put("error", failure.message ?: failure.javaClass.simpleName),
    )
  }

  private fun header(status: String): JSONObject =
    JSONObject()
      .put("set", "firsttap")
      .put("status", status)
      .put("litert_version", DecideClassifier.LITERT_VERSION)
      .put("device_model", Build.MODEL)
      .put("debuggable", BuildConfig.DEBUG)
      .put("startup_warmup_iterations", DecideClassifier.STARTUP_WARMUP_ITERATIONS)
      .put("warmup_timing", "Full pipeline repeats after s128 compilation, before Ready")
      .put("timed_classifications", if (status == "PASS") 1 else 0)

  private fun write(context: Context, report: JSONObject) {
    check(BuildConfig.DEBUG) { "First-request diagnostics require a debug APK." }
    val destination = File(context.filesDir, "gate/first_tap.json")
    destination.parentFile?.mkdirs()
    destination.writeText(report.toString(DecideGateFixtures.REPORT_JSON_INDENT) + "\n")
    Log.i(DecideGateFixtures.GATE_LOG_TAG, report.toString())
  }
}
