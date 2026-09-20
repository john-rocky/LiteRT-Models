package com.gliner25

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/** Debug-only observation of the normal startup and the same extraction used by the UI button. */
internal object GlinerFirstTapReport {
  fun begin(context: Context) {
    write(context, header("RUNNING"))
  }

  fun complete(context: Context, startupWarmUpMs: Double, result: GlinerExtractor.Result) {
    val spans =
      result.spans.map { span ->
        JSONObject()
          .put("label", span.label)
          .put("text", span.text)
          .put("start", span.start)
          .put("end", span.end)
          .put("confidence", span.confidence.toDouble())
      }
    write(
      context,
      header("PASS")
        .put("startup_warmup_ms", startupWarmUpMs)
        .put("text", result.text)
        .put("window", result.window)
        .put("encoded_tokens", result.encodedTokens)
        .put("text_words", result.textWords)
        .put("accelerator", result.backend.name)
        .put("precision", result.backend.precision)
        .put("tokenize_embed_ms", result.timing.tokenizeEmbedMs)
        .put("graph_ms", result.timing.graphMs)
        .put("decode_ms", result.timing.decodeMs)
        .put("write_ms", result.timing.writeMs)
        .put("enqueue_ms", result.timing.enqueueMs)
        .put("readback_ms", result.timing.readbackMs)
        .put("all_outputs_finite", true)
        .put("spans", JSONArray(spans)),
    )
  }

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
      .put("litert_version", GlinerExtractor.LITERT_VERSION)
      .put("device_model", Build.MODEL)
      .put("debuggable", BuildConfig.DEBUG)
      .put("startup_warmup_iterations", GlinerExtractor.STARTUP_WARMUP_ITERATIONS)
      .put("warmup_timing", "Full pipeline repeats after s128 compilation, before Ready")
      .put(
        "timed_extractions",
        if (status == "PASS") {
          1
        } else {
          0
        },
      )

  private fun write(context: Context, report: JSONObject) {
    check(BuildConfig.DEBUG) { "First-extraction diagnostics require a debug APK." }
    val destination = File(context.filesDir, "gate/first_tap.json")
    destination.parentFile?.mkdirs()
    destination.writeText(report.toString(2) + "\n")
    Log.i("GLINER_GATE", report.toString())
  }
}
