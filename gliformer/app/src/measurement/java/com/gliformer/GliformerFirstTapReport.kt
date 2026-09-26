package com.gliformer

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Real UI button handler → result state → completed Compose draw observations, also in benchmark.
 */
internal class GliformerFirstTapReport(private val context: Context) {
  private val session = "${Process.myPid()}-${SystemClock.elapsedRealtimeNanos()}"
  private val pending = HashMap<Int, JSONObject>()

  fun ready(
    launchStartedNs: Long,
    backend: GliformerExtractor.Backend,
    tableStorage: GliformerInputs.EmbeddingTable.Storage,
    loadMs: Double,
    warmup: GliformerExtractor.WarmupReport,
  ) {
    Log.i(
      "GLIFORMER_UI",
      "READY launch_to_ready_ms=${(SystemClock.elapsedRealtimeNanos() - launchStartedNs) / 1_000_000.0} backend=$backend window=128 table=${tableStorage} load_ms=$loadMs warmup_ms=${warmup.totalMs} passes=${warmup.passes} build=${BuildConfig.BUILD_TYPE}",
    )
  }

  fun begin(
    id: Int,
    tapNs: Long,
    backend: GliformerExtractor.Backend,
    storage: GliformerInputs.EmbeddingTable.Storage,
    loadMs: Double,
    warmMs: Double,
  ) {
    pending[id] =
      JSONObject()
        .put("session", session)
        .put("tap", id)
        .put("pid", Process.myPid())
        .put("status", "RUNNING")
        .put("tap_ns", tapNs)
        .put("backend", backend.name)
        .put("table", storage.name)
        .put("build_variant", BuildConfig.BUILD_TYPE)
        .put("debuggable", context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0)
        .put("device", Build.MODEL)
        .put("litert", GliformerExtractor.LITERT_VERSION)
        .put("screen_interactive", context.getSystemService(PowerManager::class.java).isInteractive)
        .put("startup_load_ms", loadMs)
        .put("startup_warmup_ms", warmMs)
        .put("request_window_load_ms", 0.0)
        .put("request_window_warmup_ms", 0.0)
        .put("startup_warmup_passes", GliformerExtractor.STARTUP_WARMUP_ITERATIONS)
        .put(
          "tap_definition",
          "SystemClock.elapsedRealtimeNanos inside actual Compose Button.onClick",
        )
        .put(
          "render_definition",
          "elapsed realtime after drawContent of the result panel; not display presentation time",
        )
    Log.i("GLIFORMER_TAP", pending.getValue(id).toString())
  }

  fun windowWarmup(id: Int, loadMs: Double, warmMs: Double) {
    pending[id]?.put("request_window_load_ms", loadMs)?.put("request_window_warmup_ms", warmMs)
  }

  fun result(id: Int, resultNs: Long, result: GliformerExtractor.Result) {
    val report = pending[id] ?: return
    report
      .put("result_ns", resultNs)
      .put("tap_to_result_ms", ms(resultNs - report.getLong("tap_ns")))
      .put("window", result.window)
      .put("encoded_tokens", result.prepared.encodedLength)
      .put("tokenize_lookup_ms", result.timings.tokenizeLookupMs)
      .put("graph_readback_ms", result.timings.graphMs)
      .put("decode_ms", result.timings.decodeMs)
      .put("pipeline_total_ms", result.timings.totalMs)
      .put(
        "entities",
        JSONArray(
          result.entities.map {
            JSONObject()
              .put("text", it.text)
              .put("label", it.label)
              .put("start", it.start)
              .put("end", it.end)
              .put("score", it.score.toDouble())
          }
        ),
      )
  }

  fun statePublished(id: Int, stateNs: Long) {
    val report = pending[id] ?: return
    report
      .put("state_ns", stateNs)
      .put("tap_to_state_ms", ms(stateNs - report.getLong("tap_ns")))
      .put("state_definition", "elapsed realtime immediately after StateFlow.value assignment")
  }

  fun rendered(id: Int, renderedNs: Long): JSONObject? {
    val report = pending[id] ?: return null
    if (!report.has("result_ns")) return null
    pending.remove(id)
    return report
      .put("status", "PASS")
      .put("render_ns", renderedNs)
      .put("tap_to_render_ms", ms(renderedNs - report.getLong("tap_ns")))
  }

  fun failure(id: Int, failure: Throwable) {
    val report = pending.remove(id) ?: return
    report.put("status", "FAIL").put("error", failure.message ?: failure.javaClass.simpleName)
    Log.e("GLIFORMER_TAP", report.toString())
  }

  fun write(report: JSONObject) {
    val directory = File(context.filesDir, "first_tap").apply { mkdirs() }
    val text = report.toString()
    File(directory, "tap_${report.getInt("tap")}.json").writeText(text + "\n")
    File(directory, "latest.json").writeText(text + "\n")
    Log.i("GLIFORMER_TAP", text)
  }

  private fun ms(ns: Long) = ns / 1_000_000.0
}
