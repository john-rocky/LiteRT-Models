// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.content.Context
import android.os.Build
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

/** Measurement output is public to adb pull, without making the release app debuggable. */
object MeasurementFiles {
  fun root(context: Context, kind: String, tag: String): File {
    require(tag.matches(Regex("[A-Za-z0-9_-]+")))
    val external =
      requireNotNull(context.getExternalFilesDir(null)) { "External app files unavailable" }
    return File(external, "$kind/$tag").apply { check(isDirectory || mkdirs()) }
  }

  fun phone(): String =
    if (Build.MODEL == "SM-S942Q") "Galaxy S26" else "${Build.MANUFACTURER} ${Build.MODEL}"

  fun base(mode: String, apkSha256: String): JSONObject =
    JSONObject()
      .put("phone", phone())
      .put("fingerprint", Build.FINGERPRINT)
      .put("pid", Process.myPid())
      .put("mode", mode)
      .put("debuggable", BuildConfig.DEBUG)
      .put("build", if (BuildConfig.DEBUG) "debug" else "release")
      .put("apk_sha256", apkSha256)
      .put("producer_only", true)
      .put("started_unix_ms", System.currentTimeMillis())
      .put("transfer_route", "external_files_plain_adb_pull")

  fun save(root: File, report: JSONObject) {
    val temporary = File(root, "index.json.tmp")
    temporary.writeText(report.toString(2) + "\n")
    check(temporary.renameTo(File(root, "index.json")))
  }

  fun sha(file: File): String {
    val hash = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { stream ->
      val buffer = ByteArray(65536)
      while (true) {
        val count = stream.read(buffer)
        if (count < 0) break
        hash.update(buffer, 0, count)
      }
    }
    return hash.digest().joinToString("") { "%02x".format(it) }
  }

  fun dump(root: File, name: String, array: FloatArray): JSONObject {
    val file = File(root, "$name.bin")
    file.parentFile?.mkdirs()
    val buffer = ByteBuffer.allocate(65536).order(ByteOrder.LITTLE_ENDIAN)
    FileOutputStream(file).use { output ->
      for (value in array) {
        if (buffer.remaining() < 4) {
          output.write(buffer.array(), 0, buffer.position())
          buffer.clear()
        }
        buffer.putFloat(value)
      }
      output.write(buffer.array(), 0, buffer.position())
    }
    return JSONObject()
      .put("path", "$name.bin")
      .put("dtype", "float32")
      .put("shape", JSONArray(listOf(array.size)))
      .put("sha256", sha(file))
      .put("bytes", array.size * 4)
  }
}

/** Records only normal UI button runs; never invokes synthesis itself or warms up a graph. */
class UiMeasurementReport(
  context: Context,
  tag: String,
  apkSha256: String,
  private val activityStartedNanos: Long,
  activityStartedUnixMs: Long,
) {
  val root = MeasurementFiles.root(context, "r8_ui", tag)
  private val rows = JSONArray()
  private val report =
    MeasurementFiles.base("first_tap_ui", apkSha256)
      .put("status", "LOADING")
      .put("rows", rows)
      .put("activity_started_unix_ms", activityStartedUnixMs)
      .put("process_started_elapsed_realtime_ms", Process.getStartElapsedRealtime())
      .put("warmup_inference_count", 0)

  init {
    MeasurementFiles.save(root, report)
  }

  @Synchronized
  fun ready(engine: SoproEngine, precision: String, contractSet: String) {
    val elapsed = (System.nanoTime() - activityStartedNanos) / 1e6
    report
      .put("status", "READY")
      .put("ready_unix_ms", System.currentTimeMillis())
      .put("activity_create_to_ready_ms", elapsed)
      .put("creation_ms", JSONObject(engine.creationMs.toMap()))
      .put(
        "process_start_to_ready_ms",
        SystemClock.elapsedRealtime() - Process.getStartElapsedRealtime(),
      )
      .put("placement", engine.placementDescription)
      .put("precision", precision)
      .put("contract_set", contractSet)
      .put(
        "ready_scope",
        "Activity.onCreate to all required graphs compiled; no inference; " +
          "host driver adds process launch overhead",
      )
    MeasurementFiles.save(root, report)
    Log.i(
      "SoproMeasurement",
      "READY activity_create_to_ready_ms=$elapsed path=${root.absolutePath}",
    )
  }

  @Synchronized
  fun tapped(seed: Long) {
    report
      .put("status", "SYNTHESIZING")
      .put("current_tap", rows.length() + 1)
      .put("tap_unix_ms", System.currentTimeMillis())
      .put("current_seed", seed)
    MeasurementFiles.save(root, report)
  }

  @Synchronized
  fun generated(state: UiState, result: SoproEngine.Output, saved: WavFiles.Saved) {
    val stats = requireNotNull(result.stats)
    val copyWav = File(root, saved.wav.name)
    saved.wav.copyTo(copyWav, overwrite = true)
    val copySidecar = File(root, saved.sidecar.name)
    saved.sidecar.copyTo(copySidecar, overwrite = true)
    rows.put(
      JSONObject()
        .put("phone", MeasurementFiles.phone())
        .put("build", if (BuildConfig.DEBUG) "debug" else "release")
        .put("apk_sha256", report.getString("apk_sha256"))
        .put("tap", rows.length() + 1)
        .put("text", state.text)
        .put("lang", state.language)
        .put("seed", stats.seed)
        .put("stats", stats.toJson().put("trim", JSONObject(result.trim.asMap())))
        .put(
          "displayed",
          JSONObject()
            .put("ttfa_ms", stats.ttfaMs)
            .put("ttfa_to_onset_ms", stats.ttfaToOnsetMs)
            .put("total_ms", stats.totalWallMs)
            .put("rtf", stats.rtf)
            .put("placement", stats.placement),
        )
        .put("wav", copyWav.name)
        .put("wav_sha256", MeasurementFiles.sha(copyWav))
        .put("sidecar", copySidecar.name)
    )
    report.put("status", "DRAINING")
    MeasurementFiles.save(root, report)
    Log.i(
      "SoproMeasurement",
      "GENERATED tap=${rows.length()} ttfa_ms=${stats.ttfaMs} total_ms=${stats.totalWallMs}",
    )
  }

  @Synchronized
  fun completed() {
    report
      .put("status", "READY")
      .put("completed_taps", rows.length())
      .put("last_completed_unix_ms", System.currentTimeMillis())
    MeasurementFiles.save(root, report)
    Log.i("SoproMeasurement", "COMPLETE tap=${rows.length()}")
  }

  @Synchronized
  fun failure(error: Throwable) {
    report.put("status", "ERROR").put("error", error.stackTraceToString())
    MeasurementFiles.save(root, report)
    Log.e("SoproMeasurement", "Measurement failed", error)
  }
}
