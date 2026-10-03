package com.kev

import android.content.Context
import android.util.Log
import java.io.File
import java.security.MessageDigest

/**
 * Timing mode on the device (debug and benchmark builds), run on [KevRuntime.dispatcher] with the
 * model card's protocol ([KevTimingCore]): every set of the conversion run's `timing_rows.json`
 * whose L is the resident window, then the request path of the bundled five-question request. The
 * engine load (cold compile with `clear_cache`) is timed too, and the report records the device
 * facts around the run. `files/STOP` ends the run after the current call.
 */
class KevTimingRunner(private val context: Context) {
  class Args(
    val rows: File,
    val backend: KevDecider.Backend,
    val report: String,
    val window: Int,
    /** Empty the app's cache directory before compiling, so that the compile is cold. */
    val clearCache: Boolean,
  )

  private val files = context.filesDir

  fun run(args: Args, loadEngine: () -> KevEngine, progress: (String) -> Unit): KevGateRunner.Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val stop = File(files, KevGateRunner.STOP_FILE)
    stop.delete()
    destination.delete()
    val report = KevDevice.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "kev_app_timing",
        "status" to "RUNNING",
        "backend" to args.backend.name.lowercase(),
        "window" to args.window,
        "graph_file" to KevFiles.graph(args.window),
        "graph_bytes" to File(files, KevFiles.graph(args.window)).length(),
        "protocol" to KevTimingCore.PROTOCOL,
        "rows_file" to args.rows.name,
        "cgroup_start" to KevDevice.cgroup(),
        "thermal_status_start" to KevDevice.thermalStatus(context),
        "battery_temperature_start" to KevDevice.batteryTemperature(context),
        "clear_cache" to args.clearCache,
      )
    )
    var error: String? = null
    var core: KevTimingCore? = null
    try {
      require(args.rows.isFile) { "Missing ${args.rows.name}" }
      val rowsBytes = args.rows.readBytes()
      report["rows_file_sha256"] = sha256(rowsBytes)
      val timingRows = KevTimingRows.parse(rowsBytes)
      if (args.clearCache) context.cacheDir.listFiles()?.forEach { it.deleteRecursively() }
      report["cache_dir_before_load"] = KevDevice.directoryUsage(context.cacheDir)
      progress("loading")
      val engine = loadEngine()
      report["cache_dir_after_load"] = KevDevice.directoryUsage(context.cacheDir)
      report["accelerator_used"] = engine.backend.name.lowercase()
      report["compile_ms"] = engine.decider.compileMs
      report["tokenizer_load_ms"] = engine.tokenizerMs
      report["head_load_ms"] = engine.headMs
      val timing = KevTimingCore(engine.pipeline, engine.decider) { stop.exists() }
      core = timing
      val sets = LinkedHashMap<String, Any?>()
      val skipped = ArrayList<Any?>()
      report["sets"] = sets
      report["skipped_sets"] = skipped
      write(partial, report)
      for (set in timingRows.sets) {
        if (set.window != engine.window) {
          skipped.add(linkedMapOf("name" to set.name, "L" to set.window, "reason" to "the resident graph is L${engine.window}"))
          continue
        }
        if (timing.stoppedEarly) break
        progress(set.name)
        Log.i(KevGateRunner.LOG_TAG, "TIMING_SET ${set.name} rows=${set.rows.size} L=${set.window}")
        sets[set.name] = timing.timeSet(set)
        write(partial, report)
      }
      if (!timing.stoppedEarly) {
        progress("request path")
        val items = KevGateChecks.parseAsset(context.assets.open(KevGateChecks.ASSET_NAME).use { it.readBytes() })
        val item = items.first { it.id == KevTimingCore.REQUEST_PATH_RECORD }
        report["request_path"] = timing.timeRequestPath(item.id, KevRequest.fromJson(item.request))
      }
    } catch (failure: Exception) {
      error = KevDecider.describe(failure)
    } catch (failure: LinkageError) {
      error = "Native runtime: ${KevDecider.describe(failure)}"
    }
    val stopped = core?.stoppedEarly == true
    val status =
      when {
        error != null -> "FAIL"
        stopped -> "STOPPED"
        else -> "DONE"
      }
    report["status"] = status
    report["stopped_early"] = stopped
    report["error"] = error
    report["cgroup_end"] = KevDevice.cgroup()
    report["thermal_status_end"] = KevDevice.thermalStatus(context)
    report["battery_temperature_end"] = KevDevice.batteryTemperature(context)
    write(partial, report)
    check(partial.renameTo(destination)) { "Could not save ${destination.absolutePath}" }
    Log.i(KevGateRunner.LOG_TAG, "TIMING_DONE status=$status path=${destination.absolutePath}")
    if (error != null) Log.i(KevGateRunner.LOG_TAG, "failed $error")
    return KevGateRunner.Summary(status, destination.absolutePath, error)
  }

  private fun write(file: File, report: Map<String, Any?>) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(KevJson.writeIndented(report, 1) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
  }

  private fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { (it.toInt() and 0xff).toString(16).padStart(2, '0') }
}
