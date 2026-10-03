package com.kev

import android.content.Context
import android.util.Log
import java.io.File

/**
 * The debug gate on the device (debug build only), run on [KevRuntime.dispatcher]: loads the engine
 * on the requested window and backend, then runs [KevGateCore] over the bundled assets
 * (`tokenizer_probes.json`, `gate_fixtures.json`) with the device facts around it (cgroup, thermal
 * status, battery temperature, cache directory, load and compile times). Android's ICU regex and
 * NFC follow other Unicode versions than the desktop JVM, so the probes run on the device too.
 *
 * The report is `files/<report>.partial` while running and `files/<report>` at the end. A file
 * `files/STOP` ends the run after the current row with `stopped_early: true`.
 */
class KevGateRunner(private val context: Context) {
  class Args(val backend: KevDecider.Backend, val report: String, val window: Int, val limit: Int)

  /** Final status (PASS, FAIL or STOPPED), the report path and the error that ended the run. */
  class Summary(val status: String, val path: String, val error: String?)

  private val files = context.filesDir

  fun run(args: Args, loadEngine: () -> KevEngine, progress: (String) -> Unit): Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val stop = File(files, STOP_FILE)
    stop.delete()
    destination.delete()
    val report = KevDevice.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "kev_app_gate",
        "status" to "RUNNING",
        "backend" to args.backend.name.lowercase(),
        "window" to args.window,
        "graph_file" to KevFiles.graph(args.window),
        "graph_bytes" to File(files, KevFiles.graph(args.window)).length(),
        "limit" to args.limit,
        "cgroup_start" to KevDevice.cgroup(),
        "thermal_status_start" to KevDevice.thermalStatus(context),
        "battery_temperature_start" to KevDevice.batteryTemperature(context),
        "infer_timing" to
          "input writes + run() + read-back of hidden (run() alone returns before the GPU work ends)",
        "near_tie" to "oracle top-2 probability gap <= ${KevGateChecks.NEAR_TIE}",
        "bar" to
          "max |dp| <= ${KevGateChecks.MAX_ABS_DP} and mean |dp| <= ${KevGateChecks.MEAN_ABS_DP} over all options " +
            "of the rows run, argmax equal outside near-ties",
      )
    )
    var error: String? = null
    var core: KevGateCore? = null
    try {
      report["cache_dir_before_load"] = KevDevice.directoryUsage(context.cacheDir)
      progress("loading")
      val engine = loadEngine()
      report["cache_dir_after_load"] = KevDevice.directoryUsage(context.cacheDir)
      report["accelerator_used"] = engine.backend.name.lowercase()
      report["tokenizer_load_ms"] = engine.tokenizerMs
      report["head_load_ms"] = engine.headMs
      report["compile_ms"] = engine.primary.compileMs
      report["avail_mem_bytes_before_compile"] = engine.loadAvailableBytes
      report["resident_windows"] = engine.windows
      val gate = KevGateCore(engine.pipeline, engine.primary, args.limit, { stop.exists() }, ::log)
      core = gate
      report["tokenizer_probes"] =
        gate.probes(KevGateChecks.parseProbes(asset(KevGateChecks.PROBES_NAME)))
      val items = KevGateChecks.parseAsset(asset(KevGateChecks.ASSET_NAME))
      report["fixtures"] =
        linkedMapOf("records" to items.size, "questions" to items.sumOf { it.questions.size })
      report["rows"] = gate.rows
      write(partial, report)
      log(
        "GATE_START backend=${args.backend.name.lowercase()} window=${args.window} limit=${args.limit} records=${items.size}"
      )
      gate.run(items) { rowsRun ->
        progress("row $rowsRun")
        write(partial, report.apply { putAll(gate.summary()) })
      }
    } catch (failure: Exception) {
      error = KevDecider.describe(failure)
    } catch (failure: LinkageError) {
      error = "Native runtime: ${KevDecider.describe(failure)}"
    }
    core?.let { report.putAll(it.summary()) }
    val stopped = core?.stoppedEarly == true
    val status =
      when {
        error == null && stopped -> "STOPPED"
        error == null && core?.passed() == true -> "PASS"
        else -> "FAIL"
      }
    report["status"] = status
    report["stopped_early"] = stopped
    report["error"] = error
    report["cgroup_end"] = KevDevice.cgroup()
    report["thermal_status_end"] = KevDevice.thermalStatus(context)
    report["battery_temperature_end"] = KevDevice.batteryTemperature(context)
    write(partial, report)
    check(partial.renameTo(destination)) { "Could not save ${destination.absolutePath}" }
    log("GATE_DONE status=$status rows_run=${report["rows_run"]} path=${destination.absolutePath}")
    if (error != null) log("failed $error")
    return Summary(status, destination.absolutePath, error)
  }

  private fun asset(name: String): ByteArray = context.assets.open(name).use { it.readBytes() }

  private fun write(file: File, report: Map<String, Any?>) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(KevJson.writeIndented(report, REPORT_INDENT) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
  }

  private fun log(line: String) {
    Log.i(LOG_TAG, line)
  }

  companion object {
    /** Logcat tag of the gate and the timing runs. */
    const val LOG_TAG = "KevGate"

    /** The file that stops a gate or timing run after the current graph call. */
    const val STOP_FILE = "STOP"

    private const val REPORT_INDENT = 1
  }
}
