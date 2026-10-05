package com.kev

import android.content.Context
import android.util.Log
import java.io.File
import java.security.MessageDigest

/**
 * Timing mode on the device (debug and benchmark builds), run on [KevRuntime.dispatcher] with the
 * model card's protocol ([KevTimingCore]): the requested sets of the timing rows file on the set
 * graph (a row window: the sets whose L is that window; the shared-state pair: the sets it takes),
 * then the request path of the bundled five-question request on the plan of the launch's graph mode
 * ([KevPlanner]; the report says which form it took). Each compile (cold with `clear_cache`) is
 * timed too, and the report records the device facts around the run. With `cool_ms`, the GPU state
 * is read before the first compile and each set and the request path wait for it ([KevCooler]).
 * `files/STOP` ends the run after the current call.
 */
class KevTimingRunner(private val context: Context) {
  class Args(
    val rows: File,
    val backend: KevDecider.Backend,
    /** The launch's precision, or null for each graph's default ([KevPrecision.defaultFor]). */
    val precision: KevPrecision?,
    val report: String,
    /** The graph the sets run on. */
    val setGraph: KevGraphKey,
    /** How the request path is planned. */
    val mode: KevGraphMode,
    /**
     * The row window of the launch: every request-path row on it in [KevGraphMode.ROWS], or null
     * for each row on its own window.
     */
    val window: Int?,
    /** Empty the app's cache directory before compiling, so that the compile is cold. */
    val clearCache: Boolean,
    /** The set names to time (null: every set for the set graph; empty: none). */
    val sets: List<String>?,
    /** Whether to time the request path after the sets. */
    val requestPath: Boolean,
    /** The longest wait for the GPU to cool before each set and the request path (0: none). */
    val coolMs: Long = 0,
  )

  private val files = context.filesDir

  fun run(
    args: Args,
    loadEngine: () -> KevEngine,
    prepare: (KevEngine, KevGraphKey) -> KevRequestGraphs,
    preparePlan: (KevEngine, KevPlan.Ready) -> KevRequestGraphs,
    progress: (String) -> Unit,
  ): KevGateRunner.Summary {
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
        "backend" to args.backend.wireName,
        "npu_libraries" to KevNpu.librariesInstalled(context),
        "memory_at_start" to KevDevice.memory(context),
        // The precision the set graph compiled with, set once it has.
        "precision" to null,
        "precision_requested" to args.precision?.wireName,
        "graph" to args.setGraph.label,
        "graph_mode" to args.mode.wireName,
        "window" to (args.setGraph as? KevGraphKey.Window)?.window,
        "graph_file" to args.setGraph.file,
        "graph_bytes" to File(files, args.setGraph.file).length(),
        "protocol" to
          if (args.setGraph is KevGraphKey.Pair) KevTimingCore.PAIR_PROTOCOL
          else KevTimingCore.PROTOCOL,
        "rows_file" to args.rows.name,
        "cgroup_start" to KevDevice.cgroup(),
        "thermal_status_start" to KevDevice.thermalStatus(context),
        "battery_temperature_start" to KevDevice.batteryTemperature(context),
        "clear_cache" to args.clearCache,
        "sets_requested" to (args.sets ?: "all"),
        "request_path_requested" to args.requestPath,
        "cool_ms" to args.coolMs,
      )
    )
    val cooler = KevCooler(args.coolMs, KevDevice::gpuState)
    // The GPU state before the first compile: the base each set waits for.
    fun readBase() {
      if (args.coolMs > 0 && !report.containsKey("gpu_state_before_compile")) {
        report["gpu_state_before_compile"] =
          cooler.readBase()?.toJson() ?: linkedMapOf("readable" to false)
      }
    }
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
      report["tokenizer_load_ms"] = engine.tokenizerMs
      report["head_load_ms"] = engine.headMs
      val sets = LinkedHashMap<String, Any?>()
      val skipped = ArrayList<Any?>()
      report["sets"] = sets
      report["skipped_sets"] = skipped
      val selection =
        when (val graph = args.setGraph) {
          is KevGraphKey.Window -> timingRows.select(args.sets, graph.window)
          is KevGraphKey.Pair -> timingRows.selectPair(args.sets, graph.shape)
        }
      selection.skipped.forEach {
        skipped.add(linkedMapOf("name" to it.name, "L" to it.window, "reason" to it.reason))
      }
      val gpuCeiling = { KevDevice.gpuState()?.maxClockMhz }
      val caps = { KevDevice.caps(context) }
      var timing =
        KevTimingCore(engine.pipeline, null, ceiling = gpuCeiling, caps = caps) { stop.exists() }
      if (selection.run.isNotEmpty()) {
        readBase()
        val graphs = prepare(engine, args.setGraph)
        val ranOn = graphs.backends.single()
        // The precision applies to a graph on the GPU only.
        report["precision"] =
          graphs.precisions.single().wireName.takeIf { ranOn == KevDecider.Backend.GPU }
        report["pair_share"] = graphs.pairShare
        report["pair_share_mode"] = engine.pairShare.wireName
        report["cache_dir_after_load"] = KevDevice.directoryUsage(context.cacheDir)
        report["accelerator_used"] = ranOn.wireName
        report["compiles"] = graphs.compiles.map { it.toJson() }
        report["compile_ms"] = engine.compileMs
        report["avail_mem_bytes_before_compile"] = graphs.availableBytes.firstOrNull()
        report["resident_graphs"] = engine.resident.map { it.label }
        val row = (graphs.runners as? KevRunners.Rows)?.graphs?.single()
        timing =
          KevTimingCore(engine.pipeline, row, ceiling = gpuCeiling, caps = caps) { stop.exists() }
        write(partial, report)
        for (set in selection.run) {
          if (timing.stoppedEarly) break
          progress(set.name)
          Log.i(
            KevGateRunner.LOG_TAG,
            "TIMING_SET ${set.name} rows=${set.rows.size} graph=${args.setGraph.label}",
          )
          val cool = cooler.waitForBase()
          sets[set.name] =
            when (val runners = graphs.runners) {
              is KevRunners.Rows -> timing.timeSet(set)
              is KevRunners.Pair -> timing.timePairSet(set, runners.graph)
            }.apply { put("cool", cool) }
          write(partial, report)
        }
      }
      core = timing
      if (!timing.stoppedEarly && args.requestPath) {
        progress("request path")
        val items =
          KevGateChecks.parseAsset(
            context.assets.open(KevGateChecks.ASSET_NAME).use { it.readBytes() }
          )
        val item = items.first { it.id == KevTimingCore.REQUEST_PATH_RECORD }
        val request = KevRequest.fromJson(item.request)
        val prepared = engine.pipeline.prepare(request)
        val fixed = if (args.mode == KevGraphMode.ROWS) args.window else null
        val plan = engine.plan(prepared, args.mode, fixed)
        KevDemo.plan(plan, engine.lastPlanInputs)
        val planReport =
          linkedMapOf<String, Any?>(
            "requested" to args.mode.wireName,
            "fixed_window" to fixed,
            "avail_mem_bytes" to engine.lastPlanInputs?.availableBytes,
            "proc_mem_available_kb" to engine.lastPlanInputs?.procAvailableKb,
            "resident" to engine.lastPlanInputs?.resident?.map { it.label },
            "form" to (plan as? KevPlan.Ready)?.form?.wireName,
            "predicted_ms" to
              (plan as? KevPlan.Ready)?.prediction?.let {
                linkedMapOf("rows" to it.rowsMs, "pair" to it.pairMs)
              },
          )
        if (plan is KevPlan.Ready) {
          readBase()
          val graphs = preparePlan(engine, plan)
          planReport["graphs"] = graphs.used.map { it.label }
          planReport["ran_on"] = graphs.backends.map { it.wireName }
          planReport["precisions"] =
            graphs.precisions.mapIndexed { index, precision ->
              precision.wireName.takeIf { graphs.backends[index] == KevDecider.Backend.GPU }
            }
          planReport["compiles"] = graphs.compiles.map { it.toJson() }
          planReport["pair_share"] = graphs.pairShare
          planReport["windows"] = graphs.windows
          planReport["compiled"] = graphs.compiled.map { it.label }
          planReport["closed"] = graphs.closed.map { it.label }
          planReport["avail_mem_bytes_before_compile"] = graphs.availableBytes
          planReport["compile_ms"] = engine.compileMs
          val cool = cooler.waitForBase()
          report["request_path"] =
            timing.timeRequestPath(item.id, request, graphs.runners).apply {
              put("plan", planReport)
              put("cool", cool)
            }
        } else {
          planReport["error"] =
            when (plan) {
              is KevPlan.NoWindow -> "a row of ${plan.missing.rowTokens} tokens has no window"
              is KevPlan.NoPair -> plan.miss.toString()
              is KevPlan.Ready -> ""
            }
          report["request_path"] = linkedMapOf("record" to item.id, "plan" to planReport)
          throw IllegalStateException("No plan for ${item.id}: $plan")
        }
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
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") {
      (it.toInt() and 0xff).toString(16).padStart(2, '0')
    }
}
