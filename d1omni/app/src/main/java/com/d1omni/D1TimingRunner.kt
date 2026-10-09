package com.d1omni

import android.content.Context
import android.util.Log
import java.io.File

/**
 * The debug timing protocol on the device, run on [D1Runtime.dispatcher]: per set of the rows file
 * (only the sets the launch names), the set's bucket compiled when first needed, then a wait for
 * the GPU (at most `cool_ms`: until kgsl's clock ceiling is back at its value before the first
 * compile, its thermal power level is 0 and its temperature at most 5 °C above that reading), the
 * warm-up calls, and `reps` rounds of the set's rows, one call per row. Every call records the
 * device's wall clock at its start, its time (input writes + `run()` + read-back), `run()` alone,
 * and kgsl's clock ceiling read right before it, outside the timed span.
 *
 * The report is `files/<report>.partial` while running and `files/<report>` at the end; a file
 * `files/STOP` ends the run after the current round.
 */
class D1TimingRunner(private val context: Context) {
  private val files = context.filesDir

  fun run(args: D1Launch.Timing, progress: (String) -> Unit): D1GateRunner.Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val stop = File(files, D1GateRunner.STOP_FILE)
    stop.delete()
    destination.delete()
    val report = D1Device.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "d1omni_app_timing",
        "status" to "RUNNING",
        "rows_file" to args.rows,
        "sets_requested" to args.sets,
        "warmup_calls_setting" to args.warmup,
        "reps" to args.reps,
        "cool_ms" to args.coolMs,
        "backend_requested" to args.backend.wireName,
        "precision_requested" to args.precision.wireName,
        "state_start" to D1Device.state(context),
        "calls_format" to
          "[device wall clock ms at the call's start, ms write + run + read, ms run only, kgsl max_clock_mhz read before the call]; a request set's calls in row order",
      )
    )
    val results = LinkedHashMap<String, Any?>()
    var error: String? = null
    var stopped = false
    var engine: D1Engine? = null
    try {
      val sets =
        D1TimingSet.parse(File(files, args.rows).readBytes()).filter {
          args.sets == null || it.name in args.sets
        }
      require(sets.isNotEmpty()) { "no timing set ${args.sets ?: ""} in ${args.rows}" }
      val loaded = D1Engine.load(context, args.backend, args.precision)
      engine = loaded
      report["engine_load_ms"] = loaded.loadMs
      val base = D1Device.gpuState()
      report["gpu_state_before_compile"] = base?.toJson()
      val core = D1GateCore(loaded.tokenizer, loaded.contract)
      report["timing"] = results
      for (set in sets) {
        if (stop.exists()) {
          stopped = true
          break
        }
        progress("timing ${set.name}: compiling L${set.length}")
        loaded.ensure(listOf(set.length), emptyList())
        val graph = loaded.graph(set.length)
        val inputs = set.rows.map { core.inputs(it, set.length) }
        progress("timing ${set.name}: cooling")
        val cool = coolDown(base, args.coolMs)
        progress("timing ${set.name}: ${set.rows.size} row(s)")
        val warmupCalls = ArrayList<List<Any?>>()
        for (index in 0 until args.warmup) {
          warmupCalls.add(timedCall(loaded, set.length, inputs[index % inputs.size]))
        }
        val calls = ArrayList<List<Any?>>()
        val requests = ArrayList<Double>()
        var finite = true
        for (round in 0 until args.reps) {
          if (stop.exists()) {
            stopped = true
            break
          }
          var total = 0.0
          for ((index, row) in set.rows.withIndex()) {
            val clock = D1Device.gpuState()?.maxClockMhz
            val wall = System.currentTimeMillis()
            val call = loaded.call(set.length, inputs[index])
            calls.add(listOf(wall, call.totalMs, call.runMs, clock))
            total += call.totalMs
            finite =
              finite &&
                D1Readout.finite(
                  D1Readout.markerScores(call.scores, row.prefixRows, row.markers, row.options)
                )
          }
          requests.add(total)
        }
        val entry =
          linkedMapOf<String, Any?>(
            "kind" to set.kind,
            "L" to set.length,
            "rows" to set.rows.size,
            "keys" to set.rows.map { it.key },
            "tokens" to set.rows.map { it.ids.size },
            "backend" to graph.backend.wireName,
            "precision" to if (graph.backend == D1Backend.GPU) graph.precision.wireName else null,
            "cool" to cool,
            "warmup_calls" to warmupCalls,
            "timed_calls" to calls,
            "per_call_ms_write_run_read" to D1GateCore.stats(calls.map { it[1] as Double }),
            "per_call_ms_run_only" to D1GateCore.stats(calls.map { it[2] as Double }),
            "finite_markers" to finite,
            "stopped_early" to stopped,
          )
        if (set.rows.size > 1) entry["request_ms_write_run_read"] = D1GateCore.stats(requests)
        results[set.name] = entry
        write(partial, report)
        Log.i(
          D1GateRunner.LOG_TAG,
          "TIMING_SET ${set.name} L=${set.length} median=${(entry["per_call_ms_write_run_read"] as Map<*, *>?)?.get("median")}",
        )
        if (stopped) break
      }
      report["compiles"] = loaded.compiles.map { it.toJson() }
      report["resident"] = loaded.resident
    } catch (failure: Exception) {
      error = D1Decider.describe(failure)
    } catch (failure: LinkageError) {
      error = "Native runtime: ${D1Decider.describe(failure)}"
    }
    val status =
      when {
        error != null -> "FAILED"
        stopped -> "STOPPED"
        else -> "DONE"
      }
    report["status"] = status
    report["stopped_early"] = stopped
    report["error"] = error
    report["state_end"] = D1Device.state(context)
    try {
      engine?.close()
    } catch (failure: Exception) {
      report["close_error"] = D1Decider.describe(failure)
    }
    write(partial, report)
    check(partial.renameTo(destination)) { "Could not save ${destination.absolutePath}" }
    Log.i(D1GateRunner.LOG_TAG, "TIMING_DONE status=$status path=${destination.absolutePath}")
    if (error != null) D1Demo.failed(error)
    return D1GateRunner.Summary(status, destination.absolutePath, error)
  }

  /** One warm-up call: [wall ms, total ms, run ms, kgsl max clock read before it]. */
  private fun timedCall(engine: D1Engine, bucket: Int, inputs: D1Inputs): List<Any?> {
    val clock = D1Device.gpuState()?.maxClockMhz
    val wall = System.currentTimeMillis()
    val call = engine.call(bucket, inputs)
    return listOf(wall, call.totalMs, call.runMs, clock)
  }

  /**
   * Waits at most [coolMs] until kgsl's clock ceiling is at least [base]'s, its thermal power
   * level 0 and its temperature at most 5 °C above [base]'s; a fixed sleep when kgsl is unreadable.
   */
  private fun coolDown(base: D1GpuState?, coolMs: Long): LinkedHashMap<String, Any?> {
    val record = linkedMapOf<String, Any?>("cool_ms" to coolMs)
    if (coolMs <= 0) return record.apply { put("mode", "none") }
    val start = System.currentTimeMillis()
    if (base == null || D1Device.gpuState() == null) {
      Thread.sleep(coolMs)
      return record.apply {
        put("mode", "fixed")
        put("waited_ms", System.currentTimeMillis() - start)
      }
    }
    fun recovered(now: D1GpuState?): Boolean =
      now != null &&
        now.maxClockMhz >= base.maxClockMhz &&
        (now.thermalPowerLevel ?: 0) == 0 &&
        now.tempMilliC <= base.tempMilliC + COOL_MARGIN_MILLI_C
    var now = D1Device.gpuState()
    while (!recovered(now) && System.currentTimeMillis() - start < coolMs) {
      Thread.sleep(COOL_POLL_MS)
      now = D1Device.gpuState()
    }
    return record.apply {
      put("mode", "kgsl")
      put("waited_ms", System.currentTimeMillis() - start)
      put("base", base.toJson())
      put("end", now?.toJson())
      put("recovered", recovered(now))
      put("thermal_status", D1Device.thermalStatus(context))
      put("cpu_caps", D1Device.cpuCaps())
    }
  }

  private fun write(file: File, report: Map<String, Any?>) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(D1Json.writeIndented(report, 1) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
  }

  private companion object {
    const val COOL_MARGIN_MILLI_C = 5000
    const val COOL_POLL_MS = 500L
  }
}
