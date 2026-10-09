package com.d1omni

import android.content.Context
import android.util.Log
import java.io.File

/**
 * The debug gate on the device, run on [D1Runtime.dispatcher]: loads the engine, compiles the
 * rows file's bucket L (after the `resident` bucket when the launch names one: two graphs at once,
 * with the memory read before and after each compile), then runs every row of the file through the
 * L graph, one call per row, the row's own IDs as the inputs. Per row: the probabilities from the
 * Kotlin read-out (float64, before any rounding), the scores at the markers, the call's times, and
 * whether this app's own tokenizer and `encode` give the row's IDs and markers again (`ids_match`,
 * Android's ICU regex) and its state's `serialize` the same text (`state_match`).
 *
 * The report is `files/<report>.partial` while running and `files/<report>` at the end. A file
 * `files/STOP` ends the run after the current row with `stopped_early: true`.
 */
class D1GateRunner(private val context: Context) {
  /** Final status (DONE, STOPPED or FAILED), the report path and the error that ended the run. */
  class Summary(val status: String, val path: String, val error: String?)

  private val files = context.filesDir

  fun run(args: D1Launch.Gate, progress: (String) -> Unit): Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val stop = File(files, STOP_FILE)
    stop.delete()
    destination.delete()
    val report = D1Device.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "d1omni_app_gate",
        "status" to "RUNNING",
        "fixture" to args.fixture,
        "limit" to args.limit,
        "backend_requested" to args.backend.wireName,
        "precision_requested" to args.precision.wireName,
        "resident_requested" to args.resident,
        "state_start" to D1Device.state(context),
        "call" to "write the six inputs + CompiledModel.run() + readFloat(scores); run() alone returns before the GPU work ends",
      )
    )
    var error: String? = null
    var stopped = false
    var engine: D1Engine? = null
    val rowsOut = ArrayList<LinkedHashMap<String, Any?>>()
    try {
      progress("loading")
      val gateRows = D1GateRows.parse(File(files, args.fixture).readBytes())
      val loaded = D1Engine.load(context, args.backend, args.precision)
      engine = loaded
      report["engine_load_ms"] = loaded.loadMs
      report["installed"] = loaded.installed
      report["L"] = gateRows.length
      report["graph_file"] = loaded.fileOf(gateRows.length)
      report["rows_in_file"] = gateRows.rows.size
      if (args.resident != null && args.resident != gateRows.length) {
        progress("compiling L${args.resident}")
        report["resident_prepare"] = prepared(loaded.ensure(listOf(args.resident), emptyList()))
        report["memory_after_resident"] = D1Device.memory(context)
      }
      progress("compiling L${gateRows.length}")
      report["prepare"] = prepared(loaded.ensure(listOf(gateRows.length), emptyList()))
      report["memory_after_prepare"] = D1Device.memory(context)
      report["resident"] = loaded.resident
      report["resident_backends"] = loaded.residentBackends.map { it.wireName }
      report["compiles"] = loaded.compiles.map { it.toJson() }
      val graph = loaded.graph(gateRows.length)
      report["accelerator_used"] = graph.backend.wireName
      report["precision"] = if (graph.backend == D1Backend.GPU) graph.precision.wireName else null
      report["gpu_failure"] = graph.gpuFailure
      val core = D1GateCore(loaded.tokenizer, loaded.contract)
      val count =
        if (args.limit > 0) minOf(args.limit, gateRows.rows.size) else gateRows.rows.size
      report["rows"] = rowsOut
      write(partial, report)
      log(
        "GATE_START L=${gateRows.length} backend=${graph.backend.wireName} precision=${report["precision"]} rows=$count resident=${loaded.resident}"
      )
      for (index in 0 until count) {
        if (stop.exists()) {
          stopped = true
          break
        }
        val row = gateRows.rows[index]
        val recheck = core.recheck(row)
        val inputs = core.inputs(row, gateRows.length)
        val wall = System.currentTimeMillis()
        val call = loaded.call(gateRows.length, inputs)
        // A GPU graph that failed to run is compiled on the CPU (`GPU_FALLBACK`): each row names where it ran.
        rowsOut.add(
          core.record(row, call, wall, recheck).apply {
            put("backend", loaded.graph(gateRows.length).backend.wireName)
          }
        )
        if (index % PROGRESS_EVERY == 0 || index == count - 1) {
          progress("row ${index + 1} / $count: ${"%.1f".format(call.totalMs)} ms")
          write(partial, report.apply { putAll(summary(rowsOut)) })
        }
      }
      report["compiles"] = loaded.compiles.map { it.toJson() }
      report["accelerator_at_end"] = loaded.graph(gateRows.length).backend.wireName
    } catch (failure: Exception) {
      error = D1Decider.describe(failure)
    } catch (failure: LinkageError) {
      error = "Native runtime: ${D1Decider.describe(failure)}"
    } catch (failure: OutOfMemoryError) {
      error = D1Decider.describe(failure)
    }
    report.putAll(summary(rowsOut))
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
    log("GATE_DONE status=$status rows=${rowsOut.size} path=${destination.absolutePath}")
    if (error != null) D1Demo.failed(error)
    return Summary(status, destination.absolutePath, error)
  }

  private fun prepared(prepared: D1Prepared): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "compiled" to prepared.compiled,
      "closed" to prepared.closed,
      "avail_mem_bytes_before_second" to prepared.availableBeforeSecond,
      "second_refused" to prepared.secondRefused,
    )

  private fun summary(rows: List<Map<String, Any?>>): LinkedHashMap<String, Any?> {
    val totals = rows.map { it["write_run_read_ms"] as Double }
    val warm = if (totals.size > WARMUP_ROWS) totals.drop(WARMUP_ROWS) else totals
    fun count(key: String, value: Any?) = rows.count { it[key] == value }
    return linkedMapOf(
      "summary" to
        linkedMapOf(
          "rows_run" to rows.size,
          "finite_rows" to count("finite", true),
          "nonfinite_rows" to count("finite", false),
          "ids_match" to "${count("ids_match", true)}/${rows.count { it["ids_match"] != null }}",
          "markers_match" to
            "${count("markers_match", true)}/${rows.count { it["markers_match"] != null }}",
          "state_match" to "${count("state_match", true)}/${rows.count { it["state_match"] != null }}",
          "first_call_write_run_read_ms" to totals.firstOrNull(),
          "warm_rows" to warm.size,
          "warm_write_run_read_ms" to D1GateCore.stats(warm),
        )
    )
  }

  private fun write(file: File, report: Map<String, Any?>) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(D1Json.writeIndented(report, REPORT_INDENT) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
  }

  private fun log(line: String) {
    Log.i(LOG_TAG, line)
  }

  companion object {
    /** Logcat tag of the gate and the timing runs. */
    const val LOG_TAG = "D1OmniGate"

    /** The file that stops a gate or timing run after the current graph call. */
    const val STOP_FILE = "STOP"

    /** Rows before the warm median (the first calls after a compile run slower). */
    private const val WARMUP_ROWS = 5

    private const val PROGRESS_EVERY = 25
    private const val REPORT_INDENT = 1
  }
}
