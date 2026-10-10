package com.d1omni

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

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
 *
 * A rows file of kind `audio` ([D1AudioRows]) holds clips instead of one bucket's rows: the decision
 * graphs its rows need are compiled (the largest first, at most two), then the audio graph of the
 * clips' bucket; per clip the wav in `files/` is read (`D1Wav`), its prefix computed on the phone
 * (`D1AudioEngine.audioPrefix`: waveform, mel, the five inputs, `audio_<T_b>`, the first P rows) and
 * compared with the Python host's sizes and mel dumps when the file names them, the prefix rows are
 * appended to `files/<report stem>.prefix.f32`, and each of the clip's rows runs on the smallest
 * compiled decision graph that holds P + n, with this app's own P. With `resident` the audio gate
 * compiles that decision bucket alone and runs every row on it (the rows must fit it).
 */
class D1GateRunner(private val context: Context) {
  /** Final status (DONE, STOPPED or FAILED), the report path and the error that ended the run. */
  class Summary(val status: String, val path: String, val error: String?)

  private val files = context.filesDir

  fun run(args: D1Launch.Gate, progress: (String) -> Unit): Summary {
    val kind = runCatching { d1RowsKind(File(files, args.fixture).readBytes()) }.getOrDefault("text")
    if (kind == D1Kind.AUDIO.wireName) return runAudio(args, progress)
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

  /** The gate of an audio rows file (see the class comment). */
  private fun runAudio(args: D1Launch.Gate, progress: (String) -> Unit): Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val prefixFile = File(files, "${args.report.removeSuffix(".json")}$PREFIX_SUFFIX")
    val stop = File(files, STOP_FILE)
    stop.delete()
    destination.delete()
    prefixFile.delete()
    val report = D1Device.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "d1omni_app_gate",
        "kind" to D1Kind.AUDIO.wireName,
        "status" to "RUNNING",
        "fixture" to args.fixture,
        "limit" to args.limit,
        "backend_requested" to args.backend.wireName,
        "precision_requested" to args.precision.wireName,
        "precision_audio_requested" to args.audioPrecision.wireName,
        "resident_requested" to args.resident,
        "state_start" to D1Device.state(context),
        "call" to "write the six inputs + CompiledModel.run() + readFloat(scores); run() alone returns before the GPU work ends",
        "audio_call" to "wav bytes -> D1Wav -> waveform -> mel (float64) -> five inputs -> audio_<T_b> (write + run + readFloat) -> the first P rows",
        "prefix_file" to prefixFile.name,
      )
    )
    var error: String? = null
    var stopped = false
    var engine: D1Engine? = null
    val rowsOut = ArrayList<LinkedHashMap<String, Any?>>()
    val recordsOut = ArrayList<LinkedHashMap<String, Any?>>()
    try {
      progress("loading")
      val audioRows = D1AudioRows.parse(File(files, args.fixture).readBytes())
      val loaded = D1Engine.load(context, args.backend, args.precision, args.audioPrecision)
      engine = loaded
      report["engine_load_ms"] = loaded.loadMs
      report["installed"] = loaded.installed
      report["audio_installed"] = loaded.audio.installed
      report["records_in_file"] = audioRows.records.size
      report["rows_in_file"] = audioRows.rowCount
      // The decision graphs first (as many as the rows need, the largest first), then the audio graph.
      val positions = audioRows.records.flatMap { record -> record.rows.map { it.prefixRows + it.ids.size } }
      progress("compiling the decision graphs")
      report["prepare"] =
        prepared(
          if (args.resident != null) loaded.ensure(listOf(args.resident), positions) else loaded.prepare(positions)
        )
      report["memory_after_prepare"] = D1Device.memory(context)
      val audioBucket =
        audioRows.records.firstNotNullOfOrNull { it.expectedInt("T_b") } ?: loaded.audio.installed.first()
      progress("compiling audio T$audioBucket")
      loaded.audio.ensure(audioBucket)
      report["memory_after_audio"] = D1Device.memory(context)
      report["resident"] = loaded.resident
      report["resident_backends"] = loaded.residentBackends.map { it.wireName }
      report["audio_resident"] = loaded.audio.resident
      report["audio_backend"] = loaded.audio.residentBackend?.wireName
      report["compiles"] = loaded.compiles.map { it.toJson() }
      val largest = loaded.graph(loaded.resident.max())
      report["accelerator_used"] = largest.backend.wireName
      report["precision"] = if (largest.backend == D1Backend.GPU) largest.precision.wireName else null
      report["precision_audio"] = loaded.audio.graph(audioBucket).let { if (it.backend == D1Backend.GPU) it.precision.wireName else null }
      report["gpu_failure"] = largest.gpuFailure
      report["audio_gpu_failure"] = loaded.audio.graph(audioBucket).gpuFailure
      val core = D1GateCore(loaded.tokenizer, loaded.contract)
      val count = if (args.limit > 0) minOf(args.limit, audioRows.rowCount) else audioRows.rowCount
      report["records"] = recordsOut
      report["rows"] = rowsOut
      write(partial, report)
      log(
        "GATE_START kind=audio L=${loaded.resident} T=${loaded.audio.resident} backend=${largest.backend.wireName} " +
          "precision=${report["precision"]} precision_audio=${report["precision_audio"]} records=${audioRows.records.size} rows=$count"
      )
      var prefixFloats = 0L
      FileOutputStream(prefixFile).use { prefixOut ->
        records@ for (record in audioRows.records) {
          if (rowsOut.size >= count) break
          if (stop.exists()) {
            stopped = true
            break
          }
          val wav = File(files, record.mediaFile)
          val readStart = System.nanoTime()
          val samples = D1Wav.parse(wav.readBytes(), record.mediaFile)
          val wavMs = (System.nanoTime() - readStart) / 1e6
          val audio = loaded.audio.audioPrefix(samples)
          val info = audio.info
          val entry =
            linkedMapOf<String, Any?>(
              "id" to record.id,
              "wav_file" to record.mediaFile,
              "wav_bytes" to wav.length(),
              "wav_sha256_match" to record.mediaSha256?.let { D1Contract.sha256(wav) == it },
              "wav_samples" to samples.size,
              "wav_read_ms" to wavMs,
              "info" to info.toJson(),
              "expected" to record.expected,
              "info_match" to D1AudioCheck.infoMatches(info, record),
              "audio_backend" to audio.backend.wireName,
              "audio_precision" to audio.precision?.wireName,
              "times" to audio.times(),
              "prefix_finite" to audio.prefix.all { it.isFinite() },
              "prefix_absmax" to audio.prefix.maxOfOrNull { Math.abs(it) },
              "prefix_offset_floats" to prefixFloats,
              "prefix_floats" to audio.prefix.size,
            )
          for ((form, name) in record.melFiles) {
            val python = File(files, name)
            entry["mel_vs_python_$form"] =
              if (python.isFile) D1AudioCheck.melDifference(audio.mel.values, D1AudioCheck.floats(python.readBytes()))
              else linkedMapOf<String, Any?>("missing" to name)
          }
          prefixOut.write(D1AudioCheck.bytes(audio.prefix))
          prefixFloats += audio.prefix.size
          recordsOut.add(entry)
          for (row in record.rows) {
            if (rowsOut.size >= count) break@records
            if (stop.exists()) {
              stopped = true
              break@records
            }
            val prefixRows = info.prefixRows
            val mine = D1AudioCheck.withPrefix(row, prefixRows)
            val recheck = D1AudioCheck.recheck(loaded.tokenizer, loaded.contract, record, row, prefixRows)
            val length =
              D1Contract.bucketFor(prefixRows + row.ids.size, loaded.resident)
                ?: throw IllegalStateException("${row.key}: no compiled decision graph holds ${prefixRows + row.ids.size} positions")
            val inputs = core.inputs(mine, length, audio.prefix)
            val wall = System.currentTimeMillis()
            val call = loaded.call(length, inputs)
            rowsOut.add(
              core.record(mine, call, wall, recheck).apply {
                put("record", record.id)
                put("L", length)
                put("P_expected", row.prefixRows)
                put("backend", loaded.graph(length).backend.wireName)
              }
            )
          }
          progress("clip ${recordsOut.size} / ${audioRows.records.size}: ${"%.1f".format(audio.totalMs)} ms to the prefix")
          write(partial, report.apply { putAll(summary(rowsOut)); putAll(audioSummary(recordsOut)) })
        }
      }
      report["prefix_file_floats"] = prefixFloats
      report["compiles"] = loaded.compiles.map { it.toJson() }
      report["accelerator_at_end"] = loaded.resident.map { loaded.graph(it).backend.wireName }
      report["audio_backend_at_end"] = loaded.audio.residentBackend?.wireName
    } catch (failure: Exception) {
      error = D1Decider.describe(failure)
    } catch (failure: LinkageError) {
      error = "Native runtime: ${D1Decider.describe(failure)}"
    } catch (failure: OutOfMemoryError) {
      error = D1Decider.describe(failure)
    }
    report.putAll(summary(rowsOut))
    report.putAll(audioSummary(recordsOut))
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
    log("GATE_DONE status=$status kind=audio records=${recordsOut.size} rows=${rowsOut.size} path=${destination.absolutePath}")
    if (error != null) D1Demo.failed(error)
    return Summary(status, destination.absolutePath, error)
  }

  /** The clips' summary: sizes and mel against the Python host, the audio graph's and the mel's times. */
  private fun audioSummary(records: List<Map<String, Any?>>): LinkedHashMap<String, Any?> {
    fun times(key: String) = records.mapNotNull { ((it["times"] as Map<*, *>?)?.get(key) as Double?) }
    fun worst(form: String) =
      records.mapNotNull { ((it["mel_vs_python_$form"] as Map<*, *>?)?.get("max_abs") as Double?) }.maxOrNull()
    val graph = times("graph_write_run_read_ms")
    return linkedMapOf(
      "audio_summary" to
        linkedMapOf(
          "records_run" to records.size,
          "info_match" to "${records.count { it["info_match"] == true }}/${records.count { it["info_match"] != null }}",
          "wav_sha256_match" to
            "${records.count { it["wav_sha256_match"] == true }}/${records.count { it["wav_sha256_match"] != null }}",
          "prefix_finite" to "${records.count { it["prefix_finite"] == true }}/${records.size}",
          "mel_max_abs_vs_python_f32" to worst("f32"),
          "mel_max_abs_vs_python_f64" to worst("f64"),
          "first_graph_write_run_read_ms" to graph.firstOrNull(),
          "graph_write_run_read_ms" to D1GateCore.stats(graph.drop(1).ifEmpty { graph }),
          "mel_ms" to D1GateCore.stats(times("mel_ms")),
          "samples_to_prefix_ms" to D1GateCore.stats(times("samples_to_prefix_ms").drop(1).ifEmpty { times("samples_to_prefix_ms") }),
        )
    )
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

    /** The audio gate's prefix rows file: `files/<report stem>.prefix.f32`, every clip's P x 1024 float32. */
    const val PREFIX_SUFFIX = ".prefix.f32"

    private const val PROGRESS_EVERY = 25
    private const val REPORT_INDENT = 1
  }
}
