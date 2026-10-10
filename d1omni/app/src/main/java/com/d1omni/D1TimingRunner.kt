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
 *
 * A rows file of kind `audio` ([D1AudioTimingSet]) times whole requests instead: per set, the
 * decision graphs of its rows and the audio graph compiled, the same wait for the GPU, then each
 * request runs the clip's wav in `files/` to every answer — read and parse the wav, waveform, mel,
 * the audio graph's inputs and call, the prefix rows, this app's encoding of every question after
 * them, and per question the decision inputs, one call and the read-out and answer — with every
 * step's wall time recorded (see [audioRequest]).
 */
class D1TimingRunner(private val context: Context) {
  private val files = context.filesDir

  fun run(args: D1Launch.Timing, progress: (String) -> Unit): D1GateRunner.Summary {
    val kind = runCatching { d1RowsKind(File(files, args.rows).readBytes()) }.getOrDefault("text")
    if (kind == D1Kind.AUDIO.wireName) return runAudio(args, progress)
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

  /** The timing protocol of an audio rows file (see the class comment). */
  private fun runAudio(args: D1Launch.Timing, progress: (String) -> Unit): D1GateRunner.Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val stop = File(files, D1GateRunner.STOP_FILE)
    stop.delete()
    destination.delete()
    val report = D1Device.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "d1omni_app_timing",
        "kind" to D1Kind.AUDIO.wireName,
        "status" to "RUNNING",
        "rows_file" to args.rows,
        "sets_requested" to args.sets,
        "warmup_calls_setting" to args.warmup,
        "reps" to args.reps,
        "cool_ms" to args.coolMs,
        "backend_requested" to args.backend.wireName,
        "precision_requested" to args.precision.wireName,
        "precision_audio_requested" to args.audioPrecision.wireName,
        "state_start" to D1Device.state(context),
        "request_format" to
          "one request = the clip's wav in files/ to every answer: wav_ms (read + parse), the audio steps (waveform, mel, inputs, graph write / run / read, prefix rows), encode_ms (this app's encode of every question after P rows), per question [ms write + run + read, write, run, read] of the decision call, decision_inputs_ms and readout_ms summed over the questions; request_ms = the whole span; wall_ms and kgsl_max_clock_mhz read before it",
      )
    )
    val results = LinkedHashMap<String, Any?>()
    var error: String? = null
    var stopped = false
    var engine: D1Engine? = null
    try {
      val sets =
        D1AudioTimingSet.parse(File(files, args.rows).readBytes()).filter { args.sets == null || it.name in args.sets }
      require(sets.isNotEmpty()) { "no timing set ${args.sets ?: ""} in ${args.rows}" }
      val loaded = D1Engine.load(context, args.backend, args.precision, args.audioPrecision)
      engine = loaded
      report["engine_load_ms"] = loaded.loadMs
      val base = D1Device.gpuState()
      report["gpu_state_before_compile"] = base?.toJson()
      report["timing"] = results
      for (set in sets) {
        if (stop.exists()) {
          stopped = true
          break
        }
        val record = set.record
        progress("timing ${set.name}: compiling")
        loaded.prepare(record.rows.map { it.prefixRows + it.ids.size })
        val bucket = record.expectedInt("T_b") ?: loaded.audio.installed.first()
        loaded.audio.ensure(bucket)
        progress("timing ${set.name}: cooling")
        val cool = coolDown(base, args.coolMs)
        progress("timing ${set.name}: requests")
        val warmup = ArrayList<LinkedHashMap<String, Any?>>()
        for (index in 0 until args.warmup) warmup.add(audioRequest(loaded, record))
        val requests = ArrayList<LinkedHashMap<String, Any?>>()
        for (round in 0 until args.reps) {
          if (stop.exists()) {
            stopped = true
            break
          }
          requests.add(audioRequest(loaded, record))
        }
        fun stat(key: String) = D1GateCore.stats(requests.mapNotNull { it[key] as Double? })
        val decision = requests.flatMap { r -> (r["decision_calls"] as List<*>).map { (it as List<*>)[0] as Double } }
        val entry =
          linkedMapOf<String, Any?>(
            "kind" to set.kind,
            "record" to record.id,
            "rows" to record.rows.size,
            "keys" to record.rows.map { it.key },
            "tokens" to record.rows.map { it.ids.size },
            "decision_buckets" to requests.firstOrNull()?.get("decision_buckets"),
            "audio_bucket" to bucket,
            "backend" to loaded.residentBackends.map { it.wireName },
            "precision" to loaded.precision.wireName,
            "audio_backend" to loaded.audio.residentBackend?.wireName,
            "precision_audio" to loaded.audio.precision.wireName,
            "cool" to cool,
            "warmup_requests" to warmup,
            "requests" to requests,
            "request_ms" to stat("request_ms"),
            "wav_ms" to stat("wav_ms"),
            "waveform_ms" to stat("waveform_ms"),
            "mel_ms" to stat("mel_ms"),
            "audio_inputs_ms" to stat("inputs_ms"),
            "audio_graph_ms_write_run_read" to stat("graph_write_run_read_ms"),
            "prefix_rows_ms" to stat("prefix_rows_ms"),
            "encode_ms" to stat("encode_ms"),
            "decision_inputs_ms" to stat("decision_inputs_ms"),
            "decision_ms" to stat("decision_ms"),
            "decision_call_ms_write_run_read" to D1GateCore.stats(decision),
            "readout_ms" to stat("readout_ms"),
            "ids_match_every_request" to requests.all { it["ids_match"] == true },
            "finite_every_request" to requests.all { it["finite"] == true },
            "stopped_early" to stopped,
          )
        results[set.name] = entry
        write(partial, report)
        Log.i(
          D1GateRunner.LOG_TAG,
          "TIMING_SET ${set.name} kind=audio request_median=${(entry["request_ms"] as Map<*, *>?)?.get("median")}",
        )
        if (stopped) break
      }
      report["compiles"] = loaded.compiles.map { it.toJson() }
      report["resident"] = loaded.resident
      report["audio_resident"] = loaded.audio.resident
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
    Log.i(D1GateRunner.LOG_TAG, "TIMING_DONE status=$status kind=audio path=${destination.absolutePath}")
    if (error != null) D1Demo.failed(error)
    return D1GateRunner.Summary(status, destination.absolutePath, error)
  }

  /**
   * One audio request, timed step by step: the wav of [record] read from `files/` and parsed, its
   * prefix ([D1AudioEngine.audioPrefix]), this app's encoding of every question after the prefix rows
   * (kind audio), then per question the decision inputs, one call on the smallest compiled graph that
   * holds P + n, the read-out and `answer`. The graphs are compiled before; nothing compiles here.
   */
  private fun audioRequest(engine: D1Engine, record: D1AudioRecord): LinkedHashMap<String, Any?> {
    val clock = D1Device.gpuState()?.maxClockMhz
    val wall = System.currentTimeMillis()
    val start = System.nanoTime()
    val samples = D1Wav.parse(File(files, record.mediaFile).readBytes(), record.mediaFile)
    val parsed = System.nanoTime()
    val audio = engine.audio.audioPrefix(samples)
    val prefixed = System.nanoTime()
    val questions = record.questions.values.toList()
    val rows =
      D1Rows.rows(engine.tokenizer, engine.contract, record.state, questions, audio.info.prefixRows, D1Kind.AUDIO)
    val encoded = System.nanoTime()
    var inputsNanos = 0L
    var readoutNanos = 0L
    val calls = ArrayList<List<Double>>()
    val buckets = ArrayList<Int>()
    val probabilities = ArrayList<List<Double>>()
    var finite = true
    for (row in rows) {
      val length =
        D1Contract.bucketFor(row.positions, engine.resident)
          ?: throw IllegalStateException("no compiled decision graph holds ${row.positions} positions")
      val inputsStart = System.nanoTime()
      val inputs = D1Rows.buildInputs(row.ids, audio.prefix, row.prefixRows, length, row.question.type)
      inputsNanos += System.nanoTime() - inputsStart
      val call = engine.call(length, inputs)
      val readStart = System.nanoTime()
      val probs = D1Readout.probabilities(call.scores, row.prefixRows, row.markers, row.question, row.calibrate, engine.contract)
      D1Prompt.answer(row.question, probs)
      readoutNanos += System.nanoTime() - readStart
      finite = finite && D1Readout.finite(D1Readout.markerScores(call.scores, row.prefixRows, row.markers, row.question.options))
      calls.add(listOf(call.totalMs, call.writeMs, call.runMs, call.readMs))
      buckets.add(length)
      probabilities.add(probs.toList())
    }
    val end = System.nanoTime()
    val expected = record.rows
    val idsMatch =
      rows.size == expected.size &&
        rows.indices.all { rows[it].ids.contentEquals(expected[it].ids) && rows[it].markers.contentEquals(expected[it].markers) }
    return linkedMapOf<String, Any?>(
        "wall_ms" to wall,
        "kgsl_max_clock_mhz" to clock,
        "request_ms" to (end - start) / 1e6,
        "wav_ms" to (parsed - start) / 1e6,
      )
      .apply {
        putAll(audio.times())
        put("audio_ms", (prefixed - parsed) / 1e6)
        put("encode_ms", (encoded - prefixed) / 1e6)
        put("decision_inputs_ms", inputsNanos / 1e6)
        put("decision_calls", calls)
        put("decision_ms", calls.sumOf { it[0] })
        put("readout_ms", readoutNanos / 1e6)
        put("decision_buckets", buckets)
        put("P", audio.info.prefixRows)
        put("audio_backend", audio.backend.wireName)
        put("ids_match", idsMatch)
        put("finite", finite)
        put("probs", probabilities)
      }
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
