package com.d1omni

import android.content.Context
import java.io.Closeable
import java.io.File

/**
 * One clip through the audio path ([D1AudioEngine.audioPrefix]): the [prefix] rows [P, 1024]
 * row-major, the clip's [info], its [mel], and the wall time of each step in nanoseconds — samples
 * to waveform, the mel, the graph's five inputs, the graph [call] (writes + `run()` + read-back) and
 * the copy of the first P rows. The graph's compile is not part of any step.
 */
class D1AudioPrefix(
  val prefix: FloatArray,
  val info: D1AudioInfo,
  val mel: D1Mel,
  val waveformNanos: Long,
  val melNanos: Long,
  val inputsNanos: Long,
  val call: D1GraphCall,
  val rowsNanos: Long,
  /** Where the audio graph ran. */
  val backend: D1Backend,
  /** Its GPU precision (null on the CPU). */
  val precision: D1Precision?,
) {
  /** Every step, samples to prefix rows, in milliseconds. */
  val totalMs: Double
    get() = (waveformNanos + melNanos + inputsNanos + rowsNanos) / NANOS_PER_MS + call.totalMs

  /** The steps in milliseconds, for reports. */
  fun times(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "waveform_ms" to waveformNanos / NANOS_PER_MS,
      "mel_ms" to melNanos / NANOS_PER_MS,
      "inputs_ms" to inputsNanos / NANOS_PER_MS,
      "graph_write_ms" to call.writeMs,
      "graph_run_ms" to call.runMs,
      "graph_read_ms" to call.readMs,
      "graph_write_run_read_ms" to call.totalMs,
      "prefix_rows_ms" to rowsNanos / NANOS_PER_MS,
      "samples_to_prefix_ms" to totalMs,
    )

  private companion object {
    const val NANOS_PER_MS = 1e6
  }
}

/**
 * The audio side of the engine: the audio graphs of `contract.json` (`files` entries of graph
 * `audio`, signature `audio_<T>`, one file per bucket T = 501 / 1001 / 2001 / 3001) found in `files/`
 * with the contract's size, at most one of them compiled at a time (a clip of another bucket closes
 * it first), and [audioPrefix]: samples -> [D1Audio]'s waveform, mel and inputs -> `audio_<T_b>` ->
 * the first P rows. Graphs compile on [backend] at [precision] and fall back to the CPU (four
 * threads) when the GPU cannot compile or run one (`GPU_FALLBACK` in logcat). Each compile is
 * appended to [compiles] with the memory before and after it; [residentDecisions] names the
 * decision graphs compiled at that moment. Use only on [D1Runtime.dispatcher].
 */
class D1AudioEngine(
  private val context: Context,
  contract: D1Contract,
  val backend: D1Backend,
  val precision: D1Precision,
  private val compiles: MutableList<D1Compile>,
  private val residentDecisions: () -> List<Int>,
) : Closeable {
  /** The audio graph file of each bucket T, ascending. */
  val bucketFiles: Map<Int, D1Contract.FileEntry> =
    contract.files
      .filter { it.graph == GRAPH && it.bucket != null && it.signature == signatureOf(it.bucket) }
      .associateBy { requireNotNull(it.bucket) }
      .toSortedMap()

  private var graph: D1Graph? = null
  private var graphBucket = 0

  /** The buckets whose file is in `files/` with the contract's size, ascending. */
  val installed: List<Int>
    get() = bucketFiles.filter { (_, entry) -> file(entry.name).length() == entry.bytes }.keys.sorted()

  /** The compiled bucket (at most one). */
  val resident: List<Int>
    get() = if (graph != null) listOf(graphBucket) else emptyList()

  /** Where the compiled graph runs, or null. */
  val residentBackend: D1Backend?
    get() = graph?.backend

  /** The compiled graph of [bucket]. */
  fun graph(bucket: Int): D1Graph {
    val current = graph
    check(current != null && graphBucket == bucket) { "audio T$bucket is not compiled" }
    return current
  }

  /** The smallest installed bucket that holds a clip of [stftFrames] STFT frames. */
  fun bucketFor(stftFrames: Int): Int {
    val buckets = installed
    check(buckets.isNotEmpty()) {
      "No audio graph installed (${bucketFiles.values.joinToString { it.name }}): run scripts/install_to_device.sh with AUDIO"
    }
    return D1Audio.bucketFor(stftFrames, buckets)
  }

  /** Compiles [bucket] unless it is the compiled one (another bucket is closed first). */
  fun ensure(bucket: Int): D1Graph {
    graph?.let { if (graphBucket == bucket) return it }
    require(bucket in installed) { "audio T$bucket is not installed (${bucketFiles[bucket]?.name})" }
    close()
    return compile(bucket, backend, null)
  }

  /**
   * [samples] (16 kHz mono int16) -> the prefix rows: waveform, mel, the graph's inputs on the
   * smallest installed bucket, one call of `audio_<T_b>` (compiled first when needed, outside the
   * timed steps), the first P rows.
   */
  fun audioPrefix(samples: ShortArray): D1AudioPrefix {
    val start = System.nanoTime()
    val waveform = D1Audio.waveform(samples)
    val waved = System.nanoTime()
    val mel = D1Audio.mel(waveform)
    val melDone = System.nanoTime()
    val bucket = bucketFor(mel.stftFrames)
    ensure(bucket)
    val inputsStart = System.nanoTime()
    val inputs = D1Audio.buildInputs(mel, bucket)
    val info = D1Audio.info(waveform.size, mel.frames, mel.stftFrames, bucket)
    val inputsDone = System.nanoTime()
    val call = call(bucket, inputs)
    val called = System.nanoTime()
    val prefix = D1Audio.prefixRows(call.values, info)
    val rowsDone = System.nanoTime()
    val ran = graph(bucket)
    return D1AudioPrefix(
      prefix,
      info,
      mel,
      waved - start,
      melDone - waved,
      inputsDone - inputsStart,
      call,
      rowsDone - called,
      ran.backend,
      if (ran.backend == D1Backend.GPU) ran.precision else null,
    )
  }

  /**
   * One call of the compiled [bucket] graph. A GPU graph that fails to run is closed and compiled on
   * the CPU, and the call is made there (`GPU_FALLBACK` in logcat).
   */
  fun call(bucket: Int, inputs: D1AudioInputs): D1GraphCall {
    val current = graph(bucket)
    return try {
      current.run(inputs.feeds())
    } catch (failure: Exception) {
      if (current.backend != D1Backend.GPU) throw failure
      val reason = "run audio T$bucket: ${D1Decider.describe(failure)}"
      D1Demo.gpuFallback(reason)
      close()
      compile(bucket, D1Backend.CPU, reason).run(inputs.feeds())
    }
  }

  override fun close() {
    val current = graph
    graph = null
    current?.close()
  }

  private fun file(name: String) = File(context.filesDir, name)

  private fun compile(bucket: Int, target: D1Backend, earlierFailure: String?): D1Graph {
    val entry = requireNotNull(bucketFiles[bucket]) { "contract.json has no audio T$bucket" }
    val before = D1Device.memory(context)
    val residentBefore = residentDecisions()
    val compiled =
      D1Graph.create(
        context,
        file(entry.name),
        signatureOf(bucket),
        D1Audio.inputShapes(bucket),
        D1Audio.outputShape(bucket),
        target,
        precision,
        fallback = target == D1Backend.GPU,
      )
    if (compiled.gpuFailure != null) D1Demo.gpuFallback("compile audio T$bucket: ${compiled.gpuFailure}")
    graph = compiled
    graphBucket = bucket
    compiles.add(
      D1Compile(
        bucket,
        target,
        compiled.backend,
        precision,
        compiled.compileMs,
        compiled.gpuFailure ?: earlierFailure,
        residentBefore,
        before,
        D1Device.memory(context),
        GRAPH,
      )
    )
    return compiled
  }

  companion object {
    /** The graph name of the audio files in `contract.json` `files`. */
    const val GRAPH = "audio"

    /**
     * The GPU precision of the audio graph when the launch names none (`precision_audio`):
     * FP16_WITH_FP32_ACCUM, the faster of the two on the Galaxy S26 (31.1 against 40.0 ms per call in
     * this app) with all 18 public audio rows inside the parity bar end to end there (max |Δp|
     * 0.00138; FP32: 0.00154).
     */
    val DEFAULT_PRECISION = D1Precision.FP16_FP32_ACCUM

    fun signatureOf(bucket: Int) = "audio_$bucket"
  }
}
