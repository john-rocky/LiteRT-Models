// SPDX-License-Identifier: Apache-2.0
package com.julia1

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors

/**
 * One Julia-1 request per graph call. The GPU path explicitly requests FP32 arithmetic: the answers
 * change under fp16 (see the model card), so no fp16 GPU precision and no NPU path is offered.
 */
class JuliaEngine(context: Context) : Closeable {
  /** Selects the accelerator explicitly; GPU creation never silently falls back. */
  enum class Backend(val accelerator: Accelerator) {
    GPU(Accelerator.GPU),
    CPU(Accelerator.CPU);

    companion object {
      /** Parses a supported intent selector, rejecting unknown values. */
      fun fromArgument(value: String): Backend =
        when (value.lowercase()) {
          "gpu" -> GPU
          "cpu" -> CPU
          else -> error("Unknown accelerator: $value; expected gpu or cpu")
        }
    }
  }

  /** Includes output readback because GPU run() can enqueue work asynchronously. */
  data class GraphTiming(val writeMs: Double, val enqueueMs: Double, val readbackMs: Double) {
    /** Total write, enqueue, and readback duration in milliseconds. */
    val totalMs: Double
      get() = writeMs + enqueueMs + readbackMs

    /** Emits named timing components for machine-readable gate reports. */
    fun toMap(): Map<String, Double> =
      linkedMapOf(
        "write_ms" to writeMs,
        "run_enqueue_ms" to enqueueMs,
        "readback_ms" to readbackMs,
        "write_run_read_ms" to totalMs,
      )
  }

  /** Raw model tensors and host/graph timings for one request. */
  data class RawResult(
    val window: Int,
    val tokenLogits: FloatArray,
    val markerLogits: DoubleArray,
    val timing: GraphTiming,
    val embeddingLookupMs: Double,
  ) {
    /** Rejects any NaN or infinity before decoding an answer. */
    val finite: Boolean
      get() = tokenLogits.all { it.isFinite() } && markerLogits.all { it.isFinite() }
  }

  /** Decoded answer with its built sequence and per-request timings. */
  data class AnswerResult(
    val sequence: JuliaSequence,
    val answer: JuliaDecoder.Answer,
    val raw: RawResult,
    val prepareMs: Double,
    val decodeMs: Double,
    val totalMs: Double,
  )

  private data class Key(val window: Int, val backend: Backend)

  private class Graph(
    val model: CompiledModel,
    val inputs: Map<String, TensorBuffer>,
    val outputs: Map<String, TensorBuffer>,
  ) : Closeable {
    override fun close() {
      try {
        (inputs.values + outputs.values).forEach { it.close() }
      } finally {
        model.close()
      }
    }
  }

  private val appContext = context.applicationContext
  private val filesDir = appContext.filesDir
  private val graphs = linkedMapOf<Key, Graph>()
  private val embeddingInputs = linkedMapOf<Int, FloatArray>()
  private val embeddings: JuliaEmbeddings
  private val tokenizer: JuliaTokenizer
  /** Tokenizer initialization duration in milliseconds. */
  val tokenizerLoadMs: Double
  /** Table mapping duration in milliseconds. */
  val embeddingLoadMs: Double

  private var closed = false

  init {
    REQUIRED_FILES.forEach { requireFile(it) }
    val started = System.nanoTime()
    tokenizer = JuliaTokenizer(requireFile(TOKENIZER_FILE))
    tokenizerLoadMs = milliseconds(System.nanoTime() - started)
    val embeddingStarted = System.nanoTime()
    embeddings = JuliaEmbeddings(requireFile(TABLE_FILE))
    embeddingLoadMs = milliseconds(System.nanoTime() - embeddingStarted)
  }

  /** True when the graph file for this window is installed; 512 is required, 1024 optional. */
  fun windowInstalled(window: Int): Boolean = File(filesDir, graphFilename(window)).isFile

  /** Compiles separately from measured calls so the first graph call stays a cold observation. */
  fun initialize(backend: Backend, window: Int = DEFAULT_WINDOW) = JuliaProcessRuntime.call {
    checkOpen()
    graph(window, backend)
    Unit
  }

  /**
   * Builds the row for the smallest installed window that fits it. A request over 512 tokens uses
   * the S1024 graph when that file is installed; otherwise the strict encoding error is rethrown.
   */
  fun prepare(state: Any?, question: JuliaQuestion): JuliaSequence {
    checkOpen()
    val builders =
      WINDOWS.filter { windowInstalled(it) }.map { JuliaSequenceBuilder(tokenizer, it) }
    var failure: EncodingException? = null
    for (builder in builders) {
      try {
        return builder.build(state, question)
      } catch (rejected: EncodingException) {
        failure = rejected
      }
    }
    throw checkNotNull(failure)
  }

  /** Builds the row for one explicit window, as the fixture gate does. */
  fun prepare(state: Any?, question: JuliaQuestion, window: Int): JuliaSequence {
    checkOpen()
    return JuliaSequenceBuilder(tokenizer, window).build(state, question)
  }

  /** Smallest supported window that holds this sequence. */
  fun windowFor(sequence: JuliaSequence): Int =
    WINDOWS.firstOrNull { sequence.ids.size <= it }
      ?: throw EncodingException("Request needs ${sequence.ids.size} tokens; the window is 1024")

  /**
   * One graph invocation including the output readback. Buffers are addressed by signature name.
   * Nonfinite output is preserved in the result; the caller decides not to decode it.
   */
  fun runRaw(
    sequence: JuliaSequence,
    backend: Backend,
    window: Int = windowFor(sequence),
  ): RawResult = JuliaProcessRuntime.call {
    checkOpen()
    require(sequence.ids.size <= window) { "Sequence exceeds window $window" }
    val graph = graph(window, backend)
    val attention = FloatArray(window) { if (it < sequence.ids.size) 1f else 0f }
    val qtype = FloatArray(3).also { it[sequence.qtype] = 1f }
    val lookupStarted = System.nanoTime()
    val embeds = embeddingInputs.getOrPut(window) { FloatArray(window * JuliaEmbeddings.WIDTH) }
    embeddings.gather(sequence.ids, window, embeds)
    val started = System.nanoTime()
    graph.inputs.getValue("inputs_embeds").writeFloat(embeds)
    graph.inputs.getValue("attention_mask").writeFloat(attention)
    graph.inputs.getValue("qtype_onehot").writeFloat(qtype)
    val written = System.nanoTime()
    graph.model.run(graph.inputs, graph.outputs, SIGNATURE)
    val enqueued = System.nanoTime()
    val tokens = graph.outputs.getValue("token_logits").readFloat()
    val read = System.nanoTime()
    require(tokens.size == window) { "Unexpected token_logits size ${tokens.size}" }
    RawResult(
      window,
      tokens,
      JuliaDecoder.gather(tokens, sequence.markers),
      GraphTiming(
        milliseconds(written - started),
        milliseconds(enqueued - written),
        milliseconds(read - enqueued),
      ),
      milliseconds(started - lookupStarted),
    )
  }

  /** Builder, graph and decoder with separate phase timings. */
  fun answer(state: Any?, question: JuliaQuestion, backend: Backend): AnswerResult {
    val started = System.nanoTime()
    val sequence = prepare(state, question)
    val prepared = System.nanoTime()
    val raw = runRaw(sequence, backend)
    check(raw.finite) {
      "Nonfinite model output on ${backend.name}; inspect the device gate report"
    }
    val decodeStarted = System.nanoTime()
    val decoded = JuliaDecoder.decode(raw.markerLogits, question)
    val finished = System.nanoTime()
    return AnswerResult(
      sequence,
      decoded,
      raw,
      milliseconds(prepared - started),
      milliseconds(finished - decodeStarted),
      milliseconds(finished - started),
    )
  }

  private fun graph(window: Int, backend: Backend): Graph =
    graphs.getOrPut(Key(window, backend)) {
      val options =
        CompiledModel.Options(backend.accelerator).apply {
          if (backend == Backend.GPU) {
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          } else {
            cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
          }
        }
      val model =
        CompiledModel.create(
          requireFile(graphFilename(window)).absolutePath,
          options,
          JuliaProcessRuntime.environment(appContext),
        )
      val inputs = linkedMapOf<String, TensorBuffer>()
      val outputs = linkedMapOf<String, TensorBuffer>()
      try {
        INPUT_NAMES.forEach { inputs[it] = model.createInputBuffer(it, SIGNATURE) }
        outputs[OUTPUT_NAME] = model.createOutputBuffer(OUTPUT_NAME, SIGNATURE)
        Graph(model, inputs, outputs)
      } catch (failure: Throwable) {
        (inputs.values + outputs.values).forEach { it.close() }
        model.close()
        throw failure
      }
    }

  private fun requireFile(name: String): File =
    File(filesDir, name).also {
      check(it.isFile) { "Missing $name. Run scripts/install_to_device.sh, then relaunch." }
    }

  private fun checkOpen() = check(!closed) { "JuliaEngine is closed" }

  /** Closes model buffers and releases table references on the native runtime thread. */
  override fun close() = JuliaProcessRuntime.call {
    if (!closed) {
      closed = true
      try {
        graphs.values.forEach { it.close() }
      } finally {
        graphs.clear()
        embeddingInputs.clear()
        embeddings.close()
      }
    }
  }

  companion object {
    /** Runtime pin shared with the version catalog and gate metadata. */
    const val LITERT_VERSION = "2.2.0"
    /** Window of the graph compiled at start; the S1024 graph is compiled on first use. */
    const val DEFAULT_WINDOW = 512
    /** Supported static windows, smallest first. */
    val WINDOWS = listOf(512, 1024)
    /** Float16 token table `[256000, 384]`, little-endian, from the model repository. */
    const val TABLE_FILE = "julia1_token_table_fp16.bin"
    /** The source repository's tokenizer.json, unchanged. */
    const val TOKENIZER_FILE = "tokenizer.json"
    private const val SIGNATURE = "serving_default"
    private const val OUTPUT_NAME = "token_logits"
    private val INPUT_NAMES = listOf("attention_mask", "inputs_embeds", "qtype_onehot")
    /** Files every launch needs; the S1024 graph is optional. */
    val REQUIRED_FILES = listOf(graphFilename(DEFAULT_WINDOW), TABLE_FILE, TOKENIZER_FILE)

    /** Graph file for a static window, as named in the model repository. */
    fun graphFilename(window: Int): String = "julia1_s${window}_fp32.tflite"

    private fun milliseconds(nanos: Long) = nanos / 1_000_000.0
  }
}

/** One native thread and one Environment for the lifetime of this application process. */
private object JuliaProcessRuntime {
  private val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "Julia-LiteRT").apply { isDaemon = true }
  }
  private var sharedEnvironment: Environment? = null

  // The same Environment options as the measuring gate app; harmless for GPU and CPU.
  fun environment(context: Context): Environment =
    sharedEnvironment
      ?: Environment.create(
          context,
          mapOf(
            Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
            Environment.Option.CompilerPluginLibraryDir to context.applicationInfo.nativeLibraryDir,
          ),
        )
        .also { sharedEnvironment = it }

  fun <T> call(block: () -> T): T =
    try {
      executor.submit(Callable { block() }).get()
    } catch (failure: ExecutionException) {
      throw failure.cause ?: failure
    }
}
