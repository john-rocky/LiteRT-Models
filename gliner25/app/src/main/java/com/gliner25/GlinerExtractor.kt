package com.gliner25

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
 * Android orchestration of the published `host_runtime.py:HostRuntime.prepare/decode` contract,
 * whose tokenizer and sparse math come from gliner2 2.0.0. Five float32 inputs enter LiteRT:
 * embeddings `[1,N,384]`, attention `[1,N]`, text routing `[1,T,N]`, query routing `[1,5,N]` and
 * text mask `[1,T]`; one `[1,1,1,1108*T+4574]` float32 buffer returns to the host decoder.
 *
 * GPU FP32 precision is mandatory: default GPU precision produces NaN for this graph. Windows N/T =
 * 128/48, 256/192 and 512/384 are selected without truncation. All native buffers live on one
 * worker thread behind a process-wide Environment; callers use the ViewModel's confined dispatcher.
 * Compiled windows stay resident until close, with no accelerator fallback.
 *
 * The three graph-contract JSON files are read from filesDir. The small, unchanged checkpoint
 * config is bundled as `gliner_config.json`. Result offsets count Unicode code points.
 */
class GlinerExtractor(context: Context) : Closeable {
  /** Explicit LiteRT execution policy; GPU always requests FP32 to avoid nonfinite graph output. */
  enum class Backend(val accelerator: Accelerator, val precision: String) {
    /** Mandatory FP32 GPU computation; model storage can still use float16 weights. */
    GPU(Accelerator.GPU, "FP32 (explicit)"),
    /** Explicit float32 CPU execution, selected by the caller rather than used as a fallback. */
    CPU(Accelerator.CPU, "FP32"),
  }

  /**
   * Millisecond wall-clock phases of the host/graph/host pipeline. [graphMs] includes the first
   * input-buffer write through output readback because `CompiledModel.run()` only enqueues work.
   * Compilation and untimed warm-up are excluded, so enqueue time alone is never GPU latency.
   */
  data class Timing(
    val tokenizeEmbedMs: Double,
    val graphMs: Double,
    val decodeMs: Double,
    val writeMs: Double,
    val enqueueMs: Double,
    val readbackMs: Double,
  )

  /**
   * Completed `HostRuntime.prepare/decode` equivalent, including source code-point spans and actual
   * capacity/length metadata. [window] counts encoded slots N, including the schema; [textWords]
   * counts upstream splitter words, which have a separate T capacity.
   */
  data class Result(
    val text: String,
    val spans: List<GlinerDecoder.Span>,
    val backend: Backend,
    val window: Int,
    val encodedTokens: Int,
    val textWords: Int,
    val timing: Timing,
  )

  private data class Key(val window: Int, val backend: Backend)

  private class Graph(
    val model: CompiledModel,
    val inputs: Map<String, TensorBuffer>,
    val outputs: Map<String, TensorBuffer>,
    var warmed: Boolean = false,
  ) : Closeable {
    override fun close() {
      try {
        (inputs.values + outputs.values).forEach { it.close() }
      } finally {
        model.close()
      }
    }
  }

  private val filesDir = context.applicationContext.filesDir
  private val graphs = linkedMapOf<Key, Graph>()
  private val inputBuilder: GlinerInputs
  private val decoder: GlinerDecoder
  private val embeddingTable: GlinerInputs.EmbeddingTable
  private var closed = false

  init {
    // Check the complete installation, including lazy windows, before opening resources.
    REQUIRED_FILES.forEach { requireFile(it) }
    val embeddings = requireFile("word_embeddings_fp32.bin")
    require(embeddings.length() == 196_624_896L) {
      "Invalid word_embeddings_fp32.bin size. Run scripts/install_to_device.sh."
    }
    inputBuilder = GlinerInputs(GlinerTokenizer(requireFile("tokenizer.json")))
    val config = context.assets.open("gliner_config.json").bufferedReader().use { it.readText() }
    decoder = GlinerDecoder(filesDir, config)
    embeddingTable = GlinerInputs.EmbeddingTable(embeddings)
  }

  /**
   * Compiles s128 at startup; its untimed warm-up uses the first actual input. Larger windows
   * remain lazy to avoid paying compilation and memory costs before the host contract needs them.
   */
  fun initialize(backend: Backend = Backend.GPU) = ProcessRuntime.call {
    checkOpen()
    graph(GlinerInputs.WINDOWS.first(), backend)
    Unit
  }

  /**
   * Exposes the same `HostRuntime.prepare` host path used by inference for captured-input checks.
   * IDs are padded to N; text/query positions select first subwords in that sequence. Validation
   * runs outside timing, so inspecting N/T = 128/48, 256/192 or 512/384 never inflates graph
   * latency.
   */
  fun inspectInputs(text: String): GlinerInputs.Prepared = ProcessRuntime.call {
    checkOpen()
    inputBuilder.prepare(text)
  }

  /**
   * Executes one untimed `HostRuntime.prepare/decode` equivalent on the smallest fitting window.
   * Explicit validation callers can warm each text; normal extraction only warms each compiled
   * window/backend once. GPU uses mandatory FP32 precision throughout both paths.
   */
  fun warmUp(text: String, backend: Backend) = ProcessRuntime.call {
    checkOpen()
    val prepared = inputBuilder.prepare(text)
    val inputs = tensors(prepared)
    val graph = graph(prepared.window, backend)
    val packed = runGraph(graph, inputs).first
    decoder.decode(packed, prepared)
    graph.warmed = true
  }

  /**
   * Runs the published host/graph/host pipeline without truncation or backend fallback. The
   * smallest N/T = 128/48, 256/192 or 512/384 window must fit both encoded tokens and words.
   * Tokenize/embed, graph through readback, and sparse decode are timed separately; one untimed
   * warm-up per compiled window/backend is excluded. Returned spans use Unicode code points.
   */
  fun extract(text: String, backend: Backend = Backend.GPU): Result = ProcessRuntime.call {
    checkOpen()
    require(text.isNotBlank()) { "Enter text before extracting entities." }
    val start = System.nanoTime()
    val prepared = inputBuilder.prepare(text)
    val inputs = tensors(prepared)
    val preparedAt = System.nanoTime()
    val graph = graph(prepared.window, backend)
    if (!graph.warmed) {
      decoder.decode(runGraph(graph, inputs).first, prepared)
      graph.warmed = true
    }
    val (packed, graphTimes) = runGraph(graph, inputs)
    val decodeStart = System.nanoTime()
    require(packed.size == prepared.window.packedFloatCount) { "Unexpected packed output size" }
    val spans = decoder.decode(packed, prepared)
    val finished = System.nanoTime()
    Result(
      text,
      spans,
      backend,
      prepared.window.sequenceLength,
      prepared.encodedLength,
      prepared.words.size,
      Timing(
        ms(preparedAt - start),
        graphTimes.sum(),
        ms(finished - decodeStart),
        graphTimes[0],
        graphTimes[1],
        graphTimes[2],
      ),
    )
  }

  private fun tensors(prepared: GlinerInputs.Prepared): List<FloatArray> =
    listOf(
      embeddingTable.lookup(prepared.inputIds),
      prepared.attentionMask,
      prepared.textRouting,
      prepared.queryRouting,
      prepared.textMask,
    )

  private fun graph(window: GlinerInputs.Window, backend: Backend): Graph {
    val key = Key(window.sequenceLength, backend)
    return graphs.getOrPut(key) {
      val options =
        CompiledModel.Options(backend.accelerator).apply {
          if (backend == Backend.GPU) {
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          } else {
            cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
          }
        }
      val path = requireFile("gliner25_small_s${window.sequenceLength}_wfp16.tflite")
      val model = CompiledModel.create(path.absolutePath, options, ProcessRuntime.environment())
      val inputs = linkedMapOf<String, TensorBuffer>()
      val outputs = linkedMapOf<String, TensorBuffer>()
      try {
        // Use the signature names explicitly instead of assuming FlatBuffer storage order.
        for (index in 0..4) {
          inputs["args_$index"] = model.createInputBuffer("args_$index", SIGNATURE)
        }
        outputs["output_0"] = model.createOutputBuffer("output_0", SIGNATURE)
        Graph(model, inputs, outputs)
      } catch (failure: Throwable) {
        (inputs.values + outputs.values).forEach { it.close() }
        model.close()
        throw failure
      }
    }
  }

  /** The measured graph interval starts before the FIRST write and ends after readback. */
  private fun runGraph(graph: Graph, inputs: List<FloatArray>): Pair<FloatArray, DoubleArray> {
    val start = System.nanoTime()
    inputs.forEachIndexed { index, values ->
      graph.inputs.getValue("args_$index").writeFloat(values)
    }
    val written = System.nanoTime()
    graph.model.run(graph.inputs, graph.outputs, SIGNATURE)
    val enqueued = System.nanoTime()
    val packed = graph.outputs.getValue("output_0").readFloat()
    val read = System.nanoTime()
    return packed to doubleArrayOf(ms(written - start), ms(enqueued - written), ms(read - enqueued))
  }

  private fun requireFile(name: String): File =
    File(filesDir, name).also {
      check(it.isFile) { "Missing $name. Run scripts/install_to_device.sh, then retry." }
    }

  private fun checkOpen() = check(!closed) { "Extractor is closed" }

  /**
   * Releases compiled windows, native tensor buffers and the embedding channel on their owning
   * worker. The single process Environment deliberately survives Activity/ViewModel lifetimes.
   */
  override fun close() = ProcessRuntime.call {
    if (!closed) {
      closed = true
      try {
        graphs.values.forEach { it.close() }
      } finally {
        graphs.clear()
        embeddingTable.close()
      }
    }
  }

  companion object {
    const val LITERT_VERSION = "2.2.0"
    private const val SIGNATURE = "serving_default"
    val REQUIRED_FILES =
      listOf(
        "gliner25_small_s128_wfp16.tflite",
        "gliner25_small_s256_wfp16.tflite",
        "gliner25_small_s512_wfp16.tflite",
        "word_embeddings_fp32.bin",
        "tokenizer.json",
        "sparse_decoder_fp32.safetensors",
        "graph_contract_s128.json",
        "graph_contract_s256.json",
        "graph_contract_s512.json",
      )

    private fun ms(nanoseconds: Long) = nanoseconds / 1_000_000.0
  }
}

/**
 * Exactly one Environment for the process lifetime. A serial coroutine dispatcher can migrate
 * threads, so native creation/run/close also use this single thread. The Environment deliberately
 * outlives Activity/ViewModel instances.
 */
private object ProcessRuntime {
  private val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "Gliner-LiteRT").apply { isDaemon = true }
  }
  private var sharedEnvironment: Environment? = null

  fun environment(): Environment =
    sharedEnvironment ?: Environment.create().also { sharedEnvironment = it }

  fun <T> call(block: () -> T): T {
    try {
      return executor.submit(Callable { block() }).get()
    } catch (failure: ExecutionException) {
      throw failure.cause ?: failure
    }
  }
}
