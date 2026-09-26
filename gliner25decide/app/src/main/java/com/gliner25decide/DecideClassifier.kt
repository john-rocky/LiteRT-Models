package com.gliner25decide

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
 * Android orchestration of the GLiNER2.5-Decide host/graph/host contract (`HOST_CONTRACT.md` in the
 * model repository): Kotlin tokenizer and schema → float16 table lookup upcast to float32 → one
 * LiteRT graph with three float32 inputs, embeddings `[1,N,1024]`, attention `[1,N]` and label
 * routing `[1,32,N]` → one `[1,1,1,32]` logits buffer → [DecideDecoder].
 *
 * GPU FP32 precision is mandatory: default GPU precision flipped 28 of 42 decisions on the S26.
 * Windows N = 128, 256 and 512 are selected without truncation. All native buffers live on one
 * worker thread behind a process-wide Environment; callers use the ViewModel's confined dispatcher.
 * Compiled windows stay resident until close, with no accelerator fallback.
 */
class DecideClassifier(context: Context) : Closeable {
  /** Explicit LiteRT execution policy; GPU always requests FP32. */
  enum class Backend(val accelerator: Accelerator, val precision: String) {
    /** Mandatory FP32 GPU computation; model storage still uses float16 weights. */
    GPU(Accelerator.GPU, "FP32 (explicit)"),
    /** Explicit float32 CPU execution, selected by the caller rather than used as a fallback. */
    CPU(Accelerator.CPU, "FP32"),
  }

  /**
   * Millisecond wall-clock phases. [graphMs] = first input-buffer write through output readback,
   * because `CompiledModel.run()` only enqueues work. Compilation and warm-up are excluded.
   */
  data class Timing(
    val tokenizeEmbedMs: Double,
    val graphMs: Double,
    val decodeMs: Double,
    val writeMs: Double,
    val enqueueMs: Double,
    val readbackMs: Double,
  )

  /** One classified request: per-task decisions, the raw 32 logits and window metadata. */
  data class Result(
    val text: String,
    val decisions: List<DecideDecoder.TaskDecision>,
    val logits: FloatArray,
    val backend: Backend,
    val window: Int,
    val encodedTokens: Int,
    val labelCount: Int,
    val timing: Timing,
  )

  private data class Key(val window: Int, val backend: Backend)

  private class Graph(
    val model: CompiledModel,
    val inputs: Map<String, TensorBuffer>,
    val embedsName: String,
    val attentionName: String,
    val routingName: String,
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
  private val inputBuilder: DecideInputs
  private val embeddingTable: DecideInputs.EmbeddingTable
  private var closed = false

  init {
    // Check the complete installation, including lazy windows, before opening resources.
    REQUIRED_FILES.forEach { requireFile(it) }
    val embeddings = requireFile(TABLE_FILE)
    require(embeddings.length() == DecideInputs.TABLE_BYTES) {
      "Invalid $TABLE_FILE size. Run scripts/install_to_device.sh."
    }
    inputBuilder = DecideInputs(GlinerTokenizer(requireFile(TOKENIZER_FILE)))
    embeddingTable = DecideInputs.EmbeddingTable(embeddings)
  }

  /** Compiles s128 before interactive warm-up or fixture validation; s256/s512 stay lazy. */
  fun initialize(backend: Backend = Backend.GPU) = ProcessRuntime.call {
    checkOpen()
    graph(DecideInputs.WINDOWS.first(), backend)
    Unit
  }

  /**
   * Exposes the host input path used by inference for captured-input checks, outside timing. With
   * [window] null the smallest fitting window is used.
   */
  fun inspectInputs(text: String, tasks: List<Task>, window: Int? = null): DecideInputs.Prepared =
    ProcessRuntime.call {
      checkOpen()
      inputBuilder.prepare(text, tasks, window)
    }

  /** One untimed full pass (tokenize, embed, graph, decode) at [window] or the smallest fit. */
  fun warmUp(text: String, tasks: List<Task>, backend: Backend, window: Int? = null) =
    ProcessRuntime.call {
      checkOpen()
      val prepared = inputBuilder.prepare(text, tasks, window)
      val graph = graph(prepared.window, backend)
      val logits = runGraph(graph, tensors(prepared)).first
      DecideDecoder.decode(logits, tasks)
      graph.warmed = true
    }

  /**
   * Repeats the full pipeline [STARTUP_WARMUP_ITERATIONS] times on the bundled example before the
   * UI reports Ready, so the first request does not pay JIT and first-dispatch costs. Returns the
   * total wall time, excluding compilation.
   */
  fun warmUpForInteraction(text: String, tasks: List<Task>, backend: Backend): Double {
    val start = System.nanoTime()
    repeat(STARTUP_WARMUP_ITERATIONS) { warmUp(text, tasks, backend) }
    return ms(System.nanoTime() - start)
  }

  /**
   * Classifies [text] with [tasks] on [backend] at [window] (null = smallest fitting window).
   * Tokenize/embed, graph through readback and decode are timed separately; one untimed pass per
   * compiled window/backend is excluded.
   */
  fun classify(
    text: String,
    tasks: List<Task>,
    backend: Backend = Backend.GPU,
    window: Int? = null,
  ): Result = ProcessRuntime.call {
    checkOpen()
    val start = System.nanoTime()
    val prepared = inputBuilder.prepare(text, tasks, window)
    val inputs = tensors(prepared)
    val preparedAt = System.nanoTime()
    val graph = graph(prepared.window, backend)
    if (!graph.warmed) {
      DecideDecoder.decode(runGraph(graph, inputs).first, tasks)
      graph.warmed = true
    }
    val (logits, graphTimes) = runGraph(graph, inputs)
    val decodeStart = System.nanoTime()
    val decisions = DecideDecoder.decode(logits, tasks)
    val finished = System.nanoTime()
    Result(
      text,
      decisions,
      logits,
      backend,
      prepared.window,
      prepared.encoded.encodedLength,
      prepared.encoded.labelCount,
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

  private class Tensors(val embeds: FloatArray, val attention: FloatArray, val routing: FloatArray)

  private fun tensors(prepared: DecideInputs.Prepared) =
    Tensors(embeddingTable.lookup(prepared.inputIds), prepared.attentionMask, prepared.labelRouting)

  private fun graph(window: Int, backend: Backend): Graph {
    val key = Key(window, backend)
    return graphs.getOrPut(key) {
      val options =
        CompiledModel.Options(backend.accelerator).apply {
          if (backend == Backend.GPU) {
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          } else {
            cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
          }
        }
      val path = requireFile(graphFile(window))
      val model = CompiledModel.create(path.absolutePath, options, ProcessRuntime.environment())
      val inputs = linkedMapOf<String, TensorBuffer>()
      val outputs = linkedMapOf<String, TensorBuffer>()
      try {
        // The converter names the inputs args_0..args_2; assign their meaning by shape rather
        // than by storage order, and check the single [1,1,1,32] output.
        val byShape = INPUT_NAMES.associateWith {
          dimensions(model.getInputTensorType(it, SIGNATURE).layout?.dimensions)
        }
        val embedsName = nameWithShape(byShape, listOf(1, window, DecideInputs.HIDDEN_SIZE))
        val attentionName = nameWithShape(byShape, listOf(1, window))
        val routingName = nameWithShape(byShape, listOf(1, DecideInputs.LABEL_SLOTS, window))
        val outputShape =
          dimensions(model.getOutputTensorType(OUTPUT_NAME, SIGNATURE).layout?.dimensions)
        check(outputShape == listOf(1, 1, 1, DecideInputs.LABEL_SLOTS)) {
          "Unexpected output shape $outputShape in ${path.name}"
        }
        for (name in INPUT_NAMES) {
          inputs[name] = model.createInputBuffer(name, SIGNATURE)
        }
        outputs[OUTPUT_NAME] = model.createOutputBuffer(OUTPUT_NAME, SIGNATURE)
        Graph(model, inputs, embedsName, attentionName, routingName, outputs)
      } catch (failure: Throwable) {
        (inputs.values + outputs.values).forEach { it.close() }
        model.close()
        throw failure
      }
    }
  }

  /** The measured graph interval starts before the FIRST write and ends after readback. */
  private fun runGraph(graph: Graph, tensors: Tensors): Pair<FloatArray, DoubleArray> {
    val start = System.nanoTime()
    graph.inputs.getValue(graph.embedsName).writeFloat(tensors.embeds)
    graph.inputs.getValue(graph.attentionName).writeFloat(tensors.attention)
    graph.inputs.getValue(graph.routingName).writeFloat(tensors.routing)
    val written = System.nanoTime()
    graph.model.run(graph.inputs, graph.outputs, SIGNATURE)
    val enqueued = System.nanoTime()
    val logits = graph.outputs.getValue(OUTPUT_NAME).readFloat()
    val read = System.nanoTime()
    check(logits.size == DecideInputs.LABEL_SLOTS) { "Unexpected logits size ${logits.size}" }
    return logits to doubleArrayOf(ms(written - start), ms(enqueued - written), ms(read - enqueued))
  }

  private fun requireFile(name: String): File =
    File(filesDir, name).also {
      check(it.isFile) { "Missing $name. Run scripts/install_to_device.sh, then retry." }
    }

  private fun checkOpen() = check(!closed) { "Classifier is closed" }

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
    /** LiteRT runtime version this sample was validated with; recorded in debug reports. */
    const val LITERT_VERSION = "2.2.0"

    /**
     * Full-pipeline passes on the bundled example before Ready (after s128 compilation). On the
     * Galaxy S26 five passes took 0.38–0.39 s and the first request after launch then ran in 72–78
     * ms end to end on GPU FP32.
     */
    const val STARTUP_WARMUP_ITERATIONS = 5

    /** Threads for the explicit CPU backend. */
    const val CPU_THREADS = 4

    /** The float16 embedding table the installer copies into `filesDir`. */
    const val TABLE_FILE = "word_embeddings_fp16.bin"
    private const val TOKENIZER_FILE = "tokenizer.json"
    private const val NANOS_PER_MILLI = 1_000_000.0
    private const val SIGNATURE = "serving_default"
    private const val OUTPUT_NAME = "output_0"
    private val INPUT_NAMES = listOf("args_0", "args_1", "args_2")

    /** File name of the float16-weight graph for encoded window [window]. */
    fun graphFile(window: Int) = "gliner25_decide_s${window}_wfp16.tflite"

    /** Every file `scripts/install_to_device.sh` installs; all are checked at construction. */
    val REQUIRED_FILES: List<String> =
      DecideInputs.WINDOWS.map { graphFile(it) } + listOf(TABLE_FILE, TOKENIZER_FILE)

    private fun dimensions(values: List<Int>?): List<Int> = values.orEmpty()

    private fun nameWithShape(byShape: Map<String, List<Int>>, shape: List<Int>): String {
      val matches = byShape.filterValues { it == shape }.keys
      check(matches.size == 1) { "Expected one input with shape $shape, found $byShape" }
      return matches.single()
    }

    private fun ms(nanoseconds: Long) = nanoseconds / NANOS_PER_MILLI
  }
}

/**
 * Exactly one Environment for the process lifetime. A serial coroutine dispatcher can migrate
 * threads, so native creation/run/close also use this single thread. The Environment deliberately
 * outlives Activity/ViewModel instances.
 */
private object ProcessRuntime {
  private val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "Decide-LiteRT").apply { isDaemon = true }
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
