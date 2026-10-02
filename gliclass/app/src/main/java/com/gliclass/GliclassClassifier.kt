package com.gliclass

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
 * Android orchestration of the GLiClass-Edge v3.0 host/graph/host contract: Kotlin tokenizer and
 * linearization → float16 table lookup upcast to float32 → one LiteRT graph with three float32
 * inputs, `inputs_embeds [1,N,384]`, `attention_mask [1,N]` and `label_routing [1,25,N]` → one
 * `logits [1,1,1,25]` buffer → [GliclassDecoder].
 *
 * GPU FP32 precision is mandatory: default GPU precision runs the graph in float16 and changed
 * decisions on a Galaxy S26. Windows N = 128 and 256 are selected without truncation. All native
 * buffers live on one worker thread behind a process-wide Environment; callers use the ViewModel's
 * confined dispatcher. Compiled windows stay resident until close, with no accelerator fallback.
 */
class GliclassClassifier(context: Context) : Closeable {
  /** Explicit LiteRT execution policy; GPU always requests FP32. */
  enum class Backend(val accelerator: Accelerator, val precision: String) {
    /** Mandatory FP32 GPU computation. */
    GPU(Accelerator.GPU, "FP32 (explicit)"),
    /** Float32 CPU execution, selected by the caller rather than used as a fallback. */
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

  /** One classified request: the decision, the n logits and window metadata. */
  data class Result(
    val text: String,
    val prompt: String?,
    val decision: GliclassDecoder.Decision,
    val backend: Backend,
    val window: Int,
    val encodedTokens: Int,
    val inputIds: IntArray,
    val labelPositions: IntArray,
    val timing: Timing,
  )

  private data class Key(val window: Int, val backend: Backend)

  /** Wall-clock phases of one graph call: input writes, `run()` enqueue, output readback. */
  private class GraphPhases(val writeMs: Double, val enqueueMs: Double, val readbackMs: Double) {
    /** The graph interval: first input write through output readback. */
    val totalMs: Double
      get() = writeMs + enqueueMs + readbackMs
  }

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
  private val inputBuilder: GliclassInputs
  private val embeddingTable: GliclassInputs.EmbeddingTable
  private var closed = false

  /** Compile time of every graph compiled so far, in milliseconds, by "s<window>_<BACKEND>". */
  val compileMs = linkedMapOf<String, Double>()

  init {
    // Check the complete installation, including the lazy s256 window, before opening resources.
    REQUIRED_FILES.forEach { requireFile(it) }
    val embeddings = requireFile(TABLE_FILE)
    require(embeddings.length() == GliclassInputs.TABLE_BYTES) {
      "Invalid $TABLE_FILE size. Run scripts/install_to_device.sh."
    }
    embeddingTable = GliclassInputs.EmbeddingTable(embeddings)
    inputBuilder =
      try {
        GliclassInputs(GliclassTokenizer(requireFile(TOKENIZER_FILE)), embeddingTable)
      } catch (failure: Throwable) {
        embeddingTable.close()
        throw failure
      }
  }

  /** Compiles s128 before interactive warm-up or fixture validation; s256 stays lazy. */
  fun initialize(backend: Backend = Backend.GPU) = ProcessRuntime.call {
    checkOpen()
    graph(GliclassInputs.WINDOWS.first(), backend)
    Unit
  }

  /**
   * Exposes the host input path used by inference for captured-input checks, outside timing. With
   * [window] null the smallest fitting window is used.
   */
  fun inspectInputs(
    text: String,
    labels: List<String>,
    prompt: String?,
    window: Int? = null,
  ): GliclassInputs.Prepared = ProcessRuntime.call {
    checkOpen()
    inputBuilder.prepare(text, labels, prompt, window)
  }

  /** One untimed full pass (tokenize, embed, graph, decode) at [window] or the smallest fit. */
  fun warmUp(
    text: String,
    labels: List<String>,
    prompt: String?,
    backend: Backend,
    window: Int? = null,
  ) = ProcessRuntime.call {
    checkOpen()
    val prepared = inputBuilder.prepare(text, labels, prompt, window)
    val graph = graph(prepared.window, backend)
    val logits = runGraph(graph, prepared).first
    GliclassDecoder.decide(logits, labels, GliclassDecoder.Mode.SINGLE_LABEL)
    graph.warmed = true
  }

  /**
   * Repeats the full pipeline [STARTUP_WARMUP_ITERATIONS] times on the bundled example before the
   * UI reports Ready, so the first request does not pay JIT and first-dispatch costs. Returns the
   * total wall time, excluding compilation.
   */
  fun warmUpForInteraction(
    text: String,
    labels: List<String>,
    prompt: String?,
    backend: Backend,
  ): Double {
    val start = System.nanoTime()
    repeat(STARTUP_WARMUP_ITERATIONS) { warmUp(text, labels, prompt, backend) }
    return ms(System.nanoTime() - start)
  }

  /**
   * Classifies [text] against [labels] on [backend] at [window] (null = smallest fitting window).
   * Tokenize/embed, graph through readback and decode are timed separately; one untimed pass per
   * compiled window/backend is excluded.
   */
  fun classify(
    text: String,
    labels: List<String>,
    prompt: String?,
    mode: GliclassDecoder.Mode,
    threshold: Double = GliclassDecoder.DEFAULT_THRESHOLD,
    backend: Backend = Backend.GPU,
    window: Int? = null,
  ): Result = ProcessRuntime.call {
    checkOpen()
    val start = System.nanoTime()
    val prepared = inputBuilder.prepare(text, labels, prompt, window)
    val preparedAt = System.nanoTime()
    val graph = graph(prepared.window, backend)
    if (!graph.warmed) {
      GliclassDecoder.decide(runGraph(graph, prepared).first, labels, mode, threshold)
      graph.warmed = true
    }
    val (logits, phases) = runGraph(graph, prepared)
    val decodeStart = System.nanoTime()
    val decision = GliclassDecoder.decide(logits, labels, mode, threshold)
    val finished = System.nanoTime()
    Result(
      text,
      prompt,
      decision,
      backend,
      prepared.window,
      prepared.encoded.encodedLength,
      prepared.encoded.inputIds,
      prepared.encoded.labelPositions,
      Timing(
        ms(preparedAt - start),
        phases.totalMs,
        ms(finished - decodeStart),
        phases.writeMs,
        phases.enqueueMs,
        phases.readbackMs,
      ),
    )
  }

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
      val compileStart = System.nanoTime()
      val model = CompiledModel.create(path.absolutePath, options, ProcessRuntime.environment())
      compileMs["s${window}_${backend.name}"] = ms(System.nanoTime() - compileStart)
      val inputs = linkedMapOf<String, TensorBuffer>()
      val outputs = linkedMapOf<String, TensorBuffer>()
      try {
        val expected =
          mapOf(
            EMBEDS_NAME to listOf(1, window, GliclassInputs.HIDDEN_SIZE),
            ATTENTION_NAME to listOf(1, window),
            ROUTING_NAME to listOf(1, GliclassInputs.LABEL_SLOTS, window),
          )
        for ((name, shape) in expected) {
          val actual = dimensions(model.getInputTensorType(name, SIGNATURE).layout?.dimensions)
          check(actual == shape) { "Input $name of ${path.name} has shape $actual, not $shape" }
          inputs[name] = model.createInputBuffer(name, SIGNATURE)
        }
        val outputShape =
          dimensions(model.getOutputTensorType(OUTPUT_NAME, SIGNATURE).layout?.dimensions)
        check(outputShape == listOf(1, 1, 1, GliclassInputs.LABEL_SLOTS)) {
          "Unexpected output shape $outputShape in ${path.name}"
        }
        outputs[OUTPUT_NAME] = model.createOutputBuffer(OUTPUT_NAME, SIGNATURE)
        Graph(model, inputs, outputs)
      } catch (failure: Throwable) {
        (inputs.values + outputs.values).forEach { it.close() }
        model.close()
        throw failure
      }
    }
  }

  /** The measured graph interval starts before the FIRST write and ends after readback. */
  private fun runGraph(
    graph: Graph,
    prepared: GliclassInputs.Prepared,
  ): Pair<FloatArray, GraphPhases> {
    val start = System.nanoTime()
    graph.inputs.getValue(EMBEDS_NAME).writeFloat(prepared.embeds)
    graph.inputs.getValue(ATTENTION_NAME).writeFloat(prepared.attentionMask)
    graph.inputs.getValue(ROUTING_NAME).writeFloat(prepared.labelRouting)
    val written = System.nanoTime()
    graph.model.run(graph.inputs, graph.outputs, SIGNATURE)
    val enqueued = System.nanoTime()
    val logits = graph.outputs.getValue(OUTPUT_NAME).readFloat()
    val read = System.nanoTime()
    check(logits.size == GliclassInputs.LABEL_SLOTS) { "Unexpected logits size ${logits.size}" }
    return logits to GraphPhases(ms(written - start), ms(enqueued - written), ms(read - enqueued))
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

    /** Full-pipeline passes on the bundled example before Ready (after s128 compilation). */
    const val STARTUP_WARMUP_ITERATIONS = 5

    /** Threads for the explicit CPU backend. */
    const val CPU_THREADS = 4

    /** The float16 embedding table the installer copies into `filesDir`. */
    const val TABLE_FILE = "tok_embeddings_fp16.bin"
    private const val TOKENIZER_FILE = "tokenizer.json"
    private const val NANOS_PER_MILLI = 1_000_000.0
    private const val SIGNATURE = "serving_default"
    private const val EMBEDS_NAME = "inputs_embeds"
    private const val ATTENTION_NAME = "attention_mask"
    private const val ROUTING_NAME = "label_routing"
    private const val OUTPUT_NAME = "logits"

    /** File name of the float32 graph for encoded window [window]. */
    fun graphFile(window: Int) = "gliclass_edge_v3_s${window}_fp32.tflite"

    /** Every file `scripts/install_to_device.sh` installs; all are checked at construction. */
    val REQUIRED_FILES: List<String> =
      GliclassInputs.WINDOWS.map { graphFile(it) } + listOf(TABLE_FILE, TOKENIZER_FILE)

    private fun dimensions(values: List<Int>?): List<Int> = values.orEmpty()

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
    Thread(runnable, "Gliclass-LiteRT").apply { isDaemon = true }
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
