package com.opendecision

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
 * Android orchestration of the host/graph/host contract of litert-community's DeBERTa-v3-large typed-decision
 * graphs: Kotlin tokenizer and sequence builder → float16 table lookup upcast to float32 → one LiteRT graph with
 * four float32 inputs (`inputs_embeds [1,N,1024]`, `attention_mask [1,N]`, `q_routing [1,128,N]`,
 * `o_routing [1,128,N]`) → one `logits [1,1,1,128]` buffer → [DecisionDecoder].
 *
 * GPU runs with explicit FP32 precision: at the default precision every output of these graphs is non-finite
 * (the attention mask constant is -3.4e38, which is -inf in fp16). Windows N = 256 and 512 are selected without
 * truncation. All native buffers live on one worker thread behind a process-wide Environment.
 */
class DecisionModel(context: Context) : Closeable {
  /** Explicit LiteRT execution policy; GPU always requests FP32. */
  enum class Backend(val accelerator: Accelerator, val precision: String) {
    GPU(Accelerator.GPU, "FP32 (explicit)"),
    CPU(Accelerator.CPU, "FP32"),
  }

  /**
   * Millisecond wall-clock phases. [graphMs] = first input-buffer write through output readback, because
   * `CompiledModel.run()` only enqueues work. Compilation and warm-up are excluded.
   */
  data class Timing(val tokenizeEmbedMs: Double, val graphMs: Double, val decodeMs: Double)

  /** One answered request. */
  data class Result(
    val state: String,
    val answers: List<DecisionDecoder.Answer>,
    val logits: FloatArray,
    val backend: Backend,
    val window: Int,
    val encodedTokens: Int,
    val optionCount: Int,
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
  val inputs: DecisionInputs
  private val table: DecisionInputs.EmbeddingTable
  private var closed = false

  init {
    REQUIRED_FILES.forEach { requireFile(it) }
    inputs = DecisionInputs(DecisionTokenizer(requireFile(TOKENIZER_FILE)))
    table = DecisionInputs.EmbeddingTable(requireFile(TABLE_FILE))
  }

  /** Compiles the s256 graph on [backend]; s512 stays lazy. */
  fun initialize(backend: Backend = Backend.GPU) = ProcessRuntime.call {
    checkOpen()
    graph(DecisionInputs.WINDOWS.first(), backend)
    Unit
  }

  /** One untimed pass at [window] or the smallest fit, so the first timed request pays no first-dispatch cost. */
  fun warmUp(state: String, questions: List<Question>, backend: Backend, window: Int? = null) =
    ProcessRuntime.call {
      checkOpen()
      val prepared = inputs.prepare(state, questions, window)
      val graph = graph(prepared.window, backend)
      runGraph(graph, prepared)
      graph.warmed = true
    }

  /** [STARTUP_WARMUP_ITERATIONS] full passes on the bundled example before Ready; returns the wall time. */
  fun warmUpForInteraction(state: String, questions: List<Question>, backend: Backend): Double {
    val start = System.nanoTime()
    repeat(STARTUP_WARMUP_ITERATIONS) { warmUp(state, questions, backend) }
    return ms(System.nanoTime() - start)
  }

  /** Answers [questions] about [state] on [backend] at [window] (null = the smallest window that fits). */
  fun decide(state: String, questions: List<Question>, backend: Backend = Backend.GPU, window: Int? = null): Result =
    ProcessRuntime.call {
      checkOpen()
      val start = System.nanoTime()
      val prepared = inputs.prepare(state, questions, window)
      val embeds = table.lookup(prepared.inputIds)
      val preparedAt = System.nanoTime()
      val graph = graph(prepared.window, backend)
      if (!graph.warmed) {
        runGraph(graph, prepared, embeds)
        graph.warmed = true
      }
      val (logits, graphMs) = runGraph(graph, prepared, embeds)
      val decodeStart = System.nanoTime()
      val answers = DecisionDecoder.decode(logits, questions)
      val finished = System.nanoTime()
      Result(
        state,
        answers,
        logits,
        backend,
        prepared.window,
        prepared.encoded.encodedLength,
        prepared.encoded.optionCount,
        Timing(ms(preparedAt - start), graphMs, ms(finished - decodeStart)),
      )
    }

  /** Raw logits for already-prepared inputs (the debug gate), with the graph interval in milliseconds. */
  fun run(prepared: DecisionInputs.Prepared, backend: Backend): Pair<FloatArray, Double> = ProcessRuntime.call {
    checkOpen()
    val graph = graph(prepared.window, backend)
    runGraph(graph, prepared)
  }

  private fun runGraph(graph: Graph, prepared: DecisionInputs.Prepared, embeds: FloatArray? = null): Pair<FloatArray, Double> {
    val rows = embeds ?: table.lookup(prepared.inputIds)
    val start = System.nanoTime()
    graph.inputs.getValue(INPUT_EMBEDS).writeFloat(rows)
    graph.inputs.getValue(INPUT_MASK).writeFloat(prepared.attentionMask)
    graph.inputs.getValue(INPUT_Q_ROUTING).writeFloat(prepared.qRouting)
    graph.inputs.getValue(INPUT_O_ROUTING).writeFloat(prepared.oRouting)
    graph.model.run(graph.inputs, graph.outputs, SIGNATURE)
    val logits = graph.outputs.getValue(OUTPUT_NAME).readFloat()
    val read = System.nanoTime()
    check(logits.size == DecisionInputs.OPTION_SLOTS) { "Unexpected logits size ${logits.size}" }
    return logits to ms(read - start)
  }

  private fun graph(window: Int, backend: Backend): Graph {
    val key = Key(window, backend)
    return graphs.getOrPut(key) {
      val options =
        CompiledModel.Options(backend.accelerator).apply {
          if (backend == Backend.GPU) {
            gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          } else {
            cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
          }
        }
      val path = requireFile(graphFile(window))
      val model = CompiledModel.create(path.absolutePath, options, ProcessRuntime.environment())
      val inputs = linkedMapOf<String, TensorBuffer>()
      val outputs = linkedMapOf<String, TensorBuffer>()
      try {
        for (name in INPUT_NAMES) inputs[name] = model.createInputBuffer(name, SIGNATURE)
        outputs[OUTPUT_NAME] = model.createOutputBuffer(OUTPUT_NAME, SIGNATURE)
        Graph(model, inputs, outputs)
      } catch (failure: Throwable) {
        (inputs.values + outputs.values).forEach { it.close() }
        model.close()
        throw failure
      }
    }
  }

  private fun requireFile(name: String): File =
    File(filesDir, name).also { check(it.isFile) { "Missing $name. Run scripts/install_to_device.sh, then retry." } }

  private fun checkOpen() = check(!closed) { "Model is closed" }

  override fun close() = ProcessRuntime.call {
    if (!closed) {
      closed = true
      try {
        graphs.values.forEach { it.close() }
      } finally {
        graphs.clear()
        table.close()
      }
    }
  }

  companion object {
    const val LITERT_VERSION = "2.2.0"
    const val STARTUP_WARMUP_ITERATIONS = 3
    const val CPU_THREADS = 4
    const val TABLE_FILE = "word_embeddings_fp16.bin"
    const val TOKENIZER_FILE = "tokenizer.json"
    private const val SIGNATURE = "serving_default"
    private const val INPUT_EMBEDS = "inputs_embeds"
    private const val INPUT_MASK = "attention_mask"
    private const val INPUT_Q_ROUTING = "q_routing"
    private const val INPUT_O_ROUTING = "o_routing"
    private const val OUTPUT_NAME = "logits"
    private val INPUT_NAMES = listOf(INPUT_EMBEDS, INPUT_MASK, INPUT_Q_ROUTING, INPUT_O_ROUTING)

    fun graphFile(window: Int) = "deberta_v3_large_decision_s${window}_wfp16.tflite"

    val REQUIRED_FILES: List<String> = DecisionInputs.WINDOWS.map { graphFile(it) } + listOf(TABLE_FILE, TOKENIZER_FILE)

    private fun ms(nanoseconds: Long) = nanoseconds / 1_000_000.0
  }
}

/** Exactly one Environment and one native thread for the process lifetime. */
private object ProcessRuntime {
  private val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "Decision-LiteRT").apply { isDaemon = true }
  }
  private var sharedEnvironment: Environment? = null

  fun environment(): Environment = sharedEnvironment ?: Environment.create().also { sharedEnvironment = it }

  fun <T> call(block: () -> T): T {
    try {
      return executor.submit(Callable { block() }).get()
    } catch (failure: ExecutionException) {
      throw failure.cause ?: failure
    }
  }
}
