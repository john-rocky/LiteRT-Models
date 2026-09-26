package com.gliformer

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext

/** Published GLiFormer graphs with one process Environment and one native owner thread. */
class GliformerExtractor(
  private val filesDir: File,
  val tableStorage: GliformerInputs.EmbeddingTable.Storage =
    GliformerInputs.EmbeddingTable.Storage.FP16,
) : Closeable {
  enum class Backend {
    GPU,
    CPU
  }

  data class Timings(
    val tokenizeLookupMs: Double,
    val graphMs: Double,
    val decodeMs: Double,
    val totalMs: Double,
    val encoderMs: Double,
    val headMs: Double,
  )

  data class Result(
    val prepared: GliformerInputs.Prepared,
    val logits: FloatArray,
    val entities: List<GliformerDecoder.Entity>,
    val timings: Timings,
    val backend: Backend,
    val window: Int,
  )

  data class LoadInfo(val window: Int, val backend: Backend, val loadMs: Double)

  data class WarmupReport(
    val window: Int,
    val backend: Backend,
    val passes: Int,
    val totalMs: Double
  )

  private class Graph(
    val model: CompiledModel,
    val inputs: Map<String, TensorBuffer>,
    val outputs: Map<String, TensorBuffer>,
  ) : Closeable {
    fun run(values: Map<String, FloatArray>): FloatArray {
      values.forEach { (name, data) -> inputs.getValue(name).writeFloat(data) }
      model.run(inputs, outputs, SIGNATURE)
      return outputs.getValue("output_0").readFloat()
    }

    override fun close() {
      try {
        (inputs.values + outputs.values).forEach { it.close() }
      } finally {
        model.close()
      }
    }
  }

  private data class Pipeline(
    val window: Int,
    val backend: Backend,
    val encoder: Graph,
    val head: Graph?
  ) : Closeable {
    var completedWarmupPasses: Int = 0

    override fun close() {
      try {
        head?.close()
      } finally {
        encoder.close()
      }
    }
  }

  private var inputBuilder: GliformerInputs? = null
  private var embeddingTable: GliformerInputs.EmbeddingTable? = null
  private var pipeline: Pipeline? = null
  @Volatile private var closed = false

  suspend fun initialize(backend: Backend = Backend.GPU, forcedWindow: Int = 128): LoadInfo =
    withContext(GliformerProcessRuntime.dispatcher) {
      val start = System.nanoTime()
      ensureHost()
      graph(forcedWindow, backend)
      LoadInfo(forcedWindow, backend, ms(System.nanoTime() - start))
    }

  suspend fun prepare(text: String, forcedWindow: Int? = null): GliformerInputs.Prepared =
    withContext(GliformerProcessRuntime.dispatcher) {
      ensureHost()
      inputBuilder!!.prepare(text, window = window(forcedWindow))
    }

  /** Twelve complete host → graph → host passes, including tokenization and table lookup. */
  suspend fun warmUp(
    backend: Backend = Backend.GPU,
    forcedWindow: Int = 128,
    passes: Int = STARTUP_WARMUP_ITERATIONS,
  ): WarmupReport =
    withContext(GliformerProcessRuntime.dispatcher) {
      require(passes > 0)
      ensureHost()
      val active = graph(forcedWindow, backend)
      val start = System.nanoTime()
      repeat(passes) { extractOnWorker(EXAMPLE_TEXT, backend, forcedWindow) }
      active.completedWarmupPasses = maxOf(active.completedWarmupPasses, passes)
      WarmupReport(forcedWindow, backend, passes, ms(System.nanoTime() - start))
    }

  suspend fun extract(
    text: String,
    backend: Backend = Backend.GPU,
    forcedWindow: Int? = null,
  ): Result =
    withContext(GliformerProcessRuntime.dispatcher) {
      extractOnWorker(text, backend, forcedWindow, ensureWarm = true)
    }

  /** Detects eviction by a second Activity's extractor instead of trusting UI window metadata. */
  suspend fun isWarm(backend: Backend, sequenceLength: Int): Boolean =
    withContext(GliformerProcessRuntime.dispatcher) {
      !closed &&
        GliformerProcessRuntime.isOwner(this@GliformerExtractor) &&
        pipeline?.let {
          it.window == sequenceLength &&
            it.backend == backend &&
            it.completedWarmupPasses >= STARTUP_WARMUP_ITERATIONS
        } == true
    }

  private fun extractOnWorker(
    text: String,
    backend: Backend,
    forcedWindow: Int?,
    ensureWarm: Boolean = false
  ): Result {
    ensureHost()
    val begin = System.nanoTime()
    val prepared = inputBuilder!!.prepare(text, window = window(forcedWindow))
    val embeddings = embeddingTable!!.lookup(prepared.inputIds)
    val preparedAt = System.nanoTime()
    val active = graph(prepared.window.sequenceLength, backend)
    if (ensureWarm && active.completedWarmupPasses < STARTUP_WARMUP_ITERATIONS) {
      // Another extractor may have claimed the worker between the UI's warm check and call.
      // Rewarm atomically on this owner thread, then time the actual extraction separately.
      repeat(STARTUP_WARMUP_ITERATIONS) { extractOnWorker(EXAMPLE_TEXT, backend, active.window) }
      active.completedWarmupPasses = STARTUP_WARMUP_ITERATIONS
      return extractOnWorker(text, backend, forcedWindow)
    }
    val graphStart = System.nanoTime()
    val logits: FloatArray
    val encoderMs: Double
    val headMs: Double
    if (active.head == null) {
      logits =
        active.encoder.run(
          linkedMapOf(
            "inputs_embeds" to embeddings,
            "attention_mask" to prepared.attentionMask,
            "text_routing" to prepared.textRouting,
            "parent_routing" to prepared.parentRouting,
            "label_routing" to prepared.labelRouting,
            "text_mask" to prepared.textMask,
          )
        )
      encoderMs = ms(System.nanoTime() - graphStart)
      headMs = 0.0
    } else {
      // Explicit host readback then head write; there is no implicit GPU/CPU buffer sharing.
      val hidden =
        active.encoder.run(
          linkedMapOf(
            "inputs_embeds" to embeddings,
            "attention_mask" to prepared.attentionMask,
          )
        )
      val encoderEnd = System.nanoTime()
      logits =
        active.head.run(
          linkedMapOf(
            "encoder_hidden" to hidden,
            "text_routing" to prepared.textRouting,
            "parent_routing" to prepared.parentRouting,
            "label_routing" to prepared.labelRouting,
            "text_mask" to prepared.textMask,
          )
        )
      encoderMs = ms(encoderEnd - graphStart)
      headMs = ms(System.nanoTime() - encoderEnd)
    }
    val readbackAt = System.nanoTime()
    check(logits.all { it.isFinite() }) { "The graph returned nonfinite logits" }
    val entities = GliformerDecoder.decode(logits, prepared)
    val done = System.nanoTime()
    return Result(
      prepared,
      logits,
      entities,
      Timings(
        ms(preparedAt - begin),
        ms(readbackAt - graphStart),
        ms(done - readbackAt),
        ms(done - begin),
        encoderMs,
        headMs,
      ),
      backend,
      active.window
    )
  }

  private fun ensureHost() {
    check(!closed) { "Extractor is closed" }
    if (inputBuilder == null) {
      val tokenizer = GliformerTokenizer(requireFile("tokenizer.json"))
      requireFile("tokenizer_config.json")
      requireFile("gliner_config.json")
      val table = requireFile("word_embeddings_${tableStorage.name.lowercase()}.bin")
      embeddingTable = GliformerInputs.EmbeddingTable(table, tableStorage)
      inputBuilder = GliformerInputs(tokenizer)
    }
  }

  private fun graph(sequenceLength: Int, backend: Backend): Pipeline {
    check(!closed) { "Extractor is closed" }
    require(sequenceLength in listOf(128, 256, 512)) { "Unsupported graph window" }
    require(sequenceLength == 128 || backend == Backend.GPU) {
      "CPU comparison supports s128 only. Choose GPU for the larger encoder/CPU-head pipeline."
    }
    pipeline?.let {
      if (
        GliformerProcessRuntime.isOwner(this) &&
          it.window == sequenceLength &&
          it.backend == backend
      )
        return it
    }
    val prefix = "gliformer_large_ner_s${sequenceLength}"
    val encoderFile =
      requireFile(
        if (sequenceLength == 128) "${prefix}_wfp16.tflite" else "${prefix}_encoder_wfp16.tflite"
      )
    val headFile = if (sequenceLength == 128) null else requireFile("${prefix}_head_wfp16.tflite")
    // Release the old window before any new native model is created.
    GliformerProcessRuntime.claim(this)
    releasePipelineOnWorker()
    Log.i("GLIFORMER_RUNTIME", "Loading s$sequenceLength $backend table=${tableStorage.name}")
    val encoder =
      openGraph(encoderFile, backend, if (headFile == null) FULL_INPUTS else ENCODER_INPUTS)
    try {
      val head = headFile?.let { openGraph(it, Backend.CPU, HEAD_INPUTS) }
      return Pipeline(sequenceLength, backend, encoder, head).also { pipeline = it }
    } catch (failure: Throwable) {
      encoder.close()
      throw failure
    }
  }

  private fun openGraph(file: File, backend: Backend, names: List<String>): Graph {
    val options =
      CompiledModel.Options(if (backend == Backend.GPU) Accelerator.GPU else Accelerator.CPU)
        .apply {
          if (backend == Backend.GPU) {
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          } else {
            cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
          }
        }
    val model =
      CompiledModel.create(file.absolutePath, options, GliformerProcessRuntime.environment())
    val inputs = linkedMapOf<String, TensorBuffer>()
    val outputs = linkedMapOf<String, TensorBuffer>()
    try {
      names.forEach { inputs[it] = model.createInputBuffer(it, SIGNATURE) }
      outputs["output_0"] = model.createOutputBuffer("output_0", SIGNATURE)
      return Graph(model, inputs, outputs)
    } catch (failure: Throwable) {
      (inputs.values + outputs.values).forEach { it.close() }
      model.close()
      throw failure
    }
  }

  private fun requireFile(name: String): File =
    File(filesDir, name).also {
      check(it.isFile) { "Missing $name. Run scripts/install_to_device.sh, then retry." }
    }

  override fun close() {
    GliformerProcessRuntime.executor.execute {
      if (!closed) {
        closed = true
        try {
          GliformerProcessRuntime.release(this)
        } finally {
          embeddingTable?.close()
          embeddingTable = null
          inputBuilder = null
        }
      }
    }
  }

  /** Called only by the singleton worker; clear first so stale owners cannot double-close. */
  internal fun releasePipelineOnWorker() {
    val previous = pipeline
    pipeline = null
    previous?.close()
  }

  companion object {
    const val LITERT_VERSION = "2.2.0"
    const val STARTUP_WARMUP_ITERATIONS = 12
    const val EXAMPLE_TEXT =
      "Mira Okafor, the founder of Halden Robotics, unveiled the Atlas Pro headset in Lisbon on 3 March 2025."
    private const val SIGNATURE = "serving_default"
    private val FULL_INPUTS =
      listOf(
        "inputs_embeds",
        "attention_mask",
        "text_routing",
        "parent_routing",
        "label_routing",
        "text_mask"
      )
    private val ENCODER_INPUTS = listOf("inputs_embeds", "attention_mask")
    private val HEAD_INPUTS =
      listOf("encoder_hidden", "text_routing", "parent_routing", "label_routing", "text_mask")

    private fun ms(ns: Long) = ns / 1_000_000.0

    private fun window(value: Int?): GliformerInputs.Window? = value?.let { requested ->
      GliformerInputs.WINDOWS.firstOrNull { it.sequenceLength == requested }
        ?: throw IllegalArgumentException("Unsupported graph window s$requested")
    }
  }
}

/** A serial coroutine dispatcher alone may migrate threads; this executor never does. */
private object GliformerProcessRuntime {
  val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "Gliformer-LiteRT").apply { isDaemon = true }
  }
  val dispatcher = executor.asCoroutineDispatcher()
  private var sharedEnvironment: Environment? = null
  private var pipelineOwner: GliformerExtractor? = null

  fun environment(): Environment =
    sharedEnvironment ?: Environment.create().also { sharedEnvironment = it }

  fun isOwner(owner: GliformerExtractor): Boolean = pipelineOwner === owner

  fun claim(owner: GliformerExtractor) {
    if (pipelineOwner === owner) return
    val previous = pipelineOwner
    pipelineOwner = null
    previous?.releasePipelineOnWorker()
    pipelineOwner = owner
  }

  fun release(owner: GliformerExtractor) {
    if (pipelineOwner === owner) pipelineOwner = null
    owner.releasePipelineOnWorker()
  }
}
