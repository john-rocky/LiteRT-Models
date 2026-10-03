package com.kev

import android.content.Context
import java.io.Closeable
import java.io.File

/** Loading stages shown while the engine starts. */
enum class LoadStage {
  TOKENIZER,
  HEAD,
  GRAPH,
}

/**
 * The tokenizer, the pointer head and the compiled graphs ([KevResidentGraphs]): the smallest
 * installed window compiled at load, then the windows each request asks for, at most two and only
 * L128 / L256 side by side. Use only on [KevRuntime.dispatcher].
 */
class KevEngine
private constructor(
  private val context: Context,
  val pipeline: KevPipeline,
  /** Wall time of loading tokenizer.json, in milliseconds. */
  val tokenizerMs: Double,
  /** Wall time of loading the head weights, in milliseconds. */
  val headMs: Double,
  private val graphs: KevResidentGraphs<KevDecider>,
  backend: KevDecider.Backend,
  private val cpuFallback: Boolean,
  /** `ActivityManager.MemoryInfo.availMem` right before the load compiled its graph, in bytes. */
  val loadAvailableBytes: Long,
) : Closeable {
  /** The backend graphs are compiled on (GPU may still fall back to CPU, see [gpuFailure]). */
  var requestedBackend: KevDecider.Backend = backend
    private set

  /** Tokenizer + head + primary graph compile of the load: the `ENGINE_READY` figure. */
  val loadMs: Double = tokenizerMs + headMs + graphs.graph(graphs.primary).compileMs

  /** Compile time of the resident primary graph (0 when it is not resident), in milliseconds. */
  val primaryCompileMs: Double
    get() = graphs.all.firstOrNull { it.length == graphs.primary }?.compileMs ?: 0.0

  /** The window compiled at load: the smallest installed one, or the one a diagnostic run names. */
  val primaryWindow: Int
    get() = graphs.primary

  /** The primary graph; gate and timing runs use only this one. */
  val primary: KevDecider
    get() = graphs.graph(graphs.primary)

  /** The resident windows, in ascending order. */
  val windows: List<Int>
    get() = graphs.windows

  /** The backend the primary graph runs on. */
  val backend: KevDecider.Backend
    get() = graphs.all.firstOrNull()?.backend ?: requestedBackend

  /** GPU's error when GPU was requested and a resident graph runs on CPU instead. */
  val gpuFailure: String?
    get() = graphs.all.firstNotNullOfOrNull { it.gpuFailure }

  /** The graphs a request with rows of [rows] tokens needs (see [KevResidentGraphs.plan]). */
  fun plan(rows: List<Int>): KevWindowPlan =
    graphs.plan(rows, KevFiles.installedWindows(context.filesDir))

  /** Every row on the one [window] (see [KevResidentGraphs.planFixed]). */
  fun planFixed(rows: List<Int>, window: Int): KevWindowPlan =
    graphs.planFixed(rows, window, KevFiles.installedWindows(context.filesDir))

  /**
   * Closes and compiles what [plan] needs (see [KevResidentGraphs.prepare]), calling [onCompile]
   * before each compile, and returns the graph of each question.
   */
  fun prepare(plan: KevWindowPlan.Ready, onCompile: (Int) -> Unit): KevWindowRun<KevDecider> =
    graphs.prepare(plan, { KevDevice.availableMemoryBytes(context) }) { window ->
      onCompile(window)
      open(window)
    }

  /**
   * Closes the compiled graphs and compiles the primary window on [backend] (another window
   * compiles again when a request needs it). GPU falls back to CPU when this engine allows it.
   */
  fun switchBackend(backend: KevDecider.Backend) {
    requestedBackend = backend
    graphs.reopen(::open)
  }

  override fun close() = graphs.close()

  private fun open(window: Int): KevDecider =
    KevDecider.create(context, window, requestedBackend, cpuFallback)

  companion object {
    /**
     * Loads tokenizer, head and the [window] graph from `files/`, reporting each stage as it
     * starts. With [cpuFallback], a graph that does not compile on the GPU runs on CPU.
     */
    fun load(
      context: Context,
      window: Int,
      backend: KevDecider.Backend,
      cpuFallback: Boolean,
      onStage: (LoadStage) -> Unit,
    ): KevEngine {
      val files = context.filesDir
      onStage(LoadStage.TOKENIZER)
      val tokenizerStart = System.nanoTime()
      val tokenizer = KevTokenizer(File(files, KevFiles.TOKENIZER))
      val tokenizerMs = KevPipeline.millis(System.nanoTime() - tokenizerStart)
      onStage(LoadStage.HEAD)
      val headStart = System.nanoTime()
      val head = KevPointerHead(File(files, KevFiles.HEAD))
      val headMs = KevPipeline.millis(System.nanoTime() - headStart)
      onStage(LoadStage.GRAPH)
      val available = KevDevice.availableMemoryBytes(context)
      val primary = KevDecider.create(context, window, backend, cpuFallback)
      return KevEngine(
        context.applicationContext,
        KevPipeline(tokenizer, head),
        tokenizerMs,
        headMs,
        KevResidentGraphs(window, primary),
        backend,
        cpuFallback,
        available,
      )
    }
  }
}
