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
 * The tokenizer, the pointer head and one resident graph. A request whose longest row needs a
 * larger window replaces the resident graph ([switchGraph]); only one graph is in memory at a time.
 * Use only on [KevRuntime.dispatcher].
 */
class KevEngine
private constructor(
  val pipeline: KevPipeline,
  decider: KevDecider,
  /** Wall time of loading tokenizer.json, in milliseconds. */
  val tokenizerMs: Double,
  /** Wall time of loading the head weights, in milliseconds. */
  val headMs: Double,
  /** When GPU was requested but failed and CPU took over: GPU's error. */
  gpuFailure: String?,
) : Closeable {
  var decider: KevDecider = decider
    private set

  var gpuFailure: String? = gpuFailure
    private set

  /** Tokenizer + head + graph compile of the startup load: the `ENGINE_READY` figure. */
  val loadMs: Double = tokenizerMs + headMs + decider.compileMs

  val window: Int
    get() = decider.length

  val backend: KevDecider.Backend
    get() = decider.backend

  /**
   * Makes the [window] graph on [backend] the resident one: closes the current graph and compiles
   * the new one (see [KevDecider.create]) unless it is already resident. A GPU request that already
   * fell back to CPU in this window keeps the CPU graph. Returns whether a graph was compiled.
   */
  fun ensureGraph(
    context: Context,
    window: Int,
    backend: KevDecider.Backend,
    cpuFallback: Boolean,
  ): Boolean {
    val resident =
      !decider.isClosed &&
        decider.length == window &&
        (decider.backend == backend || (backend == KevDecider.Backend.GPU && gpuFailure != null))
    if (resident) return false
    decider.close()
    val created = KevDecider.create(context, window, backend, cpuFallback)
    decider = created.decider
    gpuFailure = created.gpuFailure
    return true
  }

  override fun close() = decider.close()

  companion object {
    /**
     * Loads tokenizer, head and the [window] graph from `files/`, reporting each stage as it
     * starts.
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
      val created = KevDecider.create(context, window, backend, cpuFallback)
      return KevEngine(
        KevPipeline(tokenizer, head),
        created.decider,
        tokenizerMs,
        headMs,
        created.gpuFailure,
      )
    }
  }
}
