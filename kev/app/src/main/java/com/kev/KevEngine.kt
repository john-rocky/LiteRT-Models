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
 * What a plan was made with: the available memory in bytes, the resident graphs and whether a pair
 * compiles with constant tensor sharing.
 */
class KevPlanInputs(
  val availableBytes: Long,
  val resident: List<KevGraphKey>,
  val pairShare: KevPairShare = KevPairShare.AUTO,
)

/**
 * The graphs of one request after [KevEngine.prepare]: each question's row graph or the pair, and
 * what the engine compiled and closed for the request.
 */
class KevRequestGraphs(
  val plan: KevPlan.Ready,
  val runners: KevRunners,
  /** Each question's window: L of its row graph, or the pair's Lq. */
  val windows: List<Int>,
  /** The graphs the questions run on: the distinct row windows ascending, or the pair. */
  val used: List<KevGraphKey>,
  /** The GPU precision of each graph of [used], in that order. */
  val precisions: List<KevPrecision>,
  /** The graphs compiled for this request, in order. */
  val compiled: List<KevGraphKey>,
  /** The available memory read right before each compile, in bytes, in the order of [compiled]. */
  val availableBytes: List<Long>,
  /** The graphs closed for this request. */
  val closed: List<KevGraphKey>,
  /** The row plan wanted a second window but the available memory was below the limit. */
  val secondRefused: Boolean,
  /** For a pair plan, whether the pair holds one copy of the weights (constant tensor sharing). */
  val pairShare: Boolean? = null,
) {
  /** The pair, for a pair plan. */
  val pair: PairRunner?
    get() = (runners as? KevRunners.Pair)?.graph

  /** The GPU precision of the graph question [index] runs on. */
  fun precisionOf(index: Int): KevPrecision =
    when (runners) {
      is KevRunners.Pair -> precisions.single()
      is KevRunners.Rows -> precisions[used.indexOf(KevGraphKey.Window(windows[index]))]
    }
}

/**
 * The tokenizer, the pointer head and the compiled graphs ([KevResidentGraphs]). Loading reads only
 * the tokenizer and the head; each request's plan ([KevPlanner]) then compiles what it needs: row
 * windows (at most two, and only windows up to L256 side by side) or the shared-state pair alone,
 * each graph at [forcedPrecision] or else at its own default ([KevPrecision.defaultFor]). Use only
 * on [KevRuntime.dispatcher].
 */
class KevEngine
private constructor(
  private val context: Context,
  val pipeline: KevPipeline,
  /** Wall time of loading tokenizer.json, in milliseconds. */
  val tokenizerMs: Double,
  /** Wall time of loading the head weights, in milliseconds. */
  val headMs: Double,
  backend: KevDecider.Backend,
  /**
   * The GPU precision of every graph of this engine (the launch's), or null for each graph's own.
   */
  val forcedPrecision: KevPrecision?,
  /** Whether a pair compiles with constant tensor sharing (one copy of the weights on the GPU). */
  val pairShare: KevPairShare,
  private val cpuFallback: Boolean,
) : Closeable {
  private val graphs = KevResidentGraphs<KevDecider, KevPairDecider>()

  /** The backend graphs are compiled on (GPU may still fall back to CPU, see [gpuFailure]). */
  var requestedBackend: KevDecider.Backend = backend
    private set

  /**
   * Tokenizer + head + the compiles of the request prepared at startup: the `ENGINE_READY` figure,
   * null until [markLoaded].
   */
  var loadMs: Double? = null
    private set

  /** The resident graphs: windows in ascending order, then the pair. */
  val resident: List<KevGraphKey>
    get() = graphs.resident

  /** The resident windows, in ascending order. */
  val windows: List<Int>
    get() = graphs.windows

  /** The GPU precision of each resident graph, in the order of [resident]. */
  val residentPrecisions: List<KevPrecision>
    get() = graphs.all.map { it.precision } + listOfNotNull(graphs.pairRunner?.precision)

  /**
   * The GPU precision [graph] compiles with: [forcedPrecision], or the graph's default for its
   * installed file ([KevPrecision.defaultFor]).
   */
  fun precisionOf(graph: KevGraphKey): KevPrecision =
    forcedPrecision ?: KevPrecision.defaultFor(graph, File(context.filesDir, graph.file).length())

  /** Compile time of the resident graphs together, in milliseconds. */
  val compileMs: Double
    get() = graphs.all.sumOf { it.compileMs } + (graphs.pairRunner?.compileMs ?: 0.0)

  /** The backend the resident graphs run on. */
  val backend: KevDecider.Backend
    get() = graphs.all.firstOrNull()?.backend ?: graphs.pairRunner?.backend ?: requestedBackend

  /** GPU's error when GPU was requested and a resident graph runs on CPU instead. */
  val gpuFailure: String?
    get() = graphs.all.firstNotNullOfOrNull { it.gpuFailure } ?: graphs.pairRunner?.gpuFailure

  /** The available memory (bytes) and the resident graphs the last [plan] saw. */
  var lastPlanInputs: KevPlanInputs? = null
    private set

  /**
   * How [prepared] runs ([KevPlanner]) with the installed files: [mode], and every row on
   * [fixedWindow] when it is set.
   */
  fun plan(
    prepared: KevPrepared,
    mode: KevGraphMode = KevGraphMode.AUTO,
    fixedWindow: Int? = null,
  ): KevPlan {
    val available = KevDevice.availableMemoryBytes(context)
    lastPlanInputs = KevPlanInputs(available, graphs.resident, pairShare)
    return KevPlanner.plan(
      prepared.encoded.stateIds.size,
      prepared.encoded.branches.map { it.ids.size },
      KevFiles.installedWindows(context.filesDir),
      KevFiles.installedPairs(context.filesDir),
      graphs.windows,
      available,
      mode,
      fixedWindow,
      share = pairShare,
      residentPair = graphs.pair,
      residentPairShared = graphs.pairRunner?.shareConstants,
    )
  }

  /**
   * Closes and compiles what [plan] needs (see [KevResidentGraphs]), calling [onCompile] before
   * each compile, and returns the graphs of the request.
   */
  fun prepare(plan: KevPlan.Ready, onCompile: (KevGraphKey) -> Unit): KevRequestGraphs =
    when (plan) {
      is KevPlan.Rows -> {
        val run =
          graphs.prepare(plan.windows, { KevDevice.availableMemoryBytes(context) }) { window ->
            val key = KevGraphKey.Window(window)
            onCompile(key)
            KevDecider.create(context, window, requestedBackend, precisionOf(key), cpuFallback)
          }
        val used = run.windows.distinct().sorted()
        KevRequestGraphs(
          plan,
          KevRunners.Rows(run.graphs),
          run.windows,
          used.map { KevGraphKey.Window(it) },
          used.map { graphs.graph(it).precision },
          run.compiled.map { KevGraphKey.Window(it) },
          run.availableBytes,
          listOfNotNull(run.closedPair?.let { KevGraphKey.Pair(it) }) +
            run.closed.map { KevGraphKey.Window(it) },
          run.secondRefused,
        )
      }
      is KevPlan.Pair -> {
        val run =
          graphs.preparePair(plan.shape, { KevDevice.availableMemoryBytes(context) }) {
            shape,
            available ->
            val key = KevGraphKey.Pair(shape)
            onCompile(key)
            KevPairDecider.create(
              context,
              shape,
              requestedBackend,
              precisionOf(key),
              pairShare.sharesAt(available),
              cpuFallback,
            )
          }
        val key = KevGraphKey.Pair(run.shape)
        KevRequestGraphs(
          plan,
          KevRunners.Pair(run.graph),
          List(plan.questions) { run.shape.questionLength },
          listOf(key),
          listOf(run.graph.precision),
          if (run.compiled) listOf(key) else emptyList(),
          listOfNotNull(run.availableBytes),
          run.closedWindows.map { KevGraphKey.Window(it) } +
            listOfNotNull(run.closedPair?.let { KevGraphKey.Pair(it) }),
          false,
          run.graph.shareConstants,
        )
      }
    }

  /** The one row [window] resident (a gate or timing run on that window). */
  fun prepareWindow(window: Int): KevRequestGraphs {
    val plan = KevWindowPlan.Ready(listOf(window))
    return prepare(KevPlan.Rows(plan, KevPrediction(null, null))) {}
  }

  /** The pair [shape] resident alone (a gate or timing run on the pair). */
  fun preparePair(shape: KevPairShape): KevRequestGraphs =
    prepare(KevPlan.Pair(shape, 0, KevPrediction(null, null))) {}

  /** The resident row graph of [window]. */
  fun window(window: Int): KevDecider = graphs.graph(window)

  /** The resident pair, or null. */
  val pair: KevPairDecider?
    get() = graphs.pairRunner

  /** Records [loadMs] once: tokenizer + head + the resident graphs' compile times. */
  fun markLoaded() {
    if (loadMs == null) loadMs = tokenizerMs + headMs + compileMs
  }

  /** Closes the compiled graphs; the next [prepare] compiles on [backend]. */
  fun switchBackend(backend: KevDecider.Backend) {
    requestedBackend = backend
    graphs.close()
  }

  override fun close() = graphs.close()

  companion object {
    /**
     * Loads tokenizer and head from `files/`, reporting each stage as it starts; graphs compile
     * when a request needs them, at [forcedPrecision] or at their own default. With [cpuFallback],
     * a graph that does not compile on the GPU runs on CPU.
     */
    fun load(
      context: Context,
      backend: KevDecider.Backend,
      forcedPrecision: KevPrecision?,
      pairShare: KevPairShare,
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
      return KevEngine(
        context.applicationContext,
        KevPipeline(tokenizer, head),
        tokenizerMs,
        headMs,
        backend,
        forcedPrecision,
        pairShare,
        cpuFallback,
      )
    }
  }
}
