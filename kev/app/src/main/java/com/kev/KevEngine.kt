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
 * compiles with constant tensor sharing; also the kernel's `MemAvailable` at that moment (kB), for
 * the record only.
 */
class KevPlanInputs(
  val availableBytes: Long,
  val resident: List<KevGraphKey>,
  val pairShare: KevPairShare = KevPairShare.AUTO,
  val procAvailableKb: Long? = null,
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
  /** The GPU precision of each graph of [used], in that order (meaningful on the GPU only). */
  val precisions: List<KevPrecision>,
  /** Where each graph of [used] ran ([KevDecider.ranOn]), in that order. */
  val backends: List<KevDecider.Backend>,
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
  /** Every compile for this request, in order (a window compiled twice appears twice). */
  val compiles: List<KevCompiled> = emptyList(),
) {
  /** The pair, for a pair plan. */
  val pair: PairRunner?
    get() = (runners as? KevRunners.Pair)?.graph

  /** The GPU precision of the graph question [index] runs on. */
  fun precisionOf(index: Int): KevPrecision = precisions[usedIndex(index)]

  /** Where the graph question [index] runs on ran. */
  fun backendOf(index: Int): KevDecider.Backend = backends[usedIndex(index)]

  private fun usedIndex(index: Int): Int =
    when (runners) {
      is KevRunners.Pair -> 0
      is KevRunners.Rows -> used.indexOf(KevGraphKey.Window(windows[index]))
    }
}

/**
 * How a graph is about to compile, for the status line: on [backend]; for the NPU, whether the app
 * expects a first compile (minutes) rather than a load from LiteRT's JIT cache.
 */
class KevCompileStep(val graph: KevGraphKey, val backend: KevDecider.Backend, val npuFirst: Boolean)

/**
 * One compile of a request: [graph] asked for on [backend], compiled on [compiledOn] (another one
 * after a fallback, with its [failure]) and running on [ranOn], in [compileMs]; [npu] for an NPU
 * compile.
 */
class KevCompiled(
  val graph: KevGraphKey,
  val backend: KevDecider.Backend,
  val compiledOn: KevDecider.Backend,
  val ranOn: KevDecider.Backend,
  val compileMs: Double,
  val npu: KevNpuCompile?,
  val failure: String?,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "graph" to graph.label,
      "backend" to backend.wireName,
      "compiled_on" to compiledOn.wireName,
      "ran_on" to ranOn.wireName,
      "compile_ms" to compileMs,
      "failure" to failure,
      "npu" to npu?.toJson(),
    )
}

/**
 * The tokenizer, the pointer head and the compiled graphs ([KevResidentGraphs]). Loading reads only
 * the tokenizer and the head; each request's plan ([KevPlanner]) then compiles what it needs: row
 * windows (at most two, and only windows up to L256 side by side) or the shared-state pair alone,
 * each graph at [forcedPrecision] or else at its own default ([KevPrecision.defaultFor]). With the
 * NPU chosen, the windows of [KevFiles.NPU_WINDOWS] compile on the NPU and the other graphs on the
 * GPU ([KevDecider.Backend.forGraph]); a first NPU compile of a file runs alone. Use only on
 * [KevRuntime.dispatcher].
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
  /** The Qualcomm options of every graph, or null when the APK has no NPU libraries. */
  val npuOptions: KevNpuOptions?,
  /** A debug run: the pair passes its state through the host ([KevPairDecider.stateCopy]). */
  private val pairStateCopy: Boolean = false,
) : Closeable {
  private val graphs = KevResidentGraphs<KevDecider, KevPairDecider>()

  /** The backend chosen for this engine's graphs ([KevDecider.Backend.forGraph] maps each one). */
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

  /** Where each resident graph runs, in the order of [resident]. */
  val residentBackends: List<KevDecider.Backend>
    get() = graphs.all.map { it.ranOn } + listOfNotNull(graphs.pairRunner?.ranOn)

  /** The NPU compile of each resident graph that compiled on the NPU, by graph. */
  val residentNpu: Map<KevGraphKey, KevNpuCompile>
    get() =
      graphs.all
        .mapNotNull { graph -> graph.npu?.let { KevGraphKey.Window(graph.length) to it } }
        .toMap() +
        listOfNotNull(
          graphs.pairRunner?.let { pair -> pair.npu?.let { KevGraphKey.Pair(pair.shape) to it } }
        )

  /**
   * The GPU precision [graph] compiles with: [forcedPrecision], or the graph's default for its
   * installed file ([KevPrecision.defaultFor]).
   */
  fun precisionOf(graph: KevGraphKey): KevPrecision =
    forcedPrecision ?: KevPrecision.defaultFor(graph, File(context.filesDir, graph.file).length())

  /** Compile time of each resident graph, in the order of [resident], in milliseconds. */
  val residentCompileMs: List<Double>
    get() = graphs.all.map { it.compileMs } + listOfNotNull(graphs.pairRunner?.compileMs)

  /** Compile time of the resident graphs together, in milliseconds. */
  val compileMs: Double
    get() = residentCompileMs.sum()

  /** GPU's error when GPU was requested and a resident graph runs on CPU instead. */
  val gpuFailure: String?
    get() = graphs.all.firstNotNullOfOrNull { it.gpuFailure } ?: graphs.pairRunner?.gpuFailure

  /** NPU's error when NPU was requested and a resident graph runs on GPU instead. */
  val npuFailure: String?
    get() = graphs.all.firstNotNullOfOrNull { it.npuFailure }

  /** The available memory (bytes) and the resident graphs the last [plan] saw. */
  var lastPlanInputs: KevPlanInputs? = null
    private set

  /**
   * How [prepared] runs ([KevPlanner]) with the installed files: [mode], and every row on
   * [fixedWindow] when it is set; the times come from the table of the chosen backend
   * ([KevCosts.forBackend]).
   */
  fun plan(
    prepared: KevPrepared,
    mode: KevGraphMode = KevGraphMode.AUTO,
    fixedWindow: Int? = null,
  ): KevPlan {
    val available = KevDevice.availableMemoryBytes(context)
    lastPlanInputs =
      KevPlanInputs(available, graphs.resident, pairShare, KevDevice.procMemAvailableKb())
    return KevPlanner.plan(
      prepared.encoded.stateIds.size,
      prepared.encoded.branches.map { it.ids.size },
      KevFiles.installedWindows(context.filesDir),
      KevFiles.installedPairs(context.filesDir),
      graphs.windows,
      available,
      mode,
      fixedWindow,
      costs = KevCosts.forBackend(requestedBackend),
      share = pairShare,
      residentPair = graphs.pair,
      residentPairShared = graphs.pairRunner?.shareConstants,
    )
  }

  /** The backend [graph] compiles on: [exact] = the requested one as it is (a debug run). */
  private fun backendOf(graph: KevGraphKey, exact: Boolean): KevDecider.Backend =
    if (exact) requestedBackend else KevDecider.Backend.forGraph(requestedBackend, graph)

  /** Whether compiling [graph] on the NPU now is a first compile (no usable JIT cache). */
  private fun npuFirst(graph: KevGraphKey, backend: KevDecider.Backend): Boolean =
    backend == KevDecider.Backend.NPU &&
      KevNpuCompiler.state(context, File(context.filesDir, graph.file), npuOptions).firstCompile

  /**
   * Closes and compiles what [plan] needs (see [KevResidentGraphs]), calling [onCompile] before
   * each compile, and returns the graphs of the request. [exact]: every graph on the requested
   * backend as it is (a debug run that names its graph).
   */
  fun prepare(
    plan: KevPlan.Ready,
    exact: Boolean = false,
    onCompile: (KevCompileStep) -> Unit,
  ): KevRequestGraphs =
    when (plan) {
      is KevPlan.Rows -> {
        val compiles = ArrayList<KevCompiled>()
        val run =
          graphs.prepare(
            plan.windows,
            { KevDevice.availableMemoryBytes(context) },
            open = { window ->
              val key = KevGraphKey.Window(window)
              val backend = backendOf(key, exact)
              onCompile(KevCompileStep(key, backend, npuFirst(key, backend)))
              KevDecider.create(context, window, backend, precisionOf(key), cpuFallback, npuOptions)
                .also {
                  compiles.add(
                    KevCompiled(
                      key,
                      backend,
                      it.backend,
                      it.ranOn,
                      it.compileMs,
                      it.npu,
                      it.npuFailure ?: it.gpuFailure,
                    )
                  )
                }
            },
            alone = { window ->
              val key = KevGraphKey.Window(window)
              npuFirst(key, backendOf(key, exact))
            },
          )
        val used = run.windows.distinct().sorted()
        KevRequestGraphs(
          plan,
          KevRunners.Rows(run.graphs),
          run.windows,
          used.map { KevGraphKey.Window(it) },
          used.map { graphs.graph(it).precision },
          used.map { graphs.graph(it).ranOn },
          run.compiled.map { KevGraphKey.Window(it) },
          run.availableBytes,
          listOfNotNull(run.closedPair?.let { KevGraphKey.Pair(it) }) +
            run.closed.map { KevGraphKey.Window(it) },
          run.secondRefused,
          compiles = compiles,
        )
      }
      is KevPlan.Pair -> {
        val compiles = ArrayList<KevCompiled>()
        val run =
          graphs.preparePair(plan.shape, { KevDevice.availableMemoryBytes(context) }) {
            shape,
            available ->
            val key = KevGraphKey.Pair(shape)
            val backend = backendOf(key, exact)
            onCompile(KevCompileStep(key, backend, npuFirst(key, backend)))
            KevPairDecider.create(
                context,
                shape,
                backend,
                precisionOf(key),
                pairShare.sharesAt(available),
                cpuFallback,
                npuOptions,
                pairStateCopy,
              )
              .also {
                compiles.add(
                  KevCompiled(
                    key,
                    backend,
                    it.backend,
                    it.ranOn,
                    it.compileMs,
                    it.npu,
                    it.gpuFailure,
                  )
                )
              }
          }
        val key = KevGraphKey.Pair(run.shape)
        KevRequestGraphs(
          plan,
          KevRunners.Pair(run.graph),
          List(plan.questions) { run.shape.questionLength },
          listOf(key),
          listOf(run.graph.precision),
          listOf(run.graph.ranOn),
          if (run.compiled) listOf(key) else emptyList(),
          listOfNotNull(run.availableBytes),
          run.closedWindows.map { KevGraphKey.Window(it) } +
            listOfNotNull(run.closedPair?.let { KevGraphKey.Pair(it) }),
          false,
          run.graph.shareConstants,
          compiles,
        )
      }
    }

  /** The one row [window] resident on the requested backend as it is (a gate or timing run). */
  fun prepareWindow(window: Int): KevRequestGraphs {
    val plan = KevWindowPlan.Ready(listOf(window))
    return prepare(KevPlan.Rows(plan, KevPrediction(null, null)), exact = true) {}
  }

  /** The pair [shape] resident alone on the requested backend as it is (a gate or timing run). */
  fun preparePair(shape: KevPairShape): KevRequestGraphs =
    prepare(KevPlan.Pair(shape, 0, KevPrediction(null, null)), exact = true) {}

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
     * a graph that does not compile on the NPU runs on GPU, one that does not on the GPU on CPU.
     */
    fun load(
      context: Context,
      backend: KevDecider.Backend,
      forcedPrecision: KevPrecision?,
      pairShare: KevPairShare,
      cpuFallback: Boolean,
      npuOptions: KevNpuOptions?,
      pairStateCopy: Boolean = false,
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
        npuOptions,
        pairStateCopy,
      )
    }
  }
}
