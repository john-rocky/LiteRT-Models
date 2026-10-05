package com.kev

import java.util.Locale

/** The extras of a launch intent, read by name (Android-free; `MainActivity` wraps the intent). */
interface KevExtras {
  fun has(name: String): Boolean

  fun string(name: String): String?

  fun int(name: String, default: Int): Int

  fun boolean(name: String, default: Boolean): Boolean
}

/**
 * What a launch can use besides its extras: the build ([debug]: the gate; [diagnostics]: the timing
 * runs), the installed graphs (the window and pair a gate or timing run takes when it names none)
 * and whether the APK carries the NPU libraries.
 */
class KevLaunchContext(
  val debug: Boolean,
  val diagnostics: Boolean,
  val installedWindows: List<Int>,
  val installedPairs: List<KevPairShape>,
  val npuAvailable: Boolean,
)

/** What a launch intent asks for (see `MainActivity` for the extras). */
sealed interface KevLaunch {
  /**
   * The editable sample: Decide plans each request in [graph] mode, on [backend] (null: the choice
   * the app kept from its last run) and on GPU at [precision] (null: each graph at its own default,
   * [KevPrecision.defaultFor]).
   */
  data class Normal(
    val graph: KevGraphMode = KevGraphMode.AUTO,
    val precision: KevPrecision? = null,
    /** Whether a pair compiles with constant tensor sharing (the `share` extra). */
    val share: KevPairShare = KevPairShare.AUTO,
    val backend: KevDecider.Backend? = null,
  ) : KevLaunch

  /**
   * Debug build: the fixture gate on [graph] (one row window or the pair) into `files/<report>`,
   * compiled on [backend] as it is (with the NPU, any graph: a debug run); [npu] holds the Qualcomm
   * options when the APK carries the NPU libraries, [stateCopy] makes the pair pass its state
   * through the host.
   */
  data class Gate(
    val backend: KevDecider.Backend,
    val precision: KevPrecision?,
    val report: String,
    val graph: KevGraphKey,
    val limit: Int,
    val share: KevPairShare = KevPairShare.AUTO,
    val npu: KevNpuOptions? = null,
    val stateCopy: Boolean = false,
  ) : KevLaunch

  /**
   * Debug and benchmark builds: the timing protocol on the rows of `files/<rows>`, limited to the
   * [sets] named (null: all of them), on the row [window] or on the [pair] when [graph] is
   * [KevGraphMode.PAIR], compiled on [backend] as it is; then, with [requestPath], the bundled
   * request from its text on the plan of [graph] (with [KevGraphMode.ROWS], every row on [window]
   * when the launch named it, else each row on its own window as a Decide would run it), each graph
   * of that plan on the backend the app gives it. With [coolMs] > 0, each set and the request path
   * first wait for the GPU to cool ([KevCooler]).
   */
  data class Timing(
    val rows: String,
    val backend: KevDecider.Backend,
    val precision: KevPrecision?,
    val report: String,
    val window: Int,
    val pair: KevPairShape,
    val graph: KevGraphMode,
    val clearCache: Boolean,
    val sets: List<String>?,
    val requestPath: Boolean,
    val coolMs: Long = 0,
    val share: KevPairShare = KevPairShare.AUTO,
    val npu: KevNpuOptions? = null,
    val stateCopy: Boolean = false,
    /** The launch named [window] (the `window` extra). */
    val windowNamed: Boolean = true,
  ) : KevLaunch

  /**
   * The demo recording: answer the request in [fixture] on the presentation layout, every question
   * on [window] when it is set, otherwise on the plan of [graph] (the plan a Decide would use for
   * [KevGraphMode.AUTO]). [precision] and [backend] (null: the kept choice) apply when this launch
   * starts the app.
   */
  data class Autoplay(
    val fixture: String,
    val delayMs: Long,
    val gapMs: Long,
    val window: Int?,
    val graph: KevGraphMode = KevGraphMode.AUTO,
    val precision: KevPrecision? = null,
    val share: KevPairShare = KevPairShare.AUTO,
    val backend: KevDecider.Backend? = null,
  ) : KevLaunch

  /** Extras that cannot be followed; [autoplay] says which log tag reports it. */
  data class Invalid(val reason: String, val autoplay: Boolean) : KevLaunch

  companion object {
    const val EXTRA_AUTOPLAY = "autoplay"
    const val EXTRA_FIXTURE = "fixture"
    const val EXTRA_DELAY_MS = "delay_ms"
    const val EXTRA_GAP_MS = "gap_ms"
    const val EXTRA_WINDOW = "window"
    const val EXTRA_GATE = "gate"
    const val EXTRA_TIMING = "timing"
    const val EXTRA_BACKEND = "backend"
    const val EXTRA_REPORT = "report"
    const val EXTRA_LIMIT = "limit"
    const val EXTRA_ROWS = "rows"
    const val EXTRA_CLEAR_CACHE = "clear_cache"
    const val EXTRA_SETS = "sets"
    const val EXTRA_REQUEST_PATH = "request_path"
    const val EXTRA_GRAPH = "graph"
    const val EXTRA_PRECISION = "precision"
    const val EXTRA_LS = "ls"
    const val EXTRA_COOL_MS = "cool_ms"
    const val EXTRA_SHARE = "share"
    const val EXTRA_NPU_PERF = "npu_perf"
    const val EXTRA_NPU_OPT = "npu_opt"
    const val EXTRA_PAIR_STATE = "pair_state"
    const val DEFAULT_DELAY_MS = 1500
    const val DEFAULT_GAP_MS = 800

    /** Why `backend npu` cannot be followed by an APK without the NPU libraries. */
    const val NO_NPU_LIBRARIES =
      "backend npu needs the Qualcomm NPU libraries in the APK (scripts/fetch_npu_libs.sh)"

    /** Report names stay inside `files/`: letters, digits, dot, underscore and hyphen. */
    private val REPORT_NAME = Regex("[A-Za-z0-9._-]+")

    /** The `backend` extra: `gpu` (also without the extra), `npu` or `cpu`; null for others. */
    fun backend(name: String?): KevDecider.Backend? = KevDecider.Backend.of(name ?: "gpu")

    /**
     * The `precision` extra, `fp32` or `fp16acc`, for every graph of the process; null for anything
     * else. Without the extra each graph runs at its own default ([KevPrecision.defaultFor]).
     */
    fun precision(name: String): KevPrecision? =
      KevPrecision.entries.firstOrNull { it.wireName == name.trim().lowercase(Locale.ROOT) }

    /** The `share` extra: `auto`, `on` or `off` (constant tensor sharing of a pair); else null. */
    fun share(name: String): KevPairShare? = KevPairShare.of(name)

    fun reportNameValid(name: String): Boolean =
      name.matches(REPORT_NAME) && !name.endsWith(".partial")

    fun windowValid(window: Int): Boolean = window in KevFiles.WINDOWS

    /** Why a `window` extra cannot be followed. */
    fun windowInvalid(window: Int): String =
      "window $window is not one of ${KevFiles.WINDOWS.joinToString(", ")}"

    /** Why a `graph` extra cannot be followed. */
    fun graphInvalid(name: String): String =
      "graph $name is not one of ${KevGraphMode.entries.joinToString(", ") { it.wireName }}"

    /** Why a `precision` extra cannot be followed. */
    fun precisionInvalid(name: String): String =
      "precision $name is not one of ${KevPrecision.entries.joinToString(", ") { it.wireName }}"

    /** A report file name part for [graph]: "L256" or "S128Q64". */
    fun reportLabel(graph: KevGraphKey): String = graph.label.replace("+", "")

    /**
     * The `sets` extra: comma-separated set names, or `none` for no set (the request path only).
     */
    fun setNames(extra: String): List<String> =
      if (extra.trim() == "none") emptyList()
      else extra.split(',').map { it.trim() }.filter { it.isNotEmpty() }

    /** The launch [extras] ask for, given what the APK and `files/` hold ([context]). */
    fun parse(extras: KevExtras, context: KevLaunchContext): KevLaunch {
      val named = if (extras.has(EXTRA_WINDOW)) extras.int(EXTRA_WINDOW, 0) else null
      // Gate and timing runs compile one window: the named one, else the smallest installed one.
      val window =
        named ?: context.installedWindows.firstOrNull() ?: KevFiles.DEFAULT_INSTALL.first()
      val autoplay = extras.boolean(EXTRA_AUTOPLAY, false)
      val graphName = extras.string(EXTRA_GRAPH)
      val graph =
        KevGraphMode.of(graphName) ?: return Invalid(graphInvalid(graphName.orEmpty()), autoplay)
      val precisionName = extras.string(EXTRA_PRECISION)
      val precision = precisionName?.let {
        precision(it) ?: return Invalid(precisionInvalid(it), autoplay)
      }
      val shareName = extras.string(EXTRA_SHARE)
      val share =
        shareName?.let { share(it) ?: return Invalid("share $it is not auto, on or off", autoplay) }
          ?: KevPairShare.AUTO
      val backendName = extras.string(EXTRA_BACKEND)
      val backend = backendName?.let {
        backend(it) ?: return Invalid("backend $it is not gpu, npu or cpu", autoplay)
      }
      if (backend == KevDecider.Backend.NPU && !context.npuAvailable) {
        return Invalid(NO_NPU_LIBRARIES, autoplay)
      }
      // The pair of a gate or timing run: the one of the named state length, else an installed
      // pair of PAIRS (in order).
      val namedLs = if (extras.has(EXTRA_LS)) extras.int(EXTRA_LS, 0) else null
      val shapes =
        if (namedLs == null) KevFiles.PAIRS else KevFiles.PAIRS.filter { it.stateLength == namedLs }
      if (shapes.isEmpty()) {
        val known = KevFiles.PAIRS.joinToString(", ") { it.stateLength.toString() }
        return Invalid("ls $namedLs is not one of $known", autoplay)
      }
      val pair = shapes.firstOrNull { it in context.installedPairs } ?: shapes.first()
      return when {
        autoplay -> {
          val fixture = extras.string(EXTRA_FIXTURE)
          when {
            fixture.isNullOrEmpty() -> Invalid("no fixture extra", autoplay = true)
            named != null && !windowValid(named) -> Invalid(windowInvalid(named), autoplay = true)
            named != null && graph == KevGraphMode.PAIR ->
              Invalid("window $named and graph pair exclude each other", autoplay = true)
            else ->
              Autoplay(
                fixture,
                extras.int(EXTRA_DELAY_MS, DEFAULT_DELAY_MS).toLong(),
                extras.int(EXTRA_GAP_MS, DEFAULT_GAP_MS).toLong(),
                named,
                graph,
                precision,
                share,
                backend,
              )
          }
        }
        context.debug && extras.boolean(EXTRA_GATE, false) -> {
          val gateBackend = backend ?: KevDecider.Backend.GPU
          val gateGraph =
            if (graph == KevGraphMode.PAIR) KevGraphKey.Pair(pair) else KevGraphKey.Window(window)
          val report =
            extras.string(EXTRA_REPORT)
              ?: "app_gate_${gateBackend.wireName}_${reportLabel(gateGraph)}.json"
          val npu = debugNpu(extras, context) ?: return Invalid(npuInvalid(extras), false)
          when {
            !windowValid(window) -> Invalid(windowInvalid(window), autoplay = false)
            !reportNameValid(report) -> Invalid("invalid report name $report", autoplay = false)
            else ->
              Gate(
                gateBackend,
                precision,
                report,
                gateGraph,
                extras.int(EXTRA_LIMIT, 0),
                share,
                npu.options,
                npu.stateCopy,
              )
          }
        }
        context.diagnostics && extras.boolean(EXTRA_TIMING, false) -> {
          val timingBackend = backend ?: KevDecider.Backend.GPU
          val rows = extras.string(EXTRA_ROWS)
          val setGraph =
            if (graph == KevGraphMode.PAIR) KevGraphKey.Pair(pair) else KevGraphKey.Window(window)
          val report =
            extras.string(EXTRA_REPORT)
              ?: "app_timing_${timingBackend.wireName}_${reportLabel(setGraph)}.json"
          val npu = debugNpu(extras, context) ?: return Invalid(npuInvalid(extras), false)
          when {
            rows.isNullOrEmpty() -> Invalid("no rows extra", autoplay = false)
            !windowValid(window) -> Invalid(windowInvalid(window), autoplay = false)
            !reportNameValid(report) -> Invalid("invalid report name $report", autoplay = false)
            else ->
              Timing(
                rows,
                timingBackend,
                precision,
                report,
                window,
                pair,
                graph,
                extras.boolean(EXTRA_CLEAR_CACHE, false),
                extras.string(EXTRA_SETS)?.let(::setNames),
                extras.boolean(EXTRA_REQUEST_PATH, true),
                extras.int(EXTRA_COOL_MS, 0).toLong(),
                share,
                npu.options,
                npu.stateCopy,
                windowNamed = named != null,
              )
          }
        }
        else -> Normal(graph, precision, share, backend)
      }
    }

    /** The NPU extras of a debug run: the Qualcomm options (null without the libraries). */
    private class DebugNpu(val options: KevNpuOptions?, val stateCopy: Boolean)

    /** [DebugNpu] from the `npu_perf`, `npu_opt` and `pair_state` extras; null when invalid. */
    private fun debugNpu(extras: KevExtras, context: KevLaunchContext): DebugNpu? {
      val performance = extras.string(EXTRA_NPU_PERF)
      val optimization = extras.string(EXTRA_NPU_OPT)
      if (!context.npuAvailable && (performance != null || optimization != null)) return null
      val options =
        if (context.npuAvailable) KevNpuOptions.parse(performance, optimization) ?: return null
        else null
      val stateCopy =
        when (extras.string(EXTRA_PAIR_STATE)?.trim()?.lowercase(Locale.ROOT)) {
          null,
          "direct" -> false
          "copy" -> true
          else -> return null
        }
      return DebugNpu(options, stateCopy)
    }

    /** Why the NPU extras of a debug run cannot be followed. */
    private fun npuInvalid(extras: KevExtras): String =
      "npu_perf ${extras.string(EXTRA_NPU_PERF)}, npu_opt ${extras.string(EXTRA_NPU_OPT)} or " +
        "pair_state ${extras.string(EXTRA_PAIR_STATE)}: npu_perf takes a mode name or none, " +
        "npu_opt default, inference, o3 or prepare (both need the NPU libraries), pair_state " +
        "direct or copy"
  }
}

/**
 * The backend a normal or autoplay launch starts on: the launch's `backend` extra, else the choice
 * kept from the screen ("Run on"); a kept NPU in an APK without the NPU libraries goes back to the
 * GPU. Android-free; the app keeps the choice in its shared preferences.
 */
object KevBackendChoice {
  const val PREFERENCES = "kev"
  const val KEY = "backend"

  /** The backend to start on, from the launch's [extra], the [kept] wire name and the APK. */
  fun resolve(
    extra: KevDecider.Backend?,
    kept: String?,
    npuAvailable: Boolean,
  ): KevDecider.Backend {
    if (extra != null) return extra
    val saved = kept?.let { KevDecider.Backend.of(it) } ?: KevDecider.Backend.GPU
    return if (saved == KevDecider.Backend.NPU && !npuAvailable) KevDecider.Backend.GPU else saved
  }

  /** The [kept] value no longer applies (an NPU kept by an APK without the libraries). */
  fun keptUnusable(kept: String?, npuAvailable: Boolean): Boolean =
    kept?.let { KevDecider.Backend.of(it) } == KevDecider.Backend.NPU && !npuAvailable
}
