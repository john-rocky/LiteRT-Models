package com.kev

import java.util.Locale

/** What a launch intent asks for (see `MainActivity` for the extras). */
sealed interface KevLaunch {
  /**
   * The editable sample: Decide plans each request in [graph] mode, on GPU at [precision] (null:
   * each graph at its own default, [KevPrecision.defaultFor]).
   */
  data class Normal(
    val graph: KevGraphMode = KevGraphMode.AUTO,
    val precision: KevPrecision? = null,
    /** Whether a pair compiles with constant tensor sharing (the `share` extra). */
    val share: KevPairShare = KevPairShare.AUTO,
  ) : KevLaunch

  /**
   * Debug build: the fixture gate on [graph] (one row window or the pair) into `files/<report>`.
   */
  data class Gate(
    val backend: KevDecider.Backend,
    val precision: KevPrecision?,
    val report: String,
    val graph: KevGraphKey,
    val limit: Int,
    val share: KevPairShare = KevPairShare.AUTO,
  ) : KevLaunch

  /**
   * Debug and benchmark builds: the timing protocol on the rows of `files/<rows>`, limited to the
   * [sets] named (null: all of them), on the row [window] or on the [pair] when [graph] is
   * [KevGraphMode.PAIR]; then, with [requestPath], the bundled request from its text on the plan of
   * [graph] (every row on [window] when the mode is [KevGraphMode.ROWS]). With [coolMs] > 0, each
   * set and the request path first wait for the GPU to cool ([KevCooler]).
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
  ) : KevLaunch

  /**
   * The demo recording: answer the request in [fixture] on the presentation layout, every question
   * on [window] when it is set, otherwise on the plan of [graph] (the plan a Decide would use for
   * [KevGraphMode.AUTO]). [precision] applies when this launch starts the app.
   */
  data class Autoplay(
    val fixture: String,
    val delayMs: Long,
    val gapMs: Long,
    val window: Int?,
    val graph: KevGraphMode = KevGraphMode.AUTO,
    val precision: KevPrecision? = null,
    val share: KevPairShare = KevPairShare.AUTO,
  ) : KevLaunch

  /** Extras that cannot be followed; [autoplay] says which log tag reports it. */
  data class Invalid(val reason: String, val autoplay: Boolean) : KevLaunch

  companion object {
    /** Report names stay inside `files/`: letters, digits, dot, underscore and hyphen. */
    private val REPORT_NAME = Regex("[A-Za-z0-9._-]+")

    fun backend(name: String?): KevDecider.Backend? =
      KevDecider.Backend.entries.firstOrNull { it.name == (name ?: "gpu").uppercase(Locale.ROOT) }

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
  }
}
