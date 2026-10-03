package com.kev

import java.util.Locale

/** What a launch intent asks for (see `MainActivity` for the extras). */
sealed interface KevLaunch {
  /** The editable sample. */
  data object Normal : KevLaunch

  /** Debug build: the fixture gate on [window] into `files/<report>`. */
  data class Gate(val backend: KevDecider.Backend, val report: String, val window: Int, val limit: Int) : KevLaunch

  /** Debug and benchmark builds: the timing protocol on the rows of `files/<rows>`. */
  data class Timing(
    val rows: String,
    val backend: KevDecider.Backend,
    val report: String,
    val window: Int,
    val clearCache: Boolean,
  ) : KevLaunch

  /** The demo recording: answer the request in [fixture] on the presentation layout. */
  data class Autoplay(val fixture: String, val delayMs: Long, val gapMs: Long, val window: Int) : KevLaunch

  /** Extras that cannot be followed; [autoplay] says which log tag reports it. */
  data class Invalid(val reason: String, val autoplay: Boolean) : KevLaunch

  companion object {
    /** Report names stay inside `files/`: letters, digits, dot, underscore and hyphen. */
    private val REPORT_NAME = Regex("[A-Za-z0-9._-]+")

    fun backend(name: String?): KevDecider.Backend? =
      KevDecider.Backend.entries.firstOrNull { it.name == (name ?: "gpu").uppercase(Locale.ROOT) }

    fun reportNameValid(name: String): Boolean = name.matches(REPORT_NAME) && !name.endsWith(".partial")

    fun windowValid(window: Int): Boolean = window in KevEncoder.WINDOWS
  }
}
