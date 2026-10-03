package com.kev

import java.io.Closeable

/** Window choice for question rows. */
object KevWindows {
  /** The smallest of [windows] that holds a row of [tokens] tokens, or null when none does. */
  fun smallestHolding(windows: Collection<Int>, tokens: Int): Int? =
    windows.sorted().firstOrNull { tokens <= it }
}

/** The windows a request asks for, before any graph is compiled or closed. */
sealed interface KevWindowPlan {
  /**
   * Every row has an installed window: [windows] holds each question's smallest installed window
   * that holds its row (or the one window a run names), in question order.
   */
  class Ready(val windows: List<Int>) : KevWindowPlan {
    /** The largest window of the request: the one that holds its longest row. */
    val top: Int
      get() = windows.max()
  }

  /**
   * A row of [rowTokens] tokens has no installed graph: [window] is the published window that would
   * hold it, null when the row is over the largest one.
   */
  class Missing(val rowTokens: Int, val window: Int?) : KevWindowPlan
}

/** What [KevResidentGraphs.prepare] did for a request, and the graph of each question. */
class KevWindowRun<R>(
  /** The graph of each question, in question order. */
  val graphs: List<R>,
  /** The window of each question: the plan's, or the plan's top window when it took every row. */
  val windows: List<Int>,
  /** The windows compiled for this request, in order. */
  val compiled: List<Int>,
  /** The available memory read right before each compile, in bytes, in the order of [compiled]. */
  val availableBytes: List<Long>,
  /** The windows closed for this request. */
  val closed: List<Int>,
  /** The plan wanted a second graph but the available memory was below the limit. */
  val secondRefused: Boolean,
)

/**
 * The compiled graphs of the app, starting with the [primary] window (the smallest installed one,
 * compiled when the engine loads). A request asks for each question's smallest installed window
 * (the plan), with at most two graphs compiled:
 * - when the plan's largest window is over [SECOND_RESIDENT_MAX_WINDOW], that window is the only
 *   graph and every question runs on it;
 * - otherwise the plan's windows are compiled, one or two, and each question runs on its own; a
 *   second graph is compiled only when the phone has at least [SECOND_RESIDENT_MIN_AVAILABLE_BYTES]
 *   available right before, else the plan's largest window is the only graph and takes every
 *   question.
 *
 * Graphs the plan does not use are closed before a missing one compiles; a plan the compiled graphs
 * already cover compiles nothing, so a short request after a long one returns to the small window.
 * Use from one thread.
 */
class KevResidentGraphs<R>(val primary: Int, primaryGraph: R) : Closeable
  where R : RowRunner, R : Closeable {
  private val graphs = LinkedHashMap<Int, R>().apply { put(primary, primaryGraph) }

  /** The resident windows, in ascending order. */
  val windows: List<Int>
    get() = graphs.keys.sorted()

  /** The resident graphs, by ascending window. */
  val all: List<R>
    get() = windows.map { graphs.getValue(it) }

  /** The resident graph of [window]. */
  fun graph(window: Int): R = graphs[window] ?: throw IllegalStateException("L$window not resident")

  /**
   * Each question's smallest window among the [installed] ones, given its row of [rows] tokens, or
   * the longest row when no installed window holds it.
   */
  fun plan(rows: List<Int>, installed: List<Int>): KevWindowPlan {
    val windows = rows.map { KevWindows.smallestHolding(installed, it) }
    if (windows.any { it == null }) {
      val longest = rows.max()
      return KevWindowPlan.Missing(longest, KevWindows.smallestHolding(KevFiles.WINDOWS, longest))
    }
    return KevWindowPlan.Ready(windows.map { requireNotNull(it) })
  }

  /** Every row of [rows] tokens on the one [window] (a demo run that names it). */
  fun planFixed(rows: List<Int>, window: Int, installed: List<Int>): KevWindowPlan {
    val longest = rows.max()
    if (window !in installed || longest > window) return KevWindowPlan.Missing(longest, window)
    return KevWindowPlan.Ready(rows.map { window })
  }

  /**
   * Makes [plan]'s windows resident as described on the class: closes the graphs the plan does not
   * use, then opens the missing ones with [open] in descending order, reading [availableBytes]
   * right before each compile. Returns the graph of each question.
   */
  fun prepare(
    plan: KevWindowPlan.Ready,
    availableBytes: () -> Long,
    open: (Int) -> R,
  ): KevWindowRun<R> {
    val top = plan.top
    val wanted = if (top > SECOND_RESIDENT_MAX_WINDOW) setOf(top) else plan.windows.toSet()
    var windows = plan.windows.map { if (it in wanted) it else top }
    val compiled = ArrayList<Int>()
    val memory = ArrayList<Long>()
    val closed = ArrayList<Int>()
    var secondRefused = false
    fun closeAllBut(keep: Set<Int>) {
      for (window in graphs.keys.filter { it !in keep }) {
        graphs.remove(window)?.close()
        closed.add(window)
      }
    }
    fun compile(window: Int, available: Long) {
      check(graphs.size < MAX_RESIDENT) { "L$window would be graph ${graphs.size + 1}" }
      memory.add(available)
      graphs[window] = open(window)
      compiled.add(window)
    }
    if (!graphs.keys.containsAll(wanted)) {
      closeAllBut(wanted)
      for (window in wanted.sortedDescending().filter { it !in graphs }) {
        val available = availableBytes()
        if (graphs.isEmpty() || available >= SECOND_RESIDENT_MIN_AVAILABLE_BYTES) {
          compile(window, available)
          continue
        }
        // No room for a second graph: the top window alone takes every question.
        secondRefused = true
        windows = plan.windows.map { top }
        if (top !in graphs) {
          closeAllBut(emptySet())
          compile(top, availableBytes())
        }
        break
      }
    }
    return KevWindowRun(windows.map { graph(it) }, windows, compiled, memory, closed, secondRefused)
  }

  /** Closes every graph and opens the primary again with [open] (another backend). */
  fun reopen(open: (Int) -> R) {
    close()
    graphs[primary] = open(primary)
  }

  /** Closes every graph; a later [prepare] compiles what the request needs. */
  override fun close() {
    val open = graphs.values.toList()
    graphs.clear()
    var failure: Throwable? = null
    for (graph in open) {
      try {
        graph.close()
      } catch (closing: Exception) {
        failure = failure ?: closing
      }
    }
    failure?.let { throw it }
  }

  companion object {
    /** The most graphs compiled at a time. */
    const val MAX_RESIDENT = 2

    /**
     * Two graphs only when both windows are at most this one. On the 12 GB Galaxy S26 the L2048
     * graph compiling next to the resident L128 graph took MemAvailable from 4,770,840 kB down to
     * 623,824 kB, and Android's low-memory killer stopped the app (one run).
     */
    const val SECOND_RESIDENT_MAX_WINDOW = 256

    /**
     * A second graph only with at least this much available memory (`ActivityManager.MemoryInfo
     * .availMem`; 4,500,000 kB of 1,024 bytes) right before its compile. One run on the Galaxy S26
     * is the whole basis: the L256 graph compiled next to the resident L128 graph from MemAvailable
     * 4,762,332 kB, the low point was 883,380 kB, and the app stayed up.
     */
    const val SECOND_RESIDENT_MIN_AVAILABLE_BYTES = 4_500_000L * 1024
  }
}
