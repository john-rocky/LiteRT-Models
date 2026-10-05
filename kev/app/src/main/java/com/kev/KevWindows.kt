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

/** What [KevResidentGraphs.prepare] did for a row plan, and the graph of each question. */
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
  /** The pair closed before the windows compiled, or null. */
  val closedPair: KevPairShape? = null,
)

/** What [KevResidentGraphs.preparePair] did for a pair plan. */
class KevPairRun<P>(
  /** The pair every question runs on. */
  val graph: P,
  val shape: KevPairShape,
  /** The pair was compiled for this request (it was not resident). */
  val compiled: Boolean,
  /** The available memory read right before the compile, in bytes, or null without a compile. */
  val availableBytes: Long?,
  /** The windows closed for this request. */
  val closedWindows: List<Int>,
  /** Another pair closed for this request, or null. */
  val closedPair: KevPairShape?,
)

/**
 * The compiled graphs of the app: row-prefill windows or one shared-state pair, never both. A row
 * plan asks for each question's smallest installed window, with at most two graphs compiled:
 * - when the plan's largest window is over [SECOND_RESIDENT_MAX_WINDOW], that window is the only
 *   graph and every question runs on it;
 * - otherwise the plan's two largest windows are compiled (one when it asks for one) and each
 *   question runs on its own window, or on the smaller kept one when its window is not kept; a
 *   second graph is compiled only when the phone has at least [SECOND_RESIDENT_MIN_AVAILABLE_BYTES]
 *   available right before, else the plan's largest window is the only graph and takes every
 *   question ([assign] gives the same answer before anything compiles).
 *
 * A pair plan keeps the pair alone. Graphs a plan does not use are closed before a missing one
 * compiles; a plan the compiled graphs already cover compiles nothing, so a short request after a
 * long one returns to the small window. Use from one thread.
 */
class KevResidentGraphs<R, P> : Closeable
  where R : RowRunner, R : Closeable, P : PairRunner, P : Closeable {
  private val graphs = LinkedHashMap<Int, R>()
  private var pairGraph: P? = null
  private var pairShape: KevPairShape? = null

  /** The resident windows, in ascending order. */
  val windows: List<Int>
    get() = graphs.keys.sorted()

  /** The resident row graphs, by ascending window. */
  val all: List<R>
    get() = windows.map { graphs.getValue(it) }

  /** The resident pair's shape, or null. */
  val pair: KevPairShape?
    get() = pairShape

  /** The resident pair, or null. */
  val pairRunner: P?
    get() = pairGraph

  /** The resident graphs: windows in ascending order, then the pair. */
  val resident: List<KevGraphKey>
    get() =
      windows.map { KevGraphKey.Window(it) } +
        listOfNotNull(pairShape?.let { KevGraphKey.Pair(it) })

  /** The resident graph of [window]. */
  fun graph(window: Int): R = graphs[window] ?: throw IllegalStateException("L$window not resident")

  /**
   * Each question's smallest window among the [installed] ones, given its row of [rows] tokens, or
   * the longest row when no installed window holds it.
   */
  fun plan(rows: List<Int>, installed: List<Int>): KevWindowPlan = Companion.plan(rows, installed)

  /** Every row of [rows] tokens on the one [window] (a demo run that names it). */
  fun planFixed(rows: List<Int>, window: Int, installed: List<Int>): KevWindowPlan =
    Companion.planFixed(rows, window, installed)

  /**
   * See [KevResidentGraphs.Companion.secondAllowed], with the resident windows of this instance.
   */
  fun secondAllowed(plan: KevWindowPlan.Ready, availableBytes: Long): Boolean =
    Companion.secondAllowed(plan, graphs.keys, availableBytes)

  /**
   * Makes [plan]'s windows resident as described on the class: closes the pair and the windows the
   * plan does not use, then opens the missing windows with [open], reading [availableBytes] right
   * before each compile. A window whose compile must run [alone] (the first NPU compile of a file,
   * minutes long and memory hungry) opens first, with every other graph closed; then the others
   * open in descending order. Returns the graph of each question.
   */
  fun prepare(
    plan: KevWindowPlan.Ready,
    availableBytes: () -> Long,
    open: (Int) -> R,
    alone: (Int) -> Boolean = { false },
  ): KevWindowRun<R> {
    val closedPair = closePair()
    var windows = assign(plan, secondAllowed = true)
    val wanted = windows.toSet()
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
      // Ascending, so that the largest of them is the one left open.
      for (window in wanted.sorted().filter { it !in graphs && alone(it) }) {
        closeAllBut(emptySet())
        compile(window, availableBytes())
      }
      for (window in wanted.sortedDescending().filter { it !in graphs }) {
        val available = availableBytes()
        if (graphs.isEmpty() || available >= SECOND_RESIDENT_MIN_AVAILABLE_BYTES) {
          compile(window, available)
          continue
        }
        // No room for a second graph: the top window alone takes every question.
        secondRefused = true
        windows = assign(plan, secondAllowed = false)
        val top = plan.top
        if (top !in graphs) {
          closeAllBut(emptySet())
          compile(top, availableBytes())
        } else {
          closeAllBut(setOf(top))
        }
        break
      }
    }
    return KevWindowRun(
      windows.map { graph(it) },
      windows,
      compiled,
      memory,
      closed,
      secondRefused,
      closedPair,
    )
  }

  /**
   * Makes [shape] the only resident graph: closes every window and any other pair, then opens the
   * pair with [open] when it is not resident, with [availableBytes] read right before the compile.
   */
  fun preparePair(
    shape: KevPairShape,
    availableBytes: () -> Long,
    open: (KevPairShape, Long) -> P,
  ): KevPairRun<P> {
    val closedWindows = windows
    for (window in closedWindows) graphs.remove(window)?.close()
    val closedPair = if (pairShape != null && pairShape != shape) closePair() else null
    var available: Long? = null
    val current = pairGraph
    val graph =
      if (current != null) {
        current
      } else {
        val read = availableBytes()
        available = read
        open(shape, read).also {
          pairGraph = it
          pairShape = shape
        }
      }
    return KevPairRun(graph, shape, current == null, available, closedWindows, closedPair)
  }

  /** Closes the resident pair and returns its shape, or null when there is none. */
  private fun closePair(): KevPairShape? {
    val shape = pairShape ?: return null
    val graph = pairGraph
    pairGraph = null
    pairShape = null
    graph?.close()
    return shape
  }

  /** Closes every graph; a later [prepare] or [preparePair] compiles what the request needs. */
  override fun close() {
    val open: List<Closeable> = graphs.values.toList() + listOfNotNull(pairGraph)
    graphs.clear()
    pairGraph = null
    pairShape = null
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

    /**
     * A pair compiles without constant tensor sharing ([KevPairShare.AUTO]) only with at least this
     * much available memory right before the compile. Three runs on the Galaxy S26 are the basis:
     * without sharing the compile took MemAvailable from 7,649,632 kB down to 2,210,636 kB (the app
     * stayed up each time, low points 2.21, 2.46 and 2.51 GB); with sharing it took about 1.7 GB.
     */
    const val PAIR_UNSHARED_MIN_AVAILABLE_BYTES = 6_500_000L * 1024

    /** See [KevResidentGraphs.plan]. */
    fun plan(rows: List<Int>, installed: List<Int>): KevWindowPlan {
      val windows = rows.map { KevWindows.smallestHolding(installed, it) }
      if (windows.any { it == null }) {
        val longest = rows.max()
        return KevWindowPlan.Missing(longest, KevWindows.smallestHolding(KevFiles.WINDOWS, longest))
      }
      return KevWindowPlan.Ready(windows.map { requireNotNull(it) })
    }

    /** See [KevResidentGraphs.planFixed]. */
    fun planFixed(rows: List<Int>, window: Int, installed: List<Int>): KevWindowPlan {
      val longest = rows.max()
      if (window !in installed || longest > window) return KevWindowPlan.Missing(longest, window)
      return KevWindowPlan.Ready(rows.map { window })
    }

    /**
     * Whether [plan] may keep two windows: the windows [assign] keeps are all in [resident]
     * already, or [availableBytes] is at least the limit. [prepare] reads the memory again right
     * before the second compile, so a plan made with this answer can still end on one window
     * (slower than planned, never wrong).
     */
    fun secondAllowed(
      plan: KevWindowPlan.Ready,
      resident: Collection<Int>,
      availableBytes: Long,
    ): Boolean =
      resident.containsAll(assign(plan, secondAllowed = true).toSet()) ||
        availableBytes >= SECOND_RESIDENT_MIN_AVAILABLE_BYTES

    /**
     * The window each question of [plan] runs on. The plan's top window for every question when
     * that is over [SECOND_RESIDENT_MAX_WINDOW] or when a second graph is not [secondAllowed];
     * otherwise the plan's [MAX_RESIDENT] largest windows are kept, and each question runs on the
     * smallest kept window that holds its row (its own window when that is kept). [prepare] and the
     * planner's time prediction both use it.
     */
    fun assign(plan: KevWindowPlan.Ready, secondAllowed: Boolean): List<Int> {
      val top = plan.top
      if (top > SECOND_RESIDENT_MAX_WINDOW || !secondAllowed) return plan.windows.map { top }
      val kept = plan.windows.distinct().sortedDescending().take(MAX_RESIDENT)
      return plan.windows.map { window -> kept.filter { it >= window }.min() }
    }
  }
}
