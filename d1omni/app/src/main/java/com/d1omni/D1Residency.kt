package com.d1omni

import java.io.Closeable

/**
 * What one request's preparation did: the bucket each row runs on, the graphs compiled and closed
 * for it (in order), the available memory read right before each compile that had another graph
 * resident (bytes, `ActivityManager.MemoryInfo.availMem`), and whether a second graph was refused
 * for lack of memory.
 */
class D1Prepared(
  val rowBuckets: List<Int>,
  val compiled: List<Int>,
  val closed: List<Int>,
  val availableBeforeSecond: List<Long>,
  val secondRefused: Boolean,
)

/**
 * The compiled decision graphs of the process (Android-free, so the JVM tests cover the rules). At
 * most [maxResident] (2) graphs stay compiled. A request whose rows the resident graphs already
 * hold at their smallest installed bucket compiles nothing. Otherwise the request wants its largest
 * bucket and the bucket most of its other rows prefer; resident graphs it does not want stay while
 * there is a slot for them, else they are closed first. Before a graph compiles next to another
 * one, the available memory is read: under [minFreeForSecond] bytes the second graph is given up
 * and every row runs on the request's largest bucket (when that one is the graph about to compile,
 * the other graphs are closed first). Rows run on the smallest resident graph that holds them.
 */
class D1Residency<G : Closeable>(
  private val maxResident: Int = MAX_RESIDENT,
  private val minFreeForSecond: Long = MIN_FREE_FOR_SECOND,
) : Closeable {
  private val graphs = sortedMapOf<Int, G>()

  /** The resident buckets, ascending. */
  val resident: List<Int>
    get() = graphs.keys.toList()

  /** The resident graph of [bucket]. */
  fun graph(bucket: Int): G = requireNotNull(graphs[bucket]) { "L$bucket is not compiled" }

  /** The resident graphs, ascending by bucket. */
  val all: List<G>
    get() = graphs.values.toList()

  /**
   * Makes the graphs ready for rows of [positions] (P + n each) with the [installed] buckets
   * (ascending): see the class comment. [available] reads the available memory and [open]
   * compiles one bucket. Throws when a row is longer than every installed bucket.
   */
  fun prepare(
    positions: List<Int>,
    installed: List<Int>,
    available: () -> Long,
    open: (Int) -> G,
  ): D1Prepared {
    require(positions.isNotEmpty()) { "a request without rows" }
    val preferred =
      positions.map { count ->
        D1Contract.bucketFor(count, installed)
          ?: throw IllegalArgumentException(
            "a row of $count positions does not fit the installed graphs (largest L${installed.maxOrNull()})"
          )
      }
    if (preferred.all { it in graphs }) {
      return D1Prepared(preferred, emptyList(), emptyList(), emptyList(), false)
    }
    val largest = preferred.max()
    val second =
      preferred
        .filter { it != largest }
        .groupingBy { it }
        .eachCount()
        .entries
        .sortedWith(compareBy({ -it.value }, { it.key }))
        .firstOrNull()
        ?.key
    return ensure(listOfNotNull(largest, second), positions, available, open)
  }

  /**
   * Makes [target] (the largest bucket first, then at most one more) resident under the same
   * rules, then assigns each of [positions] to the smallest resident graph that holds it.
   */
  fun ensure(
    target: List<Int>,
    positions: List<Int>,
    available: () -> Long,
    open: (Int) -> G,
  ): D1Prepared {
    require(target.isNotEmpty() && target.size <= maxResident) { "target $target" }
    val largest = target.max()
    val ordered = listOf(largest) + target.filter { it != largest }.distinct()
    val missing = ordered.filter { it !in graphs }
    val compiled = ArrayList<Int>()
    val closed = ArrayList<Int>()
    val readings = ArrayList<Long>()
    // Room for the missing graphs: close resident graphs the request does not want, largest first.
    for (bucket in graphs.keys.filter { it !in ordered }.sortedDescending()) {
      if (graphs.size + missing.size <= maxResident) break
      closeGraph(bucket, closed)
    }
    var refused = false
    for (bucket in missing) {
      if (graphs.isNotEmpty()) {
        if (refused) continue
        val free = available()
        readings.add(free)
        if (free < minFreeForSecond) {
          refused = true
          if (bucket != largest) continue
          for (other in graphs.keys.toList()) closeGraph(other, closed)
        }
      }
      graphs[bucket] = open(bucket)
      compiled.add(bucket)
    }
    val buckets =
      positions.map { count ->
        graphs.keys.firstOrNull { count <= it }
          ?: throw IllegalStateException("no resident graph holds $count positions")
      }
    return D1Prepared(buckets, compiled, closed, readings, refused)
  }

  /** Closes the graph of [bucket] (no-op when it is not resident). */
  fun close(bucket: Int) {
    graphs.remove(bucket)?.close()
  }

  override fun close() {
    for (bucket in graphs.keys.toList()) close(bucket)
  }

  private fun closeGraph(bucket: Int, closed: MutableList<Int>) {
    graphs.remove(bucket)?.close()
    closed.add(bucket)
  }

  companion object {
    /** Graphs compiled at once. */
    const val MAX_RESIDENT = 2

    /**
     * The available memory (bytes) a second graph needs before it compiles. Compiling the largest
     * decision graph (L4096) for the Galaxy S26 GPU took MemAvailable from 7.30 GB to 1.64 GB
     * (contract.json `limits`), so a compile next to a resident graph keeps this margin.
     */
    const val MIN_FREE_FOR_SECOND = 2_500_000_000L
  }
}
