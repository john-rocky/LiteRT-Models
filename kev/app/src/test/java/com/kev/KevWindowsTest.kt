package com.kev

import java.io.Closeable
import java.io.File
import kotlin.io.path.createTempDirectory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The window rules, with no model data: each question asks for the smallest installed window that
 * holds its row; a plan whose largest window is L512 or more runs on that window alone; otherwise
 * the plan's windows (one or two) are compiled, a second one only with enough available memory,
 * else the largest takes every question. Stand-in graphs count how many are open at any time, and
 * the available memory is a number the test sets.
 */
class KevWindowsTest {
  /** A stand-in graph of window [length]; it is in [open] from creation until [close]. */
  private class FakeGraph(override val length: Int, private val open: MutableList<FakeGraph>) :
    RowRunner, Closeable {
    var closed = false
      private set

    init {
      open.add(this)
    }

    override fun run(ids: IntArray, valid: FloatArray) =
      FloatArray(length * KevPointerHead.HIDDEN_SIZE)

    override fun close() {
      check(!closed) { "L$length closed twice" }
      closed = true
      open.remove(this)
    }
  }

  /**
   * A stand-in pair of [shape], compiled with constant tensor sharing or not ([shared]); it is in
   * [open] from creation until [close].
   */
  private class FakePair(
    val shape: KevPairShape,
    val shared: Boolean,
    private val open: MutableList<Any>,
  ) : PairRunner, Closeable {
    override val stateLength = shape.stateLength
    override val questionLength = shape.questionLength
    var closed = false
      private set

    init {
      open.add(this)
    }

    override fun runState(ids: IntArray, valid: FloatArray) = Unit

    override fun runQuestion(ids: IntArray, valid: FloatArray) =
      FloatArray(questionLength * KevPointerHead.HIDDEN_SIZE)

    override fun close() {
      check(!closed) { "$shape closed twice" }
      closed = true
      open.remove(this)
    }
  }

  /**
   * Graphs with [primary] compiled at the start (by a request that asks for it alone) and
   * [available] bytes of free memory; records the windows compiled after that and the most graphs
   * ever open.
   */
  private class Harness(primary: Int, var available: Long = PLENTY) {
    val open = ArrayList<FakeGraph>()
    val pairsOpen = ArrayList<Any>()
    val compiled = ArrayList<Int>()
    var maxOpen = 0
    val graphs = KevResidentGraphs<FakeGraph, FakePair>()

    init {
      graphs.prepare(KevWindowPlan.Ready(listOf(primary)), { available }, ::open)
      compiled.clear()
    }

    fun open(window: Int): FakeGraph {
      compiled.add(window)
      return FakeGraph(window, open).also { maxOpen = maxOf(maxOpen, open.size + pairsOpen.size) }
    }

    fun openPair(shape: KevPairShape, shared: Boolean = false): FakePair =
      FakePair(shape, shared, pairsOpen).also {
        maxOpen = maxOf(maxOpen, open.size + pairsOpen.size)
      }

    /** Plans and prepares a request with rows of [rows] tokens over [installed]. */
    fun request(installed: List<Int>, vararg rows: Int): KevWindowRun<FakeGraph> {
      val plan = graphs.plan(rows.toList(), installed) as KevWindowPlan.Ready
      val run = graphs.prepare(plan, { available }, ::open)
      assertEquals(run.windows, run.graphs.map { it.length })
      assertTrue(run.graphs.none { it.closed })
      assertEquals(graphs.windows, open.map { it.length }.sorted())
      assertEquals(run.compiled.size, run.availableBytes.size)
      return run
    }
  }

  @Test
  fun aRowAsksForTheSmallestInstalledWindowThatHoldsIt() {
    val table =
      linkedMapOf(
        listOf(256, 512) to
          listOf(1 to 256, 128 to 256, 129 to 256, 256 to 256, 257 to 512, 512 to 512, 513 to null),
        listOf(128, 256, 512) to
          listOf(1 to 128, 93 to 128, 128 to 128, 129 to 256, 148 to 256, 257 to 512, 513 to null),
        listOf(512) to listOf(1 to 512, 300 to 512, 512 to 512, 513 to null),
        listOf(512, 2048) to
          listOf(1 to 512, 512 to 512, 513 to 2048, 1805 to 2048, 2048 to 2048, 2049 to null),
      )
    for ((installed, cases) in table) {
      for ((tokens, window) in cases) {
        assertEquals("$installed $tokens", window, KevWindows.smallestHolding(installed, tokens))
        val row = KevRow(IntArray(tokens), tokens - 1, intArrayOf())
        assertEquals("$installed $tokens", window, row.window(installed))
      }
    }
    assertEquals(256, KevWindows.smallestHolding(listOf(512, 256), 200))
    assertEquals(listOf(64, 128, 256, 512, 1024, 2048), KevFiles.WINDOWS)
    assertEquals(listOf(128, 256), KevFiles.DEFAULT_INSTALL)
    assertEquals(listOf(KevPairShape(128, 64)), KevFiles.DEFAULT_PAIRS)
    assertEquals(
      "kev-0.8b_rowprefill_L128_fp16fc_i8emb.tflite",
      KevFiles.graph(KevFiles.DEFAULT_INSTALL.first()),
    )
    // The plan: every question's own window, and the rows no installed window holds.
    val graphs = Harness(256).graphs
    val plan = graphs.plan(listOf(131, 101, 300), listOf(128, 256, 512)) as KevWindowPlan.Ready
    assertEquals(listOf(256, 128, 512), plan.windows)
    assertEquals(512, plan.top)
    val notInstalled = graphs.plan(listOf(100, 600), listOf(256, 512)) as KevWindowPlan.Missing
    assertEquals(600, notInstalled.rowTokens)
    assertEquals(1024, notInstalled.window)
    val overLargest = graphs.plan(listOf(3000), KevFiles.WINDOWS) as KevWindowPlan.Missing
    assertNull(overLargest.window)
  }

  @Test
  fun aLaunchWithNoGraphAsksForTheDefaultInstall() {
    val directory = createTempDirectory("kev-files").toFile()
    try {
      // Nothing installed: the tokenizer, the head and the default install's three graphs.
      assertEquals(
        listOf(
          KevFiles.TOKENIZER,
          KevFiles.HEAD,
          "kev-0.8b_rowprefill_L128_fp16fc_i8emb.tflite",
          "kev-0.8b_rowprefill_L256_fp16fc_i8emb.tflite",
          "kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite",
        ),
        KevFiles.missing(directory),
      )
      // Any one graph is enough to launch; a request that needs another one names it.
      for (name in listOf(KevFiles.TOKENIZER, KevFiles.HEAD, KevFiles.graph(512))) {
        File(directory, name).writeText("")
      }
      assertTrue(KevFiles.missing(directory).isEmpty())
      File(directory, KevFiles.graph(512)).delete()
      File(directory, KevFiles.pair(KevPairShape(256, 64))).writeText("")
      assertTrue(KevFiles.missing(directory).isEmpty())
      assertEquals(listOf(KevPairShape(256, 64)), KevFiles.installedPairs(directory))
    } finally {
      directory.deleteRecursively()
    }
  }

  @Test
  fun twoSmallWindowsWithEnoughMemory() {
    // L128 installed, the ticket's rows 131 / 101 / 93: L256 compiles next to the resident L128.
    val harness = Harness(128)
    val run = harness.request(KevFiles.WINDOWS, 131, 101, 93)
    assertEquals(listOf(256, 128, 128), run.windows)
    assertEquals(listOf(256), run.compiled)
    assertEquals(listOf(PLENTY), run.availableBytes)
    assertTrue(run.closed.isEmpty())
    assertFalse(run.secondRefused)
    assertEquals(listOf(128, 256), harness.graphs.windows)
    // A plan the two graphs cover compiles and closes nothing (the review: 124 / 106 / 94).
    val covered = harness.request(KevFiles.WINDOWS, 124, 106, 94)
    assertEquals(listOf(128, 128, 128), covered.windows)
    assertTrue(covered.compiled.isEmpty() && covered.closed.isEmpty())
    assertEquals(listOf(128, 256), harness.graphs.windows)
    assertEquals(2, harness.maxOpen)
  }

  @Test
  fun notEnoughMemoryLeavesOneGraph() {
    // The same request with less available memory than the limit: L256 alone takes every row.
    val harness =
      Harness(128, available = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1)
    val run = harness.request(KevFiles.WINDOWS, 131, 101, 93)
    assertTrue(run.secondRefused)
    assertEquals(listOf(256, 256, 256), run.windows)
    assertEquals(listOf(128), run.closed)
    assertEquals(listOf(256), run.compiled)
    assertEquals(listOf(256), harness.graphs.windows)
    assertEquals(1, harness.maxOpen)
    // With L256 resident, a request with a short row refused again keeps L256 for every row.
    val again = harness.request(KevFiles.WINDOWS, 131, 101)
    assertTrue(again.secondRefused)
    assertEquals(listOf(256, 256), again.windows)
    assertTrue(again.compiled.isEmpty() && again.closed.isEmpty())
    // At the limit, the second graph compiles.
    harness.available = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES
    val enough = harness.request(KevFiles.WINDOWS, 131, 101)
    assertFalse(enough.secondRefused)
    assertEquals(listOf(256, 128), enough.windows)
    assertEquals(listOf(128), enough.compiled)
    assertEquals(listOf(128, 256), harness.graphs.windows)
  }

  @Test
  fun aLargeWindowIsTheOnlyGraph() {
    // L128 and L256 resident; a row of 300 tokens asks for L512: both close, L512 takes every row.
    val harness = Harness(128)
    harness.request(KevFiles.WINDOWS, 131, 101)
    val run = harness.request(KevFiles.WINDOWS, 100, 300)
    assertEquals(listOf(512, 512), run.windows)
    assertEquals(listOf(128, 256), run.closed.sorted())
    assertEquals(listOf(512), run.compiled)
    assertEquals(listOf(512), harness.graphs.windows)
    // L2048 replaces L512 the same way, whatever the memory.
    val long = harness.request(KevFiles.WINDOWS, 1500, 100)
    assertEquals(listOf(2048, 2048), long.windows)
    assertEquals(listOf(512), long.closed)
    assertEquals(listOf(2048), harness.graphs.windows)
    assertEquals(2, harness.maxOpen)
  }

  @Test
  fun aShortRequestAfterALongOneReturnsToTheSmallWindows() {
    // The default install: L2048 (installed too) resident after a long request; the ticket asks for
    // L256 only, so L2048 closes before L256 compiles.
    val installed = listOf(256, 512, 2048)
    val harness = Harness(256)
    harness.request(installed, 1500)
    assertEquals(listOf(2048), harness.graphs.windows)
    val run = harness.request(installed, 131, 101, 93)
    assertEquals(listOf(256, 256, 256), run.windows)
    assertEquals(listOf(2048), run.closed)
    assertEquals(listOf(256), run.compiled)
    assertEquals(listOf(256), harness.graphs.windows)
    // With L128 installed: from L2048 to the two small windows, L256 compiled before L128.
    val all = Harness(128)
    all.request(KevFiles.WINDOWS, 1500)
    val mixed = all.request(KevFiles.WINDOWS, 131, 101)
    assertEquals(listOf(256, 128), mixed.windows)
    assertEquals(listOf(2048), mixed.closed)
    assertEquals(listOf(256, 128), mixed.compiled)
    assertEquals(listOf(128, 256), all.graphs.windows)
    assertEquals(2, all.maxOpen)
  }

  @Test
  fun theDefaultInstallRunsTheBundledExamplesOnL128AndL256() {
    // The bundled examples' rows (ticket 131 / 101 / 93, incident 148 / 122 / 128, review 124 /
    // 106 / 94) on the default install: rows over 128 tokens on L256, the others on L128.
    val harness = Harness(256)
    val ticket = harness.request(KevFiles.DEFAULT_INSTALL, 131, 101, 93)
    assertEquals(listOf(256, 128, 128), ticket.windows)
    assertEquals(listOf(128), ticket.compiled)
    val incident = harness.request(KevFiles.DEFAULT_INSTALL, 148, 122, 128)
    assertEquals(listOf(256, 128, 128), incident.windows)
    val review = harness.request(KevFiles.DEFAULT_INSTALL, 124, 106, 94)
    assertEquals(listOf(128, 128, 128), review.windows)
    for (run in listOf(incident, review)) {
      assertTrue(run.compiled.isEmpty() && run.closed.isEmpty())
    }
    assertEquals(listOf(128), harness.compiled)
    assertEquals(2, harness.maxOpen)
  }

  @Test
  fun aNamedWindowTakesEveryRowAlone() {
    val installed = listOf(128, 256, 512, 2048)
    val harness = Harness(128)
    harness.request(installed, 131, 101)
    val plan = harness.graphs.planFixed(listOf(100, 1500), 2048, installed) as KevWindowPlan.Ready
    val run = harness.graphs.prepare(plan, { PLENTY }, harness::open)
    assertEquals(listOf(2048, 2048), run.windows)
    assertEquals(listOf(128, 256), run.closed.sorted())
    assertEquals(listOf(2048), harness.graphs.windows)
    assertTrue(harness.graphs.planFixed(listOf(300), 256, installed) is KevWindowPlan.Missing)
    assertTrue(harness.graphs.planFixed(listOf(100), 1024, installed) is KevWindowPlan.Missing)
    assertEquals(2, harness.maxOpen)
  }

  @Test
  fun anotherBackendClosesEveryGraphAndThePlanCompilesAgain() {
    // A backend switch closes every graph; the request's plan then compiles what it asks for.
    val harness = Harness(128)
    harness.request(KevFiles.WINDOWS, 131, 101)
    val before = harness.open.toList()
    harness.graphs.close()
    assertTrue(before.all { it.closed })
    assertTrue(harness.open.isEmpty())
    assertTrue(harness.graphs.resident.isEmpty())
    val run = harness.request(KevFiles.WINDOWS, 131, 101)
    assertEquals(listOf(256, 128), run.compiled)
    assertEquals(listOf(128, 256), harness.graphs.windows)
    harness.graphs.close()
    val single = harness.request(KevFiles.WINDOWS, 100)
    assertEquals(listOf(128), single.compiled)
    assertEquals(listOf(128), harness.graphs.windows)
  }

  @Test
  fun thePairIsAloneAndRowsCloseIt() {
    val pair = KevPairShape(128, 64)
    // L128 and L256 resident; a pair plan closes both before the pair compiles.
    val harness = Harness(128)
    harness.request(KevFiles.WINDOWS, 131, 101)
    assertEquals(listOf(128, 256), harness.graphs.windows)
    val run =
      harness.graphs.preparePair(pair, { harness.available }) { shape, _ ->
        harness.openPair(shape)
      }
    assertTrue(run.compiled)
    assertEquals(PLENTY, run.availableBytes)
    assertEquals(listOf(128, 256), run.closedWindows)
    assertNull(run.closedPair)
    assertTrue(harness.open.isEmpty())
    assertEquals(listOf(KevGraphKey.Pair(pair)), harness.graphs.resident)
    assertEquals(1, harness.pairsOpen.size)
    // The pair already resident: nothing compiles or closes.
    val again =
      harness.graphs.preparePair(pair, { harness.available }) { shape, _ ->
        harness.openPair(shape)
      }
    assertFalse(again.compiled)
    assertNull(again.availableBytes)
    assertTrue(again.closedWindows.isEmpty())
    assertTrue(again.graph === run.graph)
    // A row plan closes the pair before it compiles; the second-graph rule is unchanged.
    val rows = harness.request(KevFiles.WINDOWS, 131, 101, 93)
    assertEquals(pair, rows.closedPair)
    assertTrue(run.graph.closed)
    assertTrue(harness.pairsOpen.isEmpty())
    assertEquals(listOf(256, 128), rows.compiled)
    assertEquals(listOf(256, 128, 128), rows.windows)
    assertEquals(listOf(128, 256), harness.graphs.windows)
    // The same with too little memory for a second window: L256 alone after the pair.
    harness.graphs.preparePair(pair, { harness.available }) { shape, _ -> harness.openPair(shape) }
    harness.available = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1
    val low = harness.request(KevFiles.WINDOWS, 131, 101, 93)
    assertEquals(pair, low.closedPair)
    assertTrue(low.secondRefused)
    assertEquals(listOf(256, 256, 256), low.windows)
    assertEquals(listOf(256), harness.graphs.windows)
    assertEquals(2, harness.maxOpen)
  }

  @Test
  fun aResidentPairKeepsTheSharingItWasCompiledWith() {
    // share auto: the pair compiled with plenty of memory holds its weights per signature; with
    // less memory later the resident pair runs as compiled (no compile to switch); once rows have
    // closed it, the next compile reads the memory again.
    val pair = KevPairShape(128, 64)
    val harness = Harness(128)
    fun prepare() =
      harness.graphs.preparePair(pair, { harness.available }) { shape, available ->
        harness.openPair(shape, KevPairShare.AUTO.sharesAt(available))
      }
    val first = prepare()
    assertTrue(first.compiled)
    assertFalse(first.graph.shared)
    harness.available = KevResidentGraphs.PAIR_UNSHARED_MIN_AVAILABLE_BYTES - 1
    val again = prepare()
    assertFalse(again.compiled)
    assertTrue(again.graph === first.graph)
    harness.request(KevFiles.DEFAULT_INSTALL, 131, 101, 93)
    assertTrue(first.graph.closed)
    val later = prepare()
    assertTrue(later.compiled)
    assertTrue(later.graph.shared)
    assertEquals(KevResidentGraphs.PAIR_UNSHARED_MIN_AVAILABLE_BYTES - 1, later.availableBytes)
  }

  @Test
  fun assignGivesTheWindowsPrepareRuns() {
    // The planner's prediction uses assign; prepare must end on the same windows.
    for ((rows, available) in
      listOf(
        intArrayOf(131, 101, 93) to PLENTY,
        intArrayOf(131, 101, 93) to KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1,
        intArrayOf(124, 106, 94) to KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1,
        intArrayOf(100, 300) to PLENTY,
        intArrayOf(1500, 100) to KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1,
      )) {
      val harness = Harness(512, available)
      val plan = harness.graphs.plan(rows.toList(), KevFiles.WINDOWS) as KevWindowPlan.Ready
      val second = harness.graphs.secondAllowed(plan, available)
      val run = harness.graphs.prepare(plan, { available }, harness::open)
      assertEquals(rows.toList().toString(), KevResidentGraphs.assign(plan, second), run.windows)
    }
    val ticket = KevWindowPlan.Ready(listOf(256, 128, 128))
    assertEquals(listOf(256, 128, 128), KevResidentGraphs.assign(ticket, secondAllowed = true))
    assertEquals(listOf(256, 256, 256), KevResidentGraphs.assign(ticket, secondAllowed = false))
    val long = KevWindowPlan.Ready(listOf(128, 512))
    assertEquals(listOf(512, 512), KevResidentGraphs.assign(long, secondAllowed = true))
  }

  @Test
  fun threeSmallWindowsKeepTheTwoLargest() {
    // L64, L128 and L256 installed; rows of 60 / 100 / 200 tokens ask for all three. At most two
    // graphs stay compiled: L256 and L128, and the L64 row runs on L128.
    val small = listOf(64, 128, 256)
    val harness = Harness(256)
    val run = harness.request(small, 60, 100, 200)
    assertEquals(listOf(128, 128, 256), run.windows)
    assertEquals(listOf(128), run.compiled)
    assertTrue(run.closed.isEmpty())
    assertFalse(run.secondRefused)
    assertEquals(listOf(128, 256), harness.graphs.windows)
    assertEquals(2, harness.maxOpen)
    // Both kept windows resident: memory under the limit compiles and closes nothing.
    harness.available = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1
    val covered = harness.request(small, 200, 60, 100, 50)
    assertEquals(listOf(256, 128, 128, 128), covered.windows)
    assertTrue(covered.compiled.isEmpty() && covered.closed.isEmpty())
    assertFalse(covered.secondRefused)
    // Two small windows: both stay, each row on its own.
    val fresh = Harness(64)
    assertEquals(listOf(64, 256), fresh.request(small, 60, 200).windows)
    assertEquals(listOf(64, 256), fresh.graphs.windows)
    assertEquals(listOf(64, 128, 64), Harness(64).request(small, 60, 100, 30).windows)
    // No room for a second graph: L256 alone takes every row.
    val low = Harness(64, available = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1)
    val refused = low.request(small, 60, 100, 200)
    assertTrue(refused.secondRefused)
    assertEquals(listOf(256, 256, 256), refused.windows)
    assertEquals(listOf(256), low.graphs.windows)
    assertEquals(1, low.maxOpen)
  }

  @Test
  fun assignKeepsTheTwoLargestSmallWindows() {
    // (plan windows, second allowed) to the window each question runs on.
    val table =
      listOf(
        (listOf(64, 128, 256) to true) to listOf(128, 128, 256),
        (listOf(256, 64, 128, 64) to true) to listOf(256, 128, 128, 128),
        (listOf(64, 256) to true) to listOf(64, 256),
        (listOf(64, 256, 64) to true) to listOf(64, 256, 64),
        (listOf(64, 128) to true) to listOf(64, 128),
        (listOf(64, 64) to true) to listOf(64, 64),
        (listOf(64, 128, 256) to false) to listOf(256, 256, 256),
        (listOf(64, 128) to false) to listOf(128, 128),
        (listOf(64, 512) to true) to listOf(512, 512),
      )
    for ((case, expected) in table) {
      val (windows, second) = case
      val plan = KevWindowPlan.Ready(windows)
      assertEquals("$windows $second", expected, KevResidentGraphs.assign(plan, second))
    }
    // The planner and prepare agree for three small windows, with and without memory to spare.
    for (available in listOf(PLENTY, KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1)) {
      val harness = Harness(512, available)
      val plan = harness.graphs.plan(listOf(60, 100, 200), KevFiles.WINDOWS) as KevWindowPlan.Ready
      val second = harness.graphs.secondAllowed(plan, available)
      val run = harness.graphs.prepare(plan, { available }, harness::open)
      assertEquals("$available", KevResidentGraphs.assign(plan, second), run.windows)
    }
  }

  @Test
  fun aFirstNpuCompileRunsAlone() {
    // L256 resident (compiled before, its NPU compilation cached); the ticket wants L256 + L128 and
    // L128 has never been compiled on the NPU: L256 closes, L128 compiles alone, L256 opens again.
    val open = ArrayList<FakeGraph>()
    val compiled = ArrayList<Int>()
    // How many graphs were open when each first compile started.
    val openAtFirst = ArrayList<Int>()
    val first = mutableSetOf(128)
    val graphs = KevResidentGraphs<FakeGraph, FakePair>()
    fun openGraph(window: Int): FakeGraph {
      compiled.add(window)
      if (first.remove(window)) openAtFirst.add(open.size)
      return FakeGraph(window, open)
    }
    graphs.prepare(KevWindowPlan.Ready(listOf(256)), { PLENTY }, ::openGraph)
    compiled.clear()
    val plan = graphs.plan(listOf(131, 101, 93), KevFiles.WINDOWS) as KevWindowPlan.Ready
    val run = graphs.prepare(plan, { PLENTY }, ::openGraph) { it in first }
    assertEquals(listOf(128, 256), run.compiled)
    assertEquals(listOf(256), run.closed)
    assertEquals(listOf(256, 128, 128), run.windows)
    assertEquals(listOf(0), openAtFirst)
    assertEquals(listOf(128, 256), graphs.windows)
    // Both first: each compiles alone, the smaller first, then the smaller opens again next to
    // the larger one.
    graphs.close()
    compiled.clear()
    openAtFirst.clear()
    first.addAll(listOf(128, 256))
    val both = graphs.prepare(plan, { PLENTY }, ::openGraph) { it in first }
    assertEquals(listOf(128, 256, 128), both.compiled)
    assertEquals(listOf(128), both.closed)
    assertEquals(listOf(0, 0), openAtFirst)
    assertEquals(listOf(128, 256), graphs.windows)
    // Under the second-window limit after the first compiles: the larger window takes every row.
    graphs.close()
    compiled.clear()
    first.addAll(listOf(128, 256))
    val low = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1
    val refused = graphs.prepare(plan, { low }, ::openGraph) { it in first }
    assertTrue(refused.secondRefused)
    assertEquals(listOf(128, 256), refused.compiled)
    assertEquals(listOf(256, 256, 256), refused.windows)
    assertEquals(listOf(256), graphs.windows)
    // Nothing first: the order of before (largest first), nothing closed.
    graphs.close()
    compiled.clear()
    val cached = graphs.prepare(plan, { PLENTY }, ::openGraph) { false }
    assertEquals(listOf(256, 128), cached.compiled)
    assertTrue(cached.closed.isEmpty())
  }

  private companion object {
    /** More available memory than any limit. */
    const val PLENTY = 8_000_000L * 1024
  }
}
