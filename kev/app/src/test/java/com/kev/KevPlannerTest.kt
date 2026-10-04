package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The plan of a request ([KevPlanner]) with the Galaxy S26 table ([KevCosts]): the row form's time
 * is the sum over the windows the questions are assigned ([KevResidentGraphs.assign]), the pair's
 * the state call plus one question step per question; the smaller wins, the row form on a tie, and
 * a form that cannot take the request leaves the other one.
 */
class KevPlannerTest {
  /** A request as the planner sees it: the state part (`[state]` included) and the branches. */
  private class Shape(val name: String, val state: Int, val branches: List<Int>)

  private fun plan(
    request: Shape,
    windows: List<Int>,
    pairs: List<KevPairShape> = emptyList(),
    available: Long = PLENTY,
    resident: List<Int> = emptyList(),
    mode: KevGraphMode = KevGraphMode.AUTO,
    fixedWindow: Int? = null,
    costs: KevCosts = KevCosts.GALAXY_S26_GPU_FP32,
  ): KevPlan =
    KevPlanner.plan(
      request.state,
      request.branches,
      windows,
      pairs,
      resident,
      available,
      mode,
      fixedWindow,
      costs,
    )

  @Test
  fun theBundledExamplesAndTheFiveQuestionRequestOnEachInstall() {
    // form, row ms, pair ms (null: the form cannot take the request), per install set.
    val rows = listOf(256, 512)
    val small = listOf(128, 256, 512)
    val table =
      listOf(
        Triple(TICKET, rows, NONE) to Expect(KevForm.ROW, 969.0, null),
        Triple(TICKET, small, NONE) to Expect(KevForm.ROW, 675.0, null),
        Triple(TICKET, rows, PAIR) to Expect(KevForm.PAIR, 969.0, P3),
        Triple(TICKET, small, PAIR) to Expect(KevForm.ROW, 675.0, P3),
        Triple(INCIDENT, rows, NONE) to Expect(KevForm.ROW, 969.0, null),
        Triple(INCIDENT, small, NONE) to Expect(KevForm.ROW, 675.0, null),
        Triple(INCIDENT, rows, PAIR) to Expect(KevForm.PAIR, 969.0, P3),
        Triple(INCIDENT, small, PAIR) to Expect(KevForm.ROW, 675.0, P3),
        Triple(REVIEW, rows, NONE) to Expect(KevForm.ROW, 969.0, null),
        Triple(REVIEW, small, NONE) to Expect(KevForm.ROW, 528.0, null),
        Triple(REVIEW, rows, PAIR) to Expect(KevForm.PAIR, 969.0, P3),
        Triple(REVIEW, small, PAIR) to Expect(KevForm.ROW, 528.0, P3),
        Triple(FIVE, rows, NONE) to Expect(KevForm.ROW, 1615.0, null),
        Triple(FIVE, small, NONE) to Expect(KevForm.ROW, 1321.0, null),
        Triple(FIVE, rows, PAIR) to Expect(KevForm.PAIR, 1615.0, P5),
        Triple(FIVE, small, PAIR) to Expect(KevForm.PAIR, 1321.0, P5),
      )
    for ((case, expect) in table) {
      val (request, windows, pairs) = case
      val label = "${request.name} $windows ${if (pairs.isEmpty()) "" else "+ pair"}"
      expect.check(label, plan(request, windows, pairs))
    }
  }

  @Test
  fun theTicketTakesThePairWhenASecondWindowIsNotAllowed() {
    // L128 + L256 + pair installed: two windows allowed → rows 323 + 176 + 176 = 675 ms < the
    // pair's
    // state + 3 steps; not allowed (memory under the limit, nothing resident) → rows 3 × 323 =
    // 969 ms > the pair. The Galaxy S26 table puts the pair in between.
    assertTrue(675.0 < P3 && P3 < 969.0)
    val small = listOf(128, 256, 512)
    Expect(KevForm.ROW, 675.0, P3).check("allowed", plan(TICKET, small, PAIR, PLENTY))
    Expect(KevForm.PAIR, 969.0, P3).check("not allowed", plan(TICKET, small, PAIR, LOW))
    // Both windows already resident: no compile is needed, so the memory does not matter.
    Expect(KevForm.ROW, 675.0, P3)
      .check("resident", plan(TICKET, small, PAIR, LOW, resident = listOf(128, 256)))
    // One of them resident is not enough.
    Expect(KevForm.PAIR, 969.0, P3)
      .check("one resident", plan(TICKET, small, PAIR, LOW, resident = listOf(128)))
    // The five-question request takes the pair either way (state + 5 steps < 1,321 / 1,615 ms).
    assertTrue(P5 < 1321.0)
    Expect(KevForm.PAIR, 1615.0, P5).check("five low", plan(FIVE, small, PAIR, LOW))
  }

  @Test
  fun requestsThePairCannotTakeRunOnRows() {
    val all = listOf(128, 256, 512)
    // State part of 129 tokens: over Ls.
    Expect(KevForm.ROW, 323.0, null)
      .check("state 129", plan(Shape("s", 129, listOf(20)), all, PAIR))
    // A branch of 65 tokens: over Lq (rows of 115 and 70 tokens, both on L128).
    Expect(KevForm.ROW, 352.0, null)
      .check("branch 65", plan(Shape("b", 50, listOf(65, 20)), all, PAIR))
    // One question: the row (L128 176 ms, or L256 323 ms) is faster than state + one step.
    Expect(KevForm.ROW, 176.0, P1).check("one L128", plan(Shape("q", 30, listOf(64)), all, PAIR))
    Expect(KevForm.ROW, 323.0, P1)
      .check("one L256", plan(Shape("q", 30, listOf(64)), listOf(256, 512), PAIR))
  }

  @Test
  fun thePairAloneTakesWhatNoWindowHolds() {
    // No row window installed at all.
    Expect(KevForm.PAIR, null, P3).check("no window", plan(TICKET, emptyList(), PAIR))
    // L128 only: the five-question rows (128–142 tokens) need L256.
    Expect(KevForm.PAIR, null, P5).check("L128 only", plan(FIVE, listOf(128), PAIR))
    // Neither form: the window that would hold the longest row.
    val none = plan(FIVE, listOf(128))
    assertTrue(none is KevPlan.NoWindow)
    assertEquals(142, (none as KevPlan.NoWindow).missing.rowTokens)
    assertEquals(256, none.missing.window)
  }

  @Test
  fun aTieGoesToTheRows() {
    val costs =
      KevCosts(
        rowMs = mapOf(256 to 200.0),
        pairMs = mapOf(KevPairShape(128, 64) to KevPairCost(stateMs = 100.0, questionMs = 150.0)),
      )
    val tie = Shape("tie", 50, listOf(10, 10))
    Expect(KevForm.ROW, 400.0, 400.0).check("tie", plan(tie, listOf(256), PAIR, costs = costs))
  }

  @Test
  fun aNamedModeOrWindowIsFollowed() {
    val small = listOf(128, 256, 512)
    // rows: even when the pair would be faster.
    Expect(KevForm.ROW, 1321.0, P5).check("rows", plan(FIVE, small, PAIR, mode = KevGraphMode.ROWS))
    // pair: even when the rows would be faster.
    Expect(KevForm.PAIR, 675.0, P3)
      .check("pair", plan(TICKET, small, PAIR, mode = KevGraphMode.PAIR))
    // A named window: every row on it.
    val fixed = plan(TICKET, small, PAIR, fixedWindow = 512)
    Expect(KevForm.ROW, 1845.0, P3).check("L512", fixed)
    assertEquals(listOf(512, 512, 512), (fixed as KevPlan.Rows).windows.windows)
    // The pair asked for and impossible: the reason.
    val state =
      plan(Shape("s", 129, listOf(20)), small, PAIR, mode = KevGraphMode.PAIR) as KevPlan.NoPair
    assertEquals(KevPairMiss.StateTooLong(129, 128), state.miss)
    val branch =
      plan(Shape("b", 50, listOf(20, 65)), small, PAIR, mode = KevGraphMode.PAIR) as KevPlan.NoPair
    assertEquals(KevPairMiss.QuestionTooLong(1, 65, 64), branch.miss)
    val missing = plan(TICKET, small, NONE, mode = KevGraphMode.PAIR) as KevPlan.NoPair
    assertEquals(KevPairMiss.NotInstalled, missing.miss)
  }

  @Test
  fun theTableCoversEveryPublishedGraphAndNamesTheModes() {
    val costs = KevCosts.GALAXY_S26_GPU_FP32
    assertEquals(KevFiles.WINDOWS.toSet(), costs.rowMs.keys)
    assertEquals(KevFiles.PAIRS.toSet(), costs.pairMs.keys)
    assertEquals(
      "kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite",
      KevFiles.pair(KevFiles.PAIRS.single()),
    )
    assertEquals("S128+Q64", KevGraphKey.Pair(KevFiles.PAIRS.single()).label)
    assertEquals("L256", KevGraphKey.Window(256).label)
    assertEquals(KevGraphMode.AUTO, KevGraphMode.of(null))
    assertEquals(KevGraphMode.PAIR, KevGraphMode.of("pair"))
    assertEquals(KevGraphMode.ROWS, KevGraphMode.of(" Rows "))
    assertNull(KevGraphMode.of("row"))
    assertEquals(KevPrecision.FP32, KevLaunch.precision(null))
    assertEquals(KevPrecision.FP16_FP32_ACCUM, KevLaunch.precision("fp16acc"))
    assertNull(KevLaunch.precision("fp16"))
    assertEquals("S128Q64", KevLaunch.reportLabel(KevGraphKey.Pair(KevPairShape(128, 64))))
  }

  @Test
  fun theBundledRequestsHaveTheLengthsOfTheTable() {
    // The table above uses the state and branch lengths of the bundled examples and of the
    // five-question request, as the app's encoder gives them.
    val encoder = KevEncoder(ExternalTestData.tokenizer())
    val names = listOf("example_ticket", "example_incident", "example_review")
    for ((name, shape) in names.zip(listOf(TICKET, INCIDENT, REVIEW))) {
      val fixture =
        KevFixture.parse(ExternalTestData.moduleFile("app/src/main/res/raw/$name.json").readText())
      val encoded = encoder.encode(KevRecords.toRecord(fixture.request).first)
      assertEquals(name, shape.state, encoded.stateIds.size)
      assertEquals(name, shape.branches, encoded.branches.map { it.ids.size })
    }
    val asset =
      KevGateChecks.parseAsset(
        ExternalTestData.moduleFile("app/src/debug/assets/gate_fixtures.json").readBytes()
      )
    val five = asset.single { it.id == KevTimingCore.REQUEST_PATH_RECORD }
    val encoded = encoder.encode(KevRecords.toRecord(KevRequest.fromJson(five.request)).first)
    assertEquals(FIVE.state, encoded.stateIds.size)
    assertEquals(FIVE.branches, encoded.branches.map { it.ids.size })
  }

  /** The expected [form] and predictions of a plan. */
  private class Expect(val form: KevForm, val rowsMs: Double?, val pairMs: Double?) {
    fun check(label: String, plan: KevPlan) {
      assertTrue("$label: $plan", plan is KevPlan.Ready)
      val ready = plan as KevPlan.Ready
      assertEquals(label, form, ready.form)
      assertEquals(label, rowsMs, ready.prediction.rowsMs)
      assertEquals(label, pairMs, ready.prediction.pairMs)
    }
  }

  private companion object {
    /** The bundled requests (state and branches as the encoder gives them, checked above). */
    val TICKET = Shape("ticket", 72, listOf(59, 29, 21))
    val INCIDENT = Shape("incident", 95, listOf(53, 27, 33))
    val REVIEW = Shape("review", 72, listOf(52, 34, 22))

    /** The five-question request of the timing runs (own_fiveq_09: rows of 128–142 tokens). */
    val FIVE = Shape("five", 99, listOf(33, 43, 33, 29, 29))

    val NONE = emptyList<KevPairShape>()
    val PAIR = listOf(KevPairShape(128, 64))

    /** The pair's predicted request time for 1, 3 and 5 questions: the state call + the steps. */
    private val COST = KevCosts.GALAXY_S26_GPU_FP32.pairMs.getValue(KevPairShape(128, 64))
    val P1 = COST.stateMs + COST.questionMs
    val P3 = COST.stateMs + 3 * COST.questionMs
    val P5 = COST.stateMs + 5 * COST.questionMs
    const val PLENTY = 8_000_000L * 1024
    const val LOW = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1
  }
}
