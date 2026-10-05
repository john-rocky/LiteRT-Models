package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The plan of a request ([KevPlanner]) with the Galaxy S26 table ([KevCosts]): the row form's time
 * is the sum over the windows the questions are assigned ([KevResidentGraphs.assign]), the pair's
 * the state call plus one question step per question, with or without constant tensor sharing as
 * the pair would compile ([KevPairShare]); the smaller wins, the row form on a tie, and a form that
 * cannot take the request leaves the other one.
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
    costs: KevCosts = KevCosts.GALAXY_S26_GPU,
    share: KevPairShare = KevPairShare.AUTO,
    residentPair: KevPairShape? = null,
    residentPairShared: Boolean? = null,
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
      share,
      residentPair,
      residentPairShared,
    )

  @Test
  fun theTableHasTheNumbersOfTheDecision() {
    // The request times this test expects, from the Galaxy S26 table: rows (L256 + 2 × L128 for the
    // ticket, 3 × L128 for the review, 3 × L256 + 2 × L128 for the five-question request) and
    // pairs.
    val rows = KevCosts.GALAXY_S26_GPU.rowMs
    assertEquals(400.7, rows.getValue(256) + 2 * rows.getValue(128), 1e-9)
    assertEquals(306.6, 3 * rows.getValue(128), 1e-9)
    assertEquals(793.3, 3 * rows.getValue(256) + 2 * rows.getValue(128), 1e-9)
    assertEquals(178.0, LS128_UNSHARED_1.ms, 1e-9)
    assertEquals(302.0, LS128_UNSHARED_3.ms, 1e-9)
    assertEquals(426.0, LS128_UNSHARED_5.ms, 1e-9)
    assertEquals(245.2, LS128_SHARED_1.ms, 1e-9)
    assertEquals(431.0, LS128_SHARED_3.ms, 1e-9)
    assertEquals(616.8, LS128_SHARED_5.ms, 1e-9)
    assertEquals(404.0, LS256_UNSHARED_3.ms, 1e-9)
    assertEquals(540.7, LS256_SHARED_3.ms, 1e-9)
  }

  @Test
  fun theBundledExamplesAndTheLongerRequestsOnTheDefaultInstall() {
    // form, row ms, pair prediction (null: the form cannot take the request).
    val default = KevFiles.DEFAULT_INSTALL
    val l256 = listOf(256)
    val table =
      listOf(
        Case(TICKET, default, NONE, PLENTY) to Expect(KevForm.ROW, 400.7, null),
        Case(TICKET, l256, NONE, PLENTY) to Expect(KevForm.ROW, 588.9, null),
        Case(TICKET, default, LS128, PLENTY) to Expect(KevForm.PAIR, 400.7, LS128_UNSHARED_3),
        Case(TICKET, default, LS128, MID) to Expect(KevForm.ROW, 400.7, LS128_SHARED_3),
        Case(TICKET, l256, LS128, MID) to Expect(KevForm.PAIR, 588.9, LS128_SHARED_3),
        Case(INCIDENT, default, NONE, PLENTY) to Expect(KevForm.ROW, 400.7, null),
        Case(INCIDENT, default, LS128, PLENTY) to Expect(KevForm.PAIR, 400.7, LS128_UNSHARED_3),
        Case(INCIDENT, default, LS128, MID) to Expect(KevForm.ROW, 400.7, LS128_SHARED_3),
        Case(REVIEW, default, NONE, PLENTY) to Expect(KevForm.ROW, 306.6, null),
        Case(REVIEW, default, LS128, PLENTY) to Expect(KevForm.PAIR, 306.6, LS128_UNSHARED_3),
        Case(REVIEW, default, LS128, MID) to Expect(KevForm.ROW, 306.6, LS128_SHARED_3),
        Case(FIVE, default, NONE, PLENTY) to Expect(KevForm.ROW, 793.3, null),
        Case(FIVE, default, LS128, PLENTY) to Expect(KevForm.PAIR, 793.3, LS128_UNSHARED_5),
        Case(FIVE, default, LS128, MID) to Expect(KevForm.PAIR, 793.3, LS128_SHARED_5),
        Case(FIVE, default, LS128, LOW) to Expect(KevForm.PAIR, 981.5, LS128_SHARED_5),
        Case(FIVE, default, BOTH, PLENTY) to Expect(KevForm.PAIR, 793.3, LS128_UNSHARED_5),
        // The three-question email: a state of 167 tokens, over Ls128.
        Case(EMAIL3, default, LS128, PLENTY) to Expect(KevForm.ROW, 588.9, null),
        Case(EMAIL3, default, BOTH, PLENTY) to Expect(KevForm.PAIR, 588.9, LS256_UNSHARED_3),
        Case(EMAIL3, default, BOTH, MID) to Expect(KevForm.PAIR, 588.9, LS256_SHARED_3),
      )
    for ((case, expect) in table) {
      val label = "${case.request.name} ${case.windows} ${case.pairs} ${case.available}"
      expect.check(label, plan(case.request, case.windows, case.pairs, case.available))
    }
  }

  @Test
  fun theTicketTakesThePairWithoutSharingAndTheRowsAgainstAPairThatShares() {
    val default = KevFiles.DEFAULT_INSTALL
    // At least 6,500,000 kB available: the pair compiles without sharing, 302 ms < the rows' 400.7
    // ms (L256 196.3 + L128 102.2 + L128 102.2).
    Expect(KevForm.PAIR, 400.7, LS128_UNSHARED_3)
      .check("plenty", plan(TICKET, default, LS128, PLENTY))
    // Under that: the pair would share (431 ms), the two row windows win.
    Expect(KevForm.ROW, 400.7, LS128_SHARED_3).check("mid", plan(TICKET, default, LS128, MID))
    // Under the second-window limit with nothing resident: L256 takes the three rows (588.9 ms).
    Expect(KevForm.PAIR, 588.9, LS128_SHARED_3).check("low", plan(TICKET, default, LS128, LOW))
    // Both windows already resident: no compile is needed, so the memory does not matter.
    Expect(KevForm.ROW, 400.7, LS128_SHARED_3)
      .check("resident", plan(TICKET, default, LS128, LOW, resident = listOf(128, 256)))
    // One of them resident is not enough.
    Expect(KevForm.PAIR, 588.9, LS128_SHARED_3)
      .check("one resident", plan(TICKET, default, LS128, LOW, resident = listOf(128)))
    // A resident pair is predicted as it was compiled, whatever the memory now.
    Expect(KevForm.ROW, 400.7, LS128_SHARED_3)
      .check(
        "resident shared",
        plan(TICKET, default, LS128, PLENTY, residentPair = PAIR_128, residentPairShared = true),
      )
    Expect(KevForm.PAIR, 400.7, LS128_UNSHARED_3)
      .check(
        "resident unshared",
        plan(TICKET, default, LS128, MID, residentPair = PAIR_128, residentPairShared = false),
      )
    // The share extra decides instead of the memory.
    Expect(KevForm.ROW, 400.7, LS128_SHARED_3)
      .check("on", plan(TICKET, default, LS128, PLENTY, share = KevPairShare.ON))
    Expect(KevForm.PAIR, 400.7, LS128_UNSHARED_3)
      .check("off", plan(TICKET, default, LS128, MID, share = KevPairShare.OFF))
  }

  @Test
  fun autoSharesBelowTheLimitOnly() {
    val limit = KevResidentGraphs.PAIR_UNSHARED_MIN_AVAILABLE_BYTES
    assertEquals(6_500_000L * 1024, limit)
    assertFalse(KevPairShare.AUTO.sharesAt(limit))
    assertTrue(KevPairShare.AUTO.sharesAt(limit - 1))
    assertTrue(KevPairShare.ON.sharesAt(PLENTY))
    assertFalse(KevPairShare.OFF.sharesAt(LOW))
    assertTrue(MID < limit && MID >= KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES)
  }

  @Test
  fun requestsThePairCannotTakeRunOnRows() {
    val all = listOf(128, 256, 512)
    // State part of 129 tokens: over Ls.
    Expect(KevForm.ROW, 196.3, null)
      .check("state 129", plan(Shape("s", 129, listOf(20)), all, LS128))
    // A branch of 65 tokens: over Lq (rows of 115 and 70 tokens, both on L128).
    Expect(KevForm.ROW, 204.4, null)
      .check("branch 65", plan(Shape("b", 50, listOf(65, 20)), all, LS128))
  }

  @Test
  fun oneQuestionRunsOnItsRowUnlessItNeedsL256AndThePairNeedNotShare() {
    val default = KevFiles.DEFAULT_INSTALL
    // A row of 94 tokens: L128 102.2 ms against the pair's state + one step.
    Expect(KevForm.ROW, 102.2, LS128_UNSHARED_1)
      .check("L128", plan(Shape("q", 30, listOf(64)), default, LS128, PLENTY))
    Expect(KevForm.ROW, 102.2, LS128_SHARED_1)
      .check("L128 mid", plan(Shape("q", 30, listOf(64)), default, LS128, MID))
    // A row of 160 tokens: L256 196.3 ms; the pair without sharing is 178 ms, with it 245.2 ms.
    Expect(KevForm.PAIR, 196.3, LS128_UNSHARED_1)
      .check("L256", plan(Shape("q", 100, listOf(60)), default, LS128, PLENTY))
    Expect(KevForm.ROW, 196.3, LS128_SHARED_1)
      .check("L256 mid", plan(Shape("q", 100, listOf(60)), default, LS128, MID))
  }

  @Test
  fun thePairAloneTakesWhatNoWindowHolds() {
    // No row window installed at all.
    Expect(KevForm.PAIR, null, LS128_UNSHARED_3)
      .check("no window", plan(TICKET, emptyList(), LS128))
    // L128 only: the five-question rows (128–142 tokens) need L256.
    Expect(KevForm.PAIR, null, LS128_UNSHARED_5).check("L128 only", plan(FIVE, listOf(128), LS128))
    // Neither form: the window that would hold the longest row.
    val none = plan(FIVE, listOf(128))
    assertTrue(none is KevPlan.NoWindow)
    assertEquals(142, (none as KevPlan.NoWindow).missing.rowTokens)
    assertEquals(256, none.missing.window)
  }

  @Test
  fun rowsOfSixtyFourTokensOrFewerRunOnL64() {
    val all = KevFiles.WINDOWS
    val ms = KevCosts.GALAXY_S26_GPU.rowMs
    // One question whose row (state + branch) is 60 tokens.
    Expect(KevForm.ROW, ms.getValue(64), null).check("L64", plan(Shape("q", 30, listOf(30)), all))
    // Without L64 installed, the same row runs on L128.
    Expect(KevForm.ROW, ms.getValue(128), null)
      .check("no L64", plan(Shape("q", 30, listOf(30)), all - 64))
    // Rows of 60 and 100 tokens: L64 and L128, one question each.
    Expect(KevForm.ROW, ms.getValue(64) + ms.getValue(128), null)
      .check("L64 + L128", plan(Shape("q", 30, listOf(30, 70)), all))
    // Rows of 60, 100 and 200 tokens: two graphs at most, so L128 also takes the L64 row.
    Expect(KevForm.ROW, 2 * ms.getValue(128) + ms.getValue(256), null)
      .check("three windows", plan(Shape("q", 30, listOf(30, 70, 170)), all))
    // The same with memory under the limit and nothing resident: L256 takes every row.
    Expect(KevForm.ROW, 3 * ms.getValue(256), null)
      .check("three windows, low", plan(Shape("q", 30, listOf(30, 70, 170)), all, available = LOW))
    // L128 and L256 resident: the plan they cover needs no memory.
    Expect(KevForm.ROW, 2 * ms.getValue(128) + ms.getValue(256), null)
      .check(
        "three windows, resident",
        plan(
          Shape("q", 30, listOf(30, 70, 170)),
          all,
          available = LOW,
          resident = listOf(128, 256),
        ),
      )
  }

  @Test
  fun theLs256PairTakesStatesOver128Tokens() {
    // Both pairs follow one contract; Ls256 holds 256 state tokens.
    assertEquals(listOf(PAIR_128, PAIR_256), KevFiles.PAIRS)
    assertEquals("kev-0.8b_sharedstate_Ls256_Lq64_fp16fc_i8emb.tflite", KevFiles.pair(PAIR_256))
    assertEquals("S256+Q64", KevGraphKey.Pair(PAIR_256).label)
    assertEquals("state_prefill_256", KevPairContract.stateSignature(PAIR_256))
    assertEquals("question_step_256_64", KevPairContract.questionSignature(PAIR_256))
    assertEquals(listOf(1, 2, 256, 256), KevPairContract.stateShape("k_3", 256))
    assertEquals(listOf(1, 2, 256, 256), KevPairContract.stateShape("v_23", 256))
    assertEquals(listOf(1, 16, 128, 128), KevPairContract.stateShape("gdn_state_0", 256))
    assertEquals(listOf(1, 3, 6144), KevPairContract.stateShape("conv_tail_22", 256))
    // Both installed: a state of 128 tokens takes Ls128 (302 ms < Ls256's 404 ms), 129 tokens and
    // more take Ls256, and 257 tokens neither.
    val default = KevFiles.DEFAULT_INSTALL
    for ((state, shape) in listOf(72 to PAIR_128, 128 to PAIR_128, 129 to PAIR_256)) {
      val chosen =
        plan(Shape("s", state, listOf(30, 30, 30)), default, BOTH, mode = KevGraphMode.PAIR)
      assertEquals("state $state", shape, (chosen as KevPlan.Pair).shape)
    }
    // A state of 256 tokens: its rows (286 tokens) have no default-install window.
    Expect(KevForm.PAIR, null, LS256_UNSHARED_3)
      .check("state 256", plan(Shape("s", 256, listOf(30, 30, 30)), default, BOTH))
    val over =
      plan(Shape("s", 257, listOf(30)), default, BOTH, mode = KevGraphMode.PAIR) as KevPlan.NoPair
    assertEquals(KevPairMiss.StateTooLong(257, 256), over.miss)
    // Only Ls128 installed: a state of 150 tokens is over Ls.
    val miss =
      plan(Shape("s", 150, listOf(30)), default, LS128, mode = KevGraphMode.PAIR) as KevPlan.NoPair
    assertEquals(KevPairMiss.StateTooLong(150, 128), miss.miss)
  }

  @Test
  fun aTieGoesToTheRows() {
    val costs =
      KevCosts(
        rowMs = mapOf(256 to 200.0),
        pairMs = mapOf(PAIR_128 to KevPairCost(stateMs = 100.0, questionMs = 150.0)),
      )
    val tie = Shape("tie", 50, listOf(10, 10))
    val ready = plan(tie, listOf(256), LS128, costs = costs) as KevPlan.Ready
    assertEquals(KevForm.ROW, ready.form)
    assertEquals(400.0, ready.prediction.rowsMs!!, 1e-9)
    assertEquals(400.0, ready.prediction.pairMs!!, 1e-9)
  }

  @Test
  fun aNamedModeOrWindowIsFollowed() {
    val default = KevFiles.DEFAULT_INSTALL
    // rows: even when the pair would be faster.
    Expect(KevForm.ROW, 793.3, LS128_UNSHARED_5)
      .check("rows", plan(FIVE, default, LS128, mode = KevGraphMode.ROWS))
    // pair: even when the rows would be faster.
    Expect(KevForm.PAIR, 400.7, LS128_SHARED_3)
      .check("pair", plan(TICKET, default, LS128, MID, mode = KevGraphMode.PAIR))
    // A named window: every row on it.
    val fixed = plan(TICKET, listOf(128, 256, 512), LS128, fixedWindow = 512)
    Expect(KevForm.ROW, 3 * 388.8, LS128_UNSHARED_3).check("L512", fixed)
    assertEquals(listOf(512, 512, 512), (fixed as KevPlan.Rows).windows.windows)
    // The pair asked for and impossible: the reason.
    val state =
      plan(Shape("s", 129, listOf(20)), default, LS128, mode = KevGraphMode.PAIR) as KevPlan.NoPair
    assertEquals(KevPairMiss.StateTooLong(129, 128), state.miss)
    val branch =
      plan(Shape("b", 50, listOf(20, 65)), default, LS128, mode = KevGraphMode.PAIR)
        as KevPlan.NoPair
    assertEquals(KevPairMiss.QuestionTooLong(1, 65, 64), branch.miss)
    val missing = plan(TICKET, default, NONE, mode = KevGraphMode.PAIR) as KevPlan.NoPair
    assertEquals(KevPairMiss.NotInstalled, missing.miss)
  }

  @Test
  fun theTableCoversEveryPublishedGraphAndNamesTheModes() {
    val costs = KevCosts.GALAXY_S26_GPU
    assertEquals(KevFiles.WINDOWS.toSet(), costs.rowMs.keys)
    assertEquals(KevFiles.PAIRS.toSet(), costs.pairMs.keys)
    assertEquals(KevFiles.PAIRS.toSet(), costs.unsharedPairMs.keys)
    assertEquals("kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite", KevFiles.pair(PAIR_128))
    assertEquals("S128+Q64", KevGraphKey.Pair(PAIR_128).label)
    assertEquals("L256", KevGraphKey.Window(256).label)
    assertEquals(KevGraphMode.AUTO, KevGraphMode.of(null))
    assertEquals(KevGraphMode.PAIR, KevGraphMode.of("pair"))
    assertEquals(KevGraphMode.ROWS, KevGraphMode.of(" Rows "))
    assertNull(KevGraphMode.of("row"))
    assertEquals(KevPrecision.FP32, KevLaunch.precision(" FP32 "))
    assertEquals(KevPrecision.FP16_FP32_ACCUM, KevLaunch.precision("fp16acc"))
    assertNull(KevLaunch.precision("fp16"))
    assertEquals("S128Q64", KevLaunch.reportLabel(KevGraphKey.Pair(PAIR_128)))
  }

  @Test
  fun theBundledRequestsHaveTheLengthsOfTheTable() {
    // The table above uses the state and branch lengths of the bundled examples and of the five-
    // and three-question requests of the timing runs, as the app's encoder gives them.
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
    for ((id, shape) in
      listOf(KevTimingCore.REQUEST_PATH_RECORD to FIVE, "own_email_03" to EMAIL3)) {
      val record = asset.single { it.id == id }
      val encoded = encoder.encode(KevRecords.toRecord(KevRequest.fromJson(record.request)).first)
      assertEquals(id, shape.state, encoded.stateIds.size)
      assertEquals(id, shape.branches, encoded.branches.map { it.ids.size })
    }
  }

  /** One planner input of the table: a request on [windows] and [pairs] with [available] bytes. */
  private class Case(
    val request: Shape,
    val windows: List<Int>,
    val pairs: List<KevPairShape>,
    val available: Long,
  )

  /** A pair's predicted request time: [ms] on [shape], with constant tensor sharing or not. */
  private class PairMs(val shape: KevPairShape, val shared: Boolean, val ms: Double)

  /**
   * The expected [form] and predictions of a plan; with a [pair], also the sharing its prediction
   * assumed and, for a pair plan, the shape chosen.
   */
  private class Expect(val form: KevForm, val rowsMs: Double?, val pair: PairMs?) {
    fun check(label: String, plan: KevPlan) {
      assertTrue("$label: $plan", plan is KevPlan.Ready)
      val ready = plan as KevPlan.Ready
      assertEquals(label, form, ready.form)
      close(label, rowsMs, ready.prediction.rowsMs)
      close(label, pair?.ms, ready.prediction.pairMs)
      assertEquals(label, pair?.shared, ready.prediction.pairShared)
      if (ready is KevPlan.Pair) assertEquals(label, pair?.shape, ready.shape)
    }

    /** Equal, or both numbers within 1e-9 ms (sums of the table's decimals). */
    private fun close(label: String, expected: Double?, actual: Double?) {
      if (expected == null || actual == null) assertEquals(label, expected, actual)
      else assertEquals(label, expected, actual, 1e-9)
    }
  }

  private companion object {
    /** The bundled requests (state and branches as the encoder gives them, checked above). */
    val TICKET = Shape("ticket", 72, listOf(59, 29, 21))
    val INCIDENT = Shape("incident", 95, listOf(53, 27, 33))
    val REVIEW = Shape("review", 72, listOf(52, 34, 22))

    /** The five-question request of the timing runs (own_fiveq_09: rows of 128–142 tokens). */
    val FIVE = Shape("five", 99, listOf(33, 43, 33, 29, 29))

    /** The three-question request of the Ls256 timing runs (own_email_03: rows of 198–226). */
    val EMAIL3 = Shape("email3", 167, listOf(59, 31, 54))

    val PAIR_128 = KevPairShape(128, 64)
    val PAIR_256 = KevPairShape(256, 64)
    val NONE = emptyList<KevPairShape>()
    val LS128 = listOf(PAIR_128)
    val BOTH = listOf(PAIR_128, PAIR_256)

    /** The pairs' predicted request times: the state call + one step per question. */
    private fun pairMs(shape: KevPairShape, shared: Boolean, questions: Int): PairMs {
      val costs = KevCosts.GALAXY_S26_GPU
      val cost = (if (shared) costs.pairMs else costs.unsharedPairMs).getValue(shape)
      return PairMs(shape, shared, cost.stateMs + questions * cost.questionMs)
    }

    val LS128_UNSHARED_1 = pairMs(PAIR_128, false, 1)
    val LS128_UNSHARED_3 = pairMs(PAIR_128, false, 3)
    val LS128_UNSHARED_5 = pairMs(PAIR_128, false, 5)
    val LS128_SHARED_1 = pairMs(PAIR_128, true, 1)
    val LS128_SHARED_3 = pairMs(PAIR_128, true, 3)
    val LS128_SHARED_5 = pairMs(PAIR_128, true, 5)
    val LS256_UNSHARED_3 = pairMs(PAIR_256, false, 3)
    val LS256_SHARED_3 = pairMs(PAIR_256, true, 3)

    /** At the unshared-pair limit and over: the pair compiles without sharing. */
    const val PLENTY = 8_000_000L * 1024

    /** Under the unshared-pair limit, over the second-window limit. */
    const val MID = KevResidentGraphs.PAIR_UNSHARED_MIN_AVAILABLE_BYTES - 1

    /** Under the second-window limit. */
    const val LOW = KevResidentGraphs.SECOND_RESIDENT_MIN_AVAILABLE_BYTES - 1
  }
}
