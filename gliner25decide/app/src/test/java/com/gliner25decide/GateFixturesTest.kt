package com.gliner25decide

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The debug gate asset against the oracle, and the gate's own comparison helpers. */
class GateFixturesTest {
  private fun asset() =
    DecideGateFixtures.parse(
      JSONObject(ExternalTestData.moduleFile("app/src/debug/assets/gate_fixtures.json").readText())
    )

  @Test
  fun assetMatchesTheOracleAndTheKotlinHost() {
    val root = ExternalTestData.resolve()
    val oracle = OracleFixtures.load(root).associateBy { it.id }
    val inputs = DecideInputs(GlinerTokenizer(ExternalTestData.tokenizer(root)))
    val fixtures = asset()
    assertEquals(84, fixtures.size)
    assertEquals(
      mapOf(128 to 42, 256 to 42, 512 to 42),
      DecideInputs.WINDOWS.associateWith { window -> fixtures.count { window in it.windows } },
    )
    assertTrue(fixtures.any { it.id.startsWith("readme_18") })
    assertTrue(fixtures.any { it.id.startsWith("readme_20") })
    var pairs = 0
    for (fixture in fixtures) {
      val expected = requireNotNull(oracle[fixture.id])
      assertEquals(expected.text, fixture.text)
      assertEquals(expected.tasks, fixture.tasks)
      assertEquals(expected.official, fixture.official)
      assertArrayEquals(expected.inputIds, fixture.inputIds)
      assertArrayEquals(expected.labelPositions, fixture.labelPositions)
      assertArrayEquals(expected.logits, fixture.oracleLogits, 0f)
      assertArrayEquals(expected.probabilities, fixture.oracleProbabilities, 0f)
      for (window in fixture.windows) {
        val comparison =
          DecideGateFixtures.compareInputs(
            fixture,
            inputs.prepare(fixture.text, fixture.tasks, window),
          )
        assertTrue("${fixture.id} s$window $comparison", comparison.getBoolean("identical"))
        assertEquals(fixture.oracleLogits.size, fixture.macCpuLogits.getValue(window).size)
        pairs++
      }
      val slots = FloatArray(DecideInputs.LABEL_SLOTS)
      fixture.oracleLogits.copyInto(slots)
      assertTrue(
        fixture.id,
        DecideGateFixtures.decisionsEqual(
          DecideDecoder.decode(slots, fixture.tasks),
          fixture.official,
        ),
      )
    }
    assertEquals(DecideGateRunner.EXPECTED_PAIRS, pairs)
    ExternalTestData.reportFile("gate_asset_parity.json")
      .writeText(
        JSONObject()
          .put("test", "GateFixturesTest")
          .put("status", "PASS")
          .put("fixtures", fixtures.size)
          .put("window_pairs_inputs_identical", pairs)
          .put("oracle_fields_identical", fixtures.size)
          .put("execution", "Desktop JVM; Android validation NOT RUN")
          .toString(2) + "\n"
      )
  }

  @Test
  fun inputComparisonReportsTheFirstDifference() {
    val root = ExternalTestData.resolve()
    val inputs = DecideInputs(GlinerTokenizer(ExternalTestData.tokenizer(root)))
    val fixture = asset().first()
    val prepared = inputs.prepare(fixture.text, fixture.tasks, fixture.windows.first())
    assertTrue(DecideGateFixtures.compareInputs(fixture, prepared).getBoolean("identical"))
    val ids = fixture.inputIds.copyOf().also { it[5] += 1 }
    val altered = DecideGateFixtures.compareInputs(fixture.copy(inputIds = ids), prepared)
    assertFalse(altered.getBoolean("identical"))
    assertEquals(5, altered.getJSONObject("input_ids").getInt("first_difference"))
    val positions = fixture.labelPositions.copyOf().also { it[0] += 1 }
    val shifted =
      DecideGateFixtures.compareInputs(fixture.copy(labelPositions = positions), prepared)
    assertEquals(0, shifted.getJSONObject("label_positions").getInt("first_difference"))
  }
}
