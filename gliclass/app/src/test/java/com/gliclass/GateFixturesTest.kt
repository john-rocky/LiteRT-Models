package com.gliclass

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The debug gate asset against the oracle, and the gate's own input comparison. Works for the
 * committed 152-request subset and for the full 552-request asset the conversion run regenerates.
 */
class GateFixturesTest {
  private fun assetJson() =
    JSONObject(
      ExternalTestData.moduleFile("app/src/debug/assets/${GliclassGateFixtures.ASSET_NAME}")
        .readText()
    )

  private fun asset() = GliclassGateFixtures.parse(assetJson())

  @Test
  fun assetMatchesTheOracleAndTheKotlinHost() {
    val root = ExternalTestData.resolve()
    ExternalTestData.requireFiles(root, "fixtures/tokenizer_stress.json")
    val oracle = OracleFixtures.load(root).associateBy { it.id }
    val json = assetJson()
    // parse() also checks the row count the asset declares.
    val fixtures = GliclassGateFixtures.parse(json)
    assertEquals("one row per request id", fixtures.size, fixtures.map { it.id }.toSet().size)
    GliclassInputs.EmbeddingTable(ExternalTestData.embeddingTable(root)).use { table ->
      val inputs = GliclassInputs(GliclassTokenizer(ExternalTestData.tokenizer(root)), table)
      for (fixture in fixtures) {
        val expected = requireNotNull(oracle[fixture.id])
        assertEquals(expected.text, fixture.text)
        assertEquals(expected.prompt, fixture.prompt)
        assertEquals(expected.labels, fixture.labels)
        assertArrayEquals(expected.inputIds, fixture.inputIds)
        assertArrayEquals(expected.labelPositions, fixture.labelPositions)
        assertEquals(expected.window, fixture.window)
        assertArrayEquals(expected.logits, fixture.oracleLogits, 0f)
        assertEquals(expected.single.map { it.label }, fixture.officialSingle)
        assertEquals(expected.multi.map { it.label }, fixture.officialMulti)
        val comparison =
          GliclassGateFixtures.compareInputs(
            fixture,
            inputs.prepare(fixture.text, fixture.labels, fixture.prompt),
            PAD_ID,
          )
        assertTrue("${fixture.id} $comparison", comparison.getBoolean("identical"))
      }
    }
    val windows = json.getJSONObject("windows")
    val counts = GliclassInputs.WINDOWS.associateWith { w -> fixtures.count { it.window == w } }
    for ((window, count) in counts) {
      assertEquals("rows at s$window", windows.optInt("s$window"), count)
    }
    val sources = fixtures.groupingBy { it.source }.eachCount().toSortedMap()
    ExternalTestData.reportFile("gate_asset_parity.json")
      .writeText(
        JSONObject()
          .put("test", "GateFixturesTest")
          .put("status", "PASS")
          .put("fixtures", fixtures.size)
          .put("inputs_identical", fixtures.size)
          .put("windows", JSONObject(counts.mapKeys { "s${it.key}" }))
          .put("by_source", JSONObject(sources))
          .put("execution", "Desktop JVM; Android validation separate")
          .toString(1) + "\n"
      )
    println("GLICLASS_GATE_ASSET rows=${fixtures.size} inputs_identical=${fixtures.size} $sources")
  }

  @Test
  fun inputComparisonReportsTheFirstDifference() {
    val root = ExternalTestData.resolve()
    GliclassInputs.EmbeddingTable(ExternalTestData.embeddingTable(root)).use { table ->
      val inputs = GliclassInputs(GliclassTokenizer(ExternalTestData.tokenizer(root)), table)
      val fixture = asset().first()
      val prepared = inputs.prepare(fixture.text, fixture.labels, fixture.prompt)
      assertTrue(
        GliclassGateFixtures.compareInputs(fixture, prepared, PAD_ID).getBoolean("identical")
      )
      val ids = fixture.inputIds.copyOf().also { it[5] += 1 }
      val altered =
        GliclassGateFixtures.compareInputs(fixture.copy(inputIds = ids), prepared, PAD_ID)
      assertFalse(altered.getBoolean("identical"))
      assertEquals(5, altered.getJSONObject("input_ids").getInt("first_difference"))
      val wrongWindow =
        GliclassGateFixtures.compareInputs(fixture.copy(window = 256), prepared, PAD_ID)
      assertFalse(wrongWindow.getBoolean("identical"))
    }
  }

  @Test
  fun bundledExampleParsesAndFitsTheSmallWindow() {
    assertEquals(listOf("a", "b c"), LabelEditor.parse(" a , b c ,, "))
    assertEquals(listOf("a, b", "c"), LabelEditor.parse("a, b\n\n c "))
    // The prefilled request of strings.xml is the invented oracle request inv_02.
    val strings = ExternalTestData.moduleFile("app/src/main/res/values/strings.xml").readText()
    val text = resourceString(strings, "example_text")
    val labels = LabelEditor.parse(resourceString(strings, "example_labels").replace("\\n", "\n"))
    val root = ExternalTestData.resolve()
    val oracle = OracleFixtures.load(root).first { it.id == "inv_02" }
    assertEquals(oracle.text, text)
    assertEquals(oracle.labels, labels)
    assertEquals(null, oracle.prompt)
    GliclassInputs.EmbeddingTable(ExternalTestData.embeddingTable(root)).use { table ->
      val inputs = GliclassInputs(GliclassTokenizer(ExternalTestData.tokenizer(root)), table)
      val prepared = inputs.prepare(text, labels, null)
      assertEquals(128, prepared.window)
      assertArrayEquals(oracle.inputIds, prepared.encoded.inputIds)
    }
  }

  private fun resourceString(xml: String, name: String): String =
    requireNotNull(Regex("<string name=\"$name\">(.*?)</string>").find(xml)) { name }.groupValues[1]

  private companion object {
    const val PAD_ID = 50283
  }
}
