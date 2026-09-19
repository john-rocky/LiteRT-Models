package com.gliner25

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class GlinerGateFixturesTest {
  private lateinit var root: File

  @Before
  fun locateExternalData() {
    root = ExternalTestData.resolve()
  }

  private val module = File(requireNotNull(System.getProperty("gliner.moduleRoot")))

  @Test
  fun f1AssetMatchesEveryCapturedInputAndOfficialSpan() {
    ExternalTestData.requireFiles(root, "fixtures/f1/oracle_fp32.json", "fixtures/f1/captured")
    val rows = asset().getJSONArray("fixtures")
    val oracle =
      JSONObject(File(root, "fixtures/f1/oracle_fp32.json").readText()).getJSONArray("fixtures")
    val inputs = GlinerInputs(GlinerTokenizer(File(root, "host_assets/tokenizer.json")))
    val texts = mutableSetOf<String>()
    val windows = mutableMapOf<Int, Int>()
    assertEquals(70, rows.length())
    for (index in 0 until rows.length()) {
      val row = rows.getJSONObject(index)
      val captured =
        JSONObject(File(root, "fixtures/f1/captured/%02d.json".format(index)).readText())
      val expected = oracle.getJSONObject(index)
      val text = GlinerGateFixtures.text(row)
      assertEquals(index, row.getInt("index"))
      assertEquals(captured.getString("text"), text)
      assertEquals(expected.getString("text"), text)
      texts.add(text)
      for (field in listOf("input_ids", "text_word_indices", "query_marker_indices")) {
        assertEquals(
          captured.getJSONObject(field).getJSONArray("values").getJSONArray(0).toString(),
          row.getJSONArray(field).toString(),
        )
      }
      val observed = inputs.prepare(text)
      assertTrue(
        "Fixture $index",
        GlinerGateFixtures.compareInputs(row, observed).getBoolean("identical"),
      )
      windows[observed.window.sequenceLength] =
        windows.getOrDefault(observed.window.sequenceLength, 0) + 1
      val spans = row.getJSONArray("spans")
      val expectedSpans = expected.getJSONArray("spans")
      assertEquals(expectedSpans.length(), spans.length())
      for (span in 0 until spans.length()) {
        for (field in listOf("label", "start", "end", "confidence")) {
          assertEquals(
            expectedSpans.getJSONObject(span).get(field),
            spans.getJSONObject(span).get(field),
          )
        }
      }
    }
    assertEquals(70, texts.size)
    assertEquals(mapOf(128 to 60, 256 to 5, 512 to 5), windows)
    ExternalTestData.reportFile("f1_asset_parity.json")
      .writeText(
        JSONObject()
          .put("status", "PASS")
          .put("unique_inputs", texts.size)
          .put("captured_inputs_identical", 70)
          .put("official_spans_identical", 70)
          .put("smallest_window_counts", JSONObject(windows.mapKeys { "s${it.key}" }))
          .put("execution", "Desktop JVM; Android validation NOT RUN")
          .toString(2) + "\n"
      )
  }

  @Test
  fun inputComparisonReportsTheFirstDifferenceForEveryCapturedField() {
    val fixture = asset().getJSONArray("fixtures").getJSONObject(0)
    val inputs = GlinerInputs(GlinerTokenizer(File(root, "host_assets/tokenizer.json")))
    val prepared = inputs.prepare(GlinerGateFixtures.text(fixture))
    for (field in listOf("input_ids", "text_word_indices", "query_marker_indices")) {
      val altered = JSONObject(fixture.toString())
      val array = altered.getJSONArray(field)
      array.put(0, array.getInt(0) + 1)
      val comparison = GlinerGateFixtures.compareInputs(altered, prepared)
      assertFalse(comparison.getBoolean("identical"))
      assertFalse(comparison.getJSONObject(field).getBoolean("identical"))
      assertEquals(0, comparison.getJSONObject(field).getInt("first_difference"))
    }
  }

  private fun asset() =
    JSONObject(File(module, "app/src/debug/assets/gate_f1_fixtures.json").readText())
}
