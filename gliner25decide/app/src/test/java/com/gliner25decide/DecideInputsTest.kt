package com.gliner25decide

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

/**
 * input_ids, attention length, `[P]`/`[L]` positions and padded graph inputs of every oracle
 * fixture against the captured official gliner2 2.0.0 batch.
 */
class DecideInputsTest {
  private lateinit var root: File

  @Before
  fun locateExternalData() {
    root = ExternalTestData.resolve()
  }

  @Test
  fun capturedInputsMatchForEveryOracleFixture() {
    val fixtures = OracleFixtures.load(root)
    assertEquals("oracle fixtures", 361, fixtures.size)
    val inputs = DecideInputs(GlinerTokenizer(ExternalTestData.tokenizer(root)))
    val failures = JSONArray()
    val windowCounts = linkedMapOf(128 to 0, 256 to 0, 512 to 0)
    var passed = 0
    var paddedPairs = 0
    for (fixture in fixtures) {
      val problems = JSONObject()
      val encoded =
        try {
          inputs.encode(fixture.text, fixture.tasks)
        } catch (failure: IllegalArgumentException) {
          failures.put(JSONObject().put("id", fixture.id).put("error", failure.message))
          continue
        }
      OracleFixtures.firstDifference(fixture.inputIds, encoded.inputIds)?.let { index ->
        problems.put(
          "input_ids",
          JSONObject()
            .put("first_token_index", index)
            .put("expected_id", fixture.inputIds.getOrNull(index) ?: JSONObject.NULL)
            .put("actual_id", encoded.inputIds.getOrNull(index) ?: JSONObject.NULL)
            .put("expected_length", fixture.inputIds.size)
            .put("actual_length", encoded.inputIds.size),
        )
      }
      if (encoded.encodedLength != fixture.encodedLength) {
        problems.put("attention_length", "${encoded.encodedLength} != ${fixture.encodedLength}")
      }
      OracleFixtures.firstDifference(fixture.labelPositions, encoded.labelPositions)?.let {
        problems.put("label_positions_first_difference", it)
      }
      val special =
        fixture.schemaSpecialIndices.size == encoded.schemaSpecialPositions.size &&
          fixture.schemaSpecialIndices.zip(encoded.schemaSpecialPositions).all { (a, b) ->
            a.contentEquals(b)
          }
      if (!special) {
        problems.put("schema_special_indices", "differ")
      }
      val fitting = DecideInputs.WINDOWS.filter { fixture.fits.getValue(it) }
      val smallest = runCatching { inputs.pad(encoded).window }.getOrNull()
      if (smallest != fitting.firstOrNull()) {
        problems.put("smallest_window", "$smallest != ${fitting.firstOrNull()}")
      }
      for (window in fitting) {
        val prepared = inputs.pad(encoded, window)
        if (!paddedMatches(fixture, prepared)) {
          problems.put("padded_s$window", "differs")
        }
        paddedPairs++
      }
      if (problems.length() == 0) {
        passed++
        fitting.firstOrNull()?.let { windowCounts[it] = windowCounts.getValue(it) + 1 }
      } else {
        failures.put(problems.put("id", fixture.id))
      }
    }
    val report =
      JSONObject()
        .put("test", "DecideInputsTest")
        .put("fixtures", fixtures.size)
        .put("passed", passed)
        .put("failed", fixtures.size - passed)
        .put("padded_window_pairs_checked", paddedPairs)
        .put("smallest_window_counts", JSONObject(windowCounts.mapKeys { "s${it.key}" }))
        .put(
          "compared",
          JSONArray(
            listOf(
              "input_ids (unpadded, byte-exact)",
              "attention length",
              "label_positions",
              "schema_special_indices",
              "smallest fitting window",
              "padded ids/attention/label_routing at every fitting window",
            )
          ),
        )
        .put("failures", failures)
    ExternalTestData.reportFile("decide_inputs_parity.json").writeText(report.toString(2) + "\n")
    println(
      "DECIDE_INPUTS passed=$passed/${fixtures.size} padded_pairs=$paddedPairs failures=$failures"
    )
    assertEquals("fixtures whose inputs match the captured batch", fixtures.size, passed)
  }

  private fun paddedMatches(
    fixture: OracleFixtures.Fixture,
    prepared: DecideInputs.Prepared,
  ): Boolean {
    val n = prepared.window
    val length = fixture.inputIds.size
    for (index in 0 until n) {
      val expectedId = if (index < length) fixture.inputIds[index] else 0
      val expectedAttention = if (index < length) 1f else 0f
      if (
        prepared.inputIds[index] != expectedId || prepared.attentionMask[index] != expectedAttention
      ) {
        return false
      }
    }
    val routing = FloatArray(DecideInputs.LABEL_SLOTS * n)
    fixture.labelPositions.forEachIndexed { row, position -> routing[row * n + position] = 1f }
    return routing.contentEquals(prepared.labelRouting)
  }
}
