package com.gliclass

import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject

/**
 * Pure JVM parsing and comparisons shared by the debug gate and its JVM tests. Every ordered
 * structure in the asset (labels, the official results) is a JSON array because org.json on the
 * desktop JVM does not keep object key order.
 */
internal object GliclassGateFixtures {
  /** Logcat tag of every debug gate, first-request and paced report line. */
  const val GATE_LOG_TAG = "GLICLASS_GATE"

  /** Indentation of the JSON reports written to `files/`. */
  const val REPORT_JSON_INDENT = 1

  /** The debug asset with the captured official requests. */
  const val ASSET_NAME = "gate_fixtures.json"

  /** One captured official request: inputs, raw logits and both pipeline results. */
  data class Fixture(
    val id: String,
    val source: String,
    val text: String,
    val prompt: String?,
    val labels: List<String>,
    val inputIds: IntArray,
    val labelPositions: IntArray,
    val window: Int,
    val oracleLogits: FloatArray,
    val officialSingle: List<String>,
    val officialMulti: List<String>,
  )

  /**
   * Parses every fixture of the debug asset. The asset declares its own row count, so a truncated
   * or hand-edited file fails here instead of passing as a smaller gate.
   */
  fun parse(root: JSONObject): List<Fixture> {
    val fixtures = root.getJSONArray("fixtures")
    val parsed = (0 until fixtures.length()).map { parseFixture(fixtures.getJSONObject(it)) }
    require(parsed.size == root.getInt("rows")) {
      "$ASSET_NAME declares ${root.getInt("rows")} rows but holds ${parsed.size}"
    }
    return parsed
  }

  private fun parseFixture(json: JSONObject): Fixture =
    Fixture(
      json.getString("id"),
      json.getString("source"),
      json.getString("text"),
      if (json.isNull("prompt")) null else json.getString("prompt"),
      json.getJSONArray("labels").strings(),
      json.getJSONArray("ids").ints(),
      json.getJSONArray("label_positions").ints(),
      json.getInt("window"),
      json.getJSONArray("logits").floats(),
      json.getJSONArray("pipeline_single").predictionLabels(),
      json.getJSONArray("pipeline_multi").predictionLabels(),
    )

  /**
   * Exact comparison of the host input path against the captured Python inputs: unpadded IDs,
   * `<<LABEL>>` positions and window, plus PAD IDs, attention and routing of the padded window.
   * Reports the first differing index per field.
   */
  fun compareInputs(fixture: Fixture, actual: GliclassInputs.Prepared, padId: Int): JSONObject {
    val encoded = actual.encoded
    val report = JSONObject()
    var identical = true
    for ((name, pair) in
      linkedMapOf(
        "input_ids" to (fixture.inputIds to encoded.inputIds),
        "label_positions" to (fixture.labelPositions to encoded.labelPositions),
      )) {
      val (expected, observed) = pair
      val same = expected.contentEquals(observed)
      identical = identical && same
      val first =
        (0 until maxOf(expected.size, observed.size)).firstOrNull {
          expected.getOrNull(it) != observed.getOrNull(it)
        }
      report.put(
        name,
        JSONObject()
          .put("identical", same)
          .put("expected_count", expected.size)
          .put("actual_count", observed.size)
          .put("first_difference", first ?: JSONObject.NULL)
          .put("expected_at_difference", first?.let { expected.getOrNull(it) } ?: JSONObject.NULL)
          .put("actual_at_difference", first?.let { observed.getOrNull(it) } ?: JSONObject.NULL),
      )
    }
    val n = actual.window
    val length = encoded.encodedLength
    val paddingOk =
      (length until n).all { actual.inputIds[it] == padId } &&
        (0 until n).all { actual.attentionMask[it] == if (it < length) 1f else 0f } &&
        routingOk(actual)
    return report
      .put("window", n)
      .put("window_identical", n == fixture.window)
      .put("padding_attention_routing_identical", paddingOk)
      .put("identical", identical && paddingOk && n == fixture.window)
  }

  private fun routingOk(actual: GliclassInputs.Prepared): Boolean {
    val n = actual.window
    val positions = actual.encoded.labelPositions
    for (row in 0 until GliclassInputs.LABEL_SLOTS) {
      for (column in 0 until n) {
        val expected = if (row < positions.size && positions[row] == column) 1f else 0f
        if (actual.labelRouting[row * n + column] != expected) {
          return false
        }
      }
    }
    return true
  }

  /** Max |a − b| over the first [count] values. */
  fun maxAbsDifference(
    actual: FloatArray,
    expected: FloatArray,
    count: Int = expected.size,
  ): Double =
    (0 until count).maxOfOrNull { abs(actual[it].toDouble() - expected[it].toDouble()) } ?: 0.0

  /** Report form of a decision: the returned labels and their probabilities. */
  fun predictionsJson(decision: GliclassDecoder.Decision): JSONArray =
    JSONArray(
      decision.predictions.map {
        JSONObject().put("label", it.label).put("score", it.score.toDouble())
      }
    )

  private fun JSONArray.predictionLabels() = List(length()) { getJSONObject(it).getString("label") }

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

  private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }

  private fun JSONArray.strings() = List(length()) { getString(it) }
}
