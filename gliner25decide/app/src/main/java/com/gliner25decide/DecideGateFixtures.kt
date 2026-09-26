package com.gliner25decide

import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject

/**
 * Pure JVM parsing and comparisons shared by the debug gate and its JVM tests. The asset keeps
 * every ordered structure (tasks, label descriptions, official decisions) as a JSON array because
 * org.json on the desktop JVM does not keep object key order.
 */
internal object DecideGateFixtures {
  /** Logcat tag of every debug gate, first-request and paced report line. */
  const val GATE_LOG_TAG = "DECIDE_GATE"

  /** Indentation of the JSON reports written under `files/gate/`. */
  const val REPORT_JSON_INDENT = 2

  /** The official gliner2 decision of one head, labels in gliner2's order. */
  data class OfficialDecision(val task: String, val multiLabel: Boolean, val labels: List<String>)

  /** One captured official request and the windows the device gate runs it at. */
  data class Fixture(
    val id: String,
    val text: String,
    val tasks: List<Task>,
    val inputIds: IntArray,
    val schemaSpecialIndices: List<IntArray>,
    val labelPositions: IntArray,
    val official: List<OfficialDecision>,
    val oracleLogits: FloatArray,
    val oracleProbabilities: FloatArray,
    val windows: List<Int>,
    val macCpuLogits: Map<Int, FloatArray>,
  )

  /** Parses every fixture of the `gate_fixtures.json` debug asset. */
  fun parse(root: JSONObject): List<Fixture> {
    val fixtures = root.getJSONArray("fixtures")
    return (0 until fixtures.length()).map { parseFixture(fixtures.getJSONObject(it)) }
  }

  /** Parses one fixture: request, captured inputs, official decisions and reference logits. */
  fun parseFixture(json: JSONObject): Fixture {
    val special = json.getJSONArray("schema_special_indices")
    val official = json.getJSONArray("official_result")
    val mac = json.optJSONObject("mac_cpu_wfp16_fp16_logits")
    val windows = json.getJSONArray("windows").ints().toList()
    return Fixture(
      json.getString("id"),
      json.getString("text"),
      tasks(json.getJSONArray("tasks")),
      json.getJSONArray("input_ids").ints(),
      (0 until special.length()).map { special.getJSONArray(it).ints() },
      json.getJSONArray("label_positions").ints(),
      (0 until official.length()).map {
        val row = official.getJSONObject(it)
        OfficialDecision(
          row.getString("task"),
          row.getBoolean("multi_label"),
          row.getJSONArray("labels").strings(),
        )
      },
      json.getJSONArray("oracle_logits").floats(),
      json.getJSONArray("oracle_probs").floats(),
      windows,
      windows
        .filter { mac?.has(it.toString()) == true }
        .associateWith { mac!!.getJSONArray(it.toString()).floats() },
    )
  }

  /**
   * Ordered task list: `{task, labels, multi_label, cls_threshold, class_act, prompt,
   * label_descriptions}`.
   */
  fun tasks(array: JSONArray): List<Task> =
    (0 until array.length()).map { index ->
      val task = array.getJSONObject(index)
      val descriptions =
        task.optJSONArray("label_descriptions")?.let { pairs ->
          val map = LinkedHashMap<String, String>()
          for (i in 0 until pairs.length()) {
            val pair = pairs.getJSONArray(i)
            map[pair.getString(0)] = pair.getString(1)
          }
          map
        }
      Task(
        task.getString("task"),
        task.getJSONArray("labels").strings(),
        task.getBoolean("multi_label"),
        task.getDouble("cls_threshold"),
        if (task.isNull("prompt")) null else task.getString("prompt"),
        descriptions,
        Activation.fromKey(task.optString("class_act", "auto")),
      )
    }

  /**
   * Byte-exact comparison of the host input path against the captured Python batch: unpadded input
   * IDs, `[L]` positions and each task's `[P]`/`[L]` positions, plus PAD/attention/routing padding
   * for the prepared window. Reports the first differing index per field.
   */
  fun compareInputs(fixture: Fixture, actual: DecideInputs.Prepared): JSONObject {
    val encoded = actual.encoded
    val fields =
      linkedMapOf(
        "input_ids" to (fixture.inputIds to encoded.inputIds),
        "label_positions" to (fixture.labelPositions to encoded.labelPositions),
        "schema_special_indices" to
          (fixture.schemaSpecialIndices.flatMap { it.toList() + SEPARATOR }.toIntArray() to
            encoded.schemaSpecialPositions.flatMap { it.toList() + SEPARATOR }.toIntArray()),
      )
    val report = JSONObject()
    var identical = true
    fields.forEach { (name, pair) ->
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
      (length until n).all { actual.inputIds[it] == 0 } &&
        (0 until n).all { actual.attentionMask[it] == if (it < length) 1f else 0f } &&
        routingOk(actual)
    return report
      .put("window", n)
      .put("padding_attention_routing_identical", paddingOk)
      .put("identical", identical && paddingOk)
  }

  private fun routingOk(actual: DecideInputs.Prepared): Boolean {
    val n = actual.window
    val positions = actual.encoded.labelPositions
    for (row in 0 until DecideInputs.LABEL_SLOTS) {
      for (column in 0 until n) {
        val expected = if (row < positions.size && positions[row] == column) 1f else 0f
        if (actual.labelRouting[row * n + column] != expected) {
          return false
        }
      }
    }
    return true
  }

  /** Decisions equal the official result: same heads in order, same labels in gliner2's order. */
  fun decisionsEqual(
    decisions: List<DecideDecoder.TaskDecision>,
    official: List<OfficialDecision>,
  ): Boolean =
    decisions.size == official.size &&
      decisions.zip(official).all { (actual, expected) ->
        actual.task == expected.task &&
          actual.multiLabel == expected.multiLabel &&
          actual.labels == expected.labels
      }

  /** Max |a − b| over the valid slots (the first [count] values). */
  fun maxAbsDifference(
    actual: FloatArray,
    expected: FloatArray,
    count: Int = expected.size,
  ): Double =
    (0 until count).maxOfOrNull { abs(actual[it].toDouble() - expected[it].toDouble()) } ?: 0.0

  /** Probabilities of every head concatenated in request order. */
  fun probabilities(decisions: List<DecideDecoder.TaskDecision>): FloatArray =
    decisions.flatMap { it.probabilities.toList() }.toFloatArray()

  /** Report form of [decisions]: task, multi-label flag, chosen labels and their probabilities. */
  fun decisionsJson(decisions: List<DecideDecoder.TaskDecision>): JSONArray =
    JSONArray(
      decisions.map { decision ->
        JSONObject()
          .put("task", decision.task)
          .put("multi_label", decision.multiLabel)
          .put("labels", JSONArray(decision.labels))
          .put("probabilities", JSONArray(decision.chosenProbabilities.map { it.toDouble() }))
      }
    )

  private const val SEPARATOR = -1

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

  private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }

  private fun JSONArray.strings() = List(length()) { getString(it) }
}
