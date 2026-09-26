package com.gliner25decide

import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Reads the official fp32 oracle (`fixtures/oracle_fp32.json`, gliner2 2.0.0 `classify_text` on Mac
 * CPU). org.json on the desktop JVM does not keep object key order, so request order comes from the
 * `task_results` array (the configs gliner2 resolved, in schema order); the `tasks` dict is only
 * looked up by name for `prompt` and label descriptions, whose order equals the label order.
 */
internal object OracleFixtures {
  data class TaskResult(
    val task: String,
    val labels: List<String>,
    val logits: FloatArray,
    val probabilities: FloatArray,
    val decision: List<String>,
  )

  data class Fixture(
    val id: String,
    val text: String,
    val tasks: List<Task>,
    val inputIds: IntArray,
    val schemaSpecialIndices: List<IntArray>,
    val labelPositions: IntArray,
    val encodedLength: Int,
    val fits: Map<Int, Boolean>,
    val taskResults: List<TaskResult>,
    val official: List<DecideGateFixtures.OfficialDecision>,
  ) {
    /** Oracle logits of every head concatenated in request order. */
    val logits: FloatArray
      get() = taskResults.flatMap { it.logits.toList() }.toFloatArray()

    val probabilities: FloatArray
      get() = taskResults.flatMap { it.probabilities.toList() }.toFloatArray()
  }

  fun load(root: File): List<Fixture> {
    val json = JSONObject(File(root, "fixtures/oracle_fp32.json").readText())
    val fixtures = json.getJSONArray("fixtures")
    return (0 until fixtures.length()).map { parse(fixtures.getJSONObject(it)) }
  }

  private fun parse(json: JSONObject): Fixture {
    val tasksDict = json.getJSONObject("tasks")
    val results = json.getJSONArray("task_results")
    val official = json.getJSONObject("official_result")
    val tasks = ArrayList<Task>()
    val taskResults = ArrayList<TaskResult>()
    val officialRows = ArrayList<DecideGateFixtures.OfficialDecision>()
    for (index in 0 until results.length()) {
      val result = results.getJSONObject(index)
      val name = result.getString("task")
      val labels = result.getJSONArray("labels").strings()
      val config = tasksDict.get(name)
      var prompt: String? = null
      var descriptions: Map<String, String>? = null
      if (config is JSONObject) {
        prompt =
          if (config.has("prompt") && !config.isNull("prompt")) config.getString("prompt") else null
        val labelConfig = config.get("labels")
        if (labelConfig is JSONObject) {
          descriptions =
            LinkedHashMap<String, String>().apply {
              labels.forEach { put(it, labelConfig.getString(it)) }
            }
          check(labelConfig.length() == labels.size) { json.getString("id") }
        } else {
          check((labelConfig as JSONArray).strings() == labels) { json.getString("id") }
        }
      } else {
        check((config as JSONArray).strings() == labels) { json.getString("id") }
      }
      tasks.add(
        Task(
          name,
          labels,
          result.getBoolean("multi_label"),
          result.getDouble("cls_threshold"),
          prompt,
          descriptions,
          Activation.fromKey(result.getString("class_act")),
        )
      )
      val decision = result.get("decision")
      taskResults.add(
        TaskResult(
          name,
          labels,
          result.getJSONArray("logits").floats(),
          result.getJSONArray("probs").floats(),
          if (decision is JSONArray) decision.strings() else listOf(decision as String),
        )
      )
      val value = official.get(name)
      officialRows.add(
        DecideGateFixtures.OfficialDecision(
          name,
          value is JSONArray,
          if (value is JSONArray) value.strings() else listOf(value as String),
        )
      )
    }
    check(tasksDict.length() == tasks.size && official.length() == tasks.size) {
      json.getString("id")
    }
    val special = json.getJSONArray("schema_special_indices")
    val fits = json.getJSONObject("fits")
    return Fixture(
      json.getString("id"),
      json.getString("text"),
      tasks,
      json.getJSONArray("input_ids").ints(),
      (0 until special.length()).map { special.getJSONArray(it).ints() },
      json.getJSONArray("label_positions").ints(),
      json.getInt("encoded_length"),
      DecideInputs.WINDOWS.associateWith { fits.getBoolean("s$it") },
      taskResults,
      officialRows,
    )
  }

  fun firstDifference(expected: IntArray, actual: IntArray): Int? =
    (0 until maxOf(expected.size, actual.size)).firstOrNull {
      expected.getOrNull(it) != actual.getOrNull(it)
    }

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

  private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }

  private fun JSONArray.strings() = List(length()) { getString(it) }
}
