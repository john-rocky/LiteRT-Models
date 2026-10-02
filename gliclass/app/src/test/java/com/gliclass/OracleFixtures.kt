package com.gliclass

import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Reads the official fp32 oracle (`fixtures/oracle.json`, the gliclass 0.1.20 pipeline on CPU,
 * batch 1) joined with the requests (`fixtures/requests.json`) by id.
 */
internal object OracleFixtures {
  /** One label and score of a pipeline result. */
  data class Prediction(val label: String, val score: Double)

  data class Fixture(
    val id: String,
    val source: String,
    val text: String,
    val prompt: String?,
    val labels: List<String>,
    val linearized: String,
    val inputIds: IntArray,
    val labelPositions: IntArray,
    val fitsS128: Boolean,
    val fitsS256: Boolean,
    val logits: FloatArray,
    val softmax: FloatArray,
    val sigmoid: FloatArray,
    val single: List<Prediction>,
    val multi: List<Prediction>,
    val multiThresholdZero: List<Prediction>?,
  ) {
    /** The smallest window that holds the captured IDs. */
    val window: Int
      get() = if (fitsS128) 128 else 256
  }

  fun load(root: File): List<Fixture> {
    val requests =
      JSONObject(File(root, "fixtures/requests.json").readText()).getJSONArray("requests").let {
        array ->
        (0 until array.length()).associate {
          val request = array.getJSONObject(it)
          request.getString("id") to request
        }
      }
    val records = JSONObject(File(root, "fixtures/oracle.json").readText()).getJSONArray("records")
    return (0 until records.length()).map {
      val record = records.getJSONObject(it)
      parse(record, requests.getValue(record.getString("id")))
    }
  }

  private fun parse(record: JSONObject, request: JSONObject): Fixture {
    val fits = record.getJSONObject("fits")
    return Fixture(
      record.getString("id"),
      record.getString("source"),
      request.getString("text"),
      if (request.isNull("prompt")) null else request.getString("prompt"),
      request.getJSONArray("labels").strings(),
      record.getString("string"),
      record.getJSONArray("input_ids").ints(),
      record.getJSONArray("label_positions").ints(),
      fits.getBoolean("s128"),
      fits.getBoolean("s256"),
      record.getJSONArray("logits").floats(),
      record.getJSONArray("softmax").floats(),
      record.getJSONArray("sigmoid").floats(),
      predictions(record.getJSONArray("pipeline_single")),
      predictions(record.getJSONArray("pipeline_multi")),
      record.optJSONArray("pipeline_multi_threshold_0")?.let { predictions(it) },
    )
  }

  private fun predictions(array: JSONArray) =
    (0 until array.length()).map {
      val item = array.getJSONObject(it)
      Prediction(item.getString("label"), item.getDouble("score"))
    }

  fun firstDifference(expected: IntArray, actual: IntArray): Int? =
    (0 until maxOf(expected.size, actual.size)).firstOrNull {
      expected.getOrNull(it) != actual.getOrNull(it)
    }

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

  private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }

  private fun JSONArray.strings() = List(length()) { getString(it) }
}
