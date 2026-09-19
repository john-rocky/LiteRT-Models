package com.gliner25

import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject

/** Pure JVM comparisons shared by debug validation and its asset integrity tests. */
internal object GlinerGateFixtures {
  fun text(fixture: JSONObject): String {
    val parts = fixture.getJSONArray("text_parts")
    return (0 until parts.length()).joinToString("") { parts.getString(it) }
  }

  fun compareInputs(fixture: JSONObject, actual: GlinerInputs.Prepared): JSONObject {
    val fields =
      linkedMapOf(
        "input_ids" to actual.inputIds.copyOf(actual.encodedLength),
        "text_word_indices" to actual.textWordPositions,
        "query_marker_indices" to actual.queryMarkerPositions,
      )
    val report = JSONObject()
    var identical = true
    fields.forEach { (name, observed) ->
      val expected = fixture.getJSONArray(name).ints()
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
          .put("actual", JSONArray(observed.toList()))
          .put("first_difference", first ?: JSONObject.NULL)
          .put("expected_at_difference", first?.let { expected.getOrNull(it) } ?: JSONObject.NULL)
          .put("actual_at_difference", first?.let { observed.getOrNull(it) } ?: JSONObject.NULL),
      )
    }
    val paddingZero =
      (actual.encodedLength until actual.inputIds.size).all {
        actual.inputIds[it] == 0
      }
    val windowSame = actual.window.sequenceLength == fixture.getInt("window")
    return report
      .put("padding_zero", paddingZero)
      .put("window_identical", windowSame)
      .put("identical", identical && paddingZero && windowSame)
  }

  fun compareSpans(actual: List<GlinerDecoder.Span>, expected: JSONArray): Pair<Boolean, Double> {
    val observed = actual.associateBy { Triple(it.label, it.start, it.end) }
    val reference =
      (0 until expected.length()).associate { index ->
        val span = expected.getJSONObject(index)
        Triple(span.getString("label"), span.getInt("start"), span.getInt("end")) to
          span.getDouble("confidence")
      }
    val identical = observed.size == actual.size && observed.keys == reference.keys
    val difference =
      observed.keys.intersect(reference.keys).maxOfOrNull {
        abs(observed.getValue(it).confidence.toDouble() - reference.getValue(it))
      } ?: 0.0
    return identical to difference
  }

  fun spans(values: List<GlinerDecoder.Span>): JSONArray =
    JSONArray(
      values.map {
        JSONObject()
          .put("label", it.label)
          .put("text", it.text)
          .put("start", it.start)
          .put("end", it.end)
          .put("confidence", it.confidence.toDouble())
      }
    )

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }
}
