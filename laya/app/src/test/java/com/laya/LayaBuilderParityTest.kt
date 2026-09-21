// SPDX-License-Identifier: Apache-2.0
package com.laya

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LayaBuilderParityTest {
  @Test
  fun allCapturedRowsHaveExactIdsAndMarkers() {
    val fixtures =
      LayaJson.asArray(LayaTestData.read("fixtures/ml_fixtures.json"))
        .map(LayaJson::asObject)
        .associateBy { it["id"] as String }
    val details = mutableListOf<Map<String, Any?>>()
    val failures = mutableListOf<Map<String, Any?>>()
    for (window in listOf(256, 512)) {
      val builder = LayaPromptBuilder(LayaTestData.tokenizer, maxLen = window, headMaxLen = 256)
      val rows = LayaTestData.rows(window)
      assertEquals("Captured row count for S$window", 201, rows.size)
      for (row in rows) {
        val id = row["row_id"] as String
        val fixture = fixtures[row["fixture_id"]] ?: error("Missing fixture for $id")
        val question = LayaJson.asObject(row["question"])
        val expectedIds = LayaTestData.ints(row["sequence_ids"])
        val expectedMarkers = LayaTestData.ints(row["marker_positions"])
        val detail = linkedMapOf<String, Any?>("row_id" to id, "window" to window)
        try {
          val built = builder.build(fixture["state"], question, row["question_id"] as String)
          detail["ids_exact"] = expectedIds.contentEquals(built.ids)
          detail["markers_exact"] = expectedMarkers.contentEquals(built.markers)
          detail["sequence_length"] = built.ids.size
          detail["marker_count"] = built.markers.size
          if (detail["ids_exact"] != true || detail["markers_exact"] != true) {
            detail["expected_ids"] = expectedIds
            detail["actual_ids"] = built.ids
            detail["expected_markers"] = expectedMarkers
            detail["actual_markers"] = built.markers
            failures.add(detail)
          }
        } catch (error: Exception) {
          detail["ids_exact"] = false
          detail["markers_exact"] = false
          detail["error"] = error.toString()
          failures.add(detail)
        }
        details.add(detail)
      }
    }
    LayaTestData.report(
      "jvm_builder_parity.json",
      linkedMapOf(
        "status" to if (failures.isEmpty()) "PASS" else "FAIL",
        "rows" to details.size,
        "ids_exact" to details.count { it["ids_exact"] == true },
        "markers_exact" to details.count { it["markers_exact"] == true },
        "failures" to failures,
        "details" to details,
      ),
    )
    assertEquals(402, details.size)
    LayaTestData.assertNoFailures("Builder parity", failures)
  }

  @Test
  fun schemaNormalizationKeepsStructuredCriteriaAndLabelOrder() {
    val question =
      LayaPromptBuilder.normalize(
        linkedMapOf(
          "type" to "choice",
          "instructions" to linkedMapOf("日本語" to true),
          "criteria" to listOf("later", "first", "later"),
        )
      )
    assertEquals("{\"\\u65e5\\u672c\\u8a9e\": true}", question.instructions)
    assertEquals(listOf("later", "first"), LayaJson.asObject(question.criteria).keys.toList())
    val values =
      LayaQuestion(
        "choice",
        "",
        linkedMapOf("zero" to 0, "false" to false, "empty" to "", "nil" to null),
      )
    assertEquals(
      listOf("zero: 0", "false: false", "empty", "nil"),
      LayaPromptBuilder.renderOptions(values),
    )
    val noul =
      LayaQuestion(
        "noul",
        "",
        linkedMapOf("true" to linkedMapOf("説明" to listOf(false, null)), "false" to 0),
      )
    assertEquals(
      listOf("false: 0", "true: {\"説明\": [false, null]}"),
      LayaPromptBuilder.renderOptions(noul),
    )
  }

  @Test
  fun lostMarkerRejectsButLossOfFinalSeparatorDoesNot() {
    val question =
      LayaPromptBuilder.normalize(
        linkedMapOf("type" to "choice", "instructions" to "x", "criteria" to listOf("a", "b"))
      )
    val complete = LayaPromptBuilder(LayaTestData.tokenizer).build("", question)
    val lastMarker = complete.markers.last()
    val truncated =
      LayaPromptBuilder(LayaTestData.tokenizer, maxLen = lastMarker + 1).build("", question)
    assertEquals(lastMarker + 1, truncated.ids.size)
    assertEquals(LayaPromptBuilder.MASK, truncated.ids.last())
    assertEquals(2, truncated.markers.size)
    try {
      LayaPromptBuilder(LayaTestData.tokenizer, maxLen = lastMarker)
        .build("", question, "too-small")
      throw AssertionError("Missing-marker row must reject before graph invocation")
    } catch (expected: IllegalArgumentException) {
      assertTrue(expected.message!!.contains("options exceed head_max_len=256"))
    }
  }
}
