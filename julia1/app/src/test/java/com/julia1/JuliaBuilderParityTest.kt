// SPDX-License-Identifier: Apache-2.0
package com.julia1

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class JuliaBuilderParityTest {
  @Test
  fun everyOracleRequestBuildsTheReferenceIdsOrIsRejectedStrictly() {
    val requests = JuliaTestData.requests()
    assertEquals(2100, requests.size)
    val failures = mutableListOf<Map<String, Any?>>()
    val counts = linkedMapOf<String, Int>()
    for (window in JuliaEngine.WINDOWS) {
      val builder = JuliaSequenceBuilder(JuliaTestData.tokenizer, window)
      var identical = 0
      var rejected = 0
      for (row in requests) {
        val id = row["id"] as String
        val expectedIds = JuliaTestData.ints(row["ids"])
        val fits = expectedIds.size <= window
        try {
          val built =
            builder.build(JuliaJson.asObject(row["request"])["state"], JuliaTestData.question(row))
          val same =
            built.ids.contentEquals(expectedIds) &&
              built.markers.contentEquals(JuliaTestData.ints(row["markers"])) &&
              built.qtype == (row["qtype"] as Number).toInt()
          if (!fits) {
            failures +=
              linkedMapOf("id" to id, "window" to window, "problem" to "accepted overflow")
          } else if (same) {
            identical++
          } else {
            failures +=
              linkedMapOf(
                "id" to id,
                "window" to window,
                "actual_ids" to built.ids.toList(),
                "expected_ids" to expectedIds.toList(),
                "actual_markers" to built.markers.toList(),
              )
          }
        } catch (error: EncodingException) {
          if (fits) {
            failures += linkedMapOf("id" to id, "window" to window, "problem" to error.message)
          } else {
            rejected++
          }
        }
      }
      counts["identical_s$window"] = identical
      counts["rejected_s$window"] = rejected
    }
    JuliaTestData.report(
      "jvm_builder_parity.json",
      linkedMapOf(
        "status" to if (failures.isEmpty()) "PASS" else "FAIL",
        "counts" to counts,
        "failures" to failures,
      ),
    )
    println("JULIA1_JVM_BUILDER $counts")
    assertEquals(2065, counts["identical_s512"])
    assertEquals(35, counts["rejected_s512"])
    assertEquals(2100, counts["identical_s1024"])
    JuliaTestData.assertNoFailures("Builder parity", failures)
  }

  @Test
  fun namedQuestionsRenderTheCriteriaDescriptionsAsOptions() {
    val choice =
      JuliaQuestion(
        "choice",
        "Which team?",
        linkedMapOf("billing" to "Billing and payment disputes", "access" to "Account access"),
      )
    assertEquals(listOf("billing", "access"), choice.keys)
    assertEquals(listOf("Billing and payment disputes", "Account access"), choice.options)
    val score = JuliaQuestion("score", "How urgent?", listOf("Not urgent", "Urgent"))
    assertEquals(listOf("0", "1"), score.keys)
    assertEquals(listOf("Not urgent", "Urgent"), score.options)
    val literal = JuliaQuestion("noul", "Refund?", null)
    assertEquals(listOf("false", "true"), literal.options)
    val described =
      JuliaQuestion(
        "noul",
        "Refund?",
        linkedMapOf("true" to "Refund asked", "false" to "No refund"),
      )
    assertEquals(listOf("No refund", "Refund asked"), described.options)
    for (criteria in
      listOf<Any?>(
        emptyMap<String, String>(),
        mapOf("true" to "yes"),
        mapOf("false" to "no", "true" to "yes", "other" to "maybe"),
        listOf("no", "yes"),
        mapOf("false" to "", "true" to "yes"),
      )) {
      try {
        JuliaQuestion("noul", "q", criteria)
        throw AssertionError("Invalid noul criteria must be rejected: $criteria")
      } catch (expected: IllegalArgumentException) {
        assertTrue(expected.message != null)
      } catch (expected: IllegalStateException) {
        assertTrue(expected.message != null)
      }
    }
  }

  @Test
  fun strictEncodingRejectsInsteadOfTruncating() {
    val tokenizer = JuliaTestData.tokenizer
    val question = JuliaQuestion("choice", "q", linkedMapOf("a" to "first", "b" to "second"))
    val builder = JuliaSequenceBuilder(tokenizer, 512)
    val short = builder.build("state", question)
    assertEquals(2, short.markers.size)
    assertEquals(JuliaSequenceBuilder.MASK, short.ids[short.markers[0]])
    val longState = "word ".repeat(600)
    try {
      builder.build(longState, question)
      throw AssertionError("An overflowing state must be rejected")
    } catch (expected: EncodingException) {
      assertTrue(expected.message!!.contains("the window is 512"))
    }
    val longOption = JuliaQuestion("choice", "q", linkedMapOf("a" to "x ".repeat(80), "b" to "y"))
    try {
      builder.build("state", longOption)
      throw AssertionError("A 48-token option overflow must be rejected")
    } catch (expected: EncodingException) {
      assertTrue(expected.message!!.contains("48-token"))
    }
    try {
      builder.build("<mask> in state", question)
      throw AssertionError("A reserved marker in the request must be rejected")
    } catch (expected: IllegalArgumentException) {
      assertTrue(expected.message!!.contains("Reserved model marker"))
    }
  }
}
