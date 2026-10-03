package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test

/**
 * `render`, `option_text`, `to_record` and request validation against the author's `kev.api` on
 * edge inputs (`render_cases.json`, written by scripts/make_test_data.py --probes; self-contained).
 */
class KevRecordsTest {
  private val cases =
    KevJson.parse(ExternalTestData.resource("render_cases.json").readBytes()) as Map<*, *>

  @Test
  fun renderMatchesTheAuthor() {
    val renders = cases["render"] as List<*>
    for (case in renders) {
      val entry = case as Map<*, *>
      val indent = (entry["indent"] as JsonNumber).toInt()
      assertEquals(
        "render(${KevJson.write(entry["value"])}, $indent)",
        entry["text"],
        KevRecords.render(entry["value"], indent),
      )
    }
    assertEquals(62, renders.size)
  }

  @Test
  fun optionTextMatchesTheAuthor() {
    val options = cases["option_text"] as List<*>
    for (case in options) {
      val entry = case as Map<*, *>
      assertEquals(
        "option_text(${entry["name"]}, ${KevJson.write(entry["description"])})",
        entry["text"],
        KevRecords.optionText(entry["name"] as String, entry["description"]),
      )
    }
    assertEquals(30, options.size)
  }

  @Test
  fun toRecordMatchesTheAuthor() {
    val records = cases["to_record"] as List<*>
    for (case in records) {
      val entry = case as Map<*, *>
      val (record, meta) = KevRecords.toRecord(KevRequest.fromJson(entry["request"]))
      val expectedRecord = entry["record"] as Map<*, *>
      assertEquals(expectedRecord["state"], record.state)
      val expectedQuestions = expectedRecord["questions"] as List<*>
      assertEquals(expectedQuestions.size, record.questions.size)
      for ((index, question) in record.questions.withIndex()) {
        val expected = expectedQuestions[index] as Map<*, *>
        assertEquals(expected["instr"], question.instructions)
        assertEquals(expected["options"], question.options)
      }
      val actualMeta = meta.map { question ->
        linkedMapOf<String, Any?>(
            "id" to question.id,
            "type" to question.type.wireName,
            "keys" to question.keys,
          )
          .apply {
            question.legend?.let { put("legend", it) }
          }
      }
      assertNull(OracleFixtures.jsonDifference(entry["meta"], actualMeta))
    }
    assertEquals(8, records.size)
  }

  @Test
  fun requestsPydanticRejectsAreRejected() {
    val invalid = cases["invalid"] as List<*>
    for (request in invalid) {
      try {
        KevRequest.fromJson(request)
        fail("accepted: ${KevJson.write(request)}")
      } catch (expected: IllegalArgumentException) {
        // SystemOneRequest.model_validate raises ValidationError for the same request.
      } catch (expected: ClassCastException) {
        fail("rejected with a cast error instead of a message: ${KevJson.write(request)}")
      }
    }
    assertEquals(13, invalid.size)
  }

  @Test
  fun pythonLeftStripUsesPythonWhitespace() {
    assertEquals("x ", KevRecords.pythonLeftStrip(" \u001c\u001f\u3000\u0085x "))
    assertEquals("\u200bx", KevRecords.pythonLeftStrip("\u200bx"))
  }
}
