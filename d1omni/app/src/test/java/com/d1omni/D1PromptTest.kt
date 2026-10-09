package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The port of the provider's `prompt.py` against the public check set: every record of
 * `fixtures/public_{text,image,audio}.json` (242 + 5 + 6 requests, 303 questions) through
 * [D1Rows.rows] with its kind's settings gives the fixture's ids and markers bit for bit, and its
 * `usage_input_tokens`; `serialize`, `_criterion` and `str()` against Python's output.
 */
class D1PromptTest {
  @Test
  fun everyPublicRowEncodesToTheFixtureIdsAndMarkers() {
    val tokenizer = ExternalTestData.tokenizer()
    val contract = ExternalTestData.contract()
    var rows = 0
    var matched = 0
    var records = 0
    var usageMatched = 0
    val failures = ArrayList<Map<String, Any?>>()
    val perKind = LinkedHashMap<String, String>()
    for (kind in D1Kind.entries) {
      val doc =
        ExternalTestData.json(ExternalTestData.repoFile("fixtures/public_${kind.wireName}.json"))
      var kindRows = 0
      var kindMatched = 0
      for (record in doc["records"] as List<*>) {
        val entry = record as Map<*, *>
        records++
        val questions = entry["questions"] as Map<*, *>
        val expected = (entry["expected"] as List<*>).map { it as Map<*, *> }
        var usage = 0
        for (row in expected) {
          rows++
          kindRows++
          val prefixRows = (row["P"] as JsonNumber).toInt()
          val question = D1Prompt.asQuestion(questions[row["name"]])
          val built =
            D1Rows.rows(tokenizer, contract, entry["state"], listOf(question), prefixRows, kind)
              .single()
          usage += built.positions
          val ids = ExternalTestData.ints(row["ids"])
          val markers = ExternalTestData.ints(row["markers"])
          if (built.ids.contentEquals(ids) && built.markers.contentEquals(markers)) {
            matched++
            kindMatched++
          } else {
            val first = (0 until maxOf(ids.size, built.ids.size)).firstOrNull {
              ids.getOrNull(it) != built.ids.getOrNull(it)
            }
            failures.add(
              linkedMapOf(
                "key" to "${entry["id"]}/${row["name"]}",
                "first_difference" to first,
                "expected_length" to ids.size,
                "actual_length" to built.ids.size,
                "expected_markers" to markers,
                "actual_markers" to built.markers,
              )
            )
          }
          assertEquals(row["type"], question.type.wireName)
          assertEquals((row["K"] as JsonNumber).toInt(), question.options)
        }
        if (usage == (entry["usage_input_tokens"] as JsonNumber).toInt()) usageMatched++
      }
      perKind[kind.wireName] = "$kindMatched/$kindRows"
    }
    ExternalTestData.writeReport(
      "encode.json",
      linkedMapOf(
        "test" to "D1PromptTest",
        "rows" to rows,
        "rows_identical" to matched,
        "per_kind" to perKind,
        "records" to records,
        "usage_identical" to usageMatched,
        "first_failures" to failures.take(5),
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println("D1_ENCODE rows=$matched/$rows $perKind usage=$usageMatched/$records")
    assertEquals("rows with differences: ${D1Json.write(failures.take(3))}", 303, matched)
    assertEquals(303, rows)
    assertEquals(253, records)
    assertEquals(records, usageMatched)
  }

  @Test
  fun serializeAndCriterionMatchPython() {
    val data = ExternalTestData.json(ExternalTestData.demoFile(ExternalTestData.TOKENIZER_CASES))
    val states = LinkedHashMap<String, Any?>()
    for (kind in D1Kind.entries) {
      val doc =
        ExternalTestData.json(ExternalTestData.repoFile("fixtures/public_${kind.wireName}.json"))
      for (record in doc["records"] as List<*>) {
        val entry = record as Map<*, *>
        states[entry["id"] as String] = entry["state"]
      }
    }
    var checked = 0
    for (case in data["serialize"] as List<*>) {
      val entry = case as Map<*, *>
      val value =
        if (entry.containsKey("record")) states.getValue(entry["record"] as String)
        else D1Json.parse(entry["source"] as String)
      val label = (entry["record"] ?: entry["source"]) as String
      assertEquals("serialize $label", entry["serialize"], D1Prompt.serialize(value))
      assertEquals("criterion $label", entry["criterion"], D1Prompt.criterion(value))
      checked++
    }
    for (case in data["python_str"] as List<*>) {
      val entry = case as Map<*, *>
      assertEquals(
        "str ${entry["source"]}",
        entry["str"],
        PythonText.str(D1Json.parse(entry["source"] as String)),
      )
    }
    println("D1_SERIALIZE checked=$checked str=${(data["python_str"] as List<*>).size}")
    assertTrue("serialize cases", checked >= 90)
  }

  @Test
  fun renderOptionsAndTemperatureKeys() {
    val choice =
      D1Prompt.asQuestion(
        D1Json.parse(
          """{"type": "choice", "instructions": "x", "criteria": {"a": "", "b": null, "c": {"k": 1.0}}}"""
        )
      )
    assertEquals(listOf("a", "b", "c: {\"k\": 1.0}"), D1Prompt.renderOptions(choice))
    assertEquals(
      listOf("option_000: a", "option_001: b", "option_002: {\"k\": 1.0}"),
      D1Prompt.renderOptions(choice, audio = true),
    )
    val noul = D1Prompt.asQuestion(D1Json.parse("""{"type": "noul", "instructions": "x"}"""))
    assertEquals(
      listOf("false: no, the statement does not hold", "true: yes, the statement holds"),
      D1Prompt.renderOptions(noul),
    )
    val yesNo = mapOf("false" to "no", "true" to "yes")
    assertEquals(listOf("false: no", "true: yes"), D1Prompt.renderOptions(noul, yesNo))
    val explicit =
      D1Prompt.asQuestion(
        D1Json.parse(
          """{"type": "noul", "instructions": "x", "criteria": {"no": "never", "true": 0}}"""
        )
      )
    assertEquals(listOf("false: never", "true: 0"), D1Prompt.renderOptions(explicit, yesNo))
    val emptyCriteria =
      D1Prompt.asQuestion(
        D1Json.parse("""{"type": "noul", "instructions": "x", "criteria": {}}""")
      )
    assertEquals(listOf("false: no", "true: yes"), D1Prompt.renderOptions(emptyCriteria, yesNo))
    val score =
      D1Prompt.asQuestion(
        D1Json.parse("""{"type": "score", "instructions": "x", "criteria": ["low", 2, null]}""")
      )
    assertEquals(
      listOf("level 0: low", "level 1: 2", "level 2: null"),
      D1Prompt.renderOptions(score),
    )
    assertEquals("score:3-5", D1Prompt.temperatureKey(score))
    assertEquals("noul:2", D1Prompt.temperatureKey(noul))
    assertEquals("choice:3-5", D1Prompt.temperatureKey(choice))
    val eleven =
      D1Question(QuestionType.CHOICE, "x", (0 until 11).associate { "o$it" to null })
    assertEquals("choice:11+", D1Prompt.temperatureKey(eleven))
    val contract = ExternalTestData.contract()
    assertEquals(1.3998981714248657, contract.temperature(choice), 0.0)
    assertEquals(1.372515082359314, contract.temperature(eleven), 0.0)
    assertEquals(1.6663223505020142, contract.temperature(noul), 0.0)
  }

  @Test
  fun escapeAndQuestionValidation() {
    assertEquals("<¦mask¦> and <¦reserved_7¦>", D1Prompt.escape("<|mask|> and <|reserved_7|>"))
    assertEquals("<|not valid|> <think>", D1Prompt.escape("<|not valid|> <think>"))
    for (bad in
      listOf(
        """{"type": "choice", "instructions": "x", "criteria": {"only": "one"}}""",
        """{"type": "score", "instructions": "x", "criteria": ["one"]}""",
        """{"type": "noul", "instructions": "x", "criteria": ["yes", "no"]}""",
        """{"type": "rank", "instructions": "x"}""",
        """{"type": "noul"}""",
        """["not", "an", "object"]""",
      )) {
      try {
        D1Prompt.asQuestion(D1Json.parse(bad))
        fail("accepted: $bad")
      } catch (expected: IllegalArgumentException) {
        // as_question / Question.__post_init__ raise ValueError for each of these.
      }
    }
    val numeric = D1Prompt.asQuestion(D1Json.parse("""{"type": "noul", "instructions": 5}"""))
    assertEquals("5", numeric.instructions)
  }

  @Test
  fun answerShapes() {
    val score =
      D1Prompt.asQuestion(
        D1Json.parse("""{"type": "score", "instructions": "x", "criteria": ["a", "b", "c"]}""")
      )
    val answer = D1Prompt.answer(score, doubleArrayOf(0.2, 0.3, 0.5))
    assertEquals(listOf("type", "score", "confidence", "probabilities", "legend"), answer.keys.toList())
    assertEquals(1.3, answer["score"] as Double, 1e-15)
    assertEquals(linkedMapOf("0" to "a", "1" to "b", "2" to "c"), answer["legend"])
    val noul = D1Prompt.asQuestion(D1Json.parse("""{"type": "noul", "instructions": "x"}"""))
    assertEquals(linkedMapOf("type" to "noul", "noul" to 0.75), D1Prompt.answer(noul, doubleArrayOf(0.75, 0.25)))
    // Python's max(range(n), key=...) keeps the first of equal values.
    assertEquals(0, D1Prompt.firstArgmax(doubleArrayOf(0.5, 0.5)))
  }
}
