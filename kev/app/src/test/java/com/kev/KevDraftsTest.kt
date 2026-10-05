package com.kev

import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test

/**
 * The editor's request form and the bundled examples: the three examples are the demo requests byte
 * for byte (`examples_oracle.json` records their sha256), they encode to the author's oracle rows,
 * the answers the app shows follow from the oracle's and the shipped graph's probabilities, an
 * example taken into the editor and back gives the model the same record, and malformed editor
 * input is rejected with the reason the screen shows.
 */
class KevDraftsTest {
  private val expected by lazy {
    (KevJson.parse(ExternalTestData.resource("examples_oracle.json").readBytes()) as Map<*, *>)[
      "examples"]
      as List<*>
  }

  private fun rawFile(name: String) = ExternalTestData.moduleFile("app/src/main/res/raw/$name")

  private fun example(file: String): KevFixture = KevFixture.parse(rawFile(file).readText())

  @Test
  fun examplesAreTheDemoRequests() {
    val files = expected.map { (it as Map<*, *>)["file"] }
    assertEquals(EXAMPLE_FILES, files)
    for (entry in expected) {
      val example = entry as Map<*, *>
      val bytes = rawFile(example["file"] as String).readBytes()
      val digest =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
      assertEquals(example["file"] as String, example["sha256"], digest)
      assertEquals(example["id"], KevFixture.parse(bytes.decodeToString()).id)
    }
  }

  @Test
  fun examplesEncodeToTheOracleRows() {
    val encoder = KevEncoder(ExternalTestData.tokenizer())
    var checked = 0
    for (entry in expected) {
      val example = entry as Map<*, *>
      val fixture = example(example["file"] as String)
      val encoded = encoder.encode(KevRecords.toRecord(fixture.request).first)
      assertEquals(fixture.id, (example["input_tokens"] as JsonNumber).toInt(), encoded.inputTokens)
      val questions = (example["questions"] as List<*>).map { it as Map<*, *> }
      val rows = encoded.rows()
      assertEquals(fixture.request.questions.map { it.id }, questions.map { it["qid"] })
      assertEquals(fixture.id, questions.size, rows.size)
      for ((row, question) in rows.zip(questions)) {
        val key = "${fixture.id}/${question["qid"]}"
        assertArrayEquals(key, OracleFixtures.ints(question["row_ids"]), row.ids)
        assertEquals(key, (question["decide_idx"] as JsonNumber).toInt(), row.decideIndex)
        assertArrayEquals(key, OracleFixtures.ints(question["opt_idx"]), row.optionIndices)
        assertEquals(key, if (row.length <= 128) 128 else 256, row.window(KevFiles.DEFAULT_INSTALL))
        checked++
      }
    }
    assertEquals(9, checked)
    println("KEV_EXAMPLES rows equal to the demo oracle: $checked/9")
  }

  @Test
  fun exampleAnswersFollowFromTheirProbabilities() {
    var answers = 0
    for (entry in expected) {
      val example = entry as Map<*, *>
      val meta = KevRecords.toRecord(example(example["file"] as String).request).second
      val questions = example["questions"] as List<*>
      assertEquals(questions.size, meta.size)
      for ((question, json) in meta.zip(questions)) {
        val oracle = json as Map<*, *>
        val graph = oracle["graph_l512_cpu"] as Map<*, *>
        assertEquals(question.id, oracle["qid"])
        assertEquals(question.keys, oracle["keys"])
        for ((probabilities, answer) in
          listOf(oracle["probs"] to oracle["answer"], graph["probs"] to graph["answer"])) {
          val ours =
            KevAnswers.toAnswers(listOf(OracleFixtures.doubles(probabilities)), listOf(question))[
                question.id]
          assertNull("${example["id"]}/${question.id}", OracleFixtures.jsonDifference(answer, ours))
          answers++
        }
      }
    }
    assertEquals(18, answers)
  }

  @Test
  fun editorRoundTripKeepsTheRecord() {
    for (file in EXAMPLE_FILES) {
      val request = example(file).request
      val draft = KevDrafts.fromRequest(request)
      val back = KevDrafts.toRequest(draft)
      val (record, meta) = KevRecords.toRecord(request)
      val (backRecord, backMeta) = KevRecords.toRecord(back)
      assertEquals(file, record.state, backRecord.state)
      assertEquals(
        record.questions.map { it.instructions },
        backRecord.questions.map { it.instructions },
      )
      assertEquals(record.questions.map { it.options }, backRecord.questions.map { it.options })
      assertEquals(
        meta.map { listOf(it.id, it.type, it.keys, it.legend) },
        backMeta.map { listOf(it.id, it.type, it.keys, it.legend) },
      )
    }
    // A JSON state survives the editor as JSON (pretty-printed, same key order and numbers).
    val json =
      KevRequest.parse(
        """{"state":{"order":"TB-1","items":[{"qty":2,"price":64.9},{"price":1840.0}],"ok":true},"questions":{"q":{"type":"noul"}}}"""
      )
    val draft = KevDrafts.fromRequest(json)
    assertEquals(StateFormat.JSON, KevDrafts.stateFormat(draft.state))
    assertEquals(
      KevRecords.toRecord(json).first.state,
      KevRecords.toRecord(KevDrafts.toRequest(draft)).first.state,
    )
  }

  @Test
  fun stateFormats() {
    assertEquals(StateFormat.TEXT, KevDrafts.stateFormat("Ticket #1"))
    assertEquals(StateFormat.JSON, KevDrafts.stateFormat("  {\"a\": 1}\n"))
    assertEquals(StateFormat.JSON, KevDrafts.stateFormat("[1, 2]"))
    assertEquals(StateFormat.TEXT_INVALID_JSON, KevDrafts.stateFormat("{not json"))
    assertEquals("{not json", KevDrafts.stateValue("{not json"))
  }

  @Test
  fun editorOptions() {
    val choice =
      draft(QuestionType.CHOICE, "billing: Charges, refunds\nshipping\n\n  returns :  Send back  ")
    val criteria = KevDrafts.toRequest(choice).questions.single().criteria as Map<*, *>
    assertEquals(
      linkedMapOf("billing" to "Charges, refunds", "shipping" to null, "returns" to "Send back"),
      criteria,
    )
    assertNull(KevDrafts.toRequest(draft(QuestionType.NOUL, "")).questions.single().criteria)
    assertEquals(
      linkedMapOf("true" to "yes it does", "false" to null),
      KevDrafts.toRequest(draft(QuestionType.NOUL, "true: yes it does\nfalse"))
        .questions
        .single()
        .criteria,
    )
    assertEquals(
      listOf("Calm", "Annoyed"),
      KevDrafts.toRequest(draft(QuestionType.SCORE, "Calm\n Annoyed \n"))
        .questions
        .single()
        .criteria,
    )
  }

  @Test
  fun editorProblems() {
    expect(DraftProblem.NO_QUESTIONS) { KevDrafts.toRequest(RequestDraft("s", emptyList())) }
    expect(DraftProblem.EMPTY_ID) { KevDrafts.toRequest(draft(QuestionType.NOUL, "", id = " ")) }
    expect(DraftProblem.DUPLICATE_ID) {
      val question = QuestionDraft(1, "q", QuestionType.NOUL, "", "")
      KevDrafts.toRequest(RequestDraft("s", listOf(question, question.copy(key = 2))))
    }
    expect(DraftProblem.NO_OPTIONS) { KevDrafts.toRequest(draft(QuestionType.CHOICE, "\n \n")) }
    expect(DraftProblem.NO_OPTIONS) { KevDrafts.toRequest(draft(QuestionType.SCORE, "")) }
    expect(DraftProblem.TOO_MANY_OPTIONS) {
      KevDrafts.toRequest(draft(QuestionType.SCORE, (0..255).joinToString("\n")))
    }
    expect(DraftProblem.EMPTY_OPTION_NAME) {
      KevDrafts.toRequest(draft(QuestionType.CHOICE, ": no name"))
    }
    expect(DraftProblem.DUPLICATE_OPTION) {
      KevDrafts.toRequest(draft(QuestionType.CHOICE, "a\na: again"))
    }
    expect(DraftProblem.BAD_NOUL_OPTION) {
      KevDrafts.toRequest(draft(QuestionType.NOUL, "maybe: unsure"))
    }
  }

  @Test
  fun indentedJsonIsPythonsIndentTwo() {
    val value =
      linkedMapOf(
        "model" to "kev-latest",
        "answers" to
          linkedMapOf(
            "q" to linkedMapOf("type" to "noul", "noul" to 0.5),
            "c" to
              linkedMapOf(
                "type" to "choice",
                "choice" to "b",
                "confidence" to 0.12,
                "probabilities" to linkedMapOf("a" to 0.44, "b" to 0.56),
              ),
          ),
        "usage" to linkedMapOf("input_tokens" to 3),
        "e" to linkedMapOf<String, Any?>(),
        "l" to emptyList<Any?>(),
        "n" to listOf(1, listOf(2, linkedMapOf<String, Any?>())),
        "s" to "\u00e9\"\n",
      )
    // Python: json.dumps(value, indent=2, ensure_ascii=False)
    val python =
      "{\n  \"model\": \"kev-latest\",\n  \"answers\": {\n    \"q\": {\n      \"type\": \"noul\",\n      \"noul\": 0.5\n    },\n" +
        "    \"c\": {\n      \"type\": \"choice\",\n      \"choice\": \"b\",\n      \"confidence\": 0.12,\n      \"probabilities\": {\n" +
        "        \"a\": 0.44,\n        \"b\": 0.56\n      }\n    }\n  },\n  \"usage\": {\n    \"input_tokens\": 3\n  },\n  \"e\": {},\n" +
        "  \"l\": [],\n  \"n\": [\n    1,\n    [\n      2,\n      {}\n    ]\n  ],\n  \"s\": \"\u00e9\\\"\\n\"\n}"
    assertEquals(python, KevJson.writeIndented(value))
  }

  @Test
  fun fourDecimalStrings() {
    for ((value, text) in
      listOf(
        0.373 to "0.3730",
        0.0 to "0.0000",
        1.0 to "1.0000",
        0.8205 to "0.8205",
        1e-4 to "0.0001",
        -0.0 to "0.0000",
        2.1878 to "2.1878",
      )) {
      assertEquals(text, KevAnswerView.fourDecimals(value))
    }
    // Every 4-decimal number below 3 prints back as itself.
    for (n in 0..30_000) {
      val value = KevAnswers.roundProb(n / 10_000.0)
      assertEquals(
        java.math.BigDecimal(n).movePointLeft(4).toPlainString(),
        KevAnswerView.fourDecimals(value),
      )
    }
  }

  private fun draft(type: QuestionType, options: String, id: String = "q") =
    RequestDraft("state", listOf(QuestionDraft(1, id, type, "instructions", options)))

  private fun expect(problem: DraftProblem, block: () -> Unit) {
    try {
      block()
      fail("expected $problem")
    } catch (failure: KevDraftException) {
      assertEquals(problem, failure.problem)
    }
  }

  private companion object {
    /** The bundled examples in the app's order (ticket, incident, review). */
    val EXAMPLE_FILES =
      listOf("example_ticket.json", "example_incident.json", "example_review.json")
  }
}
