package com.gliner25decide

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The two schema forms beyond plain label lists: `prompt` (readme_18) and `{label: description}`
 * (readme_20). gliner2 tokenizes each `prompt_str` as ONE token; `encodeWord(prompt_str)` must give
 * the same IDs, including `[DESCRIPTION]` found inside the string.
 */
class SchemaTest {
  @Test
  fun promptAndLabelDescriptionSchemasMatchGliner2() {
    val root = ExternalTestData.resolve()
    val oracle = OracleFixtures.load(root).associateBy { it.id }
    val expected =
      JSONObject(ExternalTestData.resource("schema_readme_18_20.json")).getJSONArray("fixtures")
    val tokenizer = GlinerTokenizer(ExternalTestData.tokenizer(root))
    val inputs = DecideInputs(tokenizer)
    val report = JSONArray()
    assertEquals(2, expected.length())
    for (index in 0 until expected.length()) {
      val row = expected.getJSONObject(index)
      val fixture = requireNotNull(oracle[row.getString("id")])
      assertEquals(row.getString("text"), fixture.text)
      val schemaTokens = row.getJSONArray("schema_tokens_list")
      assertEquals(fixture.tasks.size, schemaTokens.length())
      fixture.tasks.forEachIndexed { taskIndex, task ->
        val tokens = schemaTokens.getJSONArray(taskIndex)
        assertEquals(
          "${fixture.id} schema tokens",
          List(tokens.length()) { tokens.getString(it) },
          DecideSchema.schemaTokens(task),
        )
      }
      val encoded = inputs.encode(fixture.text, fixture.tasks)
      assertArrayEquals("${fixture.id} input_ids", fixture.inputIds, encoded.inputIds)
      assertArrayEquals(
        "${fixture.id} label positions",
        fixture.labelPositions,
        encoded.labelPositions,
      )
      assertEquals(fixture.schemaSpecialIndices.size, encoded.schemaSpecialPositions.size)
      fixture.schemaSpecialIndices.zip(encoded.schemaSpecialPositions).forEach { (a, b) ->
        assertArrayEquals("${fixture.id} schema special indices", a, b)
      }
      if (!row.isNull("text_tokens")) {
        val words = row.getJSONArray("text_tokens")
        assertEquals(List(words.length()) { words.getString(it) }, encoded.words.map { it.text })
      }
      val promptString = DecideSchema.promptString(fixture.tasks.single())
      val promptIds = tokenizer.encodeWord(promptString)
      assertArrayEquals(
        "${fixture.id} prompt_str ids",
        fixture.inputIds.copyOfRange(2, 2 + promptIds.size),
        promptIds,
      )
      report.put(
        JSONObject()
          .put("id", fixture.id)
          .put("prompt_str", promptString)
          .put("prompt_str_token_count", promptIds.size)
          .put("schema_tokens_identical", true)
          .put("input_ids_identical", true)
          .put("encoded_length", encoded.encodedLength)
      )
    }
    ExternalTestData.reportFile("schema_parity.json")
      .writeText(
        JSONObject()
          .put("test", "SchemaTest")
          .put("status", "PASS")
          .put("fixtures", report)
          .toString(2) + "\n"
      )
  }
}
