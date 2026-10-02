package com.gliclass

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

/**
 * The Kotlin tokenizer against Python: the 552 linearized oracle strings (the ids the official
 * pipeline fed the model) and every stress string of `fixtures/tokenizer_stress.json` (Python
 * `tokenizers` with special tokens).
 */
class GliclassTokenizerTest {
  private lateinit var root: File
  private lateinit var tokenizer: GliclassTokenizer

  @Before
  fun load() {
    root = ExternalTestData.resolve()
    tokenizer = GliclassTokenizer(ExternalTestData.tokenizer(root))
  }

  @Test
  fun oracleStringsTokenizeLikeThePipeline() {
    val fixtures = OracleFixtures.load(root)
    assertEquals("oracle fixtures", 552, fixtures.size)
    val failures = JSONArray()
    for (fixture in fixtures) {
      val actual = tokenizer.encode(fixture.linearized)
      OracleFixtures.firstDifference(fixture.inputIds, actual)?.let { index ->
        failures.put(failure(fixture.id, fixture.linearized, fixture.inputIds, actual, index))
      }
    }
    report("tokenizer_oracle.json", "oracle strings", fixtures.size, failures)
    assertEquals("oracle strings with different ids: $failures", 0, failures.length())
  }

  @Test
  fun stressStringsMatchPythonTokenizers() {
    ExternalTestData.requireFiles(root, "fixtures/tokenizer_stress.json")
    val cases =
      JSONObject(File(root, "fixtures/tokenizer_stress.json").readText()).getJSONArray("cases")
    val failures = JSONArray()
    for (index in 0 until cases.length()) {
      val case = cases.getJSONObject(index)
      val text = case.getString("text")
      val expected =
        case.getJSONArray("ids").let { ids -> IntArray(ids.length()) { ids.getInt(it) } }
      val actual = tokenizer.encode(text)
      OracleFixtures.firstDifference(expected, actual)?.let { first ->
        failures.put(
          failure(case.getString("id"), text, expected, actual, first)
            .put("category", case.getString("category"))
            .put("python_tokens", case.getJSONArray("tokens"))
        )
      }
    }
    assertEquals("stress cases", true, cases.length() >= 120)
    report("tokenizer_stress.json", "stress strings", cases.length(), failures)
    assertEquals("stress strings with different ids: $failures", 0, failures.length())
  }

  @Test
  fun specialTokensAndTemplate() {
    assertEquals(50370, tokenizer.vocabularySize)
    assertEquals(50281, tokenizer.clsId)
    assertEquals(50282, tokenizer.sepId)
    assertEquals(50283, tokenizer.padId)
    assertEquals(GliclassInputs.LABEL_ID, tokenizer.tokenId(GliclassInputs.LABEL_TOKEN))
    assertEquals(
      GliclassInputs.TEXT_SEPARATOR_ID,
      tokenizer.tokenId(GliclassInputs.TEXT_SEPARATOR_TOKEN),
    )
    assertEquals(listOf(50281, 50282), tokenizer.encode("").toList())
  }

  private fun failure(id: String, text: String, expected: IntArray, actual: IntArray, index: Int) =
    JSONObject()
      .put("id", id)
      .put("text", text)
      .put("first_token_index", index)
      .put("expected", JSONArray(expected.toList()))
      .put("actual", JSONArray(actual.toList()))

  private fun report(name: String, what: String, total: Int, failures: JSONArray) {
    ExternalTestData.reportFile(name)
      .writeText(
        JSONObject()
          .put("test", "GliclassTokenizerTest")
          .put("set", what)
          .put("total", total)
          .put("passed", total - failures.length())
          .put("failures", failures)
          .toString(1) + "\n"
      )
    println("GLICLASS_TOKENIZER $what passed=${total - failures.length()}/$total")
  }
}
