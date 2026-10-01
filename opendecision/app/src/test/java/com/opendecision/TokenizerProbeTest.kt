package com.opendecision

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Edge strings through `encode` against the official fast tokenizer (`tokenizers` on `tokenizer.json`,
 * `add_special_tokens=False`): whitespace stripping, the precompiled charsmap (compatibility characters, full-width
 * letters, ligatures), added tokens inside text, the normalized `[UNK]`, emoji and non-Latin scripts.
 */
class TokenizerProbeTest {
  @Test
  fun edgeStringsMatchThePythonTokenizer() {
    val root = ExternalTestData.resolve()
    val tokenizer = DecisionTokenizer(ExternalTestData.tokenizer(root))
    val probes = JSONObject(ExternalTestData.resource("tokenizer_probes.json")).getJSONArray("probes")
    val failures = JSONArray()
    for (index in 0 until probes.length()) {
      val probe = probes.getJSONObject(index)
      val text = probe.getString("text")
      val expected = probe.getJSONArray("ids").let { ids -> List(ids.length()) { ids.getInt(it) } }
      val actual = tokenizer.encode(text).toList()
      if (actual != expected) {
        failures.put(JSONObject().put("text", text).put("expected", JSONArray(expected)).put("actual", JSONArray(actual)))
      }
    }
    ExternalTestData.reportFile("tokenizer_probes.json")
      .writeText(JSONObject().put("probes", probes.length()).put("passed", probes.length() - failures.length()).put("failures", failures).toString(2) + "\n")
    println("TOKENIZER_PROBES passed=${probes.length() - failures.length()}/${probes.length()} failures=$failures")
    assertEquals("probe failures: $failures", 0, failures.length())
  }
}
