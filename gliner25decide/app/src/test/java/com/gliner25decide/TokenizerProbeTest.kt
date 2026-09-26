package com.gliner25decide

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Edge strings through `encodeWord` against gliner2's runtime tokenizer (tokenize +
 * convert_tokens_to_ids). Covers the normalized `[UNK]` added token of GLiNER2.5-Decide's
 * `tokenizer.json`, whitespace normalization, NFC and added tokens inside strings.
 */
class TokenizerProbeTest {
  @Test
  fun edgeStringsMatchThePythonTokenizer() {
    val root = ExternalTestData.resolve()
    val tokenizer = GlinerTokenizer(ExternalTestData.tokenizer(root))
    val probes =
      JSONObject(ExternalTestData.resource("tokenizer_probes.json")).getJSONArray("probes")
    val failures = JSONArray()
    for (index in 0 until probes.length()) {
      val probe = probes.getJSONObject(index)
      val text = probe.getString("text")
      val expected = probe.getJSONArray("ids").let { ids -> List(ids.length()) { ids.getInt(it) } }
      val actual = tokenizer.encodeWord(text).toList()
      if (actual != expected) {
        failures.put(
          JSONObject()
            .put("text", text)
            .put("expected", JSONArray(expected))
            .put("actual", JSONArray(actual))
        )
      }
    }
    ExternalTestData.reportFile("tokenizer_probes.json")
      .writeText(
        JSONObject()
          .put("test", "TokenizerProbeTest")
          .put("probes", probes.length())
          .put("passed", probes.length() - failures.length())
          .put("failures", failures)
          .toString(2) + "\n"
      )
    val passed = probes.length() - failures.length()
    println("TOKENIZER_PROBES passed=$passed/${probes.length()} failures=$failures")
    assertEquals("probe failures: $failures", 0, failures.length())
  }
}
