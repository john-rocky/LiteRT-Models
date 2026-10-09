package com.d1omni

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The tokenizer's contract with the repository's tokenizer.json, and every string the provider's
 * `encode` tokenizes for the public check set plus edge strings, against Hugging Face `tokenizers`
 * (`fixtures/tokenizer_probes.json`: raw text and the `escape`d text `encode` feeds it).
 */
class D1TokenizerTest {
  @Test
  fun contractMatchesTheRepositoryTokenizerJson() {
    val tokenizer = ExternalTestData.tokenizer()
    assertEquals(D1Tokenizer.TOKENIZER_REGEX, tokenizer.pretokenizerRegex)
    val file =
      ExternalTestData.json(ExternalTestData.repoFile("tokenizer.json"))["pre_tokenizer"]
        as Map<*, *>
    val fileRegex =
      (((file["pretokenizers"] as List<*>)[0] as Map<*, *>)["pattern"] as Map<*, *>)["Regex"]
    assertEquals(fileRegex, tokenizer.pretokenizerRegex)
    assertEquals(64400, tokenizer.bpeVocabularySize)
    assertEquals(64402, tokenizer.vocabularySize)
    assertEquals(509, tokenizer.addedTokenIds.size)
    // Only two added tokens are matched after normalization, and both are words.
    assertEquals(
      listOf("Mathias", "python"),
      tokenizer.addedTokenNormalized.filterValues { it }.keys.toList(),
    )
    assertEquals(64011, tokenizer.tokenId("Mathias"))
    assertEquals(64014, tokenizer.tokenId("python"))
    assertEquals(64400, tokenizer.tokenId("<think>"))
    assertEquals(64401, tokenizer.tokenId("</think>"))
    assertEquals(396, tokenizer.tokenId("<image>"))
    ExternalTestData.contract().checkTokenizer(tokenizer)
  }

  @Test
  fun javaPatternSpellsOutWhitespaceAndCaseFolding() {
    val whitespace =
      "\\x{09}-\\x{0d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"
    assertEquals(
      "(?:'[sSſ]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|" +
        " ?[^$whitespace\\p{L}\\p{N}]+[\\r\\n]*|[$whitespace]*[\\r\\n]+|[$whitespace]+(?![^$whitespace])|[$whitespace]+",
      D1Tokenizer.javaPattern(D1Tokenizer.TOKENIZER_REGEX),
    )
  }

  @Test
  fun addedTokensAndTheEscape() {
    val tokenizer = ExternalTestData.tokenizer()
    assertArrayEquals(intArrayOf(16), tokenizer.encode("<|mask|>"))
    // After the escape no delimiter or marker can come out of caller text.
    val escaped = tokenizer.encode(D1Prompt.escape("x <|mask|> <|reserved_7|> y"))
    assertTrue(escaped.none { it in setOf(16, 17, 18, 19, 20, 21) })
    // The escape touches only <|name|>: the tokenizer still cuts the other added tokens out.
    assertArrayEquals(intArrayOf(64400), tokenizer.encode(D1Prompt.escape("<think>")))
    assertArrayEquals(intArrayOf(64014), tokenizer.encode(D1Prompt.escape("python")))
    assertEquals(0, tokenizer.encode("").size)
  }

  @Test
  fun casesMatchTokenizers() {
    val tokenizer = ExternalTestData.tokenizer()
    val data = ExternalTestData.json(ExternalTestData.demoFile(ExternalTestData.TOKENIZER_CASES))
    assertEquals(
      D1Contract.sha256(ExternalTestData.repoFile("tokenizer.json")),
      data["tokenizer_json_sha256"],
    )
    val cases = data["cases"] as List<*>
    val failures = ArrayList<Map<String, Any?>>()
    val byCategory = LinkedHashMap<String, Int>()
    for (case in cases) {
      val entry = case as Map<*, *>
      val text = entry["text"] as String
      val raw = tokenizer.encode(text)
      val enc = tokenizer.encode(D1Prompt.escape(text))
      val category = entry["category"] as String
      byCategory[category] = (byCategory[category] ?: 0) + 1
      if (
        !raw.contentEquals(ExternalTestData.ints(entry["raw_ids"])) ||
          !enc.contentEquals(ExternalTestData.ints(entry["enc_ids"]))
      ) {
        failures.add(
          linkedMapOf(
            "id" to entry["id"],
            "text" to text,
            "expected_raw" to ExternalTestData.ints(entry["raw_ids"]),
            "actual_raw" to raw,
            "expected_enc" to ExternalTestData.ints(entry["enc_ids"]),
            "actual_enc" to enc,
          )
        )
      }
    }
    ExternalTestData.writeReport(
      "tokenizer.json",
      linkedMapOf(
        "test" to "D1TokenizerTest",
        "tokenizer_load_ms" to ExternalTestData.tokenizerLoadMillis(),
        "cases" to cases.size,
        "by_category" to byCategory,
        "passed" to cases.size - failures.size,
        "failures" to failures,
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println("D1_TOKENIZER cases passed=${cases.size - failures.size}/${cases.size} $byCategory")
    assertEquals("cases with different ids: ${D1Json.write(failures.take(5))}", 0, failures.size)
    assertTrue("edge cases", (byCategory["edge"] ?: 0) >= 40)
    assertTrue("fixture strings", cases.size - (byCategory["edge"] ?: 0) >= 700)
  }
}
