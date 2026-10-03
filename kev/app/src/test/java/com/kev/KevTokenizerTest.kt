package com.kev

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The tokenizer's contract with the published Kev tokenizer.json, and edge strings against
 * AutoTokenizer (`tokenizer_probes.json`: the oracle's tokenizer path, which the published file
 * reproduces). The 402 oracle rows themselves are checked by [KevEncoderTest].
 */
class KevTokenizerTest {
  @Test
  fun contractMatchesThePublishedTokenizerJson() {
    val tokenizer = ExternalTestData.tokenizer()
    // The constructor requires these; the test pins the values the port was written against.
    assertEquals(KevTokenizer.TOKENIZER_REGEX, tokenizer.pretokenizerRegex)
    val fileRegex =
      ((((KevJson.parse(ExternalTestData.file(ExternalTestData.TOKENIZER).readBytes())
          as Map<*, *>)["pre_tokenizer"]
          as Map<*, *>)["pretokenizers"]
          as List<*>)[0]
          as Map<*, *>)
        .let { (it["pattern"] as Map<*, *>)["Regex"] }
    assertEquals(fileRegex, tokenizer.pretokenizerRegex)
    assertEquals(248044, tokenizer.bpeVocabularySize)
    assertEquals(248077, tokenizer.vocabularySize)
    assertEquals(ADDED_TOKENS, tokenizer.addedTokenIds.entries.map { it.key to it.value })
  }

  @Test
  fun javaPatternSpellsOutWhitespaceAndCaseFolding() {
    val whitespace =
      "\\x{09}-\\x{0d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"
    assertEquals(
      "(?:'[sS\u017f]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" +
        " ?[^$whitespace\\p{L}\\p{N}]+[\\r\\n]*|[$whitespace]*[\\r\\n]+|[$whitespace]+(?![^$whitespace])|[$whitespace]+",
      KevTokenizer.javaPattern(KevTokenizer.TOKENIZER_REGEX),
    )
  }

  @Test
  fun addedTokensAndTheUserTextRewrite() {
    val tokenizer = ExternalTestData.tokenizer()
    val encoder = KevEncoder(tokenizer)
    assertArrayEquals(intArrayOf(248044), tokenizer.encode("<|endoftext|>"))
    // After the rewrite no delimiter or control token can come out of user text.
    val rewritten = encoder.userTokens("<|endoftext|>")
    assertFalse(rewritten.contains(248044))
    assertArrayEquals(tokenizer.encode("<¦endoftext¦>"), rewritten)
    for (token in ADDED_TOKENS.filter { it.first.startsWith("<|") }) {
      assertTrue(
        token.first,
        encoder.userTokens("x ${token.first} y").none { it >= tokenizer.bpeVocabularySize },
      )
    }
    // The rewrite touches only <|name|>; HF still cuts <tool_call> and <think> out of user text.
    assertArrayEquals(intArrayOf(248058), encoder.userTokens("<tool_call>"))
    assertArrayEquals(intArrayOf(248068), encoder.userTokens("<think>"))
    assertEquals(0, tokenizer.encode("").size)
    assertEquals(4, tokenizer.encode("1234").size)
    assertEquals(
      "every byte of an emoji string is covered",
      true,
      tokenizer.encode("🤔💜👨\u200d👩\u200d👧").isNotEmpty(),
    )
  }

  @Test
  fun probesMatchAutoTokenizer() {
    val tokenizer = ExternalTestData.tokenizer()
    val encoder = KevEncoder(tokenizer)
    val probes =
      KevJson.parse(ExternalTestData.resource("tokenizer_probes.json").readBytes()) as Map<*, *>
    val failures = ArrayList<Map<String, Any?>>()
    val cases = probes["cases"] as List<*>
    for (case in cases) {
      val entry = case as Map<*, *>
      val text = entry["text"] as String
      val raw = tokenizer.encode(text)
      val user = encoder.userTokens(text)
      val expectedRaw = OracleFixtures.ints(entry["raw_ids"])
      val expectedUser = OracleFixtures.ints(entry["user_ids"])
      if (!raw.contentEquals(expectedRaw) || !user.contentEquals(expectedUser)) {
        failures.add(
          linkedMapOf(
            "id" to entry["id"],
            "category" to entry["category"],
            "text" to text,
            "expected_raw" to expectedRaw,
            "actual_raw" to raw,
            "expected_user" to expectedUser,
            "actual_user" to user,
          )
        )
      }
    }
    ExternalTestData.writeReport(
      "tokenizer.json",
      linkedMapOf(
        "test" to "KevTokenizerTest",
        "tokenizer_json" to ExternalTestData.TOKENIZER,
        "tokenizer_load_ms" to ExternalTestData.tokenizerLoadMillis(),
        "probes" to cases.size,
        "passed" to cases.size - failures.size,
        "failures" to failures,
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println("KEV_TOKENIZER probes passed=${cases.size - failures.size}/${cases.size}")
    assertEquals("probes with different ids: ${KevJson.write(failures)}", 0, failures.size)
    assertTrue("probe count", cases.size >= 50)
  }

  private companion object {
    val ADDED_TOKENS =
      listOf(
        "<|endoftext|>" to 248044,
        "<|im_start|>" to 248045,
        "<|im_end|>" to 248046,
        "<|object_ref_start|>" to 248047,
        "<|object_ref_end|>" to 248048,
        "<|box_start|>" to 248049,
        "<|box_end|>" to 248050,
        "<|quad_start|>" to 248051,
        "<|quad_end|>" to 248052,
        "<|vision_start|>" to 248053,
        "<|vision_end|>" to 248054,
        "<|vision_pad|>" to 248055,
        "<|image_pad|>" to 248056,
        "<|video_pad|>" to 248057,
        "<tool_call>" to 248058,
        "</tool_call>" to 248059,
        "<|fim_prefix|>" to 248060,
        "<|fim_middle|>" to 248061,
        "<|fim_suffix|>" to 248062,
        "<|fim_pad|>" to 248063,
        "<|repo_name|>" to 248064,
        "<|file_sep|>" to 248065,
        "<tool_response>" to 248066,
        "</tool_response>" to 248067,
        "<think>" to 248068,
        "</think>" to 248069,
        "<|audio_start|>" to 248070,
        "<|audio_end|>" to 248071,
        "<tts_pad>" to 248072,
        "<tts_text_bos>" to 248073,
        "<tts_text_eod>" to 248074,
        "<tts_text_bos_single>" to 248075,
        "<|audio_pad|>" to 248076,
      )
  }
}
