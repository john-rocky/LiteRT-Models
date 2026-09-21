// SPDX-License-Identifier: Apache-2.0
package com.laya

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LayaTokenizerParityTest {
  @Test
  fun allThreeHundredPythonStressEncodingsMatch() {
    val fixture = LayaJson.asObject(LayaTestData.read("fixtures/tokenizer_stress.json"))
    val cases = LayaJson.asArray(fixture.getValue("cases"))
    val tokenizer = LayaTestData.tokenizer
    val rows =
      cases.map { raw ->
        val case = LayaJson.asObject(raw)
        val expected =
          LayaJson.asArray(case.getValue("ids")).map { (it as Number).toInt() }.toIntArray()
        val actual = tokenizer.encode(case.getValue("text") as String, addSpecialTokens = false)
        linkedMapOf<String, Any?>(
          "id" to case["id"],
          "exact" to expected.contentEquals(actual),
          "expected_ids" to expected,
          "actual_ids" to actual,
        )
      }
    val mismatches = rows.filter { it["exact"] != true }
    val report =
      linkedMapOf<String, Any?>(
        "device" to "Apple M4 Max (supervisor supplied)",
        "runtime" to System.getProperty("java.runtime.version"),
        "total" to rows.size,
        "exact" to rows.count { it["exact"] == true },
        "tokenizer_load_ms" to tokenizer.loadTimeMs,
        "vocabulary_size" to tokenizer.vocabularySize,
        "merge_count" to tokenizer.mergeCount,
        "regex_flags_used" to false,
        "model_inference_executed" to false,
        "rows" to rows,
      )
    LayaTestData.report("jvm_tokenizer.json", report)
    println(
      "LAYA_JVM_TOKENIZER exact=${rows.size - mismatches.size}/${rows.size} load_ms=${tokenizer.loadTimeMs}"
    )
    assertEquals("Required stress corpus size", 300, cases.size)
    assertTrue("Tokenizer mismatches: ${mismatches.take(5)}", mismatches.isEmpty())
    assertTrue(
      "JVM tokenizer load exceeded approximately 2 seconds: ${tokenizer.loadTimeMs} ms",
      tokenizer.loadTimeMs < 2000.0,
    )
  }
}
