// SPDX-License-Identifier: Apache-2.0
package com.julia1

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class JuliaTokenizerParityTest {
  @Test
  fun allThreeHundredPythonStressEncodingsMatch() {
    val fixture = JuliaJson.asObject(JuliaTestData.read("fixtures/tokenizer_stress.json"))
    val cases = JuliaJson.asArray(fixture.getValue("cases"))
    val tokenizer = JuliaTestData.tokenizer
    val rows = cases.map { raw ->
      val case = JuliaJson.asObject(raw)
      val expected = JuliaTestData.ints(case.getValue("ids"))
      val actual = tokenizer.encode(case.getValue("text") as String, addSpecialTokens = false)
      linkedMapOf<String, Any?>(
        "id" to case["id"],
        "exact" to expected.contentEquals(actual),
        "expected_ids" to expected,
        "actual_ids" to actual,
      )
    }
    val mismatches = rows.filter { it["exact"] != true }
    JuliaTestData.report(
      "jvm_tokenizer.json",
      linkedMapOf<String, Any?>(
        "runtime" to System.getProperty("java.runtime.version"),
        "total" to rows.size,
        "exact" to rows.count { it["exact"] == true },
        "tokenizer_load_ms" to tokenizer.loadTimeMs,
        "vocabulary_size" to tokenizer.vocabularySize,
        "merge_count" to tokenizer.mergeCount,
        "rows" to rows,
      ),
    )
    println(
      "JULIA1_JVM_TOKENIZER exact=${rows.size - mismatches.size}/${rows.size} " +
        "load_ms=${tokenizer.loadTimeMs}"
    )
    assertEquals("Required stress corpus size", 300, cases.size)
    assertTrue("Tokenizer mismatches: ${mismatches.take(5)}", mismatches.isEmpty())
  }
}
