// SPDX-License-Identifier: Apache-2.0
package com.laya

import org.junit.Assert.assertEquals
import org.junit.Test

class LayaSerializationParityTest {
  @Test
  fun allTwentyStatesMatchPythonJsonDumps() {
    val data = LayaJson.asObject(LayaTestData.read("fixtures/serialization_reference.json"))
    val cases = LayaJson.asArray(data["cases"]).map(LayaJson::asObject)
    assertEquals(20, cases.size)
    val details =
      cases.map { case ->
        val actual = LayaPromptBuilder.serializeState(case["state"])
        linkedMapOf<String, Any?>(
          "id" to case["id"],
          "exact" to (actual == case["serialized"]),
          "expected" to case["serialized"],
          "actual" to actual,
        )
      }
    val failures = details.filter { it["exact"] != true }
    LayaTestData.report(
      "jvm_serialization_parity.json",
      linkedMapOf(
        "status" to if (failures.isEmpty()) "PASS" else "FAIL",
        "cases" to cases.size,
        "exact" to cases.size - failures.size,
        "details" to details,
      ),
    )
    LayaTestData.assertNoFailures("serialize_state parity", failures)
  }
}
