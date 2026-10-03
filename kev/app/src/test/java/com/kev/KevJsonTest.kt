package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** The order-preserving JSON reader / writer and Python's float `repr` (self-contained). */
class KevJsonTest {
  @Test
  fun objectsKeepFileOrderAndPythonDuplicateKeySemantics() {
    val value = KevJson.parse("""{"b": 1, "a": {"z": 0, "y": [1, 2]}, "c": null}""") as Map<*, *>
    assertEquals(listOf("b", "a", "c"), value.keys.toList())
    assertEquals(listOf("z", "y"), (value["a"] as Map<*, *>).keys.toList())
    assertTrue(value.containsKey("c"))
    assertNull(value["c"])
    // Python's dict: a repeated key keeps its first position and takes the last value.
    val repeated = KevJson.parse("""{"a": 1, "b": 2, "a": 3}""") as Map<*, *>
    assertEquals(listOf("a", "b"), repeated.keys.toList())
    assertEquals("3", (repeated["a"] as JsonNumber).literal)
  }

  @Test
  fun numbersKeepIntegerAndFloatApart() {
    val numbers = KevJson.parse("[1, 1.0, -0, -0.0, 1E5, 1e+16, 64.90, 100000000000000000000, 2.5e-7]") as List<*>
    val literals = numbers.map { (it as JsonNumber).literal }
    assertEquals(listOf("1", "1.0", "-0", "-0.0", "1E5", "1e+16", "64.90", "100000000000000000000", "2.5e-7"), literals)
    assertEquals(
      listOf(true, false, true, false, false, false, false, true, false),
      numbers.map { (it as JsonNumber).isInteger },
    )
    // Python: str(json.loads(literal)).
    assertEquals(
      listOf("1", "1.0", "0", "-0.0", "100000.0", "1e+16", "64.9", "100000000000000000000", "2.5e-07"),
      numbers.map { (it as JsonNumber).pythonString() },
    )
    val special = KevJson.parse("[NaN, Infinity, -Infinity]") as List<*>
    assertEquals(listOf("nan", "inf", "-inf"), special.map { (it as JsonNumber).pythonString() })
  }

  @Test
  fun stringsDecodeEscapesAndSurrogatePairs() {
    val text = KevJson.parse("\"a\\n\\\"b\\\\ \\/ \\u00e9 \\ud83d\\ude00 \\t raw é 💜\"")
    assertEquals("a\n\"b\\ / é \uD83D\uDE00 \t raw é 💜", text)
    assertEquals("💜", KevJson.parse("\"\\ud83d\\udc9c\""))
    assertEquals("", KevJson.parse("\"\""))
  }

  @Test
  fun malformedInputIsRejectedLikePython() {
    for (bad in listOf("[1,]", "{\"a\":1,}", "[01]", "\"tab\there\"", "{\"a\" 1}", "[1] x", "", "[1.]", "[.5]", "[+1]", "{'a': 1}", "\"\\x\"")) {
      try {
        KevJson.parse(bad)
        fail("accepted: $bad")
      } catch (expected: IllegalArgumentException) {
        // Python's json.loads raises JSONDecodeError for each of these.
      }
    }
  }

  @Test
  fun writerKeepsOrderAndPrintsFloatsLikePython() {
    val value =
      linkedMapOf(
        "b" to 1.0,
        "a" to listOf("x\n\"", null, true, 0.1, 1e16, 5, -0.0),
        "é" to linkedMapOf("k" to 0.0001, "z" to 1e-5),
      )
    assertEquals(
      "{\"b\":1.0,\"a\":[\"x\\n\\\"\",null,true,0.1,1e+16,5,-0.0],\"é\":{\"k\":0.0001,\"z\":1e-05}}",
      KevJson.write(value),
    )
    val roundTrip = KevJson.parse(KevJson.write(value))
    assertNull(OracleFixtures.jsonDifference(value, roundTrip))
    assertEquals("\"\\u0001\\u001f\"", KevJson.write("\u0001\u001f"))
  }

  @Test
  fun floatReprMatchesCPython() {
    val cases = (KevJson.parse(ExternalTestData.resource("python_numbers.json").readBytes()) as Map<*, *>)["repr"] as List<*>
    var checked = 0
    for (case in cases) {
      val entry = case as Map<*, *>
      val x = (entry["x"] as JsonNumber).toDouble()
      assertEquals("repr(${entry["x"]})", entry["repr"], PythonFloat.repr(x))
      checked++
    }
    assertTrue("repr cases", checked >= 70)
    assertEquals("1e+22", PythonFloat.repr(1e22))
    assertEquals("1e+23", PythonFloat.repr(1e23))
    assertEquals("5e-324", PythonFloat.repr(Double.MIN_VALUE))
    assertEquals("1.7976931348623157e+308", PythonFloat.repr(Double.MAX_VALUE))
    assertFalse(PythonFloat.repr(0.1 + 0.2) == "0.3")
  }
}
