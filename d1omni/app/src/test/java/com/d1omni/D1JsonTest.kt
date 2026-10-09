package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** The order-preserving JSON reader / writer and Python's float `repr` (self-contained). */
class D1JsonTest {
  @Test
  fun objectsKeepFileOrderAndPythonDuplicateKeySemantics() {
    val value = D1Json.parse("""{"b": 1, "a": {"z": 0, "y": [1, 2]}, "c": null}""") as Map<*, *>
    assertEquals(listOf("b", "a", "c"), value.keys.toList())
    assertEquals(listOf("z", "y"), (value["a"] as Map<*, *>).keys.toList())
    assertTrue(value.containsKey("c"))
    assertNull(value["c"])
    // Python's dict: a repeated key keeps its first position and takes the last value.
    val repeated = D1Json.parse("""{"a": 1, "b": 2, "a": 3}""") as Map<*, *>
    assertEquals(listOf("a", "b"), repeated.keys.toList())
    assertEquals("3", (repeated["a"] as JsonNumber).literal)
  }

  @Test
  fun numbersKeepIntegerAndFloatApart() {
    val numbers =
      D1Json.parse("[1, 1.0, -0, -0.0, 1E5, 1e+16, 64.90, 100000000000000000000, 2.5e-7]")
        as List<*>
    assertEquals(
      listOf(true, false, true, false, false, false, false, true, false),
      numbers.map { (it as JsonNumber).isInteger },
    )
    // Python: str(json.loads(literal)).
    assertEquals(
      listOf("1", "1.0", "0", "-0.0", "100000.0", "1e+16", "64.9", "100000000000000000000", "2.5e-07"),
      numbers.map { (it as JsonNumber).pythonString() },
    )
    val special = D1Json.parse("[NaN, Infinity, -Infinity]") as List<*>
    assertEquals(listOf("nan", "inf", "-inf"), special.map { (it as JsonNumber).pythonString() })
  }

  @Test
  fun stringsDecodeEscapesAndSurrogatePairs() {
    val text = D1Json.parse("\"a\\n\\\"b\\\\ \\/ \\u00e9 \\ud83d\\ude00 \\t raw é 💜\"")
    assertEquals("a\n\"b\\ / é \uD83D\uDE00 \t raw é 💜", text)
    assertEquals("", D1Json.parse("\"\""))
  }

  @Test
  fun malformedInputIsRejectedLikePython() {
    for (bad in
      listOf("[1,]", "{\"a\":1,}", "[01]", "\"tab\there\"", "{\"a\" 1}", "[1] x", "", "[1.]", "[.5]")) {
      try {
        D1Json.parse(bad)
        fail("accepted: $bad")
      } catch (expected: IllegalArgumentException) {
        // Python's json.loads raises JSONDecodeError for each of these.
      }
    }
  }

  @Test
  fun dumpsUsesPythonDefaultSeparators() {
    val value = D1Json.parse("""{"a": 1, "b": 1.0, "c": [1, "x", null, true], "d": {}, "e": []}""")
    assertEquals(
      """{"a": 1, "b": 1.0, "c": [1, "x", null, true], "d": {}, "e": []}""",
      D1Json.dumps(value),
    )
    assertEquals("""{"a":1,"b":1.0,"c":[1,"x",null,true],"d":{},"e":[]}""", D1Json.write(value))
    assertEquals("\"\\u0001\\u001f\\b\\f\\n\\r\\t\\\"\\\\é\u007f\"", D1Json.dumps("\u0001\u001f\b\u000c\n\r\t\"\\é\u007f"))
    assertEquals("[NaN, Infinity, -Infinity]", D1Json.dumps(D1Json.parse("[NaN, Infinity, -Infinity]")))
  }

  @Test
  fun floatReprMatchesCPython() {
    val cases =
      mapOf(
        0.1 to "0.1",
        1.0 to "1.0",
        1e16 to "1e+16",
        1e15 to "1000000000000000.0",
        0.0001 to "0.0001",
        0.00001 to "1e-05",
        123456789.123 to "123456789.123",
        1e22 to "1e+22",
        Double.MIN_VALUE to "5e-324",
        Double.MAX_VALUE to "1.7976931348623157e+308",
        0.998469889163971 to "0.998469889163971",
        -2.5 to "-2.5",
      )
    for ((value, repr) in cases) assertEquals("repr($value)", repr, PythonFloat.repr(value))
    assertTrue(PythonFloat.repr(0.1 + 0.2) == "0.30000000000000004")
  }
}
