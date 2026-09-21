// SPDX-License-Identifier: Apache-2.0
package com.laya

import java.io.File
import org.junit.Assert.assertTrue

/** Reads an explicitly supplied fixture bundle without depending on a conversion checkout. */
object LayaTestData {
  private val dataRoot: File by lazy {
    val path =
      requireNotNull(System.getProperty("laya.testData")) {
        "Set LAYA_TEST_DATA to the fixture bundle directory; see scripts/TEST_DATA.md"
      }
    File(path).canonicalFile.also {
      require(File(it, "fixtures").isDirectory) {
        "Missing fixtures: $it; see scripts/TEST_DATA.md"
      }
    }
  }
  val tokenizer: LayaTokenizer by lazy { LayaTokenizer(required("host_assets/tokenizer.json")) }

  fun required(relative: String): File =
    File(dataRoot, relative).also {
      require(it.isFile) { "Required local test fixture is missing: $it" }
    }

  fun read(relative: String): Any? = LayaJson.parse(required(relative))

  fun rows(window: Int): List<Map<String, Any?>> =
    LayaJson.asArray(read("fixtures/ml_rows_s$window.json")).map(LayaJson::asObject)

  fun ints(value: Any?): IntArray =
    LayaJson.asArray(value).map { (it as Number).toInt() }.toIntArray()

  fun floats(value: Any?): FloatArray =
    LayaJson.asArray(value).map { (it as Number).toFloat() }.toFloatArray()

  fun report(name: String, data: Any?) {
    val directory = File(requireNotNull(System.getProperty("laya.testResults")))
    val destination = File(directory, name)
    requireNotNull(destination.parentFile).mkdirs()
    destination.writeText(LayaJson.stringify(data) + "\n", Charsets.UTF_8)
  }

  /** Exact dictionaries, with Python's numeric equality and explicit object insertion order. */
  fun difference(expected: Any?, actual: Any?, path: String = "$"): String? {
    if (expected is Number && actual is Number) {
      return if (expected.toDouble() == actual.toDouble()) null
      else "$path expected=$expected actual=$actual"
    }
    if (expected is Map<*, *> && actual is Map<*, *>) {
      if (expected.keys.toList() != actual.keys.toList())
        return "$path keys expected=${expected.keys} actual=${actual.keys}"
      for (key in expected.keys) difference(expected[key], actual[key], "$path.$key")?.let {
        return it
      }
      return null
    }
    if (expected is List<*> && actual is List<*>) {
      if (expected.size != actual.size)
        return "$path size expected=${expected.size} actual=${actual.size}"
      for (i in expected.indices) difference(expected[i], actual[i], "$path[$i]")?.let {
        return it
      }
      return null
    }
    return if (expected == actual) null else "$path expected=$expected actual=$actual"
  }

  fun assertNoFailures(label: String, failures: List<Map<String, Any?>>) {
    assertTrue("$label: ${failures.size} failures; ${failures.take(8)}", failures.isEmpty())
  }
}
