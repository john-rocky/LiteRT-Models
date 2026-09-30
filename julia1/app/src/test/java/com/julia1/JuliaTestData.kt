// SPDX-License-Identifier: Apache-2.0
package com.julia1

import java.io.File
import org.junit.Assert.assertTrue

/** Reads an explicitly supplied fixture bundle without depending on a conversion checkout. */
object JuliaTestData {
  private val dataRoot: File by lazy {
    val path =
      requireNotNull(System.getProperty("julia1.testData")) {
        "Set JULIA1_TEST_DATA to the fixture bundle directory; see scripts/TEST_DATA.md"
      }
    File(path).canonicalFile.also {
      require(File(it, "fixtures").isDirectory) {
        "Missing fixtures: $it; see scripts/TEST_DATA.md"
      }
    }
  }
  val tokenizer: JuliaTokenizer by lazy { JuliaTokenizer(required("host_assets/tokenizer.json")) }

  fun required(relative: String): File =
    File(dataRoot, relative).also {
      require(it.isFile) { "Required local test fixture is missing: $it" }
    }

  fun read(relative: String): Any? = JuliaJson.parse(required(relative))

  /** All 2,100 oracle requests with the reference host's ids, markers, qtype and logits. */
  fun requests(): List<Map<String, Any?>> =
    JuliaJson.asArray(JuliaJson.asObject(read("fixtures/gate_requests.json"))["rows"])
      .map(JuliaJson::asObject)

  fun ints(value: Any?): IntArray =
    JuliaJson.asArray(value).map { (it as Number).toInt() }.toIntArray()

  fun doubles(value: Any?): DoubleArray =
    JuliaJson.asArray(value).map { (it as Number).toDouble() }.toDoubleArray()

  /** The author's request row (state, question, type, options) of a fixture row. */
  fun question(row: Map<String, Any?>): JuliaQuestion {
    val request = JuliaJson.asObject(row["request"])
    val type = request["type"] as? String ?: "choice"
    val options = JuliaJson.asArray(request["options"]).map { it as String }
    val criteria: Any? =
      when (type) {
        "choice" -> options.indices.associateTo(LinkedHashMap()) { "option$it" to options[it] }
        "score" -> options
        else -> linkedMapOf("false" to options[0], "true" to options[1])
      }
    return JuliaQuestion(type, request["question"] as String, criteria)
  }

  fun report(name: String, data: Any?) {
    val directory = File(requireNotNull(System.getProperty("julia1.testResults")))
    val destination = File(directory, name)
    requireNotNull(destination.parentFile).mkdirs()
    destination.writeText(JuliaJson.stringify(data) + "\n", Charsets.UTF_8)
  }

  fun assertNoFailures(label: String, failures: List<Map<String, Any?>>) {
    assertTrue("$label: ${failures.size} failures; ${failures.take(8)}", failures.isEmpty())
  }
}
