package com.kev

import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile

/**
 * Readers for the author's fp32 oracle (`oracle/oracle_0.8b.json`: 402 questions and 377 requests,
 * written by the conversion run's `scripts/oracle_kev.py`), the fixture requests
 * (`fixtures/requests.json`) and the selected hidden states (`oracle/hidden_0.8b.npz`).
 */
internal object OracleFixtures {
  /** One question of the oracle: its row, readout indices, head outputs and answer. */
  class Question(
    val id: String,
    val source: String,
    val qid: String,
    val type: String,
    val keys: List<String>,
    val rowIds: IntArray,
    val decideIndex: Int,
    val optionIndices: IntArray,
    val zPre: DoubleArray,
    val zPost: DoubleArray,
    val probabilities: DoubleArray,
    val answer: Map<*, *>,
  ) {
    val key: String
      get() = "$id/$qid"
  }

  /** One request of the oracle: question count, usage and answers. */
  class Request(val id: String, val source: String, val questions: Int, val inputTokens: Int, val answers: Map<*, *>)

  class Oracle(
    val questions: List<Question>,
    val requests: List<Request>,
    val specialTokenIds: Map<*, *>,
    val padTokenId: Int,
    val temperature: Double,
  )

  /** One fixture record: id, source and the request object as parsed JSON. */
  class Record(val id: String, val source: String, val request: Any?)

  fun loadOracle(): Oracle {
    val json = KevJson.parse(ExternalTestData.file(ExternalTestData.ORACLE).readBytes()) as Map<*, *>
    val questions = (json["questions"] as List<*>).map { question(it as Map<*, *>) }
    val requests =
      (json["requests"] as List<*>).map {
        val request = it as Map<*, *>
        Request(
          request["id"] as String,
          request["source"] as String,
          (request["questions"] as JsonNumber).toInt(),
          ((request["usage"] as Map<*, *>)["input_tokens"] as JsonNumber).toInt(),
          request["answers"] as Map<*, *>,
        )
      }
    return Oracle(
      questions,
      requests,
      json["special_token_ids"] as Map<*, *>,
      (json["pad_token_id"] as JsonNumber).toInt(),
      (json["temperature"] as JsonNumber).toDouble(),
    )
  }

  fun question(json: Map<*, *>): Question =
    Question(
      json["id"] as String,
      json["source"] as String? ?: "",
      json["qid"] as String,
      json["type"] as String,
      (json["keys"] as List<*>).map { it as String },
      ints(json["row_ids"]),
      (json["decide_idx"] as JsonNumber).toInt(),
      ints(json["opt_idx"]),
      // The gate asset carries no logits.
      json["z_pre"]?.let { doubles(it) } ?: DoubleArray(0),
      json["z_post"]?.let { doubles(it) } ?: DoubleArray(0),
      doubles(json["probs"]),
      json["answer"] as Map<*, *>,
    )

  fun loadRecords(): List<Record> {
    val json = KevJson.parse(ExternalTestData.file(ExternalTestData.REQUESTS).readBytes()) as Map<*, *>
    return (json["records"] as List<*>).map {
      val record = it as Map<*, *>
      Record(record["id"] as String, record["source"] as String, record["request"])
    }
  }

  fun ints(value: Any?): IntArray = (value as List<*>).map { (it as JsonNumber).toInt() }.toIntArray()

  fun doubles(value: Any?): DoubleArray =
    (value as List<*>).map { (it as JsonNumber).toDouble() }.toDoubleArray()

  /** Index of the first difference of two ID arrays, or null when equal. */
  fun firstDifference(expected: IntArray, actual: IntArray): Int? =
    (0 until maxOf(expected.size, actual.size)).firstOrNull {
      expected.getOrNull(it) != actual.getOrNull(it)
    }

  /**
   * Structural equality of two parsed JSON values, numbers by double value: the same keys in the
   * same order, the same strings, the same array lengths. Returns the path of the first
   * difference, or null.
   */
  fun jsonDifference(expected: Any?, actual: Any?, path: String = "$"): String? {
    fun number(value: Any?): Double? =
      when (value) {
        is JsonNumber -> value.toDouble()
        is Double -> value
        is Float -> value.toDouble()
        is Int -> value.toDouble()
        is Long -> value.toDouble()
        else -> null
      }
    val expectedNumber = number(expected)
    val actualNumber = number(actual)
    if (expectedNumber != null || actualNumber != null) {
      // Doubles compare by bits so that -0.0 and 0.0 differ, as they print differently.
      return if (expectedNumber != null && actualNumber != null && expectedNumber.equals(actualNumber)) null
      else "$path: expected $expected, got $actual"
    }
    return when (expected) {
      is Map<*, *> -> {
        if (actual !is Map<*, *>) return "$path: expected an object, got $actual"
        if (expected.keys.toList() != actual.keys.toList()) {
          return "$path: keys ${expected.keys.toList()} vs ${actual.keys.toList()}"
        }
        expected.keys.firstNotNullOfOrNull { key -> jsonDifference(expected[key], actual[key], "$path.$key") }
      }
      is List<*> -> {
        if (actual !is List<*> || actual.size != expected.size) return "$path: expected $expected, got $actual"
        expected.indices.firstNotNullOfOrNull { jsonDifference(expected[it], actual[it], "$path[$it]") }
      }
      else -> if (expected == actual) null else "$path: expected $expected, got $actual"
    }
  }

  /** The `.npy` arrays of an uncompressed `.npz` (numpy.savez), read without numpy. */
  class Npz(file: File) : Closeable {
    private val zip = ZipFile(file)

    val size: Int
      get() = zip.size()

    /** The float32 array saved under [key] (`<id>/<qid>`): shape and row-major values. */
    fun floats(key: String): Pair<IntArray, FloatArray> {
      val entry = requireNotNull(zip.getEntry("$key.npy")) { "No $key in the npz" }
      val bytes = zip.getInputStream(entry).use { it.readBytes() }
      return parseNpy(bytes)
    }

    override fun close() = zip.close()

    private fun parseNpy(bytes: ByteArray): Pair<IntArray, FloatArray> {
      require(bytes.size > MAGIC_LENGTH && bytes[0] == MAGIC_FIRST && String(bytes, 1, 5, Charsets.ISO_8859_1) == "NUMPY") {
        "Not an .npy array"
      }
      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      val major = bytes[6].toInt()
      val (headerLength, headerStart) =
        if (major == 1) (buffer.getShort(8).toInt() and 0xffff) to 10 else buffer.getInt(8) to 12
      val header = String(bytes, headerStart, headerLength, Charsets.ISO_8859_1)
      require(header.contains("'descr': '<f4'") && header.contains("'fortran_order': False")) {
        "Unsupported .npy header: $header"
      }
      val shape =
        requireNotNull(Regex("'shape': \\(([0-9, ]*)\\)").find(header)) { "No shape in $header" }
          .groupValues[1]
          .split(',')
          .map { it.trim() }
          .filter { it.isNotEmpty() }
          .map { it.toInt() }
          .toIntArray()
      val count = shape.fold(1) { product, dimension -> product * dimension }
      val dataStart = headerStart + headerLength
      require(bytes.size - dataStart == count * 4) { "Array size does not match shape ${shape.toList()}" }
      val values = FloatArray(count)
      ByteBuffer.wrap(bytes, dataStart, count * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
      return shape to values
    }

    private companion object {
      const val MAGIC_LENGTH = 10
      const val MAGIC_FIRST = 0x93.toByte()
    }
  }

  /** Row [row] of a row-major [rows, columns] array. */
  fun row(values: FloatArray, row: Int, columns: Int): FloatArray =
    values.copyOfRange(row * columns, (row + 1) * columns)
}
