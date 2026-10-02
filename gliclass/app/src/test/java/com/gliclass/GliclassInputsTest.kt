package com.gliclass

import java.io.File
import java.io.RandomAccessFile
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Linearization, ids, `<<LABEL>>` positions, window choice and the padded graph inputs of every
 * oracle request against the captured official pipeline call, plus the rejection rules.
 */
class GliclassInputsTest {
  private lateinit var root: File
  private lateinit var table: GliclassInputs.EmbeddingTable
  private lateinit var inputs: GliclassInputs

  @Before
  fun load() {
    root = ExternalTestData.resolve()
    table = GliclassInputs.EmbeddingTable(ExternalTestData.embeddingTable(root))
    inputs = GliclassInputs(GliclassTokenizer(ExternalTestData.tokenizer(root)), table)
  }

  @After
  fun close() {
    if (::table.isInitialized) {
      table.close()
    }
  }

  @Test
  fun capturedInputsMatchForEveryOracleRequest() {
    val fixtures = OracleFixtures.load(root)
    assertEquals(552, fixtures.size)
    val rawTable = readTable(ExternalTestData.embeddingTable(root))
    val failures = JSONArray()
    val windows = linkedMapOf(128 to 0, 256 to 0)
    var passed = 0
    var paddedPairs = 0
    for (fixture in fixtures) {
      val problems = JSONObject()
      if (
        GliclassInputs.linearize(fixture.text, fixture.labels, fixture.prompt) != fixture.linearized
      ) {
        problems.put("linearized", "differs")
      }
      val encoded = inputs.encode(fixture.text, fixture.labels, fixture.prompt)
      OracleFixtures.firstDifference(fixture.inputIds, encoded.inputIds)?.let {
        problems.put("input_ids_first_difference", it)
      }
      OracleFixtures.firstDifference(fixture.labelPositions, encoded.labelPositions)?.let {
        problems.put("label_positions_first_difference", it)
      }
      val smallest = inputs.pad(encoded).window
      if (smallest != fixture.window) {
        problems.put("smallest_window", "$smallest != ${fixture.window}")
      }
      for (window in GliclassInputs.WINDOWS.filter { it >= fixture.window }) {
        if (!paddedMatches(fixture, inputs.pad(encoded, window), rawTable)) {
          problems.put("padded_s$window", "differs")
        }
        paddedPairs++
      }
      if (problems.length() == 0) {
        passed++
        windows[smallest] = windows.getValue(smallest) + 1
      } else {
        failures.put(problems.put("id", fixture.id))
      }
    }
    ExternalTestData.reportFile("inputs_parity.json")
      .writeText(
        JSONObject()
          .put("test", "GliclassInputsTest")
          .put("fixtures", fixtures.size)
          .put("passed", passed)
          .put("padded_window_pairs_checked", paddedPairs)
          .put("smallest_window_counts", JSONObject(windows.mapKeys { "s${it.key}" }))
          .put(
            "compared",
            JSONArray(
              listOf(
                "linearized string",
                "input_ids (unpadded, exact)",
                "label_positions",
                "smallest fitting window",
                "padded ids / attention / label_routing / float32 embeddings" +
                  " at every fitting window",
              )
            ),
          )
          .put("failures", failures)
          .toString(1) + "\n"
      )
    println(
      "GLICLASS_INPUTS passed=$passed/${fixtures.size} padded_pairs=$paddedPairs windows=$windows"
    )
    assertEquals("requests whose inputs match the captured call: $failures", fixtures.size, passed)
    assertEquals(mapOf(128 to 482, 256 to 70), windows)
  }

  @Test
  fun rejectsWhatTheGraphCannotHold() {
    val text = "A short text."
    assertThrows(IllegalArgumentException::class.java) { inputs.encode(text, emptyList()) }
    assertThrows(IllegalArgumentException::class.java) {
      inputs.encode(text, (1..26).map { "label $it" })
    }
    // 25 labels are accepted.
    assertEquals(25, inputs.encode(text, (1..25).map { "label $it" }).labelCount)
    // A marker inside the text, prompt or a label would route a label to the wrong token.
    assertThrows(IllegalArgumentException::class.java) {
      inputs.encode("x <<LABEL>> y", listOf("a", "b"))
    }
    assertThrows(IllegalArgumentException::class.java) {
      inputs.encode(text, listOf("a<<LABEL>>b"))
    }
    val long = (1..300).joinToString(" ") { "word$it" }
    assertThrows(IllegalArgumentException::class.java) { inputs.prepare(long, listOf("a", "b")) }
    val encoded = inputs.encode((1..80).joinToString(" ") { "word$it" }, listOf("a", "b"))
    assertTrue(encoded.encodedLength > 128)
    assertEquals(256, inputs.pad(encoded).window)
    assertThrows(IllegalArgumentException::class.java) { inputs.pad(encoded, 128) }
  }

  private fun paddedMatches(
    fixture: OracleFixtures.Fixture,
    prepared: GliclassInputs.Prepared,
    rawTable: ShortArray,
  ): Boolean {
    val n = prepared.window
    val length = fixture.inputIds.size
    val expectedIds = IntArray(n) { if (it < length) fixture.inputIds[it] else PAD_ID }
    if (!expectedIds.contentEquals(prepared.inputIds)) {
      return false
    }
    val attention = FloatArray(n) { if (it < length) 1f else 0f }
    if (!attention.contentEquals(prepared.attentionMask)) {
      return false
    }
    val routing = FloatArray(GliclassInputs.LABEL_SLOTS * n)
    fixture.labelPositions.forEachIndexed { row, position -> routing[row * n + position] = 1f }
    if (!routing.contentEquals(prepared.labelRouting)) {
      return false
    }
    // Embeddings: every value = the independent float16 decode of the table's raw bits.
    for (row in 0 until n) {
      val base = expectedIds[row] * GliclassInputs.HIDDEN_SIZE
      for (column in 0 until GliclassInputs.HIDDEN_SIZE) {
        val expected = decodeHalf(rawTable[base + column].toInt() and 0xffff)
        val actual = prepared.embeds[row * GliclassInputs.HIDDEN_SIZE + column]
        if (expected.toRawBits() != actual.toRawBits()) {
          return false
        }
      }
    }
    return true
  }

  @Test
  fun halfToFloatMatchesAnIndependentDecodeForEveryBitPattern() {
    for (bits in 0 until (1 shl 16)) {
      val actual = GliclassInputs.halfToFloat(bits)
      val expected = decodeHalf(bits)
      if (expected.isNaN()) {
        assertTrue(actual.isNaN())
        assertEquals(
          (bits and 0x8000 shl 16) or 0x7f800000 or ((bits and 0x3ff) shl 13),
          actual.toRawBits(),
        )
      } else {
        assertEquals("bits 0x${bits.toString(16)}", expected.toRawBits(), actual.toRawBits())
      }
    }
    assertArrayEquals(
      floatArrayOf(1f, -2f, 0.5f),
      floatArrayOf(
        GliclassInputs.halfToFloat(0x3c00),
        GliclassInputs.halfToFloat(0xc000),
        GliclassInputs.halfToFloat(0x3800),
      ),
      0f,
    )
  }

  private fun decodeHalf(bits: Int): Float {
    val sign = if (bits and 0x8000 != 0) -1.0 else 1.0
    val exponent = (bits ushr 10) and 0x1f
    val mantissa = bits and 0x3ff
    return when (exponent) {
      0x1f -> if (mantissa == 0) (sign * Double.POSITIVE_INFINITY).toFloat() else Float.NaN
      0 -> (sign * Math.scalb(mantissa.toDouble(), -24)).toFloat()
      else -> (sign * Math.scalb(1024.0 + mantissa, exponent - 25)).toFloat()
    }
  }

  private fun readTable(file: File): ShortArray {
    val bytes = ByteArray(file.length().toInt())
    RandomAccessFile(file, "r").use { it.readFully(bytes) }
    return ShortArray(bytes.size / 2) {
      ((bytes[2 * it].toInt() and 0xff) or ((bytes[2 * it + 1].toInt() and 0xff) shl 8)).toShort()
    }
  }

  private companion object {
    const val PAD_ID = 50283
  }
}
