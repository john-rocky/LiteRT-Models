package com.gliner25decide

import java.io.File
import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** float16 table + host upcast against numpy (`float16.astype(float32)`). */
class EmbeddingTableTest {
  @Test
  fun rowsZeroAnd128010UpcastExactlyAsNumpy() {
    val root = ExternalTestData.resolve()
    val file = ExternalTestData.embeddingTable(root)
    val expected = JSONObject(ExternalTestData.resource("embedding_rows_fp16.json"))
    assertEquals(expected.getLong("table_bytes"), file.length())
    assertEquals(expected.getString("table_sha256"), sha256(file))
    val rows = expected.getJSONArray("rows")
    val checked = mutableListOf<Int>()
    DecideInputs.EmbeddingTable(file).use { table ->
      for (index in 0 until rows.length()) {
        val row = rows.getJSONObject(index)
        val id = row.getInt("id")
        val halfHex = row.getString("fp16_bits_hex")
        val floatHex = row.getString("float32_bits_hex")
        val upcast = table.lookup(intArrayOf(id))
        assertEquals(DecideInputs.HIDDEN_SIZE, upcast.size)
        for (column in 0 until DecideInputs.HIDDEN_SIZE) {
          val half = halfHex.substring(column * 4, column * 4 + 4).toInt(16)
          val bits = floatHex.substring(column * 8, column * 8 + 8).toLong(16).toInt()
          assertEquals("row $id column $column fp16 bits", half, table.halfBits(id, column))
          assertEquals("row $id column $column float32 bits", bits, upcast[column].toRawBits())
        }
        checked.add(id)
      }
    }
    assertEquals(listOf(0, 128010), checked)
    ExternalTestData.reportFile("embedding_table.json")
      .writeText(
        JSONObject()
          .put("test", "EmbeddingTableTest")
          .put("status", "PASS")
          .put("table_sha256", expected.getString("table_sha256"))
          .put("rows_bit_identical_to_numpy_upcast", checked)
          .put("values_per_row", DecideInputs.HIDDEN_SIZE)
          .toString(2) + "\n"
      )
  }

  @Test
  fun halfToFloatMatchesAnIndependentDecodeForEveryBitPattern() {
    var nan = 0
    for (bits in 0 until (1 shl 16)) {
      val actual = DecideInputs.halfToFloat(bits)
      val sign = if (bits and 0x8000 != 0) -1.0 else 1.0
      val exponent = (bits ushr 10) and 0x1f
      val mantissa = bits and 0x3ff
      when (exponent) {
        0x1f ->
          if (mantissa == 0) {
            assertEquals(sign * Double.POSITIVE_INFINITY, actual.toDouble(), 0.0)
          } else {
            assertTrue(actual.isNaN())
            // numpy keeps the sign and the payload shifted into the float32 significand.
            assertEquals(
              (bits and 0x8000 shl 16) or 0x7f800000 or (mantissa shl 13),
              actual.toRawBits(),
            )
            nan++
          }
        0 -> assertExact(sign * Math.scalb(mantissa.toDouble(), -24), actual, bits)
        else -> assertExact(sign * Math.scalb(1024.0 + mantissa, exponent - 25), actual, bits)
      }
    }
    assertEquals(2046, nan)
  }

  private fun assertExact(expected: Double, actual: Float, bits: Int) {
    // Every binary16 value is exactly representable in binary32; signed zero must survive.
    assertEquals("bits 0x${bits.toString(16)}", expected.toFloat().toRawBits(), actual.toRawBits())
    assertEquals(expected, actual.toDouble(), 0.0)
  }

  private fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered(1 shl 20).use { input ->
      val buffer = ByteArray(1 shl 20)
      while (true) {
        val read = input.read(buffer)
        if (read < 0) {
          break
        }
        digest.update(buffer, 0, read)
      }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }
}
