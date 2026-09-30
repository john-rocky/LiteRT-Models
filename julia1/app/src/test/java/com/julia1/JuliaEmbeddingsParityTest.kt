// SPDX-License-Identifier: Apache-2.0
package com.julia1

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class JuliaEmbeddingsParityTest {
  @Test
  fun everyHalfBitPatternExpandsLikeNumPy() {
    val reference =
      ByteBuffer.wrap(JuliaTestData.required("fixtures/fp16_conversion_reference.bin").readBytes())
        .order(ByteOrder.LITTLE_ENDIAN)
    assertEquals(65536 * 4, reference.limit())
    var mismatches = 0
    for (bits in 0 until 65536) {
      val expected = reference.getInt(bits * 4)
      val actual = JuliaEmbeddings.halfToFloat(bits).toRawBits()
      if (expected != actual) {
        mismatches++
      }
    }
    println("JULIA1_JVM_HALF mismatches=$mismatches")
    assertEquals("binary16 patterns that differ from NumPy", 0, mismatches)
  }

  @Test
  fun gatheredRowsMatchTheNumPyLookupIncludingPadding() {
    val manifest =
      JuliaJson.asObject(JuliaTestData.read("fixtures/embedding_lookup_reference.json"))
    val ids = JuliaJson.asArray(manifest["rows"]).map { it as String }
    val window = (manifest["window"] as Number).toInt()
    val requests = JuliaTestData.requests().associateBy { it["id"] as String }
    val reference =
      ByteBuffer.wrap(JuliaTestData.required("fixtures/embedding_lookup_reference.bin").readBytes())
        .order(ByteOrder.LITTLE_ENDIAN)
    assertEquals(ids.size * window * JuliaEmbeddings.WIDTH * 4, reference.limit())
    val table = JuliaEmbeddings(JuliaTestData.required("host_assets/julia1_token_table_fp16.bin"))
    var mismatches = 0
    var offset = 0
    ids.forEach { id ->
      val gathered = table.gather(JuliaTestData.ints(requests.getValue(id)["ids"]), window)
      gathered.forEach { value ->
        if (reference.getInt(offset) != value.toRawBits()) {
          mismatches++
        }
        offset += 4
      }
    }
    table.close()
    println("JULIA1_JVM_EMBEDDINGS rows=${ids.size} values=${offset / 4} mismatches=$mismatches")
    assertTrue("Embedding values that differ from NumPy: $mismatches", mismatches == 0)
  }
}
