// SPDX-License-Identifier: Apache-2.0
package com.laya

import java.io.FileInputStream
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LayaEmbeddingsParityTest {
  @Test
  fun allBinary16PatternsMatchNumpyFloat32Conversion() {
    val expected = LayaTestData.required("fixtures/fp16_conversion_reference.bin").readBytes()
    assertEquals(65536 * 4, expected.size)
    val values = java.nio.ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
    val failures = mutableListOf<Map<String, Any?>>()
    var exact = 0
    var finite = 0
    var infinity = 0
    var nan = 0
    for (bits in 0..65535) {
      val actual = LayaEmbeddings.halfToFloat(bits)
      val expectedBits = values.getInt(bits * 4)
      if (actual.toRawBits() == expectedBits) exact++
      else if (failures.size < 32) {
        failures +=
          linkedMapOf(
            "half_bits" to bits,
            "expected_float_bits" to expectedBits,
            "actual_float_bits" to actual.toRawBits(),
          )
      }
      when {
        actual.isNaN() -> nan++
        actual.isInfinite() -> infinity++
        else -> finite++
      }
    }
    LayaTestData.report(
      "jvm_fp16_conversion.json",
      linkedMapOf(
        "status" to if (exact == 65536) "PASS" else "FAIL",
        "patterns" to 65536,
        "raw_float_bits_exact" to exact,
        "finite" to finite,
        "infinite" to infinity,
        "nan" to nan,
        "failures" to failures,
        "nan_policy" to
          "NumPy float16-to-float32 conversion: quiet signaling NaNs; preserve sign and payload",
      ),
    )
    assertEquals("All binary16 patterns versus NumPy", 65536, exact)
    assertEquals(63488, finite)
    assertEquals(2, infinity)
    assertEquals(2046, nan)
  }

  @Test
  fun all402CapturedRowsGatherExactlyIncludingPadRows() {
    val manifest = LayaJson.asObject(LayaTestData.read("fixtures/embedding_lookup_reference.json"))
    assertEquals("float32", manifest["dtype"])
    assertEquals("little", manifest["byte_order"])
    assertEquals(768, (manifest["embedding_width"] as Number).toInt())
    assertEquals(0, (manifest["pad_id"] as Number).toInt())
    val rows = LayaJson.asArray(manifest["rows"]).map(LayaJson::asObject)
    assertEquals(402, rows.size)
    val references = mapOf(256 to LayaTestData.rows(256), 512 to LayaTestData.rows(512))
    val binaryName = manifest["binary_file"] as String
    require(binaryName == "embedding_lookup_reference.bin") { "Unexpected lookup reference path" }
    val referenceFile = LayaTestData.required("fixtures/$binaryName")
    assertEquals((manifest["binary_size_bytes"] as Number).toLong(), referenceFile.length())
    val reference =
      FileInputStream(referenceFile).channel.use {
        it
          .map(FileChannel.MapMode.READ_ONLY, 0, referenceFile.length())
          .order(ByteOrder.LITTLE_ENDIAN)
      }
    val started = System.nanoTime()
    val table =
      LayaEmbeddings(
        LayaTestData.required("host_assets/token_embeddings_fp16.bin"),
        LayaTestData.required("host_assets/token_embeddings.json"),
      )
    val loadMs = (System.nanoTime() - started) / 1_000_000.0
    assertEquals(manifest["table_sha256"], table.sha256)
    val details = mutableListOf<Map<String, Any?>>()
    val failures = mutableListOf<Map<String, Any?>>()
    val destinations = mapOf(256 to FloatArray(256 * 768), 512 to FloatArray(512 * 768))
    var totalValues = 0L
    var maxError = 0.0
    table.use {
      for (row in rows) {
        val id = row["row_id"] as String
        val window = (row["window"] as Number).toInt()
        val sourceIndex = (row["source_row_index"] as Number).toInt()
        assertEquals("ml_rows_s$window.json", row["source_file"])
        val captured = references.getValue(window)[sourceIndex]
        assertEquals("Captured row identity", captured["row_id"], id)
        val ids = LayaTestData.ints(captured["sequence_ids"])
        assertTrue(
          "Captured ids in NumPy fixture $id",
          ids.contentEquals(LayaTestData.ints(row["sequence_ids"])),
        )
        val expectedPaddedIds = LayaTestData.ints(row["padded_input_ids"])
        assertTrue(
          "PAD id 0 in NumPy fixture $id",
          ids.copyOf(window).contentEquals(expectedPaddedIds),
        )
        val count = (row["expected_float_count"] as Number).toInt()
        assertEquals(window * 768, count)
        val offset = (row["offset_bytes"] as Number).toInt()
        val length = (row["length_bytes"] as Number).toInt()
        assertEquals(count * 4, length)
        assertTrue(offset >= 0 && offset.toLong() + length <= referenceFile.length())
        val actual = table.gather(ids, window, destinations.getValue(window))
        var firstMismatch = -1
        var rowMaxError = 0.0
        var finite = true
        for (index in actual.indices) {
          val expectedBits = reference.getInt(offset + index * 4)
          if (actual[index].toRawBits() != expectedBits && firstMismatch < 0) firstMismatch = index
          val expected = Float.fromBits(expectedBits)
          if (!actual[index].isFinite() || !expected.isFinite()) finite = false
          rowMaxError = maxOf(rowMaxError, abs(actual[index].toDouble() - expected.toDouble()))
        }
        maxError = maxOf(maxError, rowMaxError)
        totalValues += count
        val detail =
          linkedMapOf<String, Any?>(
            "row_id" to id,
            "window" to window,
            "source_row_index" to sourceIndex,
            "float_count" to count,
            "padded_positions" to window - ids.size,
            "raw_float_bits_exact" to (firstMismatch < 0),
            "finite" to finite,
            "max_abs_error" to rowMaxError,
            "first_difference" to firstMismatch,
          )
        if (firstMismatch >= 0 || !finite) failures += detail
        details += detail
      }
    }
    LayaTestData.report(
      "jvm_embedding_lookup.json",
      linkedMapOf(
        "status" to if (failures.isEmpty()) "PASS" else "FAIL",
        "device" to "Apple M4 Max (supervisor supplied)",
        "runtime" to System.getProperty("java.runtime.version"),
        "rows" to details.size,
        "rows_exact" to details.count { it["raw_float_bits_exact"] == true },
        "finite_rows" to details.count { it["finite"] == true },
        "float_values_compared" to totalValues,
        "max_abs_error" to maxError,
        "embedding_map_ms" to loadMs,
        "table_sha256" to table.sha256,
        "binary_sha256" to manifest["binary_sha256"],
        "padded_positions_use_token_id" to 0,
        "model_inference_executed" to false,
        "failures" to failures,
        "details" to details,
      ),
    )
    println(
      "LAYA_JVM_EMBEDDINGS exact=${details.size - failures.size}/${details.size} max_abs_error=$maxError values=$totalValues"
    )
    LayaTestData.assertNoFailures("Embedding lookup parity", failures)
    assertEquals(0.0, maxError, 0.0)
  }
}
