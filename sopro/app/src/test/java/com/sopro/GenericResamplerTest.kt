package com.sopro

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test

class GenericResamplerTest {
  @Test
  fun torchaudioSyntheticParity() {
    val directory = File(FixtureData.root.parentFile, "generic_resampler")
    val index = JSONObject(File(directory, "index.json").readText())
    val rows = index.getJSONArray("rows")
    val results = mutableListOf<Map<String, Any>>()
    var passed = true
    for (i in 0 until rows.length()) {
      val row = rows.getJSONObject(i)
      fun read(name: String): FloatArray {
        val info = row.getJSONObject("arrays").getJSONObject(name)
        val bytes = File(directory, info.getString("path")).readBytes()
        check(
          MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") {
            "%02x".format(it)
          } == info.getString("sha256")
        )
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(info.getInt("length")) { buffer.float }
      }
      val expected = read("expected")
      val actual =
        GenericResampler.resample(
          read("input"),
          row.getInt("source_rate"),
          row.getInt("target_rate"),
        )
      val error = FixtureData.maxDiff(expected, actual)
      val pass = expected.size == actual.size && error <= 1e-5
      passed = passed && pass
      results +=
        mapOf(
          "id" to row.getString("id"),
          "max_abs_diff" to error,
          "samples" to actual.size,
          "finite" to actual.all { it.isFinite() },
          "pass" to pass,
        )
    }
    FixtureData.metrics(
      "generic_resampler",
      mapOf(
        "status" to if (passed) "PASS" else "FAIL",
        "device" to "Mac JVM CPU",
        "threshold" to 1e-5,
        "rows" to results,
      ),
    )
    assertTrue("Generic resampler must match torchaudio within 1e-5", passed)
  }
}
