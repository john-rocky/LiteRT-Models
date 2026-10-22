package com.timestral

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Properties

/** Reading the test vectors and comparing float32 outputs. */
object TestData {
  /** The two-row graphs with 32 decoder patches the conversion run shipped first (B1, B2). */
  val TWO_ROW_H32 =
    listOf(
      GraphSpec("B1", "b1.tflite", 64, 32, 2, -3e4f, Backend.entries.toSet()),
      GraphSpec("B2", "b2.tflite", 256, 32, 2, -3e4f, Backend.entries.toSet()),
    )

  /** First RoPE position of the installed tables. */
  const val ROPE_FIRST = -1023

  /** Rows of the installed RoPE tables (positions -1023 to 1055). */
  const val ROPE_ROWS = 2079

  /** The bytes of a test resource. */
  fun resource(path: String): ByteArray =
    checkNotNull(TestData::class.java.classLoader.getResourceAsStream(path)) { "Missing $path" }
      .use { it.readBytes() }

  fun properties(path: String): Properties =
    Properties().apply { resource(path).inputStream().use { load(it) } }

  fun floats(bytes: ByteArray): FloatArray {
    val values = FloatArray(bytes.size / 4)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
    return values
  }

  fun bytes(values: FloatArray): ByteArray {
    val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buffer.asFloatBuffer().put(values)
    return buffer.array()
  }

  fun sha256(values: FloatArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes(values)).joinToString("") { "%02x".format(it) }

  /** A float written as `0x` + its 8 hex digits (big-endian bits). */
  fun hexFloat(text: String): Float =
    java.lang.Float.intBitsToFloat(java.lang.Long.parseLong(text.trim().removePrefix("0x"), 16).toInt())

  fun hexFloats(text: String): FloatArray = text.split(",").map { hexFloat(it) }.toFloatArray()

  /** The tables of [directory] (`-Ptimestral.host=<dir>`), or null when the property is unset. */
  fun hostTables(): HostTables? {
    val directory = System.getProperty("timestral.host") ?: return null
    fun read(name: String) = floats(File(directory, name).readBytes())
    return HostTables(
      read("forecast_query.f32"),
      read("rope_cos.f32"),
      read("rope_sin.f32"),
      ROPE_FIRST,
    )
  }

  /** Zero tables of the right size, for the inputs that do not depend on them. */
  val placeholderTables: HostTables by lazy {
    HostTables(
      FloatArray(TimestralHost.D_MODEL),
      FloatArray(ROPE_ROWS * TimestralHost.ROPE_DIM),
      FloatArray(ROPE_ROWS * TimestralHost.ROPE_DIM),
      ROPE_FIRST,
    )
  }

  /** How two float32 arrays differ. */
  class Difference(val maxRelative: Double, val maxUlp: Long, val notBitEqual: Int, val size: Int) {
    override fun toString(): String =
      "max rel ${"%.3e".format(maxRelative)}, max ulp $maxUlp, not bit-equal $notBitEqual / $size"
  }

  /** |a - b| / max(|b|, 1e-30) and the ulp distance, value by value; NaN equals NaN only. */
  fun difference(actual: FloatArray, expected: FloatArray): Difference {
    require(actual.size == expected.size) { "${actual.size} vs ${expected.size} values" }
    var maxRelative = 0.0
    var maxUlp = 0L
    var notBitEqual = 0
    for (i in actual.indices) {
      val a = actual[i]
      val b = expected[i]
      if (a.toRawBits() == b.toRawBits()) continue
      notBitEqual++
      if (a.isNaN() || b.isNaN()) {
        maxRelative = Double.POSITIVE_INFINITY
        maxUlp = Long.MAX_VALUE
        continue
      }
      val relative = Math.abs(a.toDouble() - b.toDouble()) / maxOf(Math.abs(b.toDouble()), 1e-30)
      maxRelative = maxOf(maxRelative, relative)
      maxUlp = maxOf(maxUlp, Math.abs(ordered(a) - ordered(b)))
    }
    return Difference(maxRelative, maxUlp, notBitEqual, actual.size)
  }

  /** Floats mapped to integers in value order, so the difference counts representable steps. */
  private fun ordered(value: Float): Long {
    val bits = value.toRawBits()
    return if (bits < 0) Int.MIN_VALUE.toLong() - bits.toLong() else bits.toLong()
  }
}
