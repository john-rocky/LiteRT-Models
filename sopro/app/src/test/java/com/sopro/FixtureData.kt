package com.sopro

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt
import org.json.JSONObject

object FixtureData {
  val root = File(System.getProperty("sopro.fixtureDir") ?: error("Set sopro.fixtureDir"))
  val results = File(System.getProperty("sopro.resultsDir") ?: error("Set sopro.resultsDir"))
  val index: JSONObject by lazy { JSONObject(File(root, "index.json").readText()) }
  private val verified = HashSet<String>()

  fun arrayShape(key: String): IntArray {
    val shape = index.getJSONObject("arrays").getJSONObject(key).getJSONArray("shape")
    return IntArray(shape.length()) { shape.getInt(it) }
  }

  private fun buffer(key: String, dtype: String): ByteBuffer {
    val entry = index.getJSONObject("arrays").getJSONObject(key)
    check(entry.getString("dtype") == dtype) { "$key dtype != $dtype" }
    val bytes = File(root, entry.getString("path")).readBytes()
    if (verified.add(key)) {
      val sha =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
      check(sha == entry.getString("sha256")) { "$key sha256 mismatch" }
    }
    return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
  }

  fun floats(key: String): FloatArray {
    val b = buffer(key, "float32")
    return FloatArray(b.remaining() / 4) { b.float }
  }

  fun ints(key: String): IntArray {
    val b = buffer(key, "int32")
    return IntArray(b.remaining() / 4) { b.int }
  }

  fun doubles(key: String): DoubleArray {
    val b = buffer(key, "float64")
    return DoubleArray(b.remaining() / 8) { b.double }
  }

  fun metrics(gate: String, data: Map<String, Any?>) {
    results.mkdirs()
    val json = JSONObject(data).toString(2)
    File(results, "jvm_$gate.json").writeText(json + "\n")
    println("$gate: $json")
  }

  fun writeFloats(name: String, values: FloatArray) {
    val path = File(results, "kotlin_dumps/$name.bin")
    path.parentFile.mkdirs()
    val b = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    values.forEach { b.putFloat(it) }
    path.writeBytes(b.array())
  }

  fun maxDiff(a: FloatArray, b: FloatArray): Double {
    check(a.size == b.size) { "Array lengths ${a.size} != ${b.size}" }
    var maximum = 0.0
    for (i in a.indices) {
      check(a[i].isFinite() && b[i].isFinite())
      maximum = maxOf(maximum, abs(a[i].toDouble() - b[i]))
    }
    return maximum
  }

  fun corr(a: FloatArray, b: FloatArray): Double {
    check(a.size == b.size)
    val ma = a.sumOf { it.toDouble() } / a.size
    val mb = b.sumOf { it.toDouble() } / b.size
    var ab = 0.0
    var aa = 0.0
    var bb = 0.0
    for (i in a.indices) {
      val x = a[i] - ma
      val y = b[i] - mb
      ab += x * y
      aa += x * x
      bb += y * y
    }
    return ab / sqrt(aa * bb)
  }
}
