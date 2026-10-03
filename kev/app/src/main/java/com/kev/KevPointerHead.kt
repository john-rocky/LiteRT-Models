package com.kev

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/** One question's head outputs: raw logits, temperature-scaled logits and their softmax. */
class KevScores(val zPre: FloatArray, val zPost: FloatArray, val probabilities: FloatArray)

/**
 * The author's `PointerHead` on the host in float32: two 1024 → 256 linear maps (q for the decide
 * token, k for each option's closing token), `z = (k(h_opts) @ q(h_decide)) * 1/sqrt(256)`, divided
 * by the calibration temperature, then softmax. Weights come straight from the head's
 * `.safetensors` file (four F32 tensors, row-major, little-endian).
 */
class KevPointerHead(weights: File, val temperature: Double = TEMPERATURE) {
  private val qWeight: FloatArray
  private val qBias: FloatArray
  private val kWeight: FloatArray
  private val kBias: FloatArray

  init {
    val tensors = readSafetensors(weights.readBytes())
    qWeight = tensors.require("q.weight", HEAD_DIM, HIDDEN_SIZE)
    qBias = tensors.require("q.bias", HEAD_DIM)
    kWeight = tensors.require("k.weight", HEAD_DIM, HIDDEN_SIZE)
    kBias = tensors.require("k.bias", HEAD_DIM)
  }

  /** Scores of one question from the decide token's hidden state and each option's. */
  fun score(hDecide: FloatArray, hOptions: List<FloatArray>): KevScores {
    require(hDecide.size == HIDDEN_SIZE) { "Decide hidden state has ${hDecide.size} values" }
    require(hOptions.isNotEmpty()) { "A question needs at least one option" }
    val query = linear(qWeight, qBias, hDecide)
    val zPre = FloatArray(hOptions.size)
    val zPost = FloatArray(hOptions.size)
    for ((index, hOption) in hOptions.withIndex()) {
      require(hOption.size == HIDDEN_SIZE) { "Option hidden state has ${hOption.size} values" }
      val key = linear(kWeight, kBias, hOption)
      var dot = 0.0
      for (dimension in 0 until HEAD_DIM) {
        dot += key[dimension].toDouble() * query[dimension]
      }
      // Each tensor torch materializes is float32: k(h) @ q(h), then * scale, then / T.
      zPre[index] = (dot.toFloat() * SCALE)
      zPost[index] = (zPre[index] / temperature).toFloat()
    }
    return KevScores(zPre, zPost, softmax(zPost))
  }

  /** `x @ W^T + b` for one row, accumulated in double and stored as float32. */
  private fun linear(weight: FloatArray, bias: FloatArray, input: FloatArray): FloatArray {
    val output = FloatArray(HEAD_DIM)
    for (row in 0 until HEAD_DIM) {
      var sum = 0.0
      val base = row * HIDDEN_SIZE
      for (column in 0 until HIDDEN_SIZE) {
        sum += weight[base + column].toDouble() * input[column]
      }
      output[row] = (sum + bias[row]).toFloat()
    }
    return output
  }

  private class Tensors(private val values: Map<String, Pair<IntArray, FloatArray>>) {
    fun require(name: String, vararg shape: Int): FloatArray {
      val (actualShape, data) = requireNotNull(values[name]) { "Head file has no $name" }
      require(actualShape.contentEquals(shape)) {
        "$name has shape ${actualShape.toList()}, not ${shape.toList()}"
      }
      return data
    }
  }

  companion object {
    /** Backbone hidden size of Kev-0.8B (Qwen3.5-0.8B). */
    const val HIDDEN_SIZE = 1024

    /** Pointer dimension of the head. */
    const val HEAD_DIM = 256

    /** `1 / sqrt(HEAD_DIM)`, exact in binary. */
    const val SCALE = 0.0625f

    /** The checkpoint's calibration temperature (the head's eval-mode divisor). */
    const val TEMPERATURE = 2.3510958125672174

    /** Bytes of the little-endian u64 that precedes the safetensors JSON header. */
    private const val HEADER_LENGTH_BYTES = 8
    private const val FLOAT_BYTES = 4

    /** float32 softmax: `exp(z - max) / sum`, computed in double and stored as float32. */
    fun softmax(logits: FloatArray): FloatArray {
      val max = logits.maxOrNull() ?: return FloatArray(0)
      val exponentials = DoubleArray(logits.size) { exp(logits[it].toDouble() - max) }
      val total = exponentials.sum()
      return FloatArray(logits.size) { (exponentials[it] / total).toFloat() }
    }

    /** Reads every F32 tensor of a safetensors file, checking offsets against shapes. */
    private fun readSafetensors(bytes: ByteArray): Tensors {
      require(bytes.size >= HEADER_LENGTH_BYTES) { "Not a safetensors file" }
      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      val headerLength = buffer.getLong(0)
      require(headerLength in 2..(bytes.size - HEADER_LENGTH_BYTES).toLong()) { "Bad safetensors header length" }
      val dataStart = HEADER_LENGTH_BYTES + headerLength.toInt()
      val header =
        KevJson.parse(bytes.copyOfRange(HEADER_LENGTH_BYTES, dataStart)) as? Map<*, *>
          ?: throw IllegalArgumentException("Safetensors header is not an object")
      val tensors = HashMap<String, Pair<IntArray, FloatArray>>()
      for ((name, entry) in header) {
        if (name == "__metadata__") continue
        val tensor = entry as Map<*, *>
        require(tensor["dtype"] == "F32") { "$name is ${tensor["dtype"]}, not F32" }
        val shape = (tensor["shape"] as List<*>).map { (it as JsonNumber).toInt() }.toIntArray()
        val offsets = (tensor["data_offsets"] as List<*>).map { (it as JsonNumber).toInt() }
        val count = shape.fold(1) { product, dimension -> product * dimension }
        require(offsets.size == 2 && offsets[1] - offsets[0] == count * FLOAT_BYTES) {
          "$name offsets $offsets do not match shape ${shape.toList()}"
        }
        require(offsets[0] >= 0 && dataStart + offsets[1] <= bytes.size) { "$name lies outside the file" }
        val data = FloatArray(count)
        ByteBuffer.wrap(bytes, dataStart + offsets[0], count * FLOAT_BYTES)
          .order(ByteOrder.LITTLE_ENDIAN)
          .asFloatBuffer()
          .get(data)
        tensors[name as String] = shape to data
      }
      return Tensors(tensors)
    }
  }
}
