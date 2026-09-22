package com.sopro

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import org.json.JSONObject

/** Original fp32 embedding tables remain mapped read-only; cache ownership is per synthesis. */
class ArHost(tablesBin: File, tableJson: File) : AutoCloseable {
  private val file = RandomAccessFile(tablesBin, "r")
  private val mapped =
    file.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length()).order(ByteOrder.LITTLE_ENDIAN)
  private val metadata = JSONObject(tableJson.readText())
  private val tensors = metadata.getJSONObject("tensors")
  private val textOffset = tensors.getJSONObject("text_tok_emb").getLong("offset_bytes").toInt()
  private val semanticOffset = tensors.getJSONObject("sem_emb512").getLong("offset_bytes").toInt()
  val bosId = metadata.getInt("bos_id")
  val eosId = metadata.getInt("eos_id")
  val maxTextLen = minOf(127, metadata.getInt("max_text_len"))

  data class Prefix(val values: FloatArray, val length: Int) {
    val lastIndex
      get() = length - 1
  }

  private fun rows(ids: IntArray, offset: Int, count: Int): FloatArray {
    return FloatArray(ids.size * 512) { index ->
      val id = ids[index / 512]
      require(id in 0 until count) { "Embedding token $id is outside vocabulary $count" }
      mapped.getFloat(offset + (id * 512 + index % 512) * 4)
    }
  }

  fun textEmbedding(ids: IntArray) = rows(ids, textOffset, 8192)

  fun semanticEmbedding(ids: IntArray) = rows(ids, semanticOffset, 4377)

  fun assemblePrefix(style: FloatArray, textIds: IntArray, refTokens: IntArray): Prefix {
    require(style.size == 8 * 512)
    require(textIds.size <= maxTextLen) {
      "Text has ${textIds.size} tokens; maximum is $maxTextLen with the 120-token reference prompt. Please shorten the text."
    }
    require(refTokens.size >= 120)
    val length = 8 + textIds.size + 120 + 1
    require(length <= P_MAX)
    val values = FloatArray(length * 512)
    style.copyInto(values)
    textEmbedding(textIds).copyInto(values, style.size)
    semanticEmbedding(refTokens.copyOfRange(0, 120)).copyInto(values, (8 + textIds.size) * 512)
    semanticEmbedding(intArrayOf(bosId)).copyInto(values, (length - 1) * 512)
    return Prefix(values, length)
  }

  override fun close() = file.close()

  class PackedKv {
    val keys = FloatArray(96 * CAP * 64)
    val values = FloatArray(keys.size)
    var position = 0
      private set

    fun initialize(k: FloatArray, v: FloatArray, length: Int) {
      require(length in 1..P_MAX && k.size == 96 * P_MAX * 64 && v.size == k.size)
      keys.fill(0f)
      values.fill(0f)
      for (head in 0 until 96) {
        k.copyInto(keys, head * CAP * 64, head * P_MAX * 64, (head * P_MAX + length) * 64)
        v.copyInto(values, head * CAP * 64, head * P_MAX * 64, (head * P_MAX + length) * 64)
      }
      position = length
    }

    fun writeRow(k: FloatArray, v: FloatArray, p: Int = position) {
      require(p in 0 until CAP && k.size == 96 * 64 && v.size == k.size)
      for (head in 0 until 96) {
        k.copyInto(keys, (head * CAP + p) * 64, head * 64, (head + 1) * 64)
        v.copyInto(values, (head * CAP + p) * 64, head * 64, (head + 1) * 64)
      }
      position = p + 1
    }
  }

  companion object {
    const val P_MAX = 256
    const val CAP = 1024

    /** Float selector for R6 prefill; selects the last real prefix row. */
    fun lastOneHot(validLength: Int): FloatArray {
      require(validLength in 1..P_MAX) { "Prefix length must be in 1..$P_MAX." }
      return FloatArray(P_MAX).also { it[validLength - 1] = 1f }
    }

    fun prefillBias(validLength: Int): FloatArray {
      require(validLength in 0..P_MAX)
      return FloatArray(P_MAX * P_MAX) {
        val q = it / P_MAX
        val k = it % P_MAX
        if (k <= q && k < validLength) 0f else -10000f
      }
    }

    fun stepBias(position: Int): FloatArray {
      require(position in 0 until CAP)
      return FloatArray(CAP) { if (it <= position) 0f else -10000f }
    }

    fun rotaryCosSin(): Pair<FloatArray, FloatArray> {
      val inv = FloatArray(32) { 1f / 10000.0.pow((it * 2f / 64f).toDouble()).toFloat() }
      val cosine = FloatArray(CAP * 64)
      val sine = FloatArray(cosine.size)
      for (p in 0 until CAP) for (d in 0 until 64) {
        val angle = p.toFloat() * inv[d % 32]
        cosine[p * 64 + d] = cos(angle.toDouble()).toFloat()
        sine[p * 64 + d] = sin(angle.toDouble()).toFloat()
      }
      return cosine to sine
    }
  }
}
