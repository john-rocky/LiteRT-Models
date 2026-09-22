package com.sopro

import kotlin.math.PI
import kotlin.math.cos

object AcousticHost {
  data class Bucket(val frames: Int, val tokens: Int)

  data class Prepared(
    val tokens: IntArray,
    val tokenMask: FloatArray,
    val frameToToken: IntArray,
    val x: FloatArray,
    val condVec: FloatArray,
    val condMel: FloatArray,
    val condMask: FloatArray,
    val keyBias: FloatArray,
    val validFrames: Int,
    val validTokens: Int,
    val promptFrames: Int,
    val frames: Int,
    val tokensCapacity: Int,
  )

  /** Row-major [1,N,vocab] float selectors; padded token zero still selects row zero. */
  fun semanticOneHot(tokens: IntArray, vocab: Int = 4375): FloatArray {
    require(tokens.isNotEmpty() && vocab > 0 && tokens.size.toLong() * vocab <= Int.MAX_VALUE)
    require(tokens.all { it in 0 until vocab }) {
      "Acoustic semantic token is outside vocabulary $vocab."
    }
    return FloatArray(tokens.size * vocab).also { values ->
      for (row in tokens.indices) values[row * vocab + tokens[row]] = 1f
    }
  }

  /** Row-major [T,N] float selectors for the unchanged frame-to-token map. */
  fun frameOneHot(frameToToken: IntArray, tokensCapacity: Int): FloatArray {
    require(
      frameToToken.isNotEmpty() &&
        tokensCapacity > 0 &&
        frameToToken.size.toLong() * tokensCapacity <= Int.MAX_VALUE
    )
    require(frameToToken.all { it in 0 until tokensCapacity }) {
      "Frame-to-token index exceeds token capacity $tokensCapacity."
    }
    return FloatArray(frameToToken.size * tokensCapacity).also { values ->
      for (frame in frameToToken.indices) values[frame * tokensCapacity + frameToToken[frame]] = 1f
    }
  }

  fun bucket(validFrames: Int, validTokens: Int): Bucket {
    require(validFrames > 0 && validTokens > 0)
    if (validFrames <= 2048 && validTokens <= 512) return Bucket(2048, 512)
    require(validFrames <= 4096 && validTokens <= 1024) {
      "Acoustic input exceeds 4096 frames / 1024 tokens."
    }
    return Bucket(4096, 1024)
  }

  fun prepareInputs(
    refTokens: IntArray,
    generatedTokens: IntArray,
    x0: FloatArray,
    condVec: FloatArray,
    refMel: FloatArray,
    promptFrames: Int = 938,
  ): Prepared {
    require(x0.size % 100 == 0 && refMel.size == 100 * promptFrames)
    val valid = x0.size / 100
    val n = refTokens.size + generatedTokens.size
    require(valid == promptFrames + 4 * generatedTokens.size)
    val bucket = bucket(valid, n)
    val tokens = IntArray(bucket.tokens)
    refTokens.copyInto(tokens)
    generatedTokens.copyInto(tokens, refTokens.size)
    val x = FloatArray(100 * bucket.frames)
    val condMel = FloatArray(x.size)
    for (band in 0 until 100) {
      x0.copyInto(x, band * bucket.frames, band * valid, (band + 1) * valid)
      refMel.copyInto(condMel, band * bucket.frames, band * promptFrames, (band + 1) * promptFrames)
    }
    return Prepared(
      tokens,
      FloatArray(bucket.tokens) { if (it < n) 1f else 0f },
      IntArray(bucket.frames) { minOf(it.toLong() * n / valid, (n - 1).toLong()).toInt() },
      x,
      condVec.copyOf(),
      condMel,
      FloatArray(bucket.frames) { if (it < promptFrames) 1f else 0f },
      FloatArray(bucket.frames) { if (it < valid) 0f else -10000f },
      valid,
      n,
      promptFrames,
      bucket.frames,
      bucket.tokens,
    )
  }

  fun timeGrid(): FloatArray =
    FloatArray(3) { i ->
      val t = i.toFloat() / 2f
      val angle = (0.5 * PI).toFloat() * t
      t + -1f * (cos(angle.toDouble()).toFloat() - 1f + t)
    }

  fun eulerUpdate(
    x: FloatArray,
    velocity: FloatArray,
    x0: FloatArray,
    condMel: FloatArray,
    condMask: FloatArray,
    t0: Float,
    t1: Float,
  ): FloatArray {
    require(
      x.size == velocity.size &&
        x.size == x0.size &&
        x.size == condMel.size &&
        x.size == condMask.size * 100
    )
    val delta = t1 - t0
    val sigmaWeight = 1f - (1.0 - 1e-6).toFloat() * t1
    return FloatArray(x.size) { i ->
      val mask = condMask[i % condMask.size]
      val next = x[i] + delta * velocity[i]
      val prompt = sigmaWeight * x0[i] + t1 * condMel[i]
      mask * prompt + (1f - mask) * next
    }
  }

  fun zeroBeyondValid(array: FloatArray, valid: Int, frames: Int) {
    require(array.size == 100 * frames)
    for (band in 0 until 100) array.fill(0f, band * frames + valid, (band + 1) * frames)
  }

  fun replacePrompt(x: FloatArray, prepared: Prepared): FloatArray =
    FloatArray(x.size) { i ->
      val mask = prepared.condMask[i % prepared.frames]
      mask * prepared.condMel[i] + (1f - mask) * x[i]
    }

  fun validMel(padded: FloatArray, valid: Int, frames: Int): FloatArray =
    FloatArray(100 * valid) { i -> padded[(i / valid) * frames + i % valid] }

  fun decodeMel(
    solved: FloatArray,
    validFrames: Int,
    promptFrames: Int,
    mean: FloatArray,
    std: FloatArray,
  ): FloatArray {
    require(solved.size == 100 * validFrames)
    require(mean.size == 1 || mean.size == 100)
    require(std.size == 1 || std.size == 100)
    val start = promptFrames - minOf(32, promptFrames)
    val frames = validFrames - start
    return FloatArray(100 * frames) { i ->
      val band = i / frames
      solved[band * validFrames + start + i % frames] * std[if (std.size == 1) 0 else band] +
        mean[if (mean.size == 1) 0 else band]
    }
  }
}
