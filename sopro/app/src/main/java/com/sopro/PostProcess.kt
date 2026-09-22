package com.sopro

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tanh

/** Single-segment source postprocessing, preserving strict threshold/index rules. */
object PostProcess {
  data class Trim(
    val gain: Double,
    val speechOnsetSamples: Int?,
    val leadCutSamples: Int,
    val trailEndAfterLeadSamples: Int,
    val finalSamples: Int,
    val inputSamples: Int,
    val sampleRateHz: Int,
  ) {
    fun asMap(): Map<String, Any?> =
      linkedMapOf(
        "gain" to gain,
        "speech_onset_samples" to speechOnsetSamples,
        "lead_cut_samples" to leadCutSamples,
        "trail_end_after_lead_samples" to trailEndAfterLeadSamples,
        "final_samples" to finalSamples,
        "input_samples" to inputSamples,
        "sample_rate_hz" to sampleRateHz,
      )
  }

  data class Result(val wav: FloatArray, val trim: Trim)

  data class Lead(val wav: FloatArray, val cut: Int, val onset: Int?)

  fun outputGain(referenceLevelDb: Double = -19.8): Double =
    10.0.pow((-23.0 - referenceLevelDb) / 20.0)

  private fun shortRms(wav: FloatArray, sampleRate: Int): FloatArray {
    val window = (sampleRate * .010).toInt()
    return FloatArray(wav.size / window) { frame ->
      // NumPy's float32 mean uses pairwise summation on each contiguous row.
      val squared = FloatArray(window) { wav[frame * window + it] * wav[frame * window + it] }
      sqrt(pairwiseSum(squared, 0, squared.size) / window.toFloat())
    }
  }

  internal fun pairwiseSum(x: FloatArray, start: Int = 0, size: Int = x.size): Float {
    if (size < 8) {
      var sum = -0.0f
      for (i in 0 until size) sum += x[start + i]
      return sum
    }
    if (size <= 128) {
      val r = FloatArray(8) { x[start + it] }
      var i = 8
      while (i < size - size % 8) {
        for (j in 0..7) r[j] += x[start + i + j]
        i += 8
      }
      var sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
      while (i < size) {
        sum += x[start + i]
        i++
      }
      return sum
    }
    var half = size / 2
    half -= half % 8
    return pairwiseSum(x, start, half) + pairwiseSum(x, start + half, size - half)
  }

  private fun onsetThreshold(rms: FloatArray): Double {
    var threshold = 10.0.pow(-45.0 / 20.0)
    if (rms.size >= 30) {
      val sorted = rms.sortedArray()
      val position = (sorted.size - 1) * .1
      val low = position.toInt()
      val a = sorted[low]
      val b = sorted[minOf(low + 1, sorted.lastIndex)]
      val fraction = position - low
      // NumPy 2.5 weak scalar q keeps the _lerp value arithmetic in fp32.
      val quantile =
        if (fraction < .5) a + (b - a) * fraction.toFloat()
        else b - (b - a) * (1.0 - fraction).toFloat()
      threshold = maxOf(threshold, quantile.toDouble() * 10.0.pow(15.0 / 20.0))
    }
    return threshold
  }

  fun speechOnset(wav: FloatArray, sampleRate: Int = 24000): Int? {
    val window = (sampleRate * .010).toInt()
    if (wav.size < window * 6) return null
    val rms = shortRms(wav, sampleRate)
    // A Python scalar compared to a float32 ndarray is cast to float32.
    val threshold = onsetThreshold(rms).toFloat()
    for (i in 0..rms.size - 6) {
      var hits = 0
      for (j in 0 until 6) if (rms[i + j] > threshold) hits++
      if (hits >= 5) return i * window
    }
    return null
  }

  fun trimLead(
    wav: FloatArray,
    sampleRate: Int = 24000,
    lead: Double = .08,
    skip: Double = 0.0,
  ): Lead {
    val onset = speechOnset(wav, sampleRate)
    var cut = 0
    if (onset != null) {
      cut = maxOf(onset - (lead * sampleRate).toInt(), (skip * sampleRate).toInt())
      cut = minOf(cut, maxOf(0, onset - (.02 * sampleRate).toInt()))
    }
    return Lead(wav.copyOfRange(cut, wav.size), cut, onset)
  }

  fun trimTrail(
    wav: FloatArray,
    sampleRate: Int = 24000,
    trail: Double = .30,
  ): Pair<FloatArray, Int> {
    val window = (sampleRate * .010).toInt()
    var end = wav.size
    if (wav.size >= window) {
      val rms = shortRms(wav, sampleRate)
      val threshold = onsetThreshold(rms).toFloat()
      val last = rms.indexOfLast { it > threshold }
      if (last >= 0) end = minOf(wav.size, (last + 1) * window + (trail * sampleRate).toInt())
    }
    return wav.copyOf(end) to end
  }

  fun softLimit(wav: FloatArray, knee: Double = .9): FloatArray {
    val k = knee.toFloat()
    val remaining = (1 - knee).toFloat()
    return FloatArray(wav.size) { i ->
      val value = wav[i]
      val magnitude = abs(value)
      if (magnitude > k) {
        val over = k + remaining * tanh((magnitude - k) / remaining)
        if (value < 0f) -over else over
      } else value
    }
  }

  fun fadeEdges(
    wav: FloatArray,
    sampleRate: Int = 24000,
    fadeIn: Boolean = false,
    fadeOut: Boolean = true,
    fadeSeconds: Double = .08,
  ): FloatArray {
    val fade = (fadeSeconds * sampleRate).toInt()
    if (wav.size <= 2 * fade) return wav
    val out = wav.copyOf()
    // linspace calculates the ramp in fp64 and then converts each element to fp32.
    val step = 1.0 / (fade - 1)
    for (i in 0 until fade) {
      val ramp = (i * step).toFloat()
      if (fadeIn) out[i] *= ramp
      if (fadeOut) out[out.lastIndex - i] *= ramp
    }
    return out
  }

  fun postprocessSegment(
    raw: FloatArray,
    referenceLevelDb: Double,
    sampleRate: Int = 24000,
  ): Result {
    val gain = outputGain(referenceLevelDb)
    val g = gain.toFloat()
    val leading = trimLead(FloatArray(raw.size) { raw[it] * g }, sampleRate)
    val (trimmed, end) = trimTrail(leading.wav, sampleRate)
    val final = fadeEdges(softLimit(trimmed), sampleRate)
    return Result(
      final,
      Trim(gain, leading.onset, leading.cut, end, final.size, raw.size, sampleRate),
    )
  }
}
