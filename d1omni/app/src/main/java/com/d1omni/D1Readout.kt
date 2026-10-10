package com.d1omni

import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.math.exp

/**
 * The read-out of the model repository's Python host (`d1_host.readout_f64`, the provider's
 * `_forward` tail in float64): the scores at P + each marker, in option order, the first K of them;
 * divided by the question's temperature for a text request; softmax; a noul reversed to [yes, no].
 */
object D1Readout {
  /** Decimals the screen shows for a probability. */
  const val SHOWN_DECIMALS = 3

  /** numpy sums fewer values than this one by one, more of them with 8 running sums. */
  private const val NUMPY_UNROLL = 8

  /** numpy's pairwise-sum block; no question has this many options. */
  private const val NUMPY_BLOCK = 128

  /** The K scores at P + markers[k] of a whole [scores] output, in option order. */
  fun markerScores(scores: FloatArray, prefixRows: Int, markers: IntArray, options: Int): FloatArray {
    require(options in 1..markers.size) { "K = $options options but ${markers.size} markers" }
    return FloatArray(options) { scores[prefixRows + markers[it]] }
  }

  /** [probabilities] from a whole [scores] output of the graph. */
  fun probabilities(
    scores: FloatArray,
    prefixRows: Int,
    markers: IntArray,
    question: D1Question,
    calibrate: Boolean,
    contract: D1Contract,
  ): DoubleArray =
    probabilities(
      markerScores(scores, prefixRows, markers, question.options),
      question,
      calibrate,
      contract,
    )

  /**
   * The distribution over the options from their K marker scores (float32 values widened to
   * float64): z / T for a text request ([calibrate]), softmax, a noul reversed to [yes, no].
   */
  fun probabilities(
    markerScores: FloatArray,
    question: D1Question,
    calibrate: Boolean,
    contract: D1Contract,
  ): DoubleArray {
    val temperature = if (calibrate) contract.temperature(question) else 1.0
    val z =
      DoubleArray(minOf(markerScores.size, question.options)) {
        val score = markerScores[it].toDouble()
        if (calibrate) score / temperature else score
      }
    val maximum = z.max()
    val e = DoubleArray(z.size) { exp(z[it] - maximum) }
    val total = numpySum(e)
    val p = DoubleArray(e.size) { e[it] / total }
    return if (question.type == QuestionType.NOUL) p.reversedArray() else p
  }

  /** A probability as the screen shows it: three decimals of its exact value, half to even. */
  fun shown(probability: Double, decimals: Int = SHOWN_DECIMALS): String =
    if (probability.isFinite()) {
      BigDecimal(probability).setScale(decimals, RoundingMode.HALF_EVEN).toPlainString()
    } else {
      probability.toString()
    }

  /** True when every value is finite. */
  fun finite(values: FloatArray): Boolean = values.all { it.isFinite() }

  /**
   * numpy's float64 `sum` of a short array in its order: one by one below 8 values, else 8 running
   * sums combined pairwise and the rest added one by one (`pairwise_sum` up to its block size).
   */
  fun numpySum(values: DoubleArray): Double {
    val count = values.size
    if (count < NUMPY_UNROLL) {
      var total = 0.0
      for (value in values) total += value
      return total
    }
    require(count <= NUMPY_BLOCK) { "$count options: more than numpy's pairwise block" }
    val sums = DoubleArray(NUMPY_UNROLL) { values[it] }
    var index = NUMPY_UNROLL
    while (index < count - count % NUMPY_UNROLL) {
      for (lane in 0 until NUMPY_UNROLL) sums[lane] += values[index + lane]
      index += NUMPY_UNROLL
    }
    var total = ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))
    while (index < count) {
      total += values[index]
      index++
    }
    return total
  }
}
