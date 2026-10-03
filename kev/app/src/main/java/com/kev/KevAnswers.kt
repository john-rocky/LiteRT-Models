package com.kev

import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.math.abs

/**
 * Port of the author's `to_answers` (`kev/api.py`) with the confidence formulas of TypeSafe's
 * reference adapter. Every number is a Python float computed in the same order: probabilities in
 * are the float32 softmax outputs widened exactly to double, sums follow CPython 3.12's built-in
 * `sum` (Neumaier-compensated), and outputs are rounded to 4 decimals with `round(x, 4)`.
 */
object KevAnswers {
  /** Decimals of every reported probability and derived scalar (`round_prob`). */
  private const val DECIMALS = 4

  /** `to_answers`: one answer object per question, keyed by question ID, in question order. */
  fun toAnswers(probabilities: List<DoubleArray>, meta: List<QuestionMeta>): LinkedHashMap<String, Any?> {
    require(probabilities.size == meta.size) { "${probabilities.size} distributions for ${meta.size} questions" }
    val answers = LinkedHashMap<String, Any?>()
    for ((p, question) in probabilities.zip(meta)) {
      require(p.size == question.keys.size) { "Question ${question.id}: ${p.size} probabilities, ${question.keys.size} keys" }
      answers[question.id] = answer(p, question)
    }
    return answers
  }

  /** The response body of the author's server, without its timing and output-token fields. */
  fun response(answers: Map<String, Any?>, inputTokens: Int, model: String = KevRequest.DEFAULT_MODEL): LinkedHashMap<String, Any?> =
    linkedMapOf("model" to model, "answers" to answers, "usage" to linkedMapOf("input_tokens" to inputTokens))

  private fun answer(p: DoubleArray, question: QuestionMeta): LinkedHashMap<String, Any?> =
    when (question.type) {
      QuestionType.NOUL -> linkedMapOf("type" to "noul", "noul" to roundProb(p[1]))
      QuestionType.CHOICE ->
        linkedMapOf(
          "type" to "choice",
          "choice" to question.keys[firstArgmax(p)],
          "confidence" to roundProb(choiceConfidence(p)),
          "probabilities" to rounded(question.keys, p),
        )
      QuestionType.SCORE ->
        linkedMapOf(
          "type" to "score",
          "score" to roundProb(pythonSum(DoubleArray(p.size) { it * p[it] })),
          "legend" to requireNotNull(question.legend) { "Score question ${question.id} has no legend" },
          "probabilities" to rounded(question.keys, p),
          "confidence" to roundProb(scoreConfidence(p)),
        )
    }

  private fun rounded(keys: List<String>, p: DoubleArray): LinkedHashMap<String, Double> =
    LinkedHashMap<String, Double>().apply {
      for ((index, key) in keys.withIndex()) {
        put(key, roundProb(p[index]))
      }
    }

  /** `round_prob`: Python's `round(x, 4)`, which rounds the exact binary value half-to-even. */
  fun roundProb(x: Double): Double {
    if (x == 0.0 || !x.isFinite()) return x
    val rounded = BigDecimal(x).setScale(DECIMALS, RoundingMode.HALF_EVEN).toDouble()
    // BigDecimal has no negative zero; Python keeps the sign (round(-1e-9, 4) is -0.0).
    return if (rounded == 0.0 && x < 0) -0.0 else rounded
  }

  /** `(p_max - 1/K) / (1 - 1/K)` on the normalized distribution: 0 at uniform, 1 at certainty. */
  fun choiceConfidence(p: DoubleArray): Double {
    val count = p.size
    if (count == 1) return 1.0
    val normalized = normalize(p)
    val maximum = normalized.max()
    return (maximum - 1.0 / count) / (1 - 1.0 / count)
  }

  /**
   * `max(0, 1 - E|level - mode| / D)` on the normalized distribution, where mode is the first most
   * likely level and D the mean distance of a uniform distribution from its centre.
   */
  fun scoreConfidence(p: DoubleArray): Double {
    val levels = p.size
    if (levels == 1) return 1.0
    val normalized = normalize(p)
    val mode = firstArgmax(normalized)
    val centre = (levels - 1) / 2.0
    val spread = pythonSum(DoubleArray(levels) { abs(it - centre) }) / levels
    val distance = pythonSum(DoubleArray(levels) { normalized[it] * abs(it - mode) })
    val confidence = 1.0 - distance / spread
    return if (confidence > 0.0) confidence else 0.0
  }

  /** `_normalize`: scaled to sum 1; a distribution that sums to 0 becomes uniform. */
  internal fun normalize(p: DoubleArray): DoubleArray {
    val total = pythonSum(p)
    return if (total == 0.0) DoubleArray(p.size) { 1.0 / p.size } else DoubleArray(p.size) { p[it] / total }
  }

  /** Index of the first maximum, as Python's `max(range(n), key=p.__getitem__)`. */
  internal fun firstArgmax(p: DoubleArray): Int {
    var best = 0
    for (index in 1 until p.size) {
      if (p[index] > p[best]) best = index
    }
    return best
  }

  /**
   * CPython 3.12's built-in `sum` of floats from the int start 0: the first value is taken as it
   * is, the rest are added with Neumaier's compensation, and the compensation is added at the end.
   */
  internal fun pythonSum(values: DoubleArray): Double {
    if (values.isEmpty()) return 0.0
    var total = 0.0 + values[0]
    var compensation = 0.0
    for (index in 1 until values.size) {
      val value = values[index]
      val sum = total + value
      compensation += if (abs(total) >= abs(value)) (total - sum) + value else (value - sum) + total
      total = sum
    }
    if (compensation != 0.0 && compensation.isFinite()) {
      total += compensation
    }
    return total
  }
}
