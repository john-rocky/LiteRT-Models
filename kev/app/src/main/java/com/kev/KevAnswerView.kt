package com.kev

import java.math.BigDecimal
import java.math.RoundingMode

/**
 * One answer as the screen shows it. Every number is the 4-decimal string of a value in the
 * `to_answers` entry, so the screen, the log and the demo run JSON print the same digits.
 */
class KevAnswerView(
  val type: QuestionType,
  /** choice: the chosen key; noul: p(true); score: the expected level. */
  val answer: String,
  /** choice and score: the confidence; noul: null. */
  val confidence: String?,
  /** Every option in option order (noul: only `true`, whose probability is the answer). */
  val bars: List<Bar>,
  /** The option the compact (presentation) card shows: the chosen key, `true`, or the likeliest level. */
  val headline: Bar,
) {
  /** One option: its key, its label on screen, its rounded probability and the bar length. */
  class Bar(val key: String, val label: String, val value: String, val fraction: Float)

  /**
   * The strings the compact card shows, keyed like the answer: `choice` and its probability under
   * `probabilities`, `noul`, or the likeliest score level's probability under `probabilities`.
   */
  fun shownCompact(): LinkedHashMap<String, Any?> =
    when (type) {
      QuestionType.CHOICE -> linkedMapOf("choice" to answer, "probabilities" to linkedMapOf(headline.key to headline.value))
      QuestionType.NOUL -> linkedMapOf("noul" to answer)
      QuestionType.SCORE -> linkedMapOf("probabilities" to linkedMapOf(headline.key to headline.value))
    }

  companion object {
    /**
     * The view of [answer], one `to_answers` entry for [meta]; [probabilities] (unrounded) only pick
     * the likeliest option and size the bars.
     */
    fun of(answer: Map<*, *>, meta: QuestionMeta, probabilities: DoubleArray): KevAnswerView {
      val rounded = answer["probabilities"] as Map<*, *>?
      val first = KevAnswers.firstArgmax(probabilities)
      fun bar(index: Int, label: String, value: Double) =
        Bar(meta.keys[index], label, fourDecimals(value), probabilities[index].toFloat())
      return when (meta.type) {
        QuestionType.CHOICE -> {
          val bars = meta.keys.indices.map { bar(it, meta.keys[it], rounded!![meta.keys[it]] as Double) }
          KevAnswerView(meta.type, answer["choice"] as String, fourDecimals(answer["confidence"] as Double), bars, bars[first])
        }
        QuestionType.NOUL -> {
          val noul = answer["noul"] as Double
          val yes = bar(1, meta.keys[1], noul)
          KevAnswerView(meta.type, fourDecimals(noul), null, listOf(yes), yes)
        }
        QuestionType.SCORE -> {
          val legend = requireNotNull(meta.legend)
          val bars = meta.keys.indices.map { bar(it, legend.getValue(meta.keys[it]), rounded!![meta.keys[it]] as Double) }
          KevAnswerView(
            meta.type,
            fourDecimals(answer["score"] as Double),
            fourDecimals(answer["confidence"] as Double),
            bars,
            bars[first],
          )
        }
      }
    }

    /**
     * A value already rounded by `round_prob` as its 4-decimal string (0.373 → "0.3730"). The exact
     * binary value lies within half an ulp of the 4-decimal number, so this restores its digits.
     */
    fun fourDecimals(value: Double): String =
      BigDecimal(value).setScale(DECIMALS, RoundingMode.HALF_EVEN).toPlainString()

    /** Milliseconds as the whole number the screen, the log and the run JSON share. */
    fun wholeMillis(ms: Double): Long = Math.round(ms)

    private const val DECIMALS = 4
  }
}
