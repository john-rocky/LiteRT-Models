package com.gliformer

import kotlin.math.exp

/**
 * Ports GLiFormer 0.1.2 NERDecoder / SpanDecoder for the published five-label graphs.
 *
 * These are independent start/end/inside logits, not a mutually exclusive BIO tag sequence.
 * The GLiFormer decoder includes outside-neighbor probabilities in the minimum span score;
 * GLiNER's similarly named token decoder does not implement that same scoring rule.
 */
object GliformerDecoder {
  val DEFAULT_LABELS = listOf("person", "organization", "location", "product", "date")

  /** Python-compatible Unicode code-point offsets; [end] is exclusive. */
  data class Entity(
    val start: Int,
    val end: Int,
    val text: String,
    val label: String,
    val score: Float,
  )

  private data class Span(val start: Int, val end: Int, val label: Int, val score: Float)

  /** Decode a full/head graph's flattened [1, 1, T, 15] output using prepared word maps. */
  fun decode(
    logits: FloatArray,
    prepared: GliformerInputs.Prepared,
    threshold: Float = 0.5f,
  ): List<Entity> {
    require(logits.size == prepared.window.packedFloatCount) {
      "Packed output does not match the prepared s${prepared.window.sequenceLength} window"
    }
    return decode(
      logits,
      prepared.text,
      prepared.words.map { it.start }.toIntArray(),
      prepared.words.map { it.end }.toIntArray(),
      prepared.labels,
      threshold,
    )
  }

  /**
   * Applies the upstream flat-entity, single-label policy to real word rows only.
   *
   * [startMap] and [endMap] index Unicode code points, as Python strings do. The original
   * surface is sliced after translating those indices with String.offsetByCodePoints;
   * returned offsets remain code points even when the source contains surrogate pairs.
   */
  fun decode(
    logits: FloatArray,
    text: String,
    startMap: IntArray,
    endMap: IntArray,
    labels: List<String> = DEFAULT_LABELS,
    threshold: Float = 0.5f,
  ): List<Entity> {
    require(labels.size == 5 && labels.toSet().size == 5 && labels.all { it.isNotBlank() }) {
      "The static graph requires exactly five distinct nonempty labels"
    }
    require(threshold.isFinite() && threshold in 0f..1f) {
      "Threshold must be finite and between zero and one"
    }
    require(startMap.size == endMap.size) { "Word maps must have equal lengths" }
    require(logits.size % 15 == 0 && logits.size / 15 >= startMap.size) {
      "Expected packed [1, 1, T, 15] logits with room for all real words"
    }
    require(logits.all { it.isFinite() }) { "Nonfinite logits cannot be decoded" }
    val codePointLength = text.codePointCount(0, text.length)
    for (word in startMap.indices) {
      require(startMap[word] in 0 until endMap[word] && endMap[word] <= codePointLength) {
        "Invalid code-point offsets for word $word"
      }
      require(word == 0 || startMap[word] >= endMap[word - 1]) {
        "Word maps must be ordered and nonoverlapping"
      }
    }

    // NERDecoder.decode uses `threshold = threshold or self.threshold`; zero means 0.5.
    val effectiveThreshold = if (threshold == 0f) 0.5f else threshold
    val wordCount = startMap.size
    val probabilities = FloatArray(wordCount * 15) { sigmoid(logits[it]) }
    val candidates = ArrayList<Span>()

    // torch.where enumerates (word, class) in row-major order. Keep that candidate
    // order for stable score ties, including shorter end positions before longer ones.
    for (start in 0 until wordCount) {
      for (label in labels.indices) {
        val startScore = probabilities[index(start, label, 0)]
        if (startScore <= effectiveThreshold) continue
        var insideMinimum = 1f
        for (end in start until wordCount) {
          val inside = probabilities[index(end, label, 2)]
          // Upstream rejects inside < threshold; equality is accepted.
          if (inside < effectiveThreshold) break
          insideMinimum = minOf(insideMinimum, inside)
          val endScore = probabilities[index(end, label, 1)]
          if (endScore <= effectiveThreshold) continue
          var score = minOf(insideMinimum, startScore, endScore)
          if (start > 0) {
            score = minOf(score, 1f - probabilities[index(start - 1, label, 2)])
          }
          if (end + 1 < wordCount) {
            score = minOf(score, 1f - probabilities[index(end + 1, label, 2)])
          }
          // Upstream does not threshold this final score a second time.
          candidates += Span(start, end, label, score)
        }
      }
    }

    val selected = ArrayList<Span>()
    for (candidate in candidates.sortedByDescending { it.score }) {
      if (selected.none { candidate.start <= it.end && it.start <= candidate.end }) {
        selected += candidate
      }
    }
    return selected
      .sortedBy { it.start }
      .map { span ->
        val start = startMap[span.start]
        val end = endMap[span.end]
        Entity(
          start = start,
          end = end,
          text = text.substring(text.offsetByCodePoints(0, start), text.offsetByCodePoints(0, end)),
          label = labels[span.label],
          score = span.score,
        )
      }
  }

  private fun index(word: Int, label: Int, channel: Int): Int = (word * 5 + label) * 3 + channel

  // exp is evaluated by the JVM's double math, then rounded before float32 add/divide,
  // matching the reference tensor's float32 arithmetic (rather than a double sigmoid).
  private fun sigmoid(value: Float): Float = 1f / (1f + exp(-value.toDouble()).toFloat())
}
