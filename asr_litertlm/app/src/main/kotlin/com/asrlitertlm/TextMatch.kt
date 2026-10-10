package com.asrlitertlm

import java.text.Normalizer

/**
 * Transcript comparison for the device check, the way the model cards score: NFKC, lowercase, every punctuation,
 * separator and whitespace character removed, then the character error rate against the expected text.
 */
object TextMatch {
  fun normalize(text: String): String {
    val folded = Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase()
    val out = StringBuilder()
    var i = 0
    while (i < folded.length) {
      val codePoint = folded.codePointAt(i)
      if (!isDropped(codePoint)) out.appendCodePoint(codePoint)
      i += Character.charCount(codePoint)
    }
    return out.toString()
  }

  /** Edit distance between the normalized texts over the normalized [expected] length (code points). */
  fun cer(expected: String, actual: String): Double {
    val reference = normalize(expected).codePoints().toArray()
    val hypothesis = normalize(actual).codePoints().toArray()
    if (reference.isEmpty()) return if (hypothesis.isEmpty()) 0.0 else 1.0
    return editDistance(reference, hypothesis).toDouble() / reference.size
  }

  /** True when the two texts are equal after [normalize]. */
  fun matches(expected: String, actual: String): Boolean = normalize(expected) == normalize(actual)

  private fun isDropped(codePoint: Int): Boolean =
    Character.isWhitespace(codePoint) ||
      when (Character.getType(codePoint).toByte()) {
        Character.CONNECTOR_PUNCTUATION,
        Character.DASH_PUNCTUATION,
        Character.START_PUNCTUATION,
        Character.END_PUNCTUATION,
        Character.INITIAL_QUOTE_PUNCTUATION,
        Character.FINAL_QUOTE_PUNCTUATION,
        Character.OTHER_PUNCTUATION,
        Character.SPACE_SEPARATOR,
        Character.LINE_SEPARATOR,
        Character.PARAGRAPH_SEPARATOR -> true
        else -> false
      }

  private fun editDistance(a: IntArray, b: IntArray): Int {
    var previous = IntArray(b.size + 1) { it }
    var current = IntArray(b.size + 1)
    for (i in 1..a.size) {
      current[0] = i
      for (j in 1..b.size) {
        val substitution = previous[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1
        current[j] = minOf(substitution, previous[j] + 1, current[j - 1] + 1)
      }
      val swap = previous
      previous = current
      current = swap
    }
    return previous[b.size]
  }
}
