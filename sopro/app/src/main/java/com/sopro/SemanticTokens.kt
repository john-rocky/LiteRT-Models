package com.sopro

/** Exact source FSQ indexing for the semantic graph's [1,235,27] float logits. */
object SemanticTokens {
  const val FRAME_COUNT = 235
  const val LOGITS_PER_FRAME = 27
  private val levels = intArrayOf(7, 5, 5, 5, 5)
  private val bases = intArrayOf(1, 7, 35, 175, 875)

  fun decode(logits: FloatArray): IntArray {
    require(logits.size == FRAME_COUNT * LOGITS_PER_FRAME) {
      "Semantic digit logits must have shape [1,235,27]; received ${logits.size} values."
    }
    require(logits.all { it.isFinite() }) { "Semantic digit logits contain nonfinite values." }
    return IntArray(FRAME_COUNT) { frame ->
      var start = frame * LOGITS_PER_FRAME
      var token = 0
      for (digit in levels.indices) {
        var best = 0
        for (index in 1 until levels[digit]) {
          // Torch argmax chooses the first index on an exact tie.
          if (logits[start + index] > logits[start + best]) best = index
        }
        token += best * bases[digit]
        start += levels[digit]
      }
      token
    }
  }
}
