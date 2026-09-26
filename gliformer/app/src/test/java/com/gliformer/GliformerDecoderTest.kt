package com.gliformer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Branch tests for details that ordinary high-confidence entity fixtures do not exercise. */
class GliformerDecoderTest {
  @Test
  fun outsideNeighborsPreferCompleteSpanOverSuffix() {
    val logits = emptyLogits(2)
    set(logits, 0, 0, 0, 100f)
    set(logits, 1, 0, 0, 100f)
    set(logits, 1, 0, 1, 100f)
    set(logits, 0, 0, 2, 100f)
    set(logits, 1, 0, 2, 100f)
    val result = decodeWords(logits, "Ada Lovelace", intArrayOf(0, 4), intArrayOf(3, 12))
    assertEquals(listOf("Ada Lovelace"), result.map { it.text })
    assertEquals(1f, result.single().score, 0f)
  }

  @Test
  fun lowFinalBoundaryScoreIsNotThresholdedAgain() {
    val logits = emptyLogits(2)
    for (channel in 0..2) set(logits, 0, 0, channel, 100f)
    set(logits, 1, 0, 2, 100f)
    val result = decodeWords(logits, "Ada x", intArrayOf(0, 4), intArrayOf(3, 5))
    assertEquals("Ada", result.single().text)
    assertEquals(0f, result.single().score, 0f)
  }

  @Test
  fun startAndEndAreStrictButInsideThresholdIsInclusive() {
    val logits = emptyLogits(1)
    set(logits, 0, 0, 0, 100f)
    set(logits, 0, 0, 1, 100f)
    set(logits, 0, 0, 2, 0f)
    assertEquals(0.5f, decodeWords(logits, "Ada", intArrayOf(0), intArrayOf(3)).single().score, 0f)
    set(logits, 0, 0, 0, 0f)
    assertTrue(decodeWords(logits, "Ada", intArrayOf(0), intArrayOf(3)).isEmpty())
    set(logits, 0, 0, 0, 100f)
    set(logits, 0, 0, 1, 0f)
    assertTrue(decodeWords(logits, "Ada", intArrayOf(0), intArrayOf(3)).isEmpty())
  }

  @Test
  fun stableTiesKeepClassThenEndPositionEnumeration() {
    val classes = emptyLogits(1)
    for (label in 0..1) for (channel in 0..2) set(classes, 0, label, channel, 100f)
    assertEquals("person", decodeWords(classes, "Ada", intArrayOf(0), intArrayOf(3)).single().label)

    val ends = emptyLogits(2)
    for (word in 0..1) {
      set(ends, word, 0, 0, 100f)
      set(ends, word, 0, 1, 100f)
      set(ends, word, 0, 2, 0f)
    }
    assertEquals(
      listOf("Ada", "Bob"),
      decodeWords(ends, "Ada Bob", intArrayOf(0, 4), intArrayOf(3, 7)).map { it.text },
    )
  }

  @Test
  fun paddedRowsDoNotBecomeOutsideBoundaryEvidence() {
    val logits = emptyLogits(48)
    for (channel in 0..2) set(logits, 0, 0, channel, 100f)
    set(logits, 1, 0, 2, 100f)
    val result = decodeWords(logits, "Ada", intArrayOf(0), intArrayOf(3))
    assertEquals(1f, result.single().score, 0f)
  }

  @Test
  fun entityOffsetsRemainCodePointsAroundSupplementaryCharacters() {
    val logits = emptyLogits(2)
    for (channel in 0..2) {
      set(logits, 0, 3, channel, 100f)
      set(logits, 1, 0, channel, 100f)
    }
    val result = decodeWords(logits, "😀 Ada", intArrayOf(0, 2), intArrayOf(1, 5))
    assertEquals(listOf("😀", "Ada"), result.map { it.text })
    assertEquals(listOf(0 to 1, 2 to 5), result.map { it.start to it.end })
  }

  @Test(expected = IllegalArgumentException::class)
  fun nonfinitePaddingIsRejectedLikeHostRuntime() {
    val logits = emptyLogits(48)
    logits[logits.lastIndex] = Float.NaN
    decodeWords(logits, "Ada", intArrayOf(0), intArrayOf(3))
  }

  @Test
  fun zeroThresholdUsesUpstreamDefault() {
    val logits = FloatArray(15) { -1f }
    assertTrue(
      GliformerDecoder.decode(logits, "Ada", intArrayOf(0), intArrayOf(3), threshold = 0f).isEmpty()
    )
  }

  private fun emptyLogits(words: Int) = FloatArray(words * 15) { -100f }

  private fun set(logits: FloatArray, word: Int, label: Int, channel: Int, value: Float) {
    logits[(word * 5 + label) * 3 + channel] = value
  }

  private fun decodeWords(logits: FloatArray, text: String, starts: IntArray, ends: IntArray) =
    GliformerDecoder.decode(logits, text, starts, ends)
}
