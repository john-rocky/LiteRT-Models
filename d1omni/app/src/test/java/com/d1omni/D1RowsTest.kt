package com.d1omni

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test

/**
 * `build_inputs` (host/d1_host.py) in Kotlin: [prefix rows | ids | pad] in a bucket of L positions,
 * media / pad / keep_right from P and n, the question type's one-hot; the request kinds' rules.
 */
class D1RowsTest {
  @Test
  fun textIdsStartAtZeroAndArePaddedWithZero() {
    val inputs = D1Rows.buildInputs(intArrayOf(1, 17, 21, 16), null, 0, 7, QuestionType.NOUL)
    assertArrayEquals(intArrayOf(1, 17, 21, 16, 0, 0, 0), inputs.ids)
    assertNull(inputs.prefix)
    assertArrayEquals(FloatArray(7), inputs.media, 0f)
    assertArrayEquals(floatArrayOf(1f, 1f, 1f, 1f, 0f, 0f, 0f), inputs.pad, 0f)
    // P = 0: keep_right = (t != -1) is 1 everywhere.
    assertArrayEquals(FloatArray(7) { 1f }, inputs.keepRight, 0f)
    assertArrayEquals(floatArrayOf(0f, 0f, 1f), inputs.qtypeOneHot, 0f)
  }

  @Test
  fun mediaRowsComeFirst() {
    val width = D1Rows.PREFIX_WIDTH
    val prefix = FloatArray(3 * width) { it + 0.5f }
    val inputs = D1Rows.buildInputs(intArrayOf(1, 18, 21), prefix, 3, 8, QuestionType.CHOICE)
    assertArrayEquals(intArrayOf(0, 0, 0, 1, 18, 21, 0, 0), inputs.ids)
    assertArrayEquals(floatArrayOf(1f, 1f, 1f, 0f, 0f, 0f, 0f, 0f), inputs.media, 0f)
    assertArrayEquals(floatArrayOf(1f, 1f, 1f, 1f, 1f, 1f, 0f, 0f), inputs.pad, 0f)
    assertArrayEquals(floatArrayOf(1f, 1f, 0f, 1f, 1f, 1f, 1f, 1f), inputs.keepRight, 0f)
    assertArrayEquals(floatArrayOf(1f, 0f, 0f), inputs.qtypeOneHot, 0f)
    val values = requireNotNull(inputs.prefix)
    assertEquals(8 * width, values.size)
    for (i in 0 until 3 * width) assertEquals(i + 0.5f, values[i], 0f)
    for (i in 3 * width until 8 * width) assertEquals(0f, values[i], 0f)
  }

  @Test
  fun rowsThatDoNotFitAreRefused() {
    for (bad in
      listOf<() -> Unit>(
        { D1Rows.buildInputs(IntArray(9), null, 0, 8, QuestionType.NOUL) },
        { D1Rows.buildInputs(IntArray(5), FloatArray(4 * D1Rows.PREFIX_WIDTH), 4, 8, QuestionType.NOUL) },
        { D1Rows.buildInputs(IntArray(2), FloatArray(7), 2, 8, QuestionType.NOUL) },
        { D1Rows.buildInputs(IntArray(2), null, 2, 8, QuestionType.NOUL) },
      )) {
      try {
        bad()
        fail("accepted")
      } catch (expected: IllegalArgumentException) {
        // build_inputs raises for a row that does not fit, the layout refuses bad prefixes.
      }
    }
  }

  @Test
  fun bucketsAndKinds() {
    val buckets = listOf(128, 256, 512, 1024, 2048, 4096)
    assertEquals(128, D1Contract.bucketFor(1, buckets))
    assertEquals(128, D1Contract.bucketFor(128, buckets))
    assertEquals(256, D1Contract.bucketFor(129, buckets))
    assertEquals(4096, D1Contract.bucketFor(4096, buckets))
    assertNull(D1Contract.bucketFor(4097, buckets))
    assertEquals(256, D1Contract.bucketFor(100, listOf(256, 512)))
    val tokenizer = ExternalTestData.tokenizer()
    val contract = ExternalTestData.contract()
    val question = D1Prompt.asQuestion(D1Json.parse("""{"type": "noul", "instructions": "x"}"""))
    try {
      D1Rows.rows(tokenizer, contract, "s", listOf(question), 10, D1Kind.TEXT)
      fail("a text request with a prefix was accepted")
    } catch (expected: IllegalArgumentException) {
      // D1Omni.rows: a text request has no prefix.
    }
    try {
      D1Rows.rows(tokenizer, contract, null, listOf(question), 16384 - 63, D1Kind.IMAGE)
      fail("an image request without room for text was accepted")
    } catch (expected: IllegalArgumentException) {
      // D1Omni.rows: the media take too many positions (max_len < 64).
    }
    // A null audio state becomes {} ("{}" in the row), a null text state "".
    val audio = D1Rows.rows(tokenizer, contract, null, listOf(question), 100, D1Kind.AUDIO).single()
    val text = D1Rows.rows(tokenizer, contract, null, listOf(question), 0, D1Kind.TEXT).single()
    assertArrayEquals(
      intArrayOf(1, 17) + tokenizer.encode("{}"),
      audio.ids.copyOfRange(0, 2 + tokenizer.encode("{}").size),
    )
    assertEquals(18, text.ids[2])
    assertEquals(100 + audio.ids.size, audio.positions)
    assertEquals(audio.positions + text.positions, D1Rows.inputTokens(listOf(audio, text)))
  }
}
