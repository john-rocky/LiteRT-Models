package com.gliner25

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Multi host pieces that need no external data. */
class GlinerMultiHostTest {
  // Expected (token, start, end) values were printed by gliner2 2.0.0
  // processing/word_splitter.py (CharLevelSplitter and WhitespaceTokenSplitter, lower=True).
  private val texts =
    listOf(
      "星瀬澄香は京都で霧灯研究社の新製品『星糸端末』を紹介した。",
      "ABC株式会社のPhone-X 15は2025年3月12日に発売。",
      "Email: a.b+c@x-y.io, URL www.ex.com",
      "A😀İ 東京",
      "全角ＡＢＣ１２３と半角abc_DEF",
      "tab\tnew\nline　ideo nbsp thin",
      "​zero‍width",
      "x\u001Cy\u001Fz\u0085w",
      "İstanbul'da é",
      "👨‍👩‍👧 ok!",
    )

  @Test
  fun charSplitterMatchesGliner2() {
    val expected =
      listOf(
        texts[0].mapIndexed { index, character -> w("$character", index) },
        listOf(
          w("abc", 0, 3),
          w("株", 3),
          w("式", 4),
          w("会", 5),
          w("社", 6),
          w("の", 7),
          w("phone-x", 8, 15),
          w("15", 16, 18),
          w("は", 18),
          w("2025", 19, 23),
          w("年", 23),
          w("3", 24),
          w("月", 25),
          w("12", 26, 28),
          w("日", 28),
          w("に", 29),
          w("発", 30),
          w("売", 31),
          w("。", 32),
        ),
        listOf(
          w("email", 0, 5),
          w(":", 5),
          w("a.b+c@x-y.io", 7, 19),
          w(",", 19),
          w("url", 21, 24),
          w("www.ex.com", 25, 35),
        ),
        listOf(w("a", 0), w("😀", 1), w("i̇", 2), w("東", 4), w("京", 5)),
        listOf(
          w("全", 0),
          w("角", 1),
          w("ａ", 2),
          w("ｂ", 3),
          w("ｃ", 4),
          w("１", 5),
          w("２", 6),
          w("３", 7),
          w("と", 8),
          w("半", 9),
          w("角", 10),
          w("abc_def", 11, 18),
        ),
        listOf(
          w("tab", 0, 3),
          w("new", 4, 7),
          w("line", 8, 12),
          w("ideo", 13, 17),
          w("nbsp", 18, 22),
          w("thin", 23, 27),
        ),
        listOf(w("​", 0), w("zero", 1, 5), w("‍", 5), w("width", 6, 11)),
        listOf(w("x", 0), w("y", 2), w("z", 4), w("w", 6)),
        listOf(
          w("i̇", 0),
          w("stanbul", 1, 8),
          w("'", 8),
          w("da", 9, 11),
          w("e", 12),
          w("́", 13),
        ),
        listOf(
          w("👨", 0),
          w("‍", 1),
          w("👩", 2),
          w("‍", 3),
          w("👧", 4),
          w("ok", 6, 8),
          w("!", 8),
        ),
      )
    texts.zip(expected).forEach { (text, words) ->
      assertEquals(text, words, GlinerInputs.splitCharacters(text))
      assertEquals(text, words, GlinerInputs.Splitter.CHAR.split(text))
    }
  }

  @Test
  fun whitespaceSplitterMatchesGliner2OnTheSameText() {
    val expected =
      listOf(
        listOf(
          w("星瀬澄香は京都で霧灯研究社の新製品", 0, 17),
          w("『", 17),
          w("星糸端末", 18, 22),
          w("』", 22),
          w("を紹介した", 23, 28),
          w("。", 28),
        ),
        listOf(w("abc株式会社のphone-x", 0, 15), w("15は2025年3月12日に発売", 16, 32), w("。", 32)),
        listOf(
          w("email", 0, 5),
          w(":", 5),
          w("a.b+c@x-y.io", 7, 19),
          w(",", 19),
          w("url", 21, 24),
          w("www.ex.com", 25, 35),
        ),
        listOf(w("a", 0), w("😀", 1), w("i̇", 2), w("東京", 4, 6)),
        listOf(w("全角ａｂｃ１２３と半角abc_def", 0, 18)),
        listOf(
          w("tab", 0, 3),
          w("new", 4, 7),
          w("line", 8, 12),
          w("ideo", 13, 17),
          w("nbsp", 18, 22),
          w("thin", 23, 27),
        ),
        listOf(w("​", 0), w("zero", 1, 5), w("‍", 5), w("width", 6, 11)),
        listOf(w("x", 0), w("y", 2), w("z", 4), w("w", 6)),
        listOf(w("i̇stanbul", 0, 8), w("'", 8), w("da", 9, 11), w("e", 12), w("́", 13)),
        listOf(
          w("👨", 0),
          w("‍", 1),
          w("👩", 2),
          w("‍", 3),
          w("👧", 4),
          w("ok", 6, 8),
          w("!", 8),
        ),
      )
    texts.zip(expected).forEach { (text, words) ->
      assertEquals(text, words, GlinerInputs.Splitter.WHITESPACE.split(text))
    }
  }

  @Test
  fun terminalPeriodFollowsCollateBatch() {
    assertEquals("a.", GlinerInputs.withTerminalPeriod("a"))
    assertEquals("a.", GlinerInputs.withTerminalPeriod("a."))
    assertEquals("a!", GlinerInputs.withTerminalPeriod("a!"))
    assertEquals("a?", GlinerInputs.withTerminalPeriod("a?"))
    assertEquals("紹介した。.", GlinerInputs.withTerminalPeriod("紹介した。"))
    assertEquals("a. .", GlinerInputs.withTerminalPeriod("a. "))
    assertEquals(".", GlinerInputs.withTerminalPeriod(""))
  }

  @Test
  fun halfUpcastIsExactForEveryBitPattern() {
    var finite = 0
    for (bits in 0..0xffff) {
      val actual = GlinerInputs.halfToFloat(bits)
      // The table path passes sign-extended shorts.
      assertEquals(actual.toRawBits(), GlinerInputs.halfToFloat(bits.toShort().toInt()).toRawBits())
      val exponent = (bits shr 10) and 0x1f
      val mantissa = bits and 0x3ff
      val sign =
        if (bits and 0x8000 != 0) {
          -1.0
        } else {
          1.0
        }
      when {
        exponent == 0x1f && mantissa != 0 -> assertTrue("NaN $bits", actual.isNaN())
        exponent == 0x1f -> assertEquals(sign * Double.POSITIVE_INFINITY, actual.toDouble(), 0.0)
        else -> {
          val magnitude =
            if (exponent == 0) {
              Math.scalb(mantissa.toDouble(), -24)
            } else {
              Math.scalb(1024.0 + mantissa, exponent - 25)
            }
          // Raw bits also distinguish -0.0 from 0.0.
          assertEquals("bits $bits", (sign * magnitude).toFloat().toRawBits(), actual.toRawBits())
          finite++
        }
      }
    }
    assertEquals(63488, finite)
  }

  private fun w(token: String, start: Int, end: Int = start + 1) =
    GlinerInputs.Word(token, start, end)
}
