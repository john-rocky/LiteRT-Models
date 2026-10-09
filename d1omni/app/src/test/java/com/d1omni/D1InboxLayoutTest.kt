package com.d1omni

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The presentation's layout plan with a stand-in for the device's fonts (widths per character close
 * to Arial's, a little wider than Roboto's; the phone measures with its own fonts): the 9:16 band,
 * every text inside its room, the long score level on two lines and the others on one, the photo
 * between its bounds, a larger font scale handled by a smaller text scale.
 */
class D1InboxLayoutTest {
  /** Arial-like advance widths (em) per character class; emoji one em and a half. */
  private class StandIn(private val density: Float, private val fontScale: Float) : D1TextMetrics {
    private fun em(ch: Char, bold: Boolean): Float {
      val regular =
        when {
          ch.isSurrogate() -> 0.75f
          ch.code > 0x2000 -> 0.6f
          ch == ' ' -> 0.28f
          ch in "il.,:;'|!" -> 0.25f
          ch in "frtjI()-·" -> 0.35f
          ch in "mwMW" -> 0.85f
          ch.isUpperCase() -> 0.68f
          ch.isDigit() -> 0.56f
          else -> 0.53f
        }
      return if (bold) regular * 1.06f else regular
    }

    override fun width(text: String, sp: Float, bold: Boolean): Float =
      text.sumOf { em(it, bold).toDouble() }.toFloat() * sp * fontScale * density

    override fun lines(text: String, sp: Float, bold: Boolean, widthPx: Float): Int {
      var lines = 1
      var line = 0f
      for (word in text.split(' ')) {
        val w = width(if (line == 0f) word else " $word", sp, bold)
        if (line > 0f && line + w > widthPx) {
          lines++
          line = width(word, sp, bold)
        } else {
          line += w
        }
      }
      return lines
    }
  }

  private fun input(): D1InboxLayoutInput {
    val fixture = D1InboxFixture.parse(File("src/main/res/raw/inbox_demo.json").readBytes())
    val graphs = D1InboxText.graphsLine(listOf(128, 256), listOf(1001), vision = true)
    val footer = D1InboxText.footer("Galaxy S26", "GPU", graphs, null, true).dropLast(1) + D1InboxText.footerTotalWidest()
    return D1InboxLayoutInput.of(fixture, mapOf("aud_food_03" to "Voice note · 8.7 s"), footer)
  }

  @Test
  fun band() {
    assertEquals(210 to 1920, D1InboxLayout.band(1080, 2340))
    assertEquals(0 to 1920, D1InboxLayout.band(1080, 1920))
    assertEquals(280 to 2560, D1InboxLayout.band(1440, 3120))
  }

  @Test
  fun galaxyS26Plan() {
    val plan = D1InboxLayout.plan(1080, 2340, 3.0f, 1.0f, input(), StandIn(3.0f, 1.0f))
    println("D1_LAYOUT ${D1Json.write(plan.toJson())}")
    assertTrue("fits", plan.fits)
    assertEquals(210, plan.bandTop)
    assertEquals(1920, plan.bandHeight)
    assertTrue("content ${plan.contentPx} within the band", plan.contentPx <= plan.bandHeight)
    assertTrue("photo ${plan.thumbnailPx} px", plan.thumbnailPx >= 72 * 3 && plan.thumbnailPx <= 120 * 3)
    assertTrue("scale ${plan.scale}", plan.scale >= 1.0f)
    assertEquals(2, plan.rowLines["card_text/urgency"])
    for (key in listOf("aud_food_03/topic", "aud_food_03/request", "aud_food_03/urgency", "img_dogs_01/animal",
        "img_dogs_01/water", "card_text/refund", "card_text/team")) {
      assertEquals(key, 1, plan.rowLines[key])
    }
    assertTrue(plan.texts.all { it.fits })
    assertEquals(2, plan.messageLines["card_text"])
    assertTrue("answer ${plan.answerSp} sp", plan.answerSp >= 16f)
  }

  @Test
  fun largerFontScaleShrinksTheScale() {
    val normal = D1InboxLayout.plan(1080, 2340, 3.0f, 1.0f, input(), StandIn(3.0f, 1.0f))
    val large = D1InboxLayout.plan(1080, 2340, 3.0f, 1.3f, input(), StandIn(3.0f, 1.3f))
    println("D1_LAYOUT large ${D1Json.write(large.toJson())}")
    assertTrue("${large.scale} < ${normal.scale}", large.scale < normal.scale)
    assertTrue(large.fits)
    assertTrue(large.contentPx <= large.bandHeight)
  }

  @Test
  fun aBandTooSmallDoesNotFit() {
    val plan = D1InboxLayout.plan(540, 700, 3.0f, 1.0f, input(), StandIn(3.0f, 1.0f))
    assertFalse(plan.fits)
  }
}
