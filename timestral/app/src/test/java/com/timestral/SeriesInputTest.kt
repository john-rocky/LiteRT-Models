package com.timestral

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** The input rules and the failure sentences the screen shows. */
class SeriesInputTest {
  private fun values(count: Int) = (1..count).joinToString(", ") { "${it % 7 + 0.5}" }

  @Test
  fun readsAListSeparatedByCommasSpacesAndNewLines() {
    val parsed = SeriesInput.parse("1, 2 3\n4;5\t6 " + (7..20).joinToString(" "))
    assertArrayEquals((1..20).map { it.toFloat() }.toFloatArray(), parsed.values, 0f)
    assertTrue(parsed.notes.isEmpty())
  }

  @Test
  fun readsTheLastColumnOfATableAndSkipsTheHeader() {
    val csv = "hour,temperature_c\n" + (0 until 20).joinToString("\n") { "$it,${it * 0.5}" }
    val parsed = SeriesInput.parse(csv)
    assertEquals(20, parsed.values.size)
    assertEquals(9.5f, parsed.values[19], 0f)
  }

  @Test
  fun emptyCellsAndMarkersAreMissing() {
    val csv = "minute,reading\n" + (0 until 20).joinToString("\n") { if (it % 5 == 0) "$it," else "$it,$it" }
    val parsed = SeriesInput.parse(csv)
    assertEquals(16, parsed.observedCount)
    assertTrue(parsed.values[0].isNaN())
    val markers = SeriesInput.parse(values(16) + ", NaN, nan, NA, n/a, null, -, ?, inf, Infinity, 1e39")
    assertEquals(16, markers.observedCount)
    assertEquals(26, markers.values.size)
    assertTrue(markers.notes.any { it.startsWith("10 missing values") })
  }

  @Test
  fun refusesTextThatIsNotANumber() {
    expectMessage("Value 3 (\"abc\") is not a number. Use numbers, or leave a field empty or write NaN for a missing value.") {
      SeriesInput.parse("1, 2, abc, 4")
    }
    expectMessage("Paste a series first: numbers separated by commas, spaces or new lines.") {
      SeriesInput.parse("  \n ")
    }
  }

  @Test
  fun refusesASeriesThatIsTooShort() {
    expectMessage("The series has 15 values; this sample needs at least 16.") {
      SeriesInput.parse(values(15))
    }
  }

  @Test
  fun refusesASeriesThatIsMostlyMissing() {
    val text = (0 until 100).joinToString(", ") { if (it % 10 == 0) "$it" else "NaN" }
    expectMessage(
      "Only 10 of 100 values in the context are numbers; this sample needs at least 16 observed " +
        "values (empty, NaN and infinite values are missing)."
    ) {
      SeriesInput.parse(text)
    }
  }

  @Test
  fun notesAConstantSeries() {
    val parsed = SeriesInput.parse(List(40) { "5" }.joinToString(","))
    assertEquals(
      listOf("Every observed value is 5: the forecast stays at that level with almost no spread."),
      parsed.notes,
    )
    assertNull(SeriesInput.constantValue(floatArrayOf(1f, Float.NaN, 2f)))
  }

  @Test
  fun notesALongSeries() {
    val parsed = SeriesInput.parse((0 until 9000).joinToString(" ") { "${it % 50}" })
    assertEquals(
      "Only the last 8,192 of 9,000 values are the context (the largest graph of this sample).",
      parsed.notes.first(),
    )
  }

  @Test
  fun checksTheHorizon() {
    assertEquals(64, SeriesInput.horizon(" 64 "))
    assertEquals(1024, SeriesInput.horizon("1,024"))
    val message =
      "The horizon must be a whole number of steps from 1 to 1,024 (one decoder block of 32 " +
        "patches); longer horizons are not part of this sample."
    for (text in listOf("0", "1025", "2.5", "", "-3")) {
      expectMessage(message) { SeriesInput.horizon(text) }
    }
  }

  @Test
  fun refusesAContextLongerThanTheBackendsGraphs() {
    val graphs =
      listOf(
        GraphSpec("c2048-h256", "a.tflite", 64, 8, 2, -3e4f, setOf(Backend.GPU, Backend.NPU)),
        GraphSpec("c8192-h256", "b.tflite", 256, 8, 2, -3e4f, setOf(Backend.GPU)),
        GraphSpec("c2048-h256-int8", "c.tflite", 64, 8, 2, -3e4f, setOf(Backend.CPU)),
      )
    assertNull(SeriesInput.contextTooLong(graphs, Backend.NPU, 2048))
    assertNull(SeriesInput.contextTooLong(graphs, Backend.GPU, 8192))
    assertEquals(
      "The NPU takes a context of up to 2,048 steps, and this one has 2,400. The 8,192-step " +
        "graphs run on the GPU and the CPU only: on a Galaxy S26, 8,192-step builds of this " +
        "model ran 82 to 97 ms per forecast on the NPU against 69 to 73 ms for these graphs on " +
        "the GPU. Choose another accelerator, or keep the last 2,048 values of the series.",
      SeriesInput.contextTooLong(graphs, Backend.NPU, 2400)?.message,
    )
    assertEquals(
      "The CPU takes a context of up to 2,048 steps, and this one has 2,049. Choose another " +
        "accelerator, or keep the last 2,048 values of the series.",
      SeriesInput.contextTooLong(graphs, Backend.CPU, 2049)?.message,
    )
  }

  @Test
  fun readsFullWidthDigitsCommasAndMinusSigns() {
    // What a Japanese keyboard in full-width mode types for "212,-3.5,..." (U+FF10..U+FF19 digits,
    // U+FF0C comma, U+FF0E period, U+FF0D minus), written as escapes.
    val ascii = (0 until 20).joinToString(",") { if (it == 1) "-3.5" else "${200 + it}" }
    val fullWidth =
      ascii
        .map { c ->
          when (c) {
            in '0'..'9' -> (c - '0' + 0xFF10).toChar()
            ',' -> '，'
            '.' -> '．'
            '-' -> '－'
            else -> c
          }
        }
        .joinToString("")
    assertArrayEquals(SeriesInput.parse(ascii).values, SeriesInput.parse(fullWidth).values, 0f)
    assertEquals(24, SeriesInput.horizon("２４"))
  }

  @Test
  fun readsWideTablesOfNumbersAsAList() {
    val parsed = SeriesInput.parse("1,2,3,4,5,6\n7,8,9,10,11,12\n13,14,15,16,17,18")
    assertEquals(18, parsed.values.size)
  }

  private fun expectMessage(message: String, block: () -> Unit) {
    try {
      block()
      fail("no InputException")
    } catch (failure: InputException) {
      assertEquals(message, failure.message)
    }
  }
}
