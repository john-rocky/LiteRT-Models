package com.gliner25

import java.io.File
import java.util.regex.Pattern
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class GlinerInputsTest {
  private lateinit var root: File

  @Before
  fun locateExternalData() {
    root = ExternalTestData.resolve()
  }

  @Test
  fun capturedInputsMatchAtEveryFittingWindow() {
    val tokenizer = GlinerTokenizer(File(root, "host_assets/tokenizer.json"))
    val inputs = GlinerInputs(tokenizer)
    ExternalTestData.requireFiles(root, "fixtures/captured", "fixtures/f1/captured")
    val short = capturedFiles(File(root, "fixtures/captured"))
    val f1 = capturedFiles(File(root, "fixtures/f1/captured"))
    assertEquals("short fixture files", 10, short.size)
    assertEquals("F1 fixture files", 70, f1.size)
    val records = JSONArray()
    val uniqueTexts = mutableSetOf<String>()
    val uniquePairs = mutableSetOf<Pair<Int, String>>()
    val fileWindowCounts = linkedMapOf(128 to 0, 256 to 0, 512 to 0)
    var pairCount = 0
    for (file in short + f1) {
      val fixture = JSONObject(file.readText())
      val text = fixture.getString("text")
      uniqueTexts.add(text)
      val fixtureName = file.relativeTo(root).invariantSeparatorsPath
      val expectedIds = fixture.tensorInts("input_ids")
      val expectedAttention = fixture.tensorInts("attention_mask")
      val textPositions = fixture.tensorInts("text_word_indices")
      val queryPositions = fixture.tensorInts("query_marker_indices")
      val textValid = fixture.tensorBooleans("text_word_mask")
      val queryValid = fixture.tensorBooleans("query_marker_mask")
      val offsets = fixture.getJSONArray("token_to_char")
      val fitting =
        GlinerInputs.WINDOWS.filter {
          expectedIds.size <= it.sequenceLength && textPositions.size <= it.textCapacity
        }
      assertTrue("$fixtureName must fit a published graph", fitting.isNotEmpty())
      assertEquals(
        "$fixtureName chooses smallest fitting graph",
        fitting.first(),
        inputs.prepare(text).window,
      )
      for (window in fitting) {
        val actual = inputs.prepare(text, window)
        val context = "$fixtureName / s${window.sequenceLength}"
        assertEquals("$context encoded length", expectedIds.size, actual.encodedLength)
        assertArrayEquals(
          "$context input_ids",
          expectedIds.copyOf(window.sequenceLength),
          actual.inputIds,
        )
        assertArrayEquals(
          "$context attention_mask",
          FloatArray(window.sequenceLength) { expectedAttention.getOrElse(it) { 0 }.toFloat() },
          actual.attentionMask,
          0f,
        )
        assertArrayEquals("$context text positions", textPositions, actual.textWordPositions)
        assertArrayEquals("$context query positions", queryPositions, actual.queryMarkerPositions)
        val expectedTextRouting = FloatArray(window.textCapacity * window.sequenceLength)
        for (row in textPositions.indices) {
          if (textValid[row]) {
            expectedTextRouting[row * window.sequenceLength + textPositions[row]] = 1f
          }
        }
        assertArrayEquals("$context text_routing", expectedTextRouting, actual.textRouting, 0f)
        val expectedQueryRouting = FloatArray(queryPositions.size * window.sequenceLength)
        for (row in queryPositions.indices) {
          if (queryValid[row]) {
            expectedQueryRouting[row * window.sequenceLength + queryPositions[row]] = 1f
          }
        }
        assertArrayEquals("$context query_routing", expectedQueryRouting, actual.queryRouting, 0f)
        assertArrayEquals(
          "$context text_mask",
          FloatArray(window.textCapacity) {
            if (textValid.getOrElse(it) { false }) {
              1f
            } else {
              0f
            }
          },
          actual.textMask,
          0f,
        )
        assertEquals("$context text words", offsets.length(), actual.words.size)
        for (wordIndex in actual.words.indices) {
          val pair = offsets.getJSONArray(wordIndex)
          assertEquals(
            "$context word $wordIndex start",
            pair.getInt(0),
            actual.words[wordIndex].start,
          )
          assertEquals("$context word $wordIndex end", pair.getInt(1), actual.words[wordIndex].end)
        }
        pairCount++
        uniquePairs.add(window.sequenceLength to text)
        fileWindowCounts[window.sequenceLength] =
          fileWindowCounts.getValue(window.sequenceLength) + 1
      }
      records.put(
        JSONObject()
          .put("file", fixtureName)
          .put("encoded_tokens", expectedIds.size)
          .put("text_words", textPositions.size)
          .put("fitting_windows", JSONArray(fitting.map { it.sequenceLength }))
          .put("all_fields_elementwise_identical", true)
      )
    }
    assertEquals("unique inputs", 70, uniqueTexts.size)
    assertEquals("file/window pairs", 225, pairCount)
    assertEquals("unique input/window pairs", 195, uniquePairs.size)
    val uniqueWindowCounts =
      GlinerInputs.WINDOWS.associate {
        it.sequenceLength to uniquePairs.count { pair -> pair.first == it.sequenceLength }
      }
    assertEquals(mapOf(128 to 60, 256 to 65, 512 to 70), uniqueWindowCounts)
    val report =
      JSONObject()
        .put("status", "PASS")
        .put("fixture_files", records.length())
        .put("unique_inputs", uniqueTexts.size)
        .put("file_window_pairs", pairCount)
        .put("unique_window_input_pairs", uniquePairs.size)
        .put("file_window_counts", JSONObject(fileWindowCounts.mapKeys { "s${it.key}" }))
        .put("unique_window_counts", JSONObject(uniqueWindowCounts.mapKeys { "s${it.key}" }))
        .put(
          "compared",
          JSONArray(
            listOf(
              "input_ids",
              "attention_mask",
              "text_routing",
              "query_routing",
              "text_mask",
              "token_to_char",
            )
          ),
        )
        .put("max_absolute_difference", 0)
        .put("fixtures", records)
    ExternalTestData.reportFile("tokenizer_parity.json").writeText(report.toString(2) + "\n")
    println(
      "TOKENIZER_PARITY PASS files=80/80 unique_inputs=70/70 " +
        "file_pairs=225/225 unique_pairs=195/195 max_error=0"
    )
  }

  @Test
  fun androidCompatibleNormalizerInitializesBeforeTheWorkedSentenceIsSplit() {
    // Android rejects the unsupported flag during class initialization, before processing any text.
    // Exercise normalizer initialization before preparing the current worked sentence.
    ExternalTestData.requireFiles(root, "fixtures/captured/00.json")
    val fixture = JSONObject(File(root, "fixtures/captured/00.json").readText())
    val text =
      "Maya Chen from Orvane Robotics demonstrated the Veltrix 9 in Lisbon on March 12, 2025."
    assertEquals(text, fixture.getString("text"))
    val tokenizer = GlinerTokenizer(File(root, "host_assets/tokenizer.json"))
    val field = GlinerTokenizer::class.java.getDeclaredField("REPEATED_WHITESPACE")
    field.isAccessible = true
    val pattern = field.get(null) as Pattern
    assertEquals("Android does not support UNICODE_CHARACTER_CLASS", 0, pattern.flags())
    val whiteSpace =
      intArrayOf(
        0x09,
        0x0a,
        0x0b,
        0x0c,
        0x0d,
        0x20,
        0x85,
        0xa0,
        0x1680,
        0x2000,
        0x2001,
        0x2002,
        0x2003,
        0x2004,
        0x2005,
        0x2006,
        0x2007,
        0x2008,
        0x2009,
        0x200a,
        0x2028,
        0x2029,
        0x202f,
        0x205f,
        0x3000,
      )
    for (point in whiteSpace) {
      val value = String(Character.toChars(point))
      assertEquals(
        "Repeated U+${point.toString(16)}",
        "a b",
        pattern.matcher("a$value${value}b").replaceAll(" "),
      )
      val expected =
        if (point in listOf(0x09, 0x0a, 0x0d)) {
          "a b"
        } else {
          "a${value}b"
        }
      assertEquals(
        "Single U+${point.toString(16)}",
        expected,
        pattern.matcher("a${value}b").replaceAll(" "),
      )
    }
    // Zero-width space is not Unicode White_Space and must not be collapsed.
    assertEquals("a\u200b\u200bb", pattern.matcher("a\u200b\u200bb").replaceAll(" "))
    val actual = GlinerInputs(tokenizer).prepare(text)
    assertArrayEquals(
      fixture.tensorInts("input_ids"),
      actual.inputIds.copyOf(actual.encodedLength),
    )
    assertArrayEquals(fixture.tensorInts("text_word_indices"), actual.textWordPositions)
    assertArrayEquals(fixture.tensorInts("query_marker_indices"), actual.queryMarkerPositions)
    ExternalTestData.reportFile("splitter_regression.json")
      .writeText(
        JSONObject()
          .put("status", "PASS")
          .put("first_attempted_text", text)
          .put("failure_phase", "Tokenizer class initialization, before any text was processed")
          .put("pattern_flags", pattern.flags())
          .put("unicode_whitespace_single_and_repeated_cases", whiteSpace.size * 2)
          .put("non_whitespace_negative_cases", 1)
          .put("captured_input_ids_and_positions_identical", true)
          .put("device_execution", "NOT RUN")
          .toString(2) + "\n"
      )
  }

  @Test
  fun offsetsRemainCodePointsWhenLowercaseExpandsAndUtf16UsesSurrogates() {
    val text = "A😀İ 東京"
    val words = GlinerInputs.splitWords(text)
    assertEquals(
      listOf(
        GlinerInputs.Word("a", 0, 1),
        GlinerInputs.Word("😀", 1, 2),
        GlinerInputs.Word("i\u0307", 2, 3),
        GlinerInputs.Word("東京", 4, 6),
      ),
      words,
    )
    assertEquals("İ", GlinerInputs.codePointSubstring(text, 2, 3))
    assertEquals("東京", GlinerInputs.codePointSubstring(text, 4, 6))
  }

  private fun capturedFiles(directory: File): List<File> =
    requireNotNull(directory.listFiles()) { "Missing fixture directory: $directory" }
      .filter { it.extension == "json" && it.nameWithoutExtension.toIntOrNull() != null }
      .sortedBy { it.name }

  private fun JSONObject.tensorInts(name: String): IntArray {
    val array = getJSONObject(name).getJSONArray("values").getJSONArray(0)
    return IntArray(array.length()) { array.getInt(it) }
  }

  private fun JSONObject.tensorBooleans(name: String): BooleanArray {
    val array = getJSONObject(name).getJSONArray("values").getJSONArray(0)
    return BooleanArray(array.length()) { array.getBoolean(it) }
  }
}
