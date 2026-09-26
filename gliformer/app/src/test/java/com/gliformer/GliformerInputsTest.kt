package com.gliformer

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.regex.Pattern
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class GliformerInputsTest {
  private val fixtures by lazy { ExternalTestData.resolve() }
  private val tokenizer by lazy {
    GliformerTokenizer(File(fixtures, "tokenizer.json"))
  }

  @Test
  fun all80CapturedInputsMatchElementwiseAtSelectedWindows() {
    val corpus = JSONObject(File(fixtures, "corpus.json").readText()).getJSONArray("rows")
    val rawRows =
      JSONObject(File(fixtures, "graph_inputs/manifest.json").readText()).getJSONArray("rows")
    val rawById =
      (0 until rawRows.length()).associate {
        rawRows.getJSONObject(it).getString("id") to rawRows.getJSONObject(it)
      }
    assertEquals(80, corpus.length())
    assertEquals(80, rawById.size)
    val inputs = GliformerInputs(tokenizer)
    val reportRows = JSONArray()
    val uniqueTexts = mutableSetOf<String>()
    val windows = linkedMapOf(128 to 0, 256 to 0, 512 to 0)
    for (index in 0 until corpus.length()) {
      val row = corpus.getJSONObject(index)
      val id = row.getString("id")
      val captured = row.getJSONObject("captured")
      val expectedIds = captured.getJSONArray("input_ids").getJSONArray(0).ints()
      val attention = captured.getJSONArray("attention_mask").getJSONArray(0).ints()
      val first = captured.getJSONArray("word_first_subtokens").ints()
      val parents = captured.getJSONArray("schema_positions").ints()
      val entities = captured.getJSONArray("entity_positions").ints()
      val labels = row.getJSONArray("labels").strings()
      val actual = inputs.prepare(row.getString("text"), labels)
      val n = actual.window.sequenceLength
      val t = actual.window.textCapacity
      assertEquals("$id selected window", row.getInt("window"), n)
      assertEquals("$id encoded length", expectedIds.size, actual.encodedLength)
      assertEquals("$id text word count", captured.getInt("text_word_length"), actual.words.size)
      assertArrayEquals("$id input_ids", expectedIds.copyOf(n), actual.inputIds)
      assertArrayEquals(
        "$id attention_mask",
        FloatArray(n) { attention.getOrElse(it) { 0 }.toFloat() },
        actual.attentionMask,
        0f,
      )
      assertArrayEquals("$id first-subtoken positions", first, actual.firstSubtokenPositions)
      assertArrayEquals("$id schema positions", parents, actual.parentPositions)
      assertArrayEquals("$id entity positions", entities, actual.entityPositions)
      val textRouting = FloatArray(t * n)
      first.forEachIndexed { word, position -> textRouting[word * n + position] = 1f }
      val parentRouting = FloatArray(n)
      parents.forEach { parentRouting[it] = 1f }
      val labelRouting = FloatArray(5 * n)
      entities.forEachIndexed { label, position -> labelRouting[label * n + position] = 1f }
      val textMask = FloatArray(t) { if (it < first.size) 1f else 0f }
      assertArrayEquals("$id text_routing from capture", textRouting, actual.textRouting, 0f)
      assertArrayEquals("$id parent_routing from capture", parentRouting, actual.parentRouting, 0f)
      assertArrayEquals("$id label_routing from capture", labelRouting, actual.labelRouting, 0f)
      assertArrayEquals("$id text_mask from capture", textMask, actual.textMask, 0f)
      val raw = requireNotNull(rawById[id])
      assertEquals("$id raw selected window", n, raw.getInt("window"))
      val tensors = raw.getJSONObject("tensors")
      mapOf(
          "attention_mask" to actual.attentionMask,
          "text_routing" to actual.textRouting,
          "parent_routing" to actual.parentRouting,
          "label_routing" to actual.labelRouting,
          "text_mask" to actual.textMask,
        )
        .forEach { (name, actualValues) ->
          val entry = tensors.getJSONObject(name)
          val bytes = File(fixtures, entry.getString("file")).readBytes()
          assertEquals("$id $name raw bytes", entry.getInt("bytes"), bytes.size)
          val expected = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
          val values = FloatArray(expected.remaining())
          expected.get(values)
          assertArrayEquals("$id $name original graph tensor", values, actualValues, 0f)
        }
      val tokens = captured.getJSONArray("tokens").strings()
      val starts = captured.getJSONArray("start_map").ints()
      val ends = captured.getJSONArray("end_map").ints()
      assertEquals("$id words", tokens, actual.words.map { it.text })
      actual.words.forEachIndexed { wordIndex, word ->
        assertEquals("$id word $wordIndex start", starts[wordIndex], word.start)
        assertEquals("$id word $wordIndex end", ends[wordIndex], word.end)
        assertEquals(
          "$id word $wordIndex original text",
          word.text,
          actual.substring(word.start, word.end),
        )
      }
      val expectedWordMask = captured.getJSONArray("words_mask").getJSONArray(0).ints()
      val wordMask = IntArray(actual.encodedLength)
      actual.firstSubtokenPositions.forEachIndexed { word, position ->
        wordMask[position] = word + 1
      }
      assertArrayEquals("$id words_mask", expectedWordMask, wordMask)
      uniqueTexts.add(actual.text)
      windows[n] = windows.getValue(n) + 1
      reportRows.put(
        JSONObject()
          .put("id", id)
          .put("window", n)
          .put("encoded_tokens", actual.encodedLength)
          .put("text_words", actual.words.size)
          .put("input_ids_masks_routing_offsets_identical", true)
      )
    }
    assertEquals("80 fixture files contain 70 unique texts", 70, uniqueTexts.size)
    report(
      "tokenizer_parity.json",
      JSONObject()
        .put("status", "PASS")
        .put("passed", 80)
        .put("total", 80)
        .put("unique_texts", uniqueTexts.size)
        .put("windows", JSONObject(windows.mapKeys { "s${it.key}" }))
        .put("max_absolute_error", 0)
        .put("raw_graph_tensor_comparisons", 400)
        .put("device_execution", "NOT RUN")
        .put("rows", reportRows),
    )
    println(
      "TOKENIZER_PARITY PASS 80/80 fixtures, 70 unique texts, 400 original graph tensors, max_error=0"
    )
  }

  @Test
  fun pythonUnicodeSplitterKeepsCaseAndCodePointOffsets() {
    // Supplementary emoji and a supplementary LETTER exercise both branches of the regex.
    val text = "A😀İ 東京 𐐀x cafe\u0301 foo_bar-12\u001c²"
    val expected =
      listOf(
        GliformerInputs.Word("A", 0, 1),
        GliformerInputs.Word("😀", 1, 2),
        GliformerInputs.Word("İ", 2, 3),
        GliformerInputs.Word("東京", 4, 6),
        GliformerInputs.Word("𐐀x", 7, 9),
        GliformerInputs.Word("cafe", 10, 14),
        GliformerInputs.Word("\u0301", 14, 15),
        GliformerInputs.Word("foo_bar-12", 16, 26),
        GliformerInputs.Word("²", 27, 28),
      )
    val actual = GliformerInputs.splitWords(text)
    assertEquals(expected, actual)
    actual.forEach {
      assertEquals(it.text, GliformerInputs.codePointSubstring(text, it.start, it.end))
    }
    val field = GliformerInputs::class.java.getDeclaredField("WORD_PATTERN")
    field.isAccessible = true
    assertEquals(0, (field.get(null) as Pattern).flags())
    report(
      "splitter_unicode.json",
      JSONObject()
        .put("status", "PASS")
        .put("text", text)
        .put("pattern_flags", 0)
        .put("offset_unit", "Unicode code points")
        .put("device_execution", "NOT RUN")
        .put(
          "words",
          JSONArray(
            actual.map {
              JSONObject().put("text", it.text).put("start", it.start).put("end", it.end)
            }
          ),
        ),
    )
  }

  @Test
  fun supplementaryCharactersMatchTheCapturedPythonPreparation() {
    val fixture = JSONObject(File(fixtures, "unicode.json").readText())
    val captured = fixture.getJSONObject("captured")
    val actual =
      GliformerInputs(tokenizer)
        .prepare(
          fixture.getString("text"),
          fixture.getJSONArray("labels").strings(),
        )
    val n = actual.window.sequenceLength
    val t = actual.window.textCapacity
    val ids = captured.getJSONArray("input_ids").getJSONArray(0).ints()
    val attention = captured.getJSONArray("attention_mask").getJSONArray(0).ints()
    val positions = captured.getJSONArray("word_first_subtokens").ints()
    val parents = captured.getJSONArray("schema_positions").ints()
    val entities = captured.getJSONArray("entity_positions").ints()
    assertEquals(fixture.getInt("window"), n)
    assertEquals(ids.size, actual.encodedLength)
    assertArrayEquals(ids.copyOf(n), actual.inputIds)
    assertArrayEquals(positions, actual.firstSubtokenPositions)
    assertArrayEquals(parents, actual.parentPositions)
    assertArrayEquals(entities, actual.entityPositions)
    assertArrayEquals(
      FloatArray(n) { attention.getOrElse(it) { 0 }.toFloat() },
      actual.attentionMask,
      0f,
    )
    val textRoute = FloatArray(t * n)
    positions.forEachIndexed { word, position -> textRoute[word * n + position] = 1f }
    val parentRoute = FloatArray(n)
    parents.forEach { parentRoute[it] = 1f }
    val labelRoute = FloatArray(5 * n)
    entities.forEachIndexed { label, position -> labelRoute[label * n + position] = 1f }
    assertArrayEquals(textRoute, actual.textRouting, 0f)
    assertArrayEquals(parentRoute, actual.parentRouting, 0f)
    assertArrayEquals(labelRoute, actual.labelRouting, 0f)
    assertArrayEquals(FloatArray(t) { if (it < positions.size) 1f else 0f }, actual.textMask, 0f)
    assertEquals(captured.getJSONArray("tokens").strings(), actual.words.map { it.text })
    val starts = captured.getJSONArray("start_map").ints()
    val ends = captured.getJSONArray("end_map").ints()
    actual.words.forEachIndexed { index, word ->
      assertEquals(starts[index], word.start)
      assertEquals(ends[index], word.end)
      assertEquals(word.text, actual.substring(word.start, word.end))
    }
    assertTrue(
      "Fixture must exercise supplementary code points",
      actual.text.length > actual.text.codePointCount(0, actual.text.length),
    )
    report(
      "unicode_prepare_parity.json",
      JSONObject()
        .put("status", "PASS")
        .put("text", actual.text)
        .put("encoded_tokens", actual.encodedLength)
        .put("text_words", actual.words.size)
        .put("window", n)
        .put("input_ids_masks_routing_offsets_identical", true)
        .put("max_absolute_error", 0)
        .put("device_execution", "NOT RUN"),
    )
  }

  @Test
  fun bothCapacityLimitsChooseTheWindowWithoutTruncation() {
    val inputs = GliformerInputs(tokenizer)
    fun text(words: Int) = List(words) { "a" }.joinToString(" ")
    assertEquals(128, inputs.prepare(text(48)).window.sequenceLength)
    assertEquals(256, inputs.prepare(text(49)).window.sequenceLength) // N fits128 but T does not.
    assertEquals(256, inputs.prepare(text(238)).window.sequenceLength)
    assertEquals(512, inputs.prepare(text(239)).window.sequenceLength) // T fits256 but N does not.
    assertEquals(512, inputs.prepare(text(494)).window.sequenceLength)
    expectInvalid { inputs.prepare(text(495)) }
    expectInvalid { inputs.prepare(text(49), window = GliformerInputs.WINDOWS.first()) }
    expectInvalid { inputs.prepare(" \t\n\u001c") }
    expectInvalid { inputs.prepare("a", List(5) { "person" }) }
    expectInvalid {
      inputs.prepare("a", listOf("person", "organization", "location", "product", "\u0085\u001c"))
    }
  }

  @Test
  fun mappedTablesLookupPaddingAndLastRowWithExactHalfUpcast() {
    val storageTypes = GliformerInputs.EmbeddingTable.Storage.values()
    val halfBits = intArrayOf(0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xc000, 0x7bff)
    val expected =
      floatArrayOf(
        0f,
        -0f,
        5.960464477539063e-8f,
        0.00006097555160522461f,
        0.00006103515625f,
        1f,
        -2f,
        65504f,
      )
    val directory = File(requireNotNull(System.getProperty("gliformer.buildDir")), "table-tests")
    directory.mkdirs()
    for (storage in storageTypes) {
      val file = File.createTempFile("table-${storage.name.lowercase()}-", ".bin", directory)
      try {
        RandomAccessFile(file, "rw").use { out ->
          out.setLength(128008L * 1024 * storage.bytesPerValue) // Sparse; no checkpoint copied.
          for (id in listOf(0, 128007)) {
            out.seek(id.toLong() * 1024 * storage.bytesPerValue)
            val bytes =
              ByteBuffer.allocate(expected.size * storage.bytesPerValue)
                .order(ByteOrder.LITTLE_ENDIAN)
            expected.forEachIndexed { index, value ->
              if (storage == GliformerInputs.EmbeddingTable.Storage.FP16)
                bytes.putShort(halfBits[index].toShort())
              else bytes.putFloat(value)
            }
            out.write(bytes.array())
          }
        }
        GliformerInputs.EmbeddingTable(file, storage).use { table ->
          val values = table.lookup(intArrayOf(0, 128007, 0))
          assertEquals(3072, values.size)
          for (row in 0..2) expected.forEachIndexed { column, value ->
            assertEquals(
              "${storage.name} row $row column $column bits",
              value.toRawBits(),
              values[row * 1024 + column].toRawBits(),
            )
          }
          expectInvalid { table.lookup(intArrayOf(128008)) }
        }
      } finally {
        assertTrue("Delete sparse test table", file.delete())
      }
    }
    assertTrue(GliformerInputs.EmbeddingTable.halfToFloat(0x7c00).isInfinite())
    assertTrue(GliformerInputs.EmbeddingTable.halfToFloat(0x7e00).isNaN())
    assertFalse(GliformerInputs.EmbeddingTable.halfToFloat(0x7bff).isInfinite())
  }

  private fun expectInvalid(block: () -> Unit) {
    try {
      block()
      fail("Expected IllegalArgumentException")
    } catch (_: IllegalArgumentException) {
      /* Required rejection. */
    }
  }

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

  private fun JSONArray.strings() = List(length()) { getString(it) }

  private fun report(name: String, value: JSONObject) {
    val file = ExternalTestData.reportFile(name)
    file.writeText(value.toString(2) + "\n")
  }
}
