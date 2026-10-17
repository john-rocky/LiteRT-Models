package com.gliner25

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test

/** GLiNER2.5 Multi host parity; data layout in scripts/TEST_DATA.md. */
class GlinerMultiParityTest {
  private lateinit var root: File
  private lateinit var inputs: GlinerInputs

  @Before
  fun locateExternalData() {
    root = ExternalTestData.resolveMulti()
    inputs =
      GlinerInputs(
        GlinerTokenizer(File(root, "host_assets/tokenizer.json")),
        GlinerModel.MULTI.appendsTerminalPeriod,
      )
  }

  /** Both worked sentences of the published example.json, with the published splitters. */
  @Test
  fun workedSentencesMatchThePublishedExample() {
    val exampleFile = File(root, "host_assets/example.json")
    val examples = JSONObject(exampleFile.readText()).getJSONArray("examples")
    assertEquals(2, examples.length())
    val gateAsset = File(moduleRoot(), "app/src/debug/assets/gate_multi_fixtures.json")
    val gate = JSONObject(gateAsset.readText())
    assertEquals(sha256(exampleFile), gate.getString("source_sha256"))
    val records = JSONArray()
    for (i in 0 until examples.length()) {
      val example = examples.getJSONObject(i)
      val id = example.getString("id")
      val text = example.getString("text")
      val splitter = GlinerInputs.Splitter.fromId(example.getString("splitter"))
      val actual = inputs.prepare(text, splitter = splitter)
      val n = actual.window.sequenceLength
      val t = actual.window.textCapacity
      val encoded = example.getInt("encoded_tokens")
      val slots = example.getInt("text_slots")
      assertEquals("$id window", example.getInt("seq"), n)
      assertEquals("$id encoded length", encoded, actual.encodedLength)
      assertArrayEquals("$id input_ids", example.ints("input_ids").copyOf(n), actual.inputIds)
      assertArrayEquals("$id attention_mask", mask(n, encoded), actual.attentionMask, 0f)
      val textPositions = example.ints("text_word_first_positions")
      val queryPositions = example.ints("query_marker_positions")
      assertArrayEquals("$id text positions", textPositions, actual.textWordPositions)
      assertArrayEquals("$id query positions", queryPositions, actual.queryMarkerPositions)
      val textRouting = FloatArray(t * n)
      textPositions.forEachIndexed { row, position -> textRouting[row * n + position] = 1f }
      assertArrayEquals("$id text_routing", textRouting, actual.textRouting, 0f)
      val queryRouting = FloatArray(5 * n)
      queryPositions.forEachIndexed { row, position -> queryRouting[row * n + position] = 1f }
      assertArrayEquals("$id query_routing", queryRouting, actual.queryRouting, 0f)
      assertArrayEquals("$id text_mask", mask(t, slots), actual.textMask, 0f)
      val tokens = example.getJSONArray("text_tokens")
      val offsets = example.getJSONArray("token_to_char")
      assertEquals("$id text slots", slots, actual.words.size)
      actual.words.forEachIndexed { index, word ->
        assertEquals("$id token $index", tokens.getString(index), word.text)
        assertEquals("$id start $index", offsets.getJSONArray(index).getInt(0), word.start)
        assertEquals("$id end $index", offsets.getJSONArray(index).getInt(1), word.end)
      }
      // The debug gate asset carries the same inputs and official spans.
      val fixture = gate.getJSONArray("fixtures").getJSONObject(i)
      assertEquals(id, fixture.getString("id"))
      assertEquals(text, fixture.getString("text"))
      assertTrue(GlinerGateFixtures.compareInputs(fixture, actual).getBoolean("identical"))
      assertEquals(
        spans(example.getJSONArray("oracle_spans")),
        spans(fixture.getJSONArray("spans")),
      )
      records.put(
        JSONObject()
          .put("id", id)
          .put("splitter", splitter.id)
          .put("window", n)
          .put("encoded_tokens", actual.encodedLength)
          .put("text_slots", actual.words.size)
          .put("identical", true)
      )
    }
    ExternalTestData.reportFile("multi_example_inputs.json")
      .writeText(
        JSONObject()
          .put("status", "PASS")
          .put("source", "host_assets/example.json")
          .put("source_sha256", sha256(exampleFile))
          .put(
            "compared",
            JSONArray(
              listOf(
                "input_ids",
                "attention_mask",
                "text_routing",
                "query_routing",
                "text_mask",
                "text_word_first_positions",
                "query_marker_positions",
                "text_tokens",
                "token_to_char",
              )
            ),
          )
          .put("examples", records)
          .toString(2) + "\n"
      )
    println("MULTI_EXAMPLE_INPUTS PASS 2/2 (en whitespace, ja char)")
  }

  /** Lookup rows equal an independent little-endian float16 read of the same file offsets. */
  @Test
  fun embeddingTableUpcastsTheSelectedFloat16Rows() {
    val tableFile = File(root, "host_assets/word_embeddings_fp16.bin")
    assumeTrue("Missing Multi data: host_assets/word_embeddings_fp16.bin", tableFile.isFile)
    assertEquals(GlinerModel.MULTI.tableBytes, tableFile.length())
    val model = GlinerModel.MULTI
    val prepared =
      inputs.prepare(
        "星瀬澄香は京都で霧灯研究社の新製品『星糸端末』を紹介した。",
        splitter = GlinerInputs.Splitter.CHAR,
      )
    val table =
      GlinerInputs.EmbeddingTable(tableFile, model.vocabularySize, model.hiddenSize, true)
    table.use {
      val rows = it.lookup(prepared.inputIds)
      assertEquals(prepared.inputIds.size * model.hiddenSize, rows.size)
      assertTrue(rows.all { it.isFinite() })
      RandomAccessFile(tableFile, "r").use { file ->
        val bytes = ByteArray(model.hiddenSize * 2)
        for ((row, id) in prepared.inputIds.withIndex()) {
          file.seek(id.toLong() * model.hiddenSize * 2)
          file.readFully(bytes)
          val halves = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
          for (column in 0 until model.hiddenSize) {
            val bits = halves.get(column).toInt() and 0xffff
            val exponent = (bits shr 10) and 0x1f
            val mantissa = bits and 0x3ff
            val magnitude =
              if (exponent == 0) {
                Math.scalb(mantissa.toDouble(), -24)
              } else {
                Math.scalb(1024.0 + mantissa, exponent - 25)
              }
            val expected =
              if (bits and 0x8000 != 0) {
                -magnitude
              } else {
                magnitude
              }
            assertEquals(
              "row $row id $id column $column",
              expected.toFloat().toRawBits(),
              rows[row * model.hiddenSize + column].toRawBits(),
            )
          }
        }
      }
    }
  }

  /**
   * Captured Python HostRuntime inputs for the 80 bilingual fixtures (35 English whitespace, 45
   * Japanese char) at every fitting window; optional, not distributed.
   */
  @Test
  fun bilingualCapturedInputsMatchAtEveryFittingWindow() {
    val captured = File(root, "fixtures/f1/captured")
    assumeTrue("Missing optional Multi captures: $captured", captured.isDirectory)
    val files =
      requireNotNull(captured.listFiles())
        .filter { it.extension == "json" && it.nameWithoutExtension.toIntOrNull() != null }
        .sortedBy { it.name }
    assertEquals(80, files.size)
    val bySplitter = linkedMapOf<String, Int>()
    val byWindow = linkedMapOf(128 to 0, 256 to 0, 512 to 0)
    for (file in files) {
      val fixture = JSONObject(file.readText())
      val text = fixture.getString("text")
      val splitter = GlinerInputs.Splitter.fromId(fixture.getString("splitter"))
      bySplitter[splitter.id] = (bySplitter[splitter.id] ?: 0) + 1
      val ids = fixture.tensorInts("input_ids")
      val textPositions = fixture.tensorInts("text_word_indices")
      val queryPositions = fixture.tensorInts("query_marker_indices")
      val starts = fixture.getJSONArray("start_mappings").getJSONArray(0)
      val ends = fixture.getJSONArray("end_mappings").getJSONArray(0)
      val tokens = fixture.getJSONArray("text_tokens").getJSONArray(0)
      val fitting =
        GlinerInputs.WINDOWS.filter {
          ids.size <= it.sequenceLength && textPositions.size <= it.textCapacity
        }
      assertTrue("${file.name} fits a window", fitting.isNotEmpty())
      assertEquals(fitting.first(), inputs.prepare(text, splitter = splitter).window)
      for (window in fitting) {
        val context = "${file.name} / s${window.sequenceLength}"
        val actual = inputs.prepare(text, window, splitter)
        assertEquals("$context encoded length", ids.size, actual.encodedLength)
        assertArrayEquals("$context input_ids", ids.copyOf(window.sequenceLength), actual.inputIds)
        assertArrayEquals("$context text positions", textPositions, actual.textWordPositions)
        assertArrayEquals("$context query positions", queryPositions, actual.queryMarkerPositions)
        assertEquals("$context slots", starts.length(), actual.words.size)
        actual.words.forEachIndexed { index, word ->
          assertEquals("$context token $index", tokens.getString(index), word.text)
          assertEquals("$context start $index", starts.getInt(index), word.start)
          assertEquals("$context end $index", ends.getInt(index), word.end)
        }
        byWindow[window.sequenceLength] = byWindow.getValue(window.sequenceLength) + 1
      }
    }
    assertEquals(mapOf("whitespace" to 35, "char" to 45), bySplitter)
    assertEquals(mapOf(128 to 70, 256 to 75, 512 to 80), byWindow)
    println("MULTI_CAPTURED_INPUTS PASS files=80 en=35 ja=45 pairs=225")
  }

  /**
   * Kotlin decoding of packed outputs captured from a device run (one window per directory) versus
   * the Python host decode of the same bytes and the official spans; optional, not distributed.
   */
  @Test
  fun packedOutputsDecodeLikePythonAndTheOfficialSpans() {
    val decoder =
      GlinerDecoder(File(root, "host_assets"), hiddenSize = GlinerModel.MULTI.hiddenSize)
    val windows =
      GlinerInputs.WINDOWS.filter {
        File(root, "fixtures/packed/s${it.sequenceLength}.json").isFile
      }
    assumeTrue("Missing optional Multi packed outputs under fixtures/packed/", windows.isNotEmpty())
    val failures = mutableListOf<String>()
    val summary = JSONObject()
    var pairs = 0
    for (window in windows) {
      val reference =
        JSONObject(File(root, "fixtures/packed/s${window.sequenceLength}.json").readText())
          .getJSONArray("fixtures")
      var pythonIdentical = 0
      var oracleIdentical = 0
      var maxPython = 0.0
      var maxOracle = 0.0
      for (i in 0 until reference.length()) {
        val entry = reference.getJSONObject(i)
        val id = entry.getString("id")
        val key = "s${window.sequenceLength}/$id"
        val captured = JSONObject(File(root, "fixtures/f1/captured/$id.json").readText())
        val splitter = GlinerInputs.Splitter.fromId(entry.getString("splitter"))
        val prepared = inputs.prepare(captured.getString("text"), window, splitter)
        val packed =
          GlinerDecoder.readPacked(File(root, "fixtures/packed/s${window.sequenceLength}/$id.bin"))
        assertEquals(
          "$key packed size",
          GlinerModel.MULTI.packedFloatCount(window.textCapacity),
          packed.size,
        )
        val actual =
          decoder.decode(packed, prepared).associate {
            Triple(it.label, it.start, it.end) to it.confidence.toDouble()
          }
        val python = spans(entry.getJSONArray("decoded_spans"))
        val oracle = spans(entry.getJSONArray("oracle_spans"))
        if (actual.keys == python.keys) {
          pythonIdentical++
        } else {
          failures += "$key Python span set"
        }
        if (actual.keys == oracle.keys) {
          oracleIdentical++
        } else {
          failures += "$key oracle span set"
        }
        actual.forEach { (span, confidence) ->
          python[span]?.let { maxPython = maxOf(maxPython, abs(confidence - it)) }
          oracle[span]?.let { maxOracle = maxOf(maxOracle, abs(confidence - it)) }
        }
        pairs++
      }
      if (maxPython > 1e-5) {
        failures += "s${window.sequenceLength} Python confidence $maxPython"
      }
      if (maxOracle > 5e-3) {
        failures += "s${window.sequenceLength} oracle confidence $maxOracle"
      }
      summary.put(
        "s${window.sequenceLength}",
        JSONObject()
          .put("inputs", reference.length())
          .put("python_span_sets_identical", pythonIdentical)
          .put("oracle_span_sets_identical", oracleIdentical)
          .put("max_confidence_difference_python", maxPython)
          .put("max_confidence_difference_oracle", maxOracle),
      )
      println(
        "MULTI_DECODER s${window.sequenceLength}: ${reference.length()} inputs, Python " +
          "$pythonIdentical, oracle $oracleIdentical, max diff Python=$maxPython oracle=$maxOracle"
      )
    }
    val status =
      if (failures.isEmpty()) {
        "PASS"
      } else {
        "FAIL"
      }
    ExternalTestData.reportFile("multi_decoder_parity.json")
      .writeText(
        summary
          .put("status", status)
          .put("pairs", pairs)
          .put("failures", JSONArray(failures))
          .toString(2) + "\n"
      )
    assertTrue(failures.joinToString("\n"), failures.isEmpty())
  }

  private fun spans(values: JSONArray): Map<Triple<String, Int, Int>, Double> =
    (0 until values.length()).associate {
      val span = values.getJSONObject(it)
      Triple(span.getString("label"), span.getInt("start"), span.getInt("end")) to
        span.getDouble("confidence")
    }

  /** Float 0/1 mask with ones in the first [valid] of [size] positions. */
  private fun mask(size: Int, valid: Int) =
    FloatArray(size) {
      if (it < valid) {
        1f
      } else {
        0f
      }
    }

  private fun moduleRoot() = File(requireNotNull(System.getProperty("gliner.moduleRoot")))

  private fun sha256(file: File): String =
    MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") {
      "%02x".format(it)
    }

  private fun JSONObject.ints(name: String): IntArray {
    val array = getJSONArray(name)
    return IntArray(array.length()) { array.getInt(it) }
  }

  private fun JSONObject.tensorInts(name: String): IntArray {
    val array = getJSONObject(name).getJSONArray("values").getJSONArray(0)
    return IntArray(array.length()) { array.getInt(it) }
  }
}
