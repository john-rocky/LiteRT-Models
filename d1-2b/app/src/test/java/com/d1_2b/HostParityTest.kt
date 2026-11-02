package com.d1_2b

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * The Kotlin host against the model's Python host on the same graph outputs: each fixture holds a
 * request, the Python host's sequences and answers, and every hidden state and tower output it
 * read. The Kotlin host gets those outputs from stub backends and must build the same sequences
 * and give the same answers. Run with `-Pd1.model=<model dir> -Pd1.fixtures=<fixtures dir>`
 * (README.md); without them the test is skipped.
 */
class HostParityTest {
  private val modelDir = System.getProperty("d1.model").orEmpty()
  private val fixturesDir = System.getProperty("d1.fixtures").orEmpty()

  private class Diffs {
    var options = 0.0
    var optionSum = 0.0
    var optionCount = 0
    var found = 0.0
    var edges = 0.0
    var confidence = 0.0
    var presence = 0.0
    val failures = ArrayList<String>()

    fun option(a: Double, b: Double) {
      val d = abs(a - b)
      options = max(options, d)
      optionSum += d
      optionCount++
    }
  }

  @Test
  fun hostMatchesTheModelsOwnHost() {
    assumeTrue("set -Pd1.model and -Pd1.fixtures", modelDir.isNotEmpty() && fixturesDir.isNotEmpty())
    val root = File(fixturesDir)
    @Suppress("UNCHECKED_CAST") val index = D1Json.parse(File(root, "index.json").readBytes()) as Map<String, Any?>
    val report = LinkedHashMap<String, Any?>()
    val all = Diffs()
    var sequencesChecked = 0
    var pictures = 0
    for (entry in index["requests"] as List<*>) {
      val id = (entry as Map<*, *>)["id"] as String
      val dir = File(root, id)
      @Suppress("UNCHECKED_CAST") val expected = D1Json.parse(File(dir, "expected.json").readBytes()) as Map<String, Any?>
      val request = expected["request"] as Map<*, *>
      val hidden = (expected["hidden_files"] as List<*>).map { readNpy(File(dir, it as String)) }
      val towerOut = (expected["tower_file"] as String?)?.let { readNpy(File(dir, it)) }
      var calls = 0
      val decide =
        object : DecideBackend {
          override val maxLength = 6144

          override fun decide(embeds: FloatArray, tokens: Int, stateLength: Int, question: IntArray): FloatArray {
            val out = hidden[calls++]
            assertEquals("$id: hidden rows of call $calls", tokens * HIDDEN, out.size)
            return out
          }
        }
      val tower =
        object : TowerBackend {
          override fun tower(picture: Picture): FloatArray = requireNotNull(towerOut) { "$id: no tower output recorded" }
        }
      val host = Host.load(File(modelDir), decide, tower)
      val picture =
        (expected["picture_file"] as String?)?.let { name ->
          val picture = readPicture(File(dir, name))
          val expectedValues = readNpy(File(dir, expected["pixel_values_file"] as String))
          val values = picture.pixelValues()
          assertEquals("$id: pixel values", expectedValues.size, values.size)
          var worst = 0f
          for (i in values.indices) worst = max(worst, abs(values[i] - expectedValues[i]))
          assertEquals("$id: max |pixel value difference|", 0f, worst)
          pictures++
          picture
        }
      val state = request["state"].let { if (it is Map<*, *> || it is List<*>) it else it }
      @Suppress("UNCHECKED_CAST") val questions = request["questions"] as Map<String, Any?>
      val result = host.systemOne(state, questions, picture)
      // (i) the sequences
      val sequences = expected["sequences"] as List<*>
      assertEquals("$id: sequence count", sequences.size, result.sequences.size)
      for ((n, raw) in sequences.withIndex()) {
        val want = raw as Map<*, *>
        val got = result.sequences[n]
        assertEquals("$id seq $n ids", ints(want["ids"]), got.ids.toList())
        assertEquals("$id seq $n state_len", (want["state_len"] as JsonNumber).toInt(), got.stateLength)
        assertEquals("$id seq $n tags", ints(want["question"]), got.question.toList())
        assertEquals("$id seq $n ends", (want["ends"] as List<*>).map { ints(it) }, got.ends.map { it.toList() })
        sequencesChecked++
      }
      // (iii) the answers
      val response = expected["response"] as Map<*, *>
      assertEquals("$id usage", ((response["usage"] as Map<*, *>)["input_tokens"] as JsonNumber).toInt(), result.inputTokens)
      val own = Diffs()
      for ((name, raw) in response["answers"] as Map<*, *>) {
        compare("$id.$name", raw as Map<*, *>, result.answers.getValue(name as String), own)
      }
      report[id] =
        linkedMapOf(
          "sequences" to sequences.size,
          "tokens" to result.inputTokens,
          "max_dp" to own.options,
          "max_dfound" to own.found,
          "max_dedge" to own.edges,
          "max_dconfidence" to own.confidence,
          "max_dpresence" to own.presence,
          "failures" to own.failures,
        )
      all.options = max(all.options, own.options)
      all.optionSum += own.optionSum
      all.optionCount += own.optionCount
      all.found = max(all.found, own.found)
      all.edges = max(all.edges, own.edges)
      all.confidence = max(all.confidence, own.confidence)
      all.presence = max(all.presence, own.presence)
      all.failures.addAll(own.failures)
    }
    val summary =
      linkedMapOf(
        "pass" to all.failures.isEmpty(),
        "requests" to report.size,
        "sequences_identical" to sequencesChecked,
        "pictures_identical" to pictures,
        "max_dp" to all.options,
        "mean_dp" to if (all.optionCount > 0) all.optionSum / all.optionCount else 0.0,
        "options" to all.optionCount,
        "max_dfound" to all.found,
        "max_dedge" to all.edges,
        "max_dconfidence" to all.confidence,
        "max_dpresence" to all.presence,
        "failures" to all.failures,
        "per_request" to report,
      )
    File(root, "parity_report.json").writeText(D1Json.writeIndented(summary, 1) + "\n")
    assertTrue("failures: ${all.failures}", all.failures.isEmpty())
    assertTrue("mean |dp| ${summary["mean_dp"]}", (summary["mean_dp"] as Double) <= 0.002)
  }

  @Suppress("UNCHECKED_CAST")
  private fun compare(where: String, want: Map<*, *>, got: Map<String, Any?>, d: Diffs) {
    fun fail(what: String) {
      d.failures.add("$where: $what")
    }
    val type = want["type"] as String
    if (type != got["type"]) return fail("type ${got["type"]}")
    when {
      type == "noul" -> {
        val a = num(want["noul"])
        val b = num(got["noul"])
        d.option(a, b)
        if (abs(a - b) > 0.02 || (a >= 0.5) != (b >= 0.5)) fail("noul $a vs $b")
      }
      type == "choice" && want.containsKey("choices") -> {
        if (want["choices"] != got["choices"]) fail("choices ${want["choices"]} vs ${got["choices"]}")
        probabilities(want, got, d, ::fail)
      }
      type == "choice" -> {
        if (want["choice"] != got["choice"]) fail("choice ${want["choice"]} vs ${got["choice"]}")
        probabilities(want, got, d, ::fail)
      }
      type == "score" -> {
        if (num(want["score"]) != num(got["score"])) fail("score ${want["score"]} vs ${got["score"]}")
        probabilities(want, got, d, ::fail)
      }
      type == "number" && want.containsKey("interval") -> {
        if (num(want["number"]) != num(got["number"])) fail("estimate ${want["number"]} vs ${got["number"]}")
        val wi = (want["interval"] as List<*>).map(::num)
        val gi = (got["interval"] as List<*>).map(::num)
        if (wi != gi) fail("interval $wi vs $gi")
        if (want["unit"] != got["unit"]) fail("unit")
      }
      type == "text" || type == "number" || type == "date" -> {
        val a = num(want["found"])
        val b = num(got["found"])
        d.found = max(d.found, abs(a - b))
        if (abs(a - b) > 0.02 || (a >= 0.5) != (b >= 0.5)) fail("found $a vs $b")
        val wantSpan = (want["span"] as List<*>?)?.map(::num)
        val gotSpan = (got["span"] as List<*>?)?.map(::num)
        if (wantSpan != gotSpan) fail("span $wantSpan vs $gotSpan")
        val wv = want[type]
        val gv = got[type]
        val same = if (wv is JsonNumber || gv is Number) wv != null && gv != null && num(wv) == num(gv) else wv == gv
        if (!same && !(wv == null && gv == null)) fail("value $wv vs $gv")
        if (type != "text" && want["text"] != got["text"]) fail("text ${want["text"]} vs ${got["text"]}")
        if (type == "number" && want["unit"] != got["unit"]) fail("unit ${want["unit"]} vs ${got["unit"]}")
        if (want["path"] != got["path"]) fail("path ${want["path"]} vs ${got["path"]}")
      }
      type == "box" -> box(want, got, d, ::fail)
      type == "boxes" -> {
        if (num(want["count"]) != num(got["count"])) fail("count ${want["count"]} vs ${got["count"]}")
        if (want["at_least"] != got["at_least"]) fail("at_least")
        val wp = (want["presence"] as List<*>).map(::num)
        val gp = (got["presence"] as List<*>).map(::num)
        for (i in wp.indices) d.presence = max(d.presence, abs(wp[i] - gp[i]))
        if (wp.indices.any { abs(wp[it] - gp[it]) > 0.02 }) fail("presence")
        val wb = want["boxes"] as List<*>
        val gb = got["boxes"] as List<*>
        if (wb.size == gb.size) wb.indices.forEach { box(wb[it] as Map<*, *>, gb[it] as Map<String, Any?>, d, ::fail) }
      }
      else -> fail("unknown type $type")
    }
  }

  private fun probabilities(want: Map<*, *>, got: Map<String, Any?>, d: Diffs, fail: (String) -> Unit) {
    val wp = want["probabilities"] as Map<*, *>
    val gp = got["probabilities"] as Map<*, *>
    if (wp.keys != gp.keys) return fail("option names")
    for (k in wp.keys) {
      val a = num(wp[k])
      val b = num(gp[k])
      d.option(a, b)
      if (abs(a - b) > 0.02) fail("p($k) $a vs $b")
    }
  }

  private fun box(want: Map<*, *>, got: Map<String, Any?>, d: Diffs, fail: (String) -> Unit) {
    val a = num(want["found"])
    val b = num(got["found"])
    d.presence = max(d.presence, abs(a - b))
    if (abs(a - b) > 0.02 || (a >= 0.5) != (b >= 0.5)) fail("box found $a vs $b")
    val wb = want["box"] as Map<*, *>?
    val gb = got["box"] as Map<*, *>?
    if ((wb == null) != (gb == null)) return fail("box presence")
    if (wb != null && gb != null) {
      for (edge in Boxes.EDGES) {
        val e = abs(num(wb[edge]) - num(gb[edge]))
        d.edges = max(d.edges, e)
        if (e > 0.002 + 1e-12) fail("edge $edge ${wb[edge]} vs ${gb[edge]}")
      }
    }
    val c = abs(num(want["confidence"]) - num(got["confidence"]))
    d.confidence = max(d.confidence, c)
    if (c > 0.02) fail("confidence ${want["confidence"]} vs ${got["confidence"]}")
  }

  private fun num(value: Any?): Double =
    when (value) {
      is JsonNumber -> value.toDouble()
      is Number -> value.toDouble()
      else -> throw IllegalArgumentException("not a number: $value")
    }

  private fun ints(value: Any?): List<Int> =
    when (value) {
      is IntArray -> value.toList()
      is List<*> -> value.map { (it as JsonNumber).toInt() }
      else -> throw IllegalArgumentException("not a list: $value")
    }

  companion object {
    /** An .npy file's header and the offset of its data. */
    private fun npyHeader(file: File, bytes: ByteArray): Pair<String, Int> {
      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      val major = bytes[6].toInt()
      val headerLength = if (major == 1) buffer.getShort(8).toInt() and 0xffff else buffer.getInt(8)
      val offset = (if (major == 1) 10 else 12) + headerLength
      val header = String(bytes, offset - headerLength, headerLength, Charsets.ISO_8859_1)
      require("'fortran_order': False" in header) { "${file.name}: $header" }
      return header to offset
    }

    /** A float32 .npy, row-major. */
    fun readNpy(file: File): FloatArray {
      val bytes = file.readBytes()
      val (header, offset) = npyHeader(file, bytes)
      require("'<f4'" in header) { "${file.name}: $header" }
      val out = FloatArray((bytes.size - offset) / 4)
      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      buffer.position(offset)
      buffer.asFloatBuffer().get(out)
      return out
    }

    /**
     * The picture as the model's image code decoded it: a uint8 .npy [height, width, 3] of the file's
     * stored RGB values (no color management), resized and normalized here by the Kotlin port.
     */
    fun readPicture(file: File): Picture {
      val bytes = file.readBytes()
      val (header, offset) = npyHeader(file, bytes)
      require("'|u1'" in header) { "${file.name}: $header" }
      val shape = Regex("'shape': \\((\\d+), (\\d+), 3\\)").find(header) ?: error("${file.name}: $header")
      val height = shape.groupValues[1].toInt()
      val width = shape.groupValues[2].toInt()
      val argb = IntArray(width * height) { i ->
        val r = bytes[offset + i * 3].toInt() and 0xff
        val g = bytes[offset + i * 3 + 1].toInt() and 0xff
        val b = bytes[offset + i * 3 + 2].toInt() and 0xff
        (0xff shl 24) or (r shl 16) or (g shl 8) or b
      }
      return Picture.of(argb, width, height)
    }
  }
}
