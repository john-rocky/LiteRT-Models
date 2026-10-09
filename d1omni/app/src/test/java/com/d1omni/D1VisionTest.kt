package com.d1omni

import java.io.File
import kotlin.math.abs
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The picture path's host steps against the Python host (`host/d1_vision_host.py`) on the public
 * image check set, bit for bit (float32 values compared as their bytes = raw bits): the decoded RGB
 * of the five PNGs, layout() on 6,000+ sizes, the resample weights and the float-path resize on 40
 * random arrays, every crop's uint8 pixels and tower inputs (pixels / pos / mask, grid), the
 * position table resized to all 1,466 even grids of up to 1,024 patches, the unshuffle and projector
 * input, the prefix assembled from the Mac CPU's graph outputs, each question's six decision inputs
 * and its read-out; EXIF orientations against Pillow's `exif_transpose`. Data: `d1omni.demo`
 * `fixtures/vision/` (demo/scripts/vision_dump_v.py) and `device/r3/` rows files.
 */
class D1VisionTest {
  private val records = listOf("img_dogs_01", "img_cat_02", "img_bike_03", "img_02", "img_03")

  private fun vision(path: String): File = ExternalTestData.demoFile("fixtures/vision/$path")

  private fun meta(id: String): Map<*, *> = ExternalTestData.json(vision("$id/meta.json"))

  private fun int(value: Any?): Int = (value as JsonNumber).toInt()

  private fun table(): FloatArray =
    D1Npy.positionTable(
      ExternalTestData.repoFile("host/vision_position_table.npy"),
      D1VisionContract.read(ExternalTestData.repoFile(D1Contract.FILE)).tableSha256,
    )

  private fun rgbOf(id: String): D1Rgb {
    val m = meta(id)
    return D1Rgb(int(m["w"]), int(m["h"]), vision("$id/rgb.u8").readBytes())
  }

  /** Fails with where and how much [actual] differs from the little-endian float32 file [expected]. */
  private fun assertBits(label: String, actual: FloatArray, expected: ByteArray) {
    val bytes = D1VisionChecks.floatBytes(actual)
    if (bytes.contentEquals(expected)) return
    if (bytes.size != expected.size) fail("$label: ${bytes.size} bytes, expected ${expected.size}")
    val reference = D1VisionChecks.floats(expected)
    var count = 0
    var first = -1
    var largest = 0.0
    for (i in actual.indices) {
      if (actual[i].toRawBits() != reference[i].toRawBits()) {
        count++
        if (first < 0) first = i
        largest = maxOf(largest, abs(actual[i].toDouble() - reference[i].toDouble()))
      }
    }
    fail("$label: $count of ${actual.size} values differ, first at $first (${actual[first]} vs ${reference[first]}), max |d| $largest")
  }

  private fun assertBytes(label: String, actual: ByteArray, expected: ByteArray) {
    if (actual.contentEquals(expected)) return
    fail("$label: ${D1VisionChecks.byteDiff(actual, expected)}")
  }

  @Test
  fun layoutMatchesPython() {
    val doc = ExternalTestData.json(vision("layout_cases.json"))
    val ratios = (doc["ratios"] as List<*>).map { pair -> (pair as List<*>).map { int(it) } }
    assertEquals(ratios, D1Vision.GRIDS.map { listOf(it.first, it.second) })
    var cases = 0
    for (case in doc["cases"] as List<*>) {
      val v = (case as List<*>).map { int(it) }
      val plan = D1Vision.layout(v[0], v[1])
      assertEquals("layout(${v[0]}, ${v[1]})", D1Layout(v[2], v[3], v[4], v[5], v[6] == 1), plan)
      cases++
    }
    println("D1_VISION layout $cases/$cases")
    assertTrue(cases > 6000)
  }

  @Test
  fun resampleWeightsAndResizeMatchPython() {
    val doc = ExternalTestData.json(vision("resize_cases.json"))
    var weights = 0
    for (entry in doc["weights"] as List<*>) {
      val w = entry as Map<*, *>
      val plan = D1Vision.linearWeightsF32(int(w["in"]), int(w["out"]))
      val label = "weights ${w["in"]} -> ${w["out"]}"
      assertArrayEquals(label, ExternalTestData.ints(w["xmins"]), plan.xmins)
      assertArrayEquals(label, ExternalTestData.ints(w["sizes"]), plan.sizes)
      val rows = (w["weights_bits"] as List<*>).map { row -> (row as List<*>).map { (it as JsonNumber).literal.toLong().toInt() } }
      assertEquals(label, rows.first().size, plan.maxTaps)
      for ((i, row) in rows.withIndex()) {
        for ((j, bits) in row.withIndex()) assertEquals("$label [$i, $j]", bits, plan.weight(i, j).toRawBits())
      }
      weights++
    }
    val blob = vision(doc["bin"] as String).readBytes()
    assertEquals(doc["bin_sha256"], D1VisionChecks.sha256(blob))
    var resized = 0
    for (entry in doc["cases"] as List<*>) {
      val c = entry as Map<*, *>
      val (ih, iw) = ExternalTestData.ints(c["in_hw"]).let { it[0] to it[1] }
      val (oh, ow) = ExternalTestData.ints(c["out_hw"]).let { it[0] to it[1] }
      val start = int(c["in_offset"])
      val source = D1Rgb(iw, ih, blob.copyOfRange(start, start + int(c["in_bytes"])))
      val out = D1Vision.resizeFloat(source, oh, ow)
      val outStart = int(c["out_offset"])
      assertBytes("resize ${ih}x$iw -> ${oh}x$ow", out.data, blob.copyOfRange(outStart, outStart + int(c["out_bytes"])))
      resized++
    }
    println("D1_VISION weights $weights resize $resized")
    assertEquals(40, resized)
  }

  @Test
  fun decodeMatchesPillow() {
    var stripped = 0
    for (id in records) {
      val png = ExternalTestData.repoFile("fixtures/media/$id.png").readBytes()
      val expected = vision("$id/rgb.u8").readBytes()
      val decoded = D1ImageJvm.decode(png)
      assertEquals(int(meta(id)["w"]), decoded.width)
      assertEquals(int(meta(id)["h"]), decoded.height)
      assertBytes("decode $id", decoded.data, expected)
      // The PNG the phone's decoder gets (colour chunks dropped) holds the same samples.
      val (bytes, removed) = D1ImageOps.stripPngColorChunks(png)
      if (removed.isNotEmpty()) stripped++
      assertTrue(removed.all { it in D1ImageOps.COLOR_CHUNKS })
      assertBytes("decode stripped $id", D1ImageJvm.decode(bytes).data, expected)
      assertTrue(if (removed.isEmpty()) bytes.contentEquals(png) else bytes.size < png.size)
      assertTrue(D1ImageOps.stripPngColorChunks(bytes).second.isEmpty())
    }
    println("D1_VISION decode ${records.size}/${records.size} (colour chunks dropped from $stripped)")
    assertEquals(3, stripped)
  }

  @Test
  fun cropsPatchesPositionsAndMaskMatchPython() {
    val table = table()
    var crops = 0
    for (id in records) {
      val m = meta(id)
      val plan = m["plan"] as Map<*, *>
      val (list, layout) = D1Vision.crops(rgbOf(id))
      val grid = ExternalTestData.ints(plan["grid"])
      val thumb = ExternalTestData.ints(plan["thumbnail"])
      assertEquals(D1Layout(grid[0], grid[1], thumb[0], thumb[1], plan["tiled"] as Boolean), layout)
      val expected = m["crops"] as List<*>
      assertEquals(expected.size, list.size)
      var rows = 0
      for ((k, crop) in list.withIndex()) {
        val c = expected[k] as Map<*, *>
        assertEquals(ExternalTestData.ints(c["hw"]).toList(), listOf(crop.height, crop.width))
        assertBytes("$id crop$k", crop.data, vision("$id/crop$k.u8").readBytes())
        val patches = D1Vision.toPatches(crop)
        assertEquals(ExternalTestData.ints(c["grid"]).toList(), listOf(patches.gridHeight, patches.gridWidth))
        assertEquals(int(c["cells"]), patches.cells)
        assertBits("$id pixels$k", patches.pixels, vision("$id/pixels$k.f32").readBytes())
        assertBits("$id mask$k", patches.mask, vision("$id/mask$k.f32").readBytes())
        val pos = D1Vision.positionsPadded(table, patches.gridHeight, patches.gridWidth)
        assertBits("$id pos$k", pos, vision("$id/pos$k.f32").readBytes())
        assertEquals(c["pixels_sha256"], D1VisionChecks.sha256(patches.pixels))
        assertEquals(c["pos_sha256"], D1VisionChecks.sha256(pos))
        rows += patches.cells
        crops++
      }
      assertEquals(int(m["P"]), rows)
    }
    println("D1_VISION crops / pixels / pos / mask $crops/$crops")
    assertEquals(11, crops)
  }

  @Test
  fun unshuffleMatchesPython() {
    var crops = 0
    for (id in records) {
      for (entry in meta(id)["crops"] as List<*>) {
        val c = entry as Map<*, *>
        val k = int(c["k"])
        val grid = ExternalTestData.ints(c["grid"])
        val features = D1VisionChecks.floats(vision("$id/features$k.f32").readBytes())
        val cells = D1Vision.pixelUnshuffle(features, grid[0], grid[1])
        val soft = D1Vision.projectorInput(cells, int(c["cells"]))
        assertBits("$id soft$k", soft, vision("$id/soft$k.f32").readBytes())
        crops++
      }
    }
    println("D1_VISION unshuffle $crops/$crops")
    assertEquals(11, crops)
  }

  @Test
  fun prefixAndDecisionRowsMatchPython() {
    val table = table()
    val tokenizer = ExternalTestData.tokenizer()
    val contract = ExternalTestData.contract()
    val fixture = ExternalTestData.json(ExternalTestData.repoFile("fixtures/public_image.json"))
    val byId = (fixture["records"] as List<*>).associateBy { (it as Map<*, *>)["id"] as String }
    var questions = 0
    var worst = 0.0
    for (id in records) {
      val m = meta(id)
      var towerCalls = 0
      var projectorCalls = 0
      val run =
        D1VisionPrefix.run(
          rgbOf(id),
          positions = { h, w -> D1Vision.positionsPadded(table, h, w) to false },
          tower = { patches, pos ->
            val k = towerCalls++
            assertBits("$id tower input pixels$k", patches.pixels, vision("$id/pixels$k.f32").readBytes())
            assertBits("$id tower input pos$k", pos, vision("$id/pos$k.f32").readBytes())
            D1VisionPrefix.Output(D1VisionChecks.floats(vision("$id/features$k.f32").readBytes()), null)
          },
          projector = { soft ->
            val k = projectorCalls++
            assertBits("$id projector input soft$k", soft, vision("$id/soft$k.f32").readBytes())
            D1VisionPrefix.Output(D1VisionChecks.floats(vision("$id/projected$k.f32").readBytes()), null)
          },
        )
      assertEquals((m["crops"] as List<*>).size, towerCalls)
      assertEquals(int(m["P"]), run.rows)
      assertBits("$id prefix", run.prefix, vision("$id/prefix.f32").readBytes())
      val record = byId.getValue(id) as Map<*, *>
      val named = record["questions"] as Map<*, *>
      for (entry in m["questions"] as List<*>) {
        val q = entry as Map<*, *>
        val question = D1Prompt.asQuestion(named[q["name"]])
        val encoded =
          D1Rows.rows(tokenizer, contract, record["state"], listOf(question), run.rows, D1Kind.IMAGE).single()
        assertArrayEquals("$id ids", ExternalTestData.ints(q["ids"]), encoded.ids)
        assertArrayEquals("$id markers", ExternalTestData.ints(q["markers"]), encoded.markers)
        val length = int(q["L"])
        assertEquals(length, D1Contract.bucketFor(run.rows + encoded.ids.size, contract.buckets))
        val inputs = D1Rows.buildInputs(encoded.ids, run.prefix, run.rows, length, question.type)
        val hashes = q["inputs_sha256"] as Map<*, *>
        val idsBytes = java.nio.ByteBuffer.allocate(length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        idsBytes.asIntBuffer().put(inputs.ids)
        assertEquals("$id ids input", hashes["ids"], D1VisionChecks.sha256(idsBytes.array()))
        assertEquals("$id prefix input", hashes["prefix"], D1VisionChecks.sha256(requireNotNull(inputs.prefix)))
        assertEquals("$id media", hashes["media"], D1VisionChecks.sha256(inputs.media))
        assertEquals("$id pad", hashes["pad"], D1VisionChecks.sha256(inputs.pad))
        assertEquals("$id keep_right", hashes["keep_right"], D1VisionChecks.sha256(inputs.keepRight))
        assertEquals("$id qtype_onehot", hashes["qtype_onehot"], D1VisionChecks.sha256(inputs.qtypeOneHot))
        val scores = (q["mac_cpu_scores"] as List<*>).map { (it as JsonNumber).toDouble().toFloat() }.toFloatArray()
        val probabilities =
          D1Readout.probabilities(scores, run.rows, encoded.markers, question, false, contract)
        val expected = ExternalTestData.doubles(q["mac_cpu_probs"])
        for (i in expected.indices) worst = maxOf(worst, abs(probabilities[i] - expected[i]))
        questions++
      }
    }
    println("D1_VISION prefix ${records.size}/${records.size}, decision inputs and read-out $questions rows, max |dp| $worst")
    assertEquals(12, questions)
    assertTrue("read-out max |dp| $worst", worst <= 1e-12)
  }

  @Test
  fun positionTableSweepMatchesPython() {
    val table = table()
    val doc = ExternalTestData.json(vision("positions_sweep.json"))
    var grids = 0
    for (entry in doc["grids"] as List<*>) {
      val g = entry as List<*>
      val h = int(g[0])
      val w = int(g[1])
      assertEquals("positions ($h, $w)", g[2], D1VisionChecks.sha256(D1Vision.positionsPadded(table, h, w)))
      grids++
    }
    println("D1_VISION positions sweep $grids/$grids")
    assertEquals(1466, grids)
  }

  @Test
  fun exifOrientationMatchesPillow() {
    val doc = ExternalTestData.json(vision("orient/orient.json"))
    val hw = ExternalTestData.ints(doc["src_hw"])
    val source = D1Rgb(hw[1], hw[0], vision("orient/src.u8").readBytes())
    var cases = 0
    for (entry in doc["cases"] as List<*>) {
      val c = entry as Map<*, *>
      val k = int(c["orientation"])
      val out = D1ImageOps.orient(source, k)
      val expectedHw = ExternalTestData.ints(c["out_hw"])
      assertEquals("orientation $k", expectedHw.toList(), listOf(out.height, out.width))
      assertBytes("orientation $k", out.data, vision("orient/${c["out"]}").readBytes())
      cases++
    }
    println("D1_VISION EXIF orientations $cases/8")
    assertEquals(8, cases)
  }

  @Test
  fun rowsFilesParseAndEncodeAgain() {
    val tokenizer = ExternalTestData.tokenizer()
    val contract = ExternalTestData.contract()
    val counts = LinkedHashMap<String, Int>()
    var matched = 0
    var rows = 0
    for (name in listOf("rows_image.json", "rows_image_small.json", "rows_image_L2048.json")) {
      val records = D1VisionRows.parse(ExternalTestData.demoFile("device/r3/$name").readBytes())
      counts[name] = records.sumOf { it.rows.size }
      for (record in records) {
        val reference = requireNotNull(record.reference)
        val m = meta(record.id)
        assertEquals(m["rgb_sha256"], reference.rgbSha256)
        assertEquals(m["prefix_sha256"], reference.prefixSha256)
        assertEquals(int(m["P"]), reference.prefixRows)
        for (row in record.rows) {
          rows++
          val encoded =
            D1Rows.rows(tokenizer, contract, record.state, listOf(row.question), row.prefixRows, D1Kind.IMAGE).single()
          if (encoded.ids.contentEquals(row.ids) && encoded.markers.contentEquals(row.markers)) matched++
          assertTrue(!row.calibrate)
        }
      }
    }
    val sets = D1VisionRows.parseTiming(ExternalTestData.demoFile("device/r3/timing_image.json").readBytes())
    assertEquals(listOf("dogs2"), sets.map { it.name })
    assertEquals(listOf("img_dogs_01/count", "img_dogs_01/water"), sets.single().record.rows.map { it.key })
    println("D1_VISION rows files $counts re-encoded $matched/$rows")
    assertEquals(mapOf("rows_image.json" to 12, "rows_image_small.json" to 9, "rows_image_L2048.json" to 3), counts)
    assertEquals(rows, matched)
  }
}
