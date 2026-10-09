package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The checks of the one-signature graph wrapper ([D1GraphIo]) without a graph: declarations, the
 * graph's tensor types against them, a call's feeds, and the audio graph's own declarations from
 * contract.json (every `audio_<T>` file's inputs and output against [D1Audio.inputShapes] /
 * [D1Audio.outputShape]), and the picture path's graphs as contract.json declares them
 * ([D1VisionContract]).
 */
class D1GraphTest {
  @Test
  fun elementCountsAndDeclarations() {
    assertEquals(128 * 1001, D1GraphIo.elementCount(listOf(1, 128, 1001)))
    assertEquals(126 * 1024, D1GraphIo.elementCount(listOf(1, 126, 1024)))
    for (bad in listOf(emptyList(), listOf(1, 0), listOf(-1), listOf(65536, 65536))) {
      refused("shape $bad") { D1GraphIo.elementCount(bad) }
    }
    D1GraphIo.checkDeclaration(D1Audio.inputShapes(1001), D1Audio.outputShape(1001))
    refused("no inputs") { D1GraphIo.checkDeclaration(emptyMap(), "y" to listOf(1)) }
    refused("output among the inputs") { D1GraphIo.checkDeclaration(mapOf("x" to listOf(1)), "x" to listOf(1)) }
  }

  @Test
  fun tensorTypesAndSignatureCounts() {
    D1GraphIo.checkTensor("mel", listOf(1, 128, 1001), true, listOf(1, 128, 1001))
    refused("an int32 tensor") { D1GraphIo.checkTensor("ids", listOf(1, 128), false, listOf(1, 128)) }
    refused("other dimensions") { D1GraphIo.checkTensor("mel", listOf(1, 128, 1001), true, listOf(1, 128, 501)) }
    refused("no layout") { D1GraphIo.checkTensor("mel", listOf(1, 128, 1001), true, null) }
    D1GraphIo.checkComplete(5, 5, 1)
    refused("an undeclared input") { D1GraphIo.checkComplete(5, 6, 1) }
    refused("two outputs") { D1GraphIo.checkComplete(5, 5, 2) }
  }

  @Test
  fun feedsMustNameEveryInputWithItsSize() {
    val shapes = D1Audio.inputShapes(501)
    val mel = D1Mel(FloatArray(128 * 400) { it * 1e-3f }, 399, 400)
    val feeds = D1Audio.buildInputs(mel, 501).feeds()
    assertEquals(shapes.keys.toList(), feeds.keys.toList())
    D1GraphIo.checkFeeds(shapes, feeds)
    refused("a missing input") { D1GraphIo.checkFeeds(shapes, feeds.filterKeys { it != "v3" }) }
    refused("an extra input") { D1GraphIo.checkFeeds(shapes, feeds + ("x" to FloatArray(1))) }
    refused("a short input") { D1GraphIo.checkFeeds(shapes, feeds + ("v1" to FloatArray(250))) }
    refused("the wrong bucket") { D1GraphIo.checkFeeds(D1Audio.inputShapes(1001), feeds) }
    val call = D1GraphCall(FloatArray(2), 1_000_000, 2_500_000, 500_000)
    assertEquals(4.0, call.totalMs, 1e-12)
    assertEquals(2.5, call.runMs, 1e-12)
  }

  @Test
  fun audioDeclarationsMatchTheContract() {
    val contract = ExternalTestData.json(ExternalTestData.repoFile(D1Contract.FILE))
    val parsed = ExternalTestData.contract()
    var checked = 0
    for (file in contract["files"] as List<*>) {
      val entry = file as Map<*, *>
      if (entry["graph"] != D1AudioEngine.GRAPH) continue
      val bucket = (entry["bucket"] as JsonNumber).toInt()
      assertEquals(D1AudioEngine.signatureOf(bucket), entry["signature"])
      val inputs =
        (entry["inputs"] as List<*>).associate {
          val tensor = it as Map<*, *>
          assertEquals("FLOAT32", tensor["dtype"])
          tensor["name"] as String to ExternalTestData.ints(tensor["shape"]).toList()
        }
      assertEquals("$bucket inputs", D1Audio.inputShapes(bucket).toList(), inputs.toList())
      val output = (entry["outputs"] as List<*>).single() as Map<*, *>
      assertEquals(D1Audio.outputShape(bucket), output["name"] as String to ExternalTestData.ints(output["shape"]).toList())
      checked++
    }
    assertEquals(listOf(501, 1001, 2001, 3001), D1Audio.T_BUCKETS)
    assertEquals(4, checked)
    // The engine finds the same four files through D1Contract.files.
    val engineFiles =
      parsed.files.filter { it.graph == D1AudioEngine.GRAPH && it.signature == D1AudioEngine.signatureOf(it.bucket ?: 0) }
    assertEquals(D1Audio.T_BUCKETS, engineFiles.map { it.bucket })
    assertEquals(225525600L, engineFiles.single { it.bucket == 1001 }.bytes)
  }


  private val tower =
    linkedMapOf("pixels" to listOf(1, 1024, 768), "pos" to listOf(1, 1024, 768), "mask" to listOf(1, 1024))

  @Test
  fun declarationsAndFeeds() {
    assertEquals(786432, D1GraphIo.elementCount(listOf(1, 1024, 768)))
    D1GraphIo.checkDeclaration(tower, "features" to listOf(1, 1024, 768))
    refused("no inputs") { D1GraphIo.checkDeclaration(linkedMapOf(), "y" to listOf(1)) }
    refused("a zero dimension") { D1GraphIo.checkDeclaration(linkedMapOf("x" to listOf(1, 0)), "y" to listOf(1)) }
    refused("an empty shape") { D1GraphIo.checkDeclaration(linkedMapOf("x" to emptyList()), "y" to listOf(1)) }
    refused("too many values") { D1GraphIo.checkDeclaration(linkedMapOf("x" to listOf(65536, 65536)), "y" to listOf(1)) }
    refused("an output named like an input") { D1GraphIo.checkDeclaration(tower, "mask" to listOf(1, 1024)) }
    val feeds = mapOf("pixels" to FloatArray(786432), "pos" to FloatArray(786432), "mask" to FloatArray(1024))
    D1GraphIo.checkFeeds(tower, feeds)
    refused("a missing input") { D1GraphIo.checkFeeds(tower, feeds - "mask") }
    refused("an extra input") { D1GraphIo.checkFeeds(tower, feeds + ("soft" to FloatArray(1))) }
    refused("a short input") { D1GraphIo.checkFeeds(tower, feeds + ("mask" to FloatArray(1023))) }
    refused("a long input") { D1GraphIo.checkFeeds(tower, feeds + ("pos" to FloatArray(786433))) }
  }

  @Test
  fun graphTensorsMustBeTheDeclaredOnes() {
    D1GraphIo.checkTensor("pixels", listOf(1, 1024, 768), true, listOf(1, 1024, 768))
    refused("an int32 tensor") { D1GraphIo.checkTensor("pixels", listOf(1, 1024, 768), false, listOf(1, 1024, 768)) }
    refused("another shape") { D1GraphIo.checkTensor("pixels", listOf(1, 1024, 768), true, listOf(1, 512, 768)) }
    refused("no layout") { D1GraphIo.checkTensor("pixels", listOf(1, 1024, 768), true, null) }
    D1GraphIo.checkComplete(3, 3, 1)
    refused("an undeclared input") { D1GraphIo.checkComplete(3, 4, 1) }
    refused("two outputs") { D1GraphIo.checkComplete(3, 3, 2) }
  }

  @Test
  fun theContractsPictureGraphs() {
    val vision = D1VisionContract.read(ExternalTestData.repoFile(D1Contract.FILE))
    assertEquals("d1-omni-600M_vision_tower_fp16.tflite", vision.tower.file)
    assertEquals("vision_tower", vision.tower.signature)
    assertEquals(171563424L, vision.tower.bytes)
    assertTrue(vision.tower.sha256.startsWith("6835066c"))
    assertEquals(tower.toList(), vision.tower.inputs.toList())
    assertEquals("features" to listOf(1, 1024, 768), vision.tower.output)
    D1GraphIo.checkDeclaration(vision.tower.inputs, vision.tower.output)
    assertEquals("d1-omni-600M_projector_fp16.tflite", vision.projector.file)
    assertEquals("projector", vision.projector.signature)
    assertEquals(16791504L, vision.projector.bytes)
    assertTrue(vision.projector.sha256.startsWith("9224b508"))
    assertEquals(listOf("soft" to listOf(1, 256, 3072)), vision.projector.inputs.toList())
    assertEquals("prefix" to listOf(1, 256, 1024), vision.projector.output)
    D1GraphIo.checkDeclaration(vision.projector.inputs, vision.projector.output)
    assertEquals("host/vision_position_table.npy", vision.tableFile)
    assertTrue(vision.tableSha256.startsWith("76d764aa"))
    // A file entry whose input is not the app's layout is refused.
    @Suppress("UNCHECKED_CAST")
    val root = D1Json.parse(ExternalTestData.repoFile(D1Contract.FILE).readBytes()) as MutableMap<String, Any?>
    @Suppress("UNCHECKED_CAST")
    val entry = (root["files"] as List<*>).map { it as MutableMap<String, Any?> }.first { it["name"] == vision.tower.file }
    @Suppress("UNCHECKED_CAST")
    val pixels = (entry["inputs"] as List<*>).map { it as MutableMap<String, Any?> }.first { it["name"] == "pixels" }
    pixels["shape"] = listOf(JsonNumber("1"), JsonNumber("512"), JsonNumber("768"))
    refused("a tower of another shape") { D1VisionContract(root) }
  }

  private fun refused(what: String, block: () -> Unit) {
    try {
      block()
      fail("accepted: $what")
    } catch (expected: IllegalArgumentException) {
      assertTrue(what, !expected.message.isNullOrEmpty())
    }
  }
}
