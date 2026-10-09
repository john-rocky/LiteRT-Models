package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * What can be checked of a single-signature graph without Android: the declared tensors, the graph's
 * tensor types and the feeds a call takes ([D1GraphIo]), and the picture path's graphs as
 * contract.json declares them ([D1VisionContract]).
 */
class D1GraphTest {
  private fun refused(what: String, block: () -> Unit) {
    try {
      block()
      fail("$what was accepted")
    } catch (expected: IllegalArgumentException) {
      // refused with its reason
    }
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
}
