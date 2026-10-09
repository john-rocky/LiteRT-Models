package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The checks of the one-signature graph wrapper ([D1GraphIo]) without a graph: declarations, the
 * graph's tensor types against them, a call's feeds, and the audio graph's own declarations from
 * contract.json (every `audio_<T>` file's inputs and output against [D1Audio.inputShapes] /
 * [D1Audio.outputShape]).
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

  private fun refused(what: String, block: () -> Unit) {
    try {
      block()
      fail("accepted: $what")
    } catch (expected: IllegalArgumentException) {
      assertTrue(what, !expected.message.isNullOrEmpty())
    }
  }
}
