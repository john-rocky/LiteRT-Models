package com.timestral

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Which graph a forecast takes: the backend's graph with the fewest tokens that holds it. */
class GraphSpecTest {
  private val all = Backend.entries.toSet()
  private val specs =
    listOf(
      GraphSpec("B1", "b1.tflite", 64, 32, 2, -3e4f, all),
      GraphSpec("B2", "b2.tflite", 256, 32, 2, -3e4f, all),
      GraphSpec("B1h8", "b1h8.tflite", 64, 8, 2, -3e4f, setOf(Backend.CPU, Backend.GPU)),
      GraphSpec("B2h8", "b2h8.tflite", 256, 8, 2, -3e4f, setOf(Backend.CPU, Backend.GPU)),
      GraphSpec("B1h8 one-row", "b1h8r1.tflite", 64, 8, 1, -3e4f, setOf(Backend.NPU)),
    )

  private fun choose(backend: Backend, contextPatches: Int, horizonPatches: Int) =
    GraphSpec.choose(specs, backend, contextPatches, horizonPatches)?.name

  @Test
  fun takesTheFewestTokensThatHoldTheForecast() {
    assertEquals("B1h8", choose(Backend.GPU, 10, 3))
    assertEquals("B1h8", choose(Backend.GPU, 64, 8))
    assertEquals("B1", choose(Backend.GPU, 64, 9))
    assertEquals("B2h8", choose(Backend.CPU, 65, 8))
    assertEquals("B2", choose(Backend.CPU, 200, 32))
    assertNull(choose(Backend.CPU, 257, 1))
  }

  @Test
  fun keepsToTheBackendsGraphs() {
    assertEquals("B1h8 one-row", choose(Backend.NPU, 10, 3))
    assertEquals("B1", choose(Backend.NPU, 10, 9))
    assertEquals("B2", choose(Backend.NPU, 100, 2))
  }

  @Test
  fun takesTheFirstListedAmongEqualTokens() {
    val tied =
      listOf(
        GraphSpec("first", "a.tflite", 64, 32, 1, -3e4f, all),
        GraphSpec("second", "b.tflite", 64, 32, 2, -3e4f, all),
      )
    assertEquals("first", GraphSpec.choose(tied, Backend.CPU, 1, 1)?.name)
  }

  @Test
  fun countsTheCallsOfAForecast() {
    assertEquals(1, specs[0].calls)
    assertEquals(2, specs[4].calls)
  }
}
