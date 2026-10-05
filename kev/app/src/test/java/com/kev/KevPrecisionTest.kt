package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The GPU precision each graph compiles with when the launch names none: FP16_WITH_FP32_ACCUM for
 * the graphs whose gate passes at it, FP32 for a file of a size the model repository published
 * before the fp16-safe kernel rewrite.
 */
class KevPrecisionTest {
  @Test
  fun eachGraphRunsAtItsDefaultAndPreRewriteFilesAtFp32() {
    // The files of the model repository's next upload (the rewritten kernel), by graph.
    val current =
      mapOf(
        KevGraphKey.Window(64) to 1_258_031_552L,
        KevGraphKey.Window(128) to 1_258_444_912L,
        KevGraphKey.Window(256) to 1_259_246_704L,
        KevGraphKey.Window(512) to 1_261_233_328L,
        KevGraphKey.Window(1024) to 1_266_799_568L,
        KevGraphKey.Window(2048) to 1_284_223_520L,
        KevGraphKey.Pair(KevPairShape(128, 64)) to 1_261_368_160L,
        KevGraphKey.Pair(KevPairShape(256, 64)) to 1_261_918_016L,
      )
    assertEquals(
      (KevFiles.WINDOWS.map { KevGraphKey.Window(it) } +
          KevFiles.PAIRS.map { KevGraphKey.Pair(it) })
        .toSet(),
      current.keys,
    )
    // Every one of them passes its gate at FP16_WITH_FP32_ACCUM.
    assertEquals(current.keys, KevPrecision.FP16_FP32_ACCUM_GRAPHS)
    for ((graph, bytes) in current) {
      assertEquals(graph.label, KevPrecision.FP16_FP32_ACCUM, KevPrecision.defaultFor(graph, bytes))
    }
    // The three files published before the rewrite run at FP32, whatever the gate of the new ones.
    assertEquals(
      mapOf(
        KevGraphKey.Window(512) to 1_264_068_368L,
        KevGraphKey.Window(1024) to 1_269_023_216L,
        KevGraphKey.Window(2048) to 1_285_227_888L,
      ),
      KevFiles.PRE_REWRITE_BYTES,
    )
    for ((graph, bytes) in KevFiles.PRE_REWRITE_BYTES) {
      assertEquals(graph.label, KevPrecision.FP32, KevPrecision.defaultFor(graph, bytes))
    }
    // An old size counts only for its own graph.
    assertEquals(
      KevPrecision.defaultFor(KevGraphKey.Window(256), current.getValue(KevGraphKey.Window(256))),
      KevPrecision.defaultFor(KevGraphKey.Window(256), 1_264_068_368L),
    )
    // A graph the app has no gate for runs at FP32.
    assertEquals(
      KevPrecision.FP32,
      KevPrecision.defaultFor(KevGraphKey.Pair(KevPairShape(512, 64)), 1_263_000_000L),
    )
  }

  @Test
  fun aLineNamesOnePrecisionOnlyWhenItsGraphsShareIt() {
    val fp32 = KevPrecision.FP32
    val fp16 = KevPrecision.FP16_FP32_ACCUM
    assertNull(KevPrecision.common(emptyList(), null))
    assertEquals(fp32, KevPrecision.common(emptyList(), fp32))
    assertEquals(fp16, KevPrecision.common(listOf(fp16, fp16), null))
    assertEquals(fp32, KevPrecision.common(listOf(fp32), fp16))
    assertNull(KevPrecision.common(listOf(fp16, fp32), fp32))
  }
}
