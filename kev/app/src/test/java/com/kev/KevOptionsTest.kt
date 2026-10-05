package com.kev

import com.google.ai.edge.litert.CompiledModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The compile options of the graphs: GPU always with an explicit precision; a pair with constant
 * tensor sharing on or off as the `share` extra (or, with `auto`, the available memory) says, a row
 * graph without the setting; CPU with four threads. Only the option objects are built, no graph is
 * compiled.
 */
class KevOptionsTest {
  @Test
  fun aPairSharesItsWeightsAsTheShareModeSays() {
    val shared = KevDecider.options(KevDecider.Backend.GPU, KevPrecision.FP16_FP32_ACCUM, true)
    assertEquals(true, shared.gpuOptions?.constantTensorSharing)
    assertEquals(
      CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM,
      shared.gpuOptions?.precision,
    )
    val unshared = KevDecider.options(KevDecider.Backend.GPU, KevPrecision.FP32, false)
    assertEquals(false, unshared.gpuOptions?.constantTensorSharing)
    assertEquals(CompiledModel.GpuOptions.Precision.FP32, unshared.gpuOptions?.precision)
    // A row graph has one signature: the setting stays LiteRT's.
    val row = KevDecider.options(KevDecider.Backend.GPU, KevPrecision.FP32)
    assertNull(row.gpuOptions?.constantTensorSharing)
    val cpu = KevDecider.options(KevDecider.Backend.CPU, KevPrecision.FP32, true)
    assertNull(cpu.gpuOptions)
    assertEquals(KevDecider.CPU_THREADS, cpu.cpuOptions?.numThreads)
    // The `share` extra: auto (the default) decides by the memory available at the compile.
    assertEquals(KevPairShare.AUTO, KevLaunch.share("auto"))
    assertEquals(KevPairShare.ON, KevLaunch.share("on"))
    assertEquals(KevPairShare.OFF, KevLaunch.share(" OFF "))
    assertNull(KevLaunch.share("yes"))
  }
}
