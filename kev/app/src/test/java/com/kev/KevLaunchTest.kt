package com.kev

import com.google.ai.edge.litert.CompiledModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The launch extras ([KevLaunch.parse]) with the `backend` extra on every kind of launch, the NPU
 * debug extras, an APK without the NPU libraries, and the "Run on" choice kept between launches.
 */
class KevLaunchTest {
  /** Launch extras as a map: strings, ints and booleans by name. */
  private class Extras(private val values: Map<String, Any>) : KevExtras {
    override fun has(name: String) = name in values

    override fun string(name: String) = values[name] as String?

    override fun int(name: String, default: Int) = values[name] as Int? ?: default

    override fun boolean(name: String, default: Boolean) = values[name] as Boolean? ?: default
  }

  private fun parse(vararg extras: Pair<String, Any>, npu: Boolean = true): KevLaunch =
    KevLaunch.parse(
      Extras(extras.toMap()),
      KevLaunchContext(
        debug = true,
        diagnostics = true,
        installedWindows = listOf(128, 256),
        installedPairs = listOf(KevPairShape(128, 64)),
        npuAvailable = npu,
      ),
    )

  @Test
  fun everyLaunchTakesTheNpuBackend() {
    val npu = KevDecider.Backend.NPU
    // Normal: the extra, or null for the kept choice.
    assertEquals(KevLaunch.Normal(backend = npu), parse("backend" to "npu"))
    assertEquals(KevLaunch.Normal(), parse())
    // Autoplay.
    val autoplay = parse("autoplay" to true, "fixture" to "demo_ticket_01.json", "backend" to "npu")
    assertEquals(npu, (autoplay as KevLaunch.Autoplay).backend)
    assertNull((parse("autoplay" to true, "fixture" to "f.json") as KevLaunch.Autoplay).backend)
    // Gate: the named graph on the NPU with BURST, the debug extras on top.
    val gate = parse("gate" to true, "backend" to "npu", "window" to 64) as KevLaunch.Gate
    assertEquals(npu, gate.backend)
    assertEquals(KevGraphKey.Window(64), gate.graph)
    assertEquals(KevNpuOptions(), gate.npu)
    assertEquals("app_gate_npu_L64.json", gate.report)
    val o3 = parse("gate" to true, "backend" to "npu", "npu_opt" to "o3") as KevLaunch.Gate
    assertEquals(
      CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_INFERENCE_O3,
      o3.npu?.optimization,
    )
    val noPerf = parse("gate" to true, "backend" to "npu", "npu_perf" to "none") as KevLaunch.Gate
    assertNull(noPerf.npu?.performance)
    // The pair on the NPU (a debug run), with the state through the host.
    val pair =
      parse("gate" to true, "backend" to "npu", "graph" to "pair", "pair_state" to "copy")
        as KevLaunch.Gate
    assertEquals(KevGraphKey.Pair(KevPairShape(128, 64)), pair.graph)
    assertTrue(pair.stateCopy)
    // Timing.
    val timing =
      parse("timing" to true, "rows" to "timing_rows.json", "backend" to "npu", "window" to 256)
        as KevLaunch.Timing
    assertEquals(npu, timing.backend)
    assertEquals(256, timing.window)
    assertEquals(KevNpuOptions(), timing.npu)
    assertFalse(timing.clearCache)
    assertFalse(timing.stateCopy)
    assertTrue(timing.windowNamed)
    // Without a window extra the request path's rows take the app's own row plan.
    val rows =
      parse("timing" to true, "rows" to "r.json", "graph" to "rows", "sets" to "none")
        as KevLaunch.Timing
    assertFalse(rows.windowNamed)
    assertEquals(emptyList<String>(), rows.sets)
    // Gate and timing without the extra stay on the GPU.
    assertEquals(KevDecider.Backend.GPU, (parse("gate" to true) as KevLaunch.Gate).backend)
  }

  @Test
  fun anApkWithoutTheNpuLibrariesRefusesTheNpu() {
    for (extras in
      listOf(
        arrayOf<Pair<String, Any>>("backend" to "npu"),
        arrayOf("autoplay" to true, "fixture" to "f.json", "backend" to "npu"),
        arrayOf("gate" to true, "backend" to "npu"),
        arrayOf("timing" to true, "rows" to "r.json", "backend" to "npu"),
      )) {
      val launch = parse(*extras, npu = false)
      assertEquals(
        extras.toList().toString(),
        KevLaunch.NO_NPU_LIBRARIES,
        (launch as KevLaunch.Invalid).reason,
      )
    }
    // The NPU debug extras need the libraries too; without any NPU extra nothing changes.
    assertTrue(parse("gate" to true, "npu_opt" to "o3", npu = false) is KevLaunch.Invalid)
    assertNull((parse("gate" to true, npu = false) as KevLaunch.Gate).npu)
    assertEquals(
      KevLaunch.Normal(backend = KevDecider.Backend.CPU),
      parse("backend" to "cpu", npu = false),
    )
    // Values that are not extras of the app.
    assertTrue(parse("backend" to "tpu") is KevLaunch.Invalid)
    assertTrue(parse("gate" to true, "backend" to "npu", "npu_opt" to "fast") is KevLaunch.Invalid)
    assertTrue(parse("gate" to true, "pair_state" to "share") is KevLaunch.Invalid)
  }

  @Test
  fun theKeptChoiceGoesBackToTheGpuWhereTheNpuCannotRun() {
    val gpu = KevDecider.Backend.GPU
    val npu = KevDecider.Backend.NPU
    // Nothing kept: the GPU.
    assertEquals(gpu, KevBackendChoice.resolve(null, null, npuAvailable = true))
    // The kept choice, when this APK can run it.
    assertEquals(npu, KevBackendChoice.resolve(null, "npu", npuAvailable = true))
    assertEquals(
      KevDecider.Backend.CPU,
      KevBackendChoice.resolve(null, "cpu", npuAvailable = false),
    )
    // A kept NPU in an APK without the libraries: the GPU, and the kept value is replaced.
    assertEquals(gpu, KevBackendChoice.resolve(null, "npu", npuAvailable = false))
    assertTrue(KevBackendChoice.keptUnusable("npu", npuAvailable = false))
    assertFalse(KevBackendChoice.keptUnusable("npu", npuAvailable = true))
    assertFalse(KevBackendChoice.keptUnusable("gpu", npuAvailable = false))
    // The launch's extra wins over the kept choice.
    assertEquals(gpu, KevBackendChoice.resolve(gpu, "npu", npuAvailable = true))
    assertEquals(gpu, KevBackendChoice.resolve(null, "garbage", npuAvailable = true))
  }
}
