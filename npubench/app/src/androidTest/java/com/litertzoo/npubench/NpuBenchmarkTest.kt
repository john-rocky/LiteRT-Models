package com.litertzoo.npubench

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters

/**
 * Benchmark entry point for a cloud device farm, where the only way in is an
 * instrumentation test. Every figure is printed to logcat under the NpuBench tag.
 *
 * Method order is fixed and the NPU cases run first, because LiteRT's Environment is
 * shared across the process: whichever model loads first fixes the dispatch options for
 * every later load. A CPU case running first leaves the NPU without its Qualcomm
 * options, which cost 6x on ssdlite before this ordering was pinned.
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class NpuBenchmarkTest {

  private val context = InstrumentationRegistry.getInstrumentation().targetContext
  private val bench = NpuBenchmark(context)

  @Test
  fun a01_npu_ssdlite() = report(Accelerator.NPU, "ssdlite_npu_aot_sm8650.tflite")

  @Test
  fun a02_npu_twinlite() = report(Accelerator.NPU, "twinlite_npu_aot_sm8650.tflite")

  @Test
  fun a03_npu_silentface() = report(Accelerator.NPU, "silentface_npu_aot_sm8650.tflite")

  @Test
  fun b01_gpu_ssdlite() = report(Accelerator.GPU, "ssdlite_stock.tflite")

  @Test
  fun b02_gpu_twinlite() = report(Accelerator.GPU, "twinlite_stock.tflite")

  @Test
  fun b03_gpu_silentface() = report(Accelerator.GPU, "silentface_stock.tflite")

  @Test
  fun c01_cpu_ssdlite() = report(Accelerator.CPU, "ssdlite_stock.tflite")

  @Test
  fun c02_cpu_twinlite() = report(Accelerator.CPU, "twinlite_stock.tflite")

  @Test
  fun c03_cpu_silentface() = report(Accelerator.CPU, "silentface_stock.tflite")

  /**
   * Same backend, same options, first and last in one process. Anything separating the
   * two NPU figures is drift over the session — thermal or governor state — because the
   * option state cannot differ between them.
   */
  @Test
  fun d01_drift_npu_first() = report(Accelerator.NPU, "ssdlite_npu_aot_sm8650.tflite")

  @Test
  fun d02_drift_gpu_middle() = report(Accelerator.GPU, "ssdlite_stock.tflite")

  @Test
  fun d03_drift_npu_last() = report(Accelerator.NPU, "ssdlite_npu_aot_sm8650.tflite")

  @Test
  fun z_reportDevice() {
    Log.i(TAG, "DEVICE ${android.os.Build.MODEL} / SoC ${android.os.Build.SOC_MODEL}")
    Log.i(TAG, "NATIVE_LIBS ${bench.listNativeLibs()}")
  }

  private fun report(accelerator: Accelerator, asset: String) {
    try {
      val r = bench.run(asset, accelerator)
      Log.i(
        TAG,
        "BENCH ${r.accelerator} [${r.asset}] median=${"%.3f".format(r.medianMs)}ms " +
          "min=${"%.3f".format(r.minMs)}ms max=${"%.3f".format(r.maxMs)}ms " +
          "load=${"%.1f".format(r.loadMs)}ms runs=${r.runs} " +
          "thermal=${r.thermalBefore}->${r.thermalAfter} " +
          "headroom=${r.headroomBefore}->${r.headroomAfter}",
      )
    } catch (e: Throwable) {
      // A failure here is a result, not a crash to hide: record it and move on so the
      // other cases still report.
      Log.e(TAG, "BENCH ${accelerator.name} [$asset] FAILED: ${e::class.java.simpleName}: ${e.message}", e)
    }
  }

  companion object {
    private const val TAG = NpuBenchmark.TAG
  }
}
