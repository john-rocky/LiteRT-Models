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


  /**
   * Sweep entry point: model and accelerator come from instrumentation arguments, so a
   * whole zoo can be measured without packing it into the APK.
   *
   *   -e model /data/local/tmp/npubench/foo.tflite -e accel npu
   */
  @Test
  fun sweep() {
    val args = InstrumentationRegistry.getArguments()
    val path = args.getString("model") ?: error("pass -e model <path>")
    val accel = when (args.getString("accel")?.lowercase()) {
      "npu" -> Accelerator.NPU
      "cpu" -> Accelerator.CPU
      else -> Accelerator.GPU
    }
    val iterations = args.getString("iters")?.toIntOrNull() ?: 50
    reportPath(accel, path, iterations)
  }

  private fun reportPath(accelerator: Accelerator, path: String, iterations: Int) {
    try {
      val r = bench.runPath(path, accelerator, iterations = iterations)
      Log.i(
        TAG,
        "SWEEP ${r.accelerator} [${r.asset}] median=${"%.3f".format(r.medianMs)}ms " +
          "min=${"%.3f".format(r.minMs)}ms max=${"%.3f".format(r.maxMs)}ms " +
          "load=${"%.1f".format(r.loadMs)}ms runs=${r.runs} " +
          "thermal=${r.thermalBefore}->${r.thermalAfter} " +
          "headroom=${r.headroomBefore}->${r.headroomAfter}",
      )
    } catch (e: Throwable) {
      Log.e(TAG, "SWEEP ${accelerator.name} [$path] FAILED: ${e::class.java.simpleName}: ${e.message}")
    }
  }

  /**
   * Parity + latency for one signature with a real input:
   *
   *   -e model /data/local/tmp/gsctc/m.tflite -e accel cpu -e sig transcribe_10s
   *   -e input /data/local/tmp/gsctc/in_10s.raw -e outkinds i,f [-e iters 20]
   *
   * Multi-input graphs pass one raw f32 file per input, in signature order, as a
   * comma-separated list: `-e inputs a.raw,b.raw,c.raw`. `-e gpuprec fp32` requests
   * FP32 GPU compute (default is the delegate's own, fp16 on most GPUs).
   *
   * Dumps each output to the app's external files dir under <tag>/out<i>.bin and
   * logs a PARITY line with the bench figures and the dump dir.
   */
  @Test
  fun parity() {
    val args = InstrumentationRegistry.getArguments()
    val path = args.getString("model") ?: error("pass -e model <path>")
    val accel = when (args.getString("accel")?.lowercase()) {
      "npu" -> Accelerator.NPU
      "cpu" -> Accelerator.CPU
      else -> Accelerator.GPU
    }
    val sig = args.getString("sig") ?: ""
    val inputs =
      args.getString("inputs")?.split(",")?.filter { it.isNotEmpty() }
        ?: listOf(args.getString("input") ?: error("pass -e input <raw f32 file> or -e inputs a,b,c"))
    val outKinds = (args.getString("outkinds") ?: "f").split(",")
    val iterations = args.getString("iters")?.toIntOrNull() ?: 20
    val gpuFp32 = args.getString("gpuprec")?.lowercase() == "fp32"
    val tag = args.getString("tag") ?: "parity"
    val outDir = java.io.File(context.getExternalFilesDir(null), tag)
    try {
      val r =
        bench.parityBench(
          path, accel, sig, inputs, outDir, outKinds, iterations = iterations, gpuFp32 = gpuFp32,
        )
      Log.i(
        TAG,
        "PARITY ${r.accelerator} [${r.asset}] median=${"%.3f".format(r.medianMs)}ms " +
          "min=${"%.3f".format(r.minMs)}ms max=${"%.3f".format(r.maxMs)}ms " +
          "load=${"%.1f".format(r.loadMs)}ms runs=${r.runs} " +
          "thermal=${r.thermalBefore}->${r.thermalAfter} " +
          "headroom=${r.headroomBefore}->${r.headroomAfter} outdir=${outDir.absolutePath}",
      )
    } catch (e: Throwable) {
      Log.e(TAG, "PARITY ${accel.name} [$path#$sig] FAILED: ${e::class.java.simpleName}: ${e.message}")
    }
  }

  /**
   * Three resident models chained on the host (SmolVLA action chunk), with memory:
   *
   *   -e vision v.tflite -e prefix p.tflite -e expert e.tflite -e chaindir <dir>
   *   [-e accel gpu|cpu] [-e gpuprec fp32] [-e iters 5] [-e tag chain]
   */
  @Test
  fun chain() {
    val args = InstrumentationRegistry.getArguments()
    val accel = if (args.getString("accel")?.lowercase() == "cpu") Accelerator.CPU else Accelerator.GPU
    val tag = args.getString("tag") ?: "chain"
    try {
      bench.chainBench(
        visionPath = args.getString("vision") ?: error("pass -e vision <path>"),
        prefixPath = args.getString("prefix") ?: error("pass -e prefix <path>"),
        expertPath = args.getString("expert") ?: error("pass -e expert <path>"),
        dir = java.io.File(args.getString("chaindir") ?: error("pass -e chaindir <dir>")),
        accelerator = accel,
        gpuFp32 = args.getString("gpuprec")?.lowercase() == "fp32",
        outDir = java.io.File(context.getExternalFilesDir(null), tag),
        iterations = args.getString("iters")?.toIntOrNull() ?: 5,
      )
    } catch (e: Throwable) {
      Log.e(TAG, "CHAIN ${accel.name} FAILED: ${e::class.java.simpleName}: ${e.message}")
    }
  }

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
