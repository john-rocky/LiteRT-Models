package com.litertzoo.npubench

import android.content.Context
import android.os.PowerManager
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/** One accelerator's timings for a single model. */
data class BenchResult(
  val accelerator: String,
  val asset: String,
  val loadMs: Double,
  val medianMs: Double,
  val minMs: Double,
  val maxMs: Double,
  val runs: Int,
  val thermalBefore: String,
  val thermalAfter: String,
  val headroomBefore: Float,
  val headroomAfter: Float,
)

/**
 * Runs a LiteRT model on a chosen accelerator and reports latency.
 *
 * The NPU path needs the dispatch library directory. LiteRT does not default it: an
 * unset value only logs a warning and the run silently continues without NPU, so the
 * directory is always passed explicitly here.
 */
class NpuBenchmark(private val context: Context) {

  private val libDir: String
    get() = context.applicationInfo.nativeLibraryDir

  fun listNativeLibs(): List<String> =
    File(libDir).listFiles()?.map { it.name }?.sorted() ?: emptyList()

  /**
   * Continuous thermal headroom, where 1.0 is the throttling threshold.
   *
   * The status bucket below is device-level mitigation and stays NONE through warming
   * that still slows a rail, so headroom is the value that decides whether thermal is
   * involved. It is rate limited to about once a second and returns NaN when called
   * faster than that or on a device that does not support it — NaN is reported as NaN,
   * never as 0.0, which would read as "no headroom left".
   */
  fun thermalHeadroom(forecastSeconds: Int = 0): Float {
    val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    return try {
      pm.getThermalHeadroom(forecastSeconds)
    } catch (e: Exception) {
      Float.NaN
    }
  }

  /**
   * Device thermal state, so throttling is observed rather than inferred. A stable min
   * with a worse median is also the signature of intermittent throttling, which timings
   * alone cannot separate from other causes.
   */
  fun thermalStatus(): String {
    val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    return when (val s = pm.currentThermalStatus) {
      PowerManager.THERMAL_STATUS_NONE -> "NONE"
      PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
      PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
      PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
      PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
      PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
      PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
      else -> "UNKNOWN($s)"
    }
  }

  fun runPath(
    modelPath: String,
    accelerator: Accelerator,
    warmup: Int = 5,
    iterations: Int = 50,
  ): BenchResult = runInternal(modelPath, accelerator, warmup, iterations, fromPath = true)

  fun run(
    assetName: String,
    accelerator: Accelerator,
    warmup: Int = 5,
    iterations: Int = 50,
  ): BenchResult = runInternal(assetName, accelerator, warmup, iterations, fromPath = false)

  private fun runInternal(
    assetName: String,
    accelerator: Accelerator,
    warmup: Int,
    iterations: Int,
    fromPath: Boolean,
  ): BenchResult {
    val thermalBefore = thermalStatus()
    val headroomBefore = thermalHeadroom()
    Log.i(TAG, "nativeLibraryDir=$libDir thermalBefore=$thermalBefore headroomBefore=$headroomBefore")

    // DispatchLibraryDir also becomes ADSP_LIBRARY_PATH inside LiteRT's QNN manager,
    // which is how the Hexagon skel next to it gets found. CompilerPluginLibraryDir is a
    // separate option and is what an un-compiled model needs: without it LiteRT cannot
    // find libLiteRtCompilerPlugin_Qualcomm.so, and a stock model asked for on the NPU
    // lands on XNNPACK instead — with no error, just a CPU-speed number.
    val envOptions =
      mapOf(
        Environment.Option.DispatchLibraryDir to libDir,
        Environment.Option.CompilerPluginLibraryDir to libDir,
      )

    Environment.create(context, envOptions).use { env ->
      Log.i(TAG, "availableAccelerators=${env.getAvailableAccelerators()}")

      val options = CompiledModel.Options(accelerator)
      if (accelerator == Accelerator.NPU) {
        // Without this the dispatch API logs "Null Qualcomm options" and the HTP runs
        // at its default clock, which costs roughly an order of magnitude.
        options.qualcommOptions =
          CompiledModel.QualcommOptions(
            htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
          )
      }

      val loadStart = System.nanoTime()
      val model =
        if (fromPath) CompiledModel.create(assetName, options, env)
        else CompiledModel.create(context.assets, assetName, options, env)
      val loadMs = (System.nanoTime() - loadStart) / 1e6

      model.use {
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()

        repeat(warmup) {
          model.run(inputs, outputs)
          outputs.forEach { buf -> buf.readFloat() }
        }

        val samples = DoubleArray(iterations)
        for (i in 0 until iterations) {
          val t0 = System.nanoTime()
          model.run(inputs, outputs)
          // run() only enqueues; reading every output forces the compute to finish.
          outputs.forEach { buf -> buf.readFloat() }
          samples[i] = (System.nanoTime() - t0) / 1e6
        }
        samples.sort()

        val result =
          BenchResult(
            accelerator = accelerator.name,
            asset = assetName,
            loadMs = loadMs,
            medianMs = samples[iterations / 2],
            minMs = samples.first(),
            maxMs = samples.last(),
            runs = iterations,
            thermalBefore = thermalBefore,
            thermalAfter = thermalStatus(),
            headroomBefore = headroomBefore,
            // getThermalHeadroom is rate limited to about once a second and returns NaN
            // if called sooner. A short model finishes 50 iterations well inside that
            // window, so wait past the limit rather than record an unusable NaN.
            headroomAfter = run { Thread.sleep(1100); thermalHeadroom() },
          )
        Log.i(TAG, "RESULT $result")
        return result
      }
    }
  }

  companion object {
    const val TAG = "NpuBench"
  }
}
