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

  /**
   * Parity + latency for one named signature with a REAL input tensor.
   *
   * Reads a raw little-endian float32 file into input buffer 0, times
   * warmup+iterations runs, then reads every output once more and dumps each to
   * `outDir/out<i>.bin` (raw LE), so the host can compare bytes against its own
   * reference. `outKinds[i]` picks the read call per output: "i" = int32, "f" =
   * float32 — TensorBuffer has no type introspection, and readInt on a float
   * tensor would not fail, it would lie.
   */
  fun parityBench(
    modelPath: String,
    accelerator: Accelerator,
    signature: String,
    inputFiles: List<String>,
    outDir: File,
    outKinds: List<String>,
    warmup: Int = 5,
    iterations: Int = 20,
    gpuFp32: Boolean = false,
  ): BenchResult {
    val thermalBefore = thermalStatus()
    val headroomBefore = thermalHeadroom()

    val envOptions =
      mapOf(
        Environment.Option.DispatchLibraryDir to libDir,
        Environment.Option.CompilerPluginLibraryDir to libDir,
      )
    Environment.create(context, envOptions).use { env ->
      val options = CompiledModel.Options(accelerator)
      if (accelerator == Accelerator.GPU && gpuFp32) {
        // Default GPU compute is fp16 on most delegates; FP32 is the precision-vs-bug
        // discriminator: a wrong output that stays wrong under FP32 is not fp16 rounding.
        options.gpuOptions =
          CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
      }
      val loadStart = System.nanoTime()
      val model = CompiledModel.create(modelPath, options, env)
      val loadMs = (System.nanoTime() - loadStart) / 1e6

      model.use {
        val inputs =
          if (signature.isEmpty()) model.createInputBuffers()
          else model.createInputBuffers(signature)
        val outputs =
          if (signature.isEmpty()) model.createOutputBuffers()
          else model.createOutputBuffers(signature)

        // One raw little-endian float32 file per input, in signature input order. A
        // multi-input graph (KV-cache step graphs) needs every buffer written, not just
        // the first; a count mismatch is an error, not a silent partial write.
        require(inputFiles.size == inputs.size) {
          "model has ${inputs.size} inputs but ${inputFiles.size} input files were given"
        }
        inputFiles.forEachIndexed { i, inputFile ->
          val bytes = File(inputFile).readBytes()
          val floats = FloatArray(bytes.size / 4)
          java.nio.ByteBuffer.wrap(bytes)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()
            .get(floats)
          inputs[i].writeFloat(floats)
        }

        fun runOnce() {
          if (signature.isEmpty()) model.run(inputs, outputs)
          else model.run(inputs, outputs, signature)
        }

        repeat(warmup) { runOnce() }
        val samples = DoubleArray(iterations)
        for (i in 0 until iterations) {
          val t0 = System.nanoTime()
          runOnce()
          // Force completion by reading every output before stopping the clock
          // (same convention as runInternal: run() only enqueues).
          outputs.forEachIndexed { j, buf ->
            if (outKinds.getOrNull(j) == "i") buf.readInt() else buf.readFloat()
          }
          samples[i] = (System.nanoTime() - t0) / 1e6
        }
        samples.sort()

        outDir.mkdirs()
        outputs.forEachIndexed { i, buf ->
          val f = File(outDir, "out$i.bin")
          val bb: java.nio.ByteBuffer
          if (outKinds.getOrNull(i) == "i") {
            val v = buf.readInt()
            bb = java.nio.ByteBuffer.allocate(v.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            v.forEach { bb.putInt(it) }
          } else {
            val v = buf.readFloat()
            bb = java.nio.ByteBuffer.allocate(v.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            v.forEach { bb.putFloat(it) }
          }
          f.writeBytes(bb.array())
        }

        val result =
          BenchResult(
            accelerator = accelerator.name,
            asset = "$modelPath#$signature",
            loadMs = loadMs,
            medianMs = samples[iterations / 2],
            minMs = samples.first(),
            maxMs = samples.last(),
            runs = iterations,
            thermalBefore = thermalBefore,
            thermalAfter = thermalStatus(),
            headroomBefore = headroomBefore,
            headroomAfter = run { Thread.sleep(1100); thermalHeadroom() },
          )
        Log.i(TAG, "RESULT $result")
        return result
      }
    }
  }

  /** Process memory now: total PSS, graphics (GPU buffers) and native heap, in MB. */
  private fun memorySnapshot(): String {
    val mi = android.os.Debug.MemoryInfo()
    android.os.Debug.getMemoryInfo(mi)
    fun stat(key: String) = (mi.getMemoryStat(key)?.toLongOrNull() ?: -1L) / 1024
    return "pss=${stat("summary.total-pss")}MB graphics=${stat("summary.graphics")}MB " +
      "native=${stat("summary.native-heap")}MB"
  }

  /**
   * SmolVLA action chunk in one process with its three graphs resident, chained on the
   * host the way an app runs them: vision -> prefix -> 10 expert steps, x += -0.1 * v.
   *
   * `dir` holds raw little-endian float32 inputs: v_image, v_pos (vision); p_lang, p_state,
   * p_bias, p_cos, p_sin (prefix, after img_emb); e_bias_self, e_bias_cross, e_cos_s,
   * e_sin_s, e_cos_c, e_sin_c (expert, after x_t, time_emb, k_all, v_all); noise and
   * temb0..temb9 (all `.bin`). The prefix K/V are written into the expert's input buffers
   * once per chunk; only x_t and time_emb change per step. Returns the CHAIN log line and
   * writes the last chunk's x to outDir/x_final.bin.
   */
  fun chainBench(
    visionPath: String,
    prefixPath: String,
    expertPath: String,
    dir: File,
    accelerator: Accelerator,
    gpuFp32: Boolean,
    outDir: File,
    warmup: Int = 2,
    iterations: Int = 5,
  ): String {
    fun load(name: String): FloatArray {
      val bytes = File(dir, "$name.bin").readBytes()
      val floats = FloatArray(bytes.size / 4)
      java.nio.ByteBuffer.wrap(bytes)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        .asFloatBuffer()
        .get(floats)
      return floats
    }

    val thermalBefore = thermalStatus()
    val headroomBefore = thermalHeadroom()
    val memBefore = memorySnapshot()
    val envOptions =
      mapOf(
        Environment.Option.DispatchLibraryDir to libDir,
        Environment.Option.CompilerPluginLibraryDir to libDir,
      )
    Environment.create(context, envOptions).use { env ->
      fun options() =
        CompiledModel.Options(accelerator).also {
          if (accelerator == Accelerator.GPU && gpuFp32) {
            it.gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          }
        }
      val loadStart = System.nanoTime()
      val vision = CompiledModel.create(visionPath, options(), env)
      val prefix = CompiledModel.create(prefixPath, options(), env)
      val expert = CompiledModel.create(expertPath, options(), env)
      val loadMs = (System.nanoTime() - loadStart) / 1e6
      val memLoaded = memorySnapshot()

      vision.use {
        prefix.use {
          expert.use {
            val vIn = vision.createInputBuffers()
            val vOut = vision.createOutputBuffers()
            val pIn = prefix.createInputBuffers()
            val pOut = prefix.createOutputBuffers()
            val eIn = expert.createInputBuffers()
            val eOut = expert.createOutputBuffers()
            vIn[0].writeFloat(load("v_image"))
            vIn[1].writeFloat(load("v_pos"))
            listOf("p_lang", "p_state", "p_bias", "p_cos", "p_sin").forEachIndexed { i, n ->
              pIn[i + 1].writeFloat(load(n))
            }
            listOf("e_bias_self", "e_bias_cross", "e_cos_s", "e_sin_s", "e_cos_c", "e_sin_c")
              .forEachIndexed { i, n -> eIn[i + 4].writeFloat(load(n)) }
            val noise = load("noise")
            val tembs = (0 until 10).map { load("temb$it") }
            val dt = -0.1f

            val stageMs = DoubleArray(3)
            fun chunk(): FloatArray {
              var t = System.nanoTime()
              vision.run(vIn, vOut)
              val imgEmb = vOut[0].readFloat()
              stageMs[0] = (System.nanoTime() - t) / 1e6
              t = System.nanoTime()
              pIn[0].writeFloat(imgEmb)
              prefix.run(pIn, pOut)
              eIn[2].writeFloat(pOut[0].readFloat())
              eIn[3].writeFloat(pOut[1].readFloat())
              stageMs[1] = (System.nanoTime() - t) / 1e6
              t = System.nanoTime()
              val x = noise.copyOf()
              for (step in 0 until 10) {
                eIn[0].writeFloat(x)
                eIn[1].writeFloat(tembs[step])
                expert.run(eIn, eOut)
                val v = eOut[0].readFloat()
                for (i in x.indices) x[i] = x[i] + dt * v[i]
              }
              stageMs[2] = (System.nanoTime() - t) / 1e6
              return x
            }

            repeat(warmup) { chunk() }
            val total = DoubleArray(iterations)
            val perStage = Array(3) { DoubleArray(iterations) }
            var x = FloatArray(0)
            for (i in 0 until iterations) {
              val t0 = System.nanoTime()
              x = chunk()
              total[i] = (System.nanoTime() - t0) / 1e6
              for (s in 0 until 3) perStage[s][i] = stageMs[s]
            }
            val memRun = memorySnapshot()
            outDir.mkdirs()
            val bb = java.nio.ByteBuffer.allocate(x.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            x.forEach { bb.putFloat(it) }
            File(outDir, "x_final.bin").writeBytes(bb.array())

            fun median(a: DoubleArray) = a.sorted()[a.size / 2]
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            val sys = android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
            val line =
              "CHAIN ${accelerator.name}${if (gpuFp32) "/FP32" else ""} " +
                "chunk median=${"%.1f".format(median(total))}ms min=${"%.1f".format(total.min())}ms " +
                "vision=${"%.1f".format(median(perStage[0]))}ms " +
                "prefix=${"%.1f".format(median(perStage[1]))}ms " +
                "expert10=${"%.1f".format(median(perStage[2]))}ms load=${"%.0f".format(loadMs)}ms " +
                "runs=$iterations | mem before [$memBefore] loaded [$memLoaded] running [$memRun] " +
                "| device RAM total=${sys.totalMem / (1 shl 20)}MB avail=${sys.availMem / (1 shl 20)}MB " +
                "| thermal=$thermalBefore->${thermalStatus()} headroom=$headroomBefore->" +
                "${run { Thread.sleep(1100); thermalHeadroom() }} outdir=${outDir.absolutePath}"
            Log.i(TAG, line)
            return line
          }
        }
      }
    }
  }

  companion object {
    const val TAG = "NpuBench"
  }
}
