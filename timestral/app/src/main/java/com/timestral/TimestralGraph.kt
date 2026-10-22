package com.timestral

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.Closeable
import java.io.File

/** One graph call: the raw output `[R, H, 32, 21]` and the wall time of write + run + read back. */
class GraphRun(val raw: FloatArray, val ms: Double)

/** The NPU did not take a graph asked for on it; the message says what the log showed. */
class NpuNotUsedException(message: String) : Exception(message)

/**
 * One compiled Timestral graph file ([spec]) on LiteRT `CompiledModel`: five float32 inputs
 * (`patches`, `dec`, `bias`, `cos`, `sin`) and one output (`output_0`, `[R, H, 32, 21]`), signature
 * `serving_default`. The buffers are created once and reused for every call. Create, run and close
 * only on [TimestralRuntime.dispatcher].
 *
 * The GPU always gets an explicit precision; there is no CPU fallback: a graph that does not
 * compile on the chosen backend is an error the screen shows.
 */
class TimestralGraph
private constructor(
  val file: File,
  val spec: GraphSpec,
  val backend: Backend,
  /** The GPU precision the graph was compiled with (meaningful on the GPU only). */
  val precision: GpuPrecision,
  /** Wall time of `CompiledModel.create`, in milliseconds. */
  val compileMs: Double,
  /** What this process's log said about the NPU compile (the NPU backend only). */
  val npu: NpuEvidence?,
  private val model: CompiledModel,
  private val inputs: Map<String, TensorBuffer>,
  private val output: TensorBuffer,
) : Closeable {
  private var closed = false

  /** Writes the five inputs, runs the graph and reads the output back. */
  fun run(graphInputs: GraphInputs): GraphRun {
    check(!closed) { "The graph is closed" }
    val start = System.nanoTime()
    for ((name, values) in graphInputs.byName()) inputs.getValue(name).writeFloat(values)
    model.run(inputs, mapOf(OUTPUT to output), SIGNATURE)
    // run() only enqueues the work on the GPU; reading the output waits for it.
    val raw = output.readFloat()
    val ms = (System.nanoTime() - start) / 1e6
    check(raw.size == outputSize(spec)) { "output_0 has ${raw.size} values" }
    return GraphRun(raw, ms)
  }

  override fun close() {
    if (closed) return
    closed = true
    try {
      inputs.values.forEach { it.close() }
      output.close()
    } finally {
      model.close()
    }
  }

  companion object {
    private const val SIGNATURE = "serving_default"
    private const val OUTPUT = "output_0"

    /** Values of `output_0` for [spec]. */
    fun outputSize(spec: GraphSpec): Int =
      spec.rows * spec.horizonPatches * TimestralHost.PATCH * TimestralHost.QUANTILES

    /**
     * Compiles [file] ([spec]) on the backend of [config]. The NPU runs together with the CPU
     * (LiteRT 2.2.0 refuses a graph whose NPU partition leaves an op behind when the NPU is the only
     * accelerator); the app then reads the log and throws [NpuNotUsedException] when the HTP took
     * none of the graph.
     */
    fun create(
      context: Context,
      file: File,
      spec: GraphSpec,
      config: BackendConfig,
      npuLibraries: Boolean,
    ): TimestralGraph {
      val accelerators =
        when (config.backend) {
          Backend.CPU -> arrayOf(Accelerator.CPU)
          Backend.GPU -> arrayOf(Accelerator.GPU)
          Backend.NPU -> arrayOf(Accelerator.NPU, Accelerator.CPU)
        }
      val gpuPrecision = config.precisionOf(spec)
      val options =
        CompiledModel.Options(*accelerators).apply {
          when (config.backend) {
            Backend.CPU -> cpuOptions = CompiledModel.CpuOptions(numThreads = config.threads)
            Backend.GPU -> gpuOptions = CompiledModel.GpuOptions(precision = precision(gpuPrecision))
            Backend.NPU -> Unit
          }
          if (npuLibraries) qualcommOptions = NpuSupport.options()
        }
      val since = System.currentTimeMillis()
      val start = System.nanoTime()
      val model = CompiledModel.create(file.absolutePath, options, TimestralRuntime.environment(context))
      val compileMs = (System.nanoTime() - start) / 1e6
      val buffers = ArrayList<TensorBuffer>()
      try {
        val npu = if (config.backend == Backend.NPU) NpuLog.since(since) else null
        if (npu != null && npu.lines != null && npu.dispatched == null) {
          throw NpuNotUsedException(
            "The NPU did not take the graph (no DispatchDelegate line in this app's log" +
              (if (npu.cpuTookOver) "; LiteRT placed it on the CPU" else "") +
              "). The sample stops instead of running it on the CPU."
          )
        }
        val s = spec.tokens
        val expected =
          mapOf(
            "patches" to listOf(spec.rows, spec.contextPatches, 2 * TimestralHost.PATCH),
            "dec" to listOf(spec.rows, spec.horizonPatches, TimestralHost.D_MODEL),
            "bias" to listOf(1, 1, s, s),
            "cos" to listOf(1, 1, s, TimestralHost.ROPE_DIM),
            "sin" to listOf(1, 1, s, TimestralHost.ROPE_DIM),
          )
        val inputs = LinkedHashMap<String, TensorBuffer>()
        for ((name, shape) in expected) {
          checkTensor(model.getInputTensorType(name, SIGNATURE), shape, name, file)
          inputs[name] = model.createInputBuffer(name, SIGNATURE).also { buffers.add(it) }
        }
        checkTensor(
          model.getOutputTensorType(OUTPUT, SIGNATURE),
          listOf(spec.rows, spec.horizonPatches, TimestralHost.PATCH, TimestralHost.QUANTILES),
          OUTPUT,
          file,
        )
        val output = model.createOutputBuffer(OUTPUT, SIGNATURE).also { buffers.add(it) }
        return TimestralGraph(
          file,
          spec,
          config.backend,
          gpuPrecision,
          compileMs,
          npu,
          model,
          inputs,
          output,
        )
      } catch (failure: Throwable) {
        buffers.forEach { it.close() }
        model.close()
        throw failure
      }
    }

    private fun precision(precision: GpuPrecision): CompiledModel.GpuOptions.Precision =
      when (precision) {
        GpuPrecision.FP32 -> CompiledModel.GpuOptions.Precision.FP32
        GpuPrecision.FP16 -> CompiledModel.GpuOptions.Precision.FP16
        GpuPrecision.FP16_FP32_ACCUM -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
      }

    private fun checkTensor(type: TensorType, shape: List<Int>, name: String, file: File) {
      val dimensions = type.layout?.dimensions.orEmpty()
      check(type.elementType == TensorType.ElementType.FLOAT && dimensions == shape) {
        "${file.name}: tensor $name is ${type.elementType} $dimensions, expected FLOAT $shape"
      }
    }
  }
}
