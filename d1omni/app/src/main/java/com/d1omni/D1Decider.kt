package com.d1omni

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.Closeable
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher

/** Where a graph runs. */
enum class D1Backend(val wireName: String) {
  GPU("gpu"),
  CPU("cpu");

  companion object {
    fun of(name: String): D1Backend? =
      entries.firstOrNull { it.wireName == name.trim().lowercase(Locale.ROOT) }
  }
}

/**
 * The GPU precision of a graph. [FP32] is the default: on the Galaxy S26 it gives the provider's
 * answers within the parity bar on every public text row (253). [FP16_FP32_ACCUM] (float16 storage,
 * float32 accumulation) is about 10 % faster per call but moves one of those rows past the bar
 * (max |Δp| 0.0400 against 0.02); a launch extra selects it. The delegate's default precision
 * (float16 activations) is never used.
 */
enum class D1Precision(val wireName: String) {
  FP16_FP32_ACCUM("fp16acc"),
  FP32("fp32");

  companion object {
    /** The precision a launch without the `precision` extra uses. */
    val DEFAULT = FP32

    fun of(name: String): D1Precision? =
      entries.firstOrNull { it.wireName == name.trim().lowercase(Locale.ROOT) }
  }
}

/** One graph call: its [scores] and the wall time of each part, in nanoseconds. */
class D1Call(
  val scores: FloatArray,
  val writeNanos: Long,
  val runNanos: Long,
  val readNanos: Long,
) {
  /** Input writes + `run()` + output read-back: the number a call shows, in milliseconds. */
  val totalMs: Double
    get() = (writeNanos + runNanos + readNanos) / NANOS_PER_MS

  val writeMs: Double
    get() = writeNanos / NANOS_PER_MS

  /** `run()` alone: on the GPU it returns before the work ends; reading the output waits. */
  val runMs: Double
    get() = runNanos / NANOS_PER_MS

  val readMs: Double
    get() = readNanos / NANOS_PER_MS

  private companion object {
    const val NANOS_PER_MS = 1e6
  }
}

/**
 * One compiled decision graph `decide_<L>` on LiteRT `CompiledModel` (one file, one signature): six
 * inputs (`ids` int32 [1, L], `prefix` float32 [1, L, 1024], `media` / `pad` / `keep_right` float32
 * [1, L], `qtype_onehot` float32 [1, 3]) and `scores` float32 [1, L]. The seven buffers are created
 * once and reused for every row. Create, run and close only on [D1Runtime.dispatcher].
 */
class D1Decider
private constructor(
  /** Where the graph was compiled (the CPU after a GPU failure). */
  val backend: D1Backend,
  /** The GPU precision it was compiled with (the CPU ignores it). */
  val precision: D1Precision,
  val length: Int,
  val file: File,
  /** Wall time of `CompiledModel.create` (load and compile), in milliseconds. */
  val compileMs: Double,
  /** The GPU's error when the GPU was asked for and the graph runs on the CPU instead. */
  val gpuFailure: String?,
  private val model: CompiledModel,
  private val inputs: Map<String, TensorBuffer>,
  private val scores: TensorBuffer,
) : Closeable {
  private val signature = signatureOf(length)
  private val outputs = mapOf(OUTPUT to scores)
  // Every text row's prefix input: L x 1024 zeros, written as they are.
  private val zeroPrefix = FloatArray(length * D1Rows.PREFIX_WIDTH)
  private var closed = false

  val isClosed: Boolean
    get() = closed

  /** Writes the six inputs, runs the graph and reads `scores` back ([length] floats). */
  fun run(row: D1Inputs): D1Call {
    check(!closed) { "The L$length graph is closed" }
    require(row.length == length) { "a row for L = ${row.length} on the L$length graph" }
    val start = System.nanoTime()
    inputs.getValue(IDS).writeInt(row.ids)
    inputs.getValue(PREFIX).writeFloat(row.prefix ?: zeroPrefix)
    inputs.getValue(MEDIA).writeFloat(row.media)
    inputs.getValue(PAD).writeFloat(row.pad)
    inputs.getValue(KEEP_RIGHT).writeFloat(row.keepRight)
    inputs.getValue(QTYPE).writeFloat(row.qtypeOneHot)
    val written = System.nanoTime()
    model.run(inputs, outputs, signature)
    val ran = System.nanoTime()
    // run() only enqueues the work on the GPU; reading the output waits for it.
    val values = scores.readFloat()
    val read = System.nanoTime()
    check(values.size == length) { "scores has ${values.size} values, expected $length" }
    return D1Call(values, written - start, ran - written, read - ran)
  }

  override fun close() {
    if (closed) return
    closed = true
    try {
      (inputs.values + scores).forEach { it.close() }
    } finally {
      model.close()
    }
  }

  companion object {
    /** LiteRT version this sample is built and measured with. */
    const val LITERT_VERSION = "2.2.0"

    /** Threads of the CPU backend and of the fallback. */
    const val CPU_THREADS = 4

    private const val IDS = "ids"
    private const val PREFIX = "prefix"
    private const val MEDIA = "media"
    private const val PAD = "pad"
    private const val KEEP_RIGHT = "keep_right"
    private const val QTYPE = "qtype_onehot"
    private const val OUTPUT = "scores"

    /** The six inputs in the graph's tensor order. */
    val INPUTS = listOf(IDS, PREFIX, MEDIA, PAD, KEEP_RIGHT, QTYPE)

    fun signatureOf(length: Int) = "decide_$length"

    /**
     * Compiles [file] (`decide_<length>`) on [backend] at [precision]. With [fallback], a GPU
     * failure is kept as [gpuFailure] and the graph compiled on the CPU (four threads) instead.
     */
    fun create(
      context: Context,
      file: File,
      length: Int,
      backend: D1Backend,
      precision: D1Precision,
      fallback: Boolean,
    ): D1Decider {
      check(file.isFile) { "Missing ${file.name}" }
      if (backend == D1Backend.CPU || !fallback) {
        return compile(context, file, length, backend, precision, null)
      }
      return try {
        compile(context, file, length, D1Backend.GPU, precision, null)
      } catch (failure: Exception) {
        compile(context, file, length, D1Backend.CPU, precision, describe(failure))
      } catch (failure: LinkageError) {
        compile(context, file, length, D1Backend.CPU, precision, describe(failure))
      }
    }

    fun describe(failure: Throwable): String =
      "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}"

    /**
     * The options of one graph: GPU at [precision] (no constant tensor sharing: one signature,
     * nothing to share), or CPU with [CPU_THREADS] threads.
     */
    fun options(backend: D1Backend, precision: D1Precision): CompiledModel.Options =
      when (backend) {
        D1Backend.GPU ->
          CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions =
              CompiledModel.GpuOptions(
                precision =
                  when (precision) {
                    D1Precision.FP32 -> CompiledModel.GpuOptions.Precision.FP32
                    D1Precision.FP16_FP32_ACCUM ->
                      CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                  }
              )
          }
        D1Backend.CPU ->
          CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
          }
      }

    /** Throws when a graph tensor is not [element] with [shape]. */
    private fun checkTensor(
      type: TensorType,
      element: TensorType.ElementType,
      shape: List<Int>,
      name: String,
    ) {
      val dimensions = type.layout?.dimensions.orEmpty()
      check(type.elementType == element && dimensions == shape) {
        "Graph tensor $name is ${type.elementType} $dimensions, expected $element $shape"
      }
    }

    private fun compile(
      context: Context,
      file: File,
      length: Int,
      backend: D1Backend,
      precision: D1Precision,
      gpuFailure: String?,
    ): D1Decider {
      val signature = signatureOf(length)
      val start = System.nanoTime()
      val model =
        CompiledModel.create(
          file.absolutePath,
          options(backend, precision),
          D1Runtime.environment(context),
        )
      val compileMs = (System.nanoTime() - start) / 1e6
      val buffers = ArrayList<TensorBuffer>()
      try {
        val float = TensorType.ElementType.FLOAT
        checkTensor(
          model.getInputTensorType(IDS, signature),
          TensorType.ElementType.INT,
          listOf(1, length),
          IDS,
        )
        checkTensor(
          model.getInputTensorType(PREFIX, signature),
          float,
          listOf(1, length, D1Rows.PREFIX_WIDTH),
          PREFIX,
        )
        for (name in listOf(MEDIA, PAD, KEEP_RIGHT)) {
          checkTensor(model.getInputTensorType(name, signature), float, listOf(1, length), name)
        }
        checkTensor(model.getInputTensorType(QTYPE, signature), float, listOf(1, 3), QTYPE)
        checkTensor(model.getOutputTensorType(OUTPUT, signature), float, listOf(1, length), OUTPUT)
        val inputs = LinkedHashMap<String, TensorBuffer>()
        for (name in INPUTS) {
          inputs[name] = model.createInputBuffer(name, signature).also { buffers.add(it) }
        }
        val scores = model.createOutputBuffer(OUTPUT, signature).also { buffers.add(it) }
        return D1Decider(
          backend,
          precision,
          length,
          file,
          compileMs,
          gpuFailure,
          model,
          inputs,
          scores,
        )
      } catch (failure: Throwable) {
        buffers.forEach { it.close() }
        model.close()
        throw failure
      }
    }
  }
}

/**
 * One LiteRT Environment and one thread for the whole process. Every CompiledModel call (create,
 * run, close) runs on [dispatcher]: the deciders reuse their native buffers, so two calls at once
 * would overwrite each other's inputs.
 */
object D1Runtime {
  private val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "D1Omni-LiteRT").apply { isDaemon = true }
  }

  /** The single worker thread as a coroutine dispatcher. */
  val dispatcher: CoroutineDispatcher = executor.asCoroutineDispatcher()

  private var environment: Environment? = null

  /** The process Environment (`Environment.create(context)`). */
  fun environment(context: Context): Environment =
    environment
      ?: Environment.create(context.applicationContext).also { environment = it }
}
