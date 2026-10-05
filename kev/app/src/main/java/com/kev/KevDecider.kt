package com.kev

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher

/**
 * One compiled Kev-0.8B row-prefill graph on LiteRT `CompiledModel`: `ids` int32 `[1, L]` and
 * `valid` float32 `[1, L]` in, `hidden` float32 `[1, L, 1024]` (after the final RMSNorm) out,
 * signature `serving_default`. The three buffers are created once and reused for every row.
 *
 * GPU always requests an explicit precision ([KevPrecision]): the default GPU precision computes in
 * float16, which gave non-finite hidden states on part of the rows with the earlier kernel and
 * missed the parity bar on a desktop GPU with the rewritten one. CPU runs four threads. Create, run
 * and close only on [KevRuntime.dispatcher].
 */
class KevDecider
private constructor(
  val backend: Backend,
  /** The GPU precision the graph was compiled with (CPU ignores it). */
  val precision: KevPrecision,
  override val length: Int,
  val file: File,
  /** Wall time of `CompiledModel.create` (graph load and compilation), in milliseconds. */
  val compileMs: Double,
  /** GPU's error when GPU was requested and this graph was compiled on CPU instead. */
  val gpuFailure: String?,
  private val model: CompiledModel,
  private val ids: TensorBuffer,
  private val valid: TensorBuffer,
  private val hidden: TensorBuffer,
) : RowRunner, Closeable {
  /** The two execution policies; GPU never runs at default precision. */
  enum class Backend(val accelerator: Accelerator) {
    GPU(Accelerator.GPU),
    CPU(Accelerator.CPU),
  }

  private val inputs = mapOf(IDS to ids, VALID to valid)
  private val outputs = mapOf(HIDDEN to hidden)
  private var closed = false

  val isClosed: Boolean
    get() = closed

  /** Writes both inputs, runs the graph and reads `hidden` back (L × 1024 floats). */
  override fun run(ids: IntArray, valid: FloatArray): FloatArray {
    check(!closed) { "The graph is closed" }
    require(ids.size == length && valid.size == length) { "Inputs must have $length entries" }
    this.ids.writeInt(ids)
    this.valid.writeFloat(valid)
    model.run(inputs, outputs, SIGNATURE)
    // run() only enqueues the work on the GPU; reading the output waits for it.
    val values = hidden.readFloat()
    check(values.size == length * KevPointerHead.HIDDEN_SIZE) {
      "hidden has ${values.size} values, expected ${length * KevPointerHead.HIDDEN_SIZE}"
    }
    return values
  }

  override fun close() {
    if (closed) return
    closed = true
    try {
      listOf(ids, valid, hidden).forEach { it.close() }
    } finally {
      model.close()
    }
  }

  companion object {
    /** LiteRT version this sample is built and measured with. */
    const val LITERT_VERSION = "2.2.0"

    /** Threads of the CPU backend. */
    const val CPU_THREADS = 4

    private const val SIGNATURE = "serving_default"
    private const val IDS = "ids"
    private const val VALID = "valid"
    private const val HIDDEN = "hidden"

    /**
     * Compiles the [window] graph from `files/` on [backend] ([precision] on GPU). With
     * [cpuFallback], a GPU failure is kept as [gpuFailure] and the graph is compiled on CPU
     * instead.
     */
    fun create(
      context: Context,
      window: Int,
      backend: Backend,
      precision: KevPrecision,
      cpuFallback: Boolean,
    ): KevDecider {
      val file = File(context.filesDir, KevFiles.graph(window))
      check(file.isFile) { "Missing ${file.name}" }
      if (backend == Backend.CPU || !cpuFallback)
        return compile(context, file, window, backend, precision, null)
      return try {
        compile(context, file, window, Backend.GPU, precision, null)
      } catch (failure: Exception) {
        compile(context, file, window, Backend.CPU, precision, describe(failure))
      } catch (failure: LinkageError) {
        compile(context, file, window, Backend.CPU, precision, describe(failure))
      }
    }

    fun describe(failure: Throwable): String =
      "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}"

    /**
     * The options of one graph: GPU with [precision] and, for a graph of several signatures,
     * [shareConstants] (true: the weights held once for all signatures; null: LiteRT's default), or
     * CPU with [CPU_THREADS] threads.
     */
    fun options(
      backend: Backend,
      precision: KevPrecision,
      shareConstants: Boolean? = null,
    ): CompiledModel.Options =
      CompiledModel.Options(backend.accelerator).apply {
        when (backend) {
          Backend.GPU ->
            gpuOptions =
              CompiledModel.GpuOptions(
                constantTensorSharing = shareConstants,
                precision =
                  when (precision) {
                    KevPrecision.FP32 -> CompiledModel.GpuOptions.Precision.FP32
                    KevPrecision.FP16_FP32_ACCUM ->
                      CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                  },
              )
          Backend.CPU -> cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
        }
      }

    /** Throws when a graph tensor is not [element] with [shape]. */
    fun checkTensor(
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
      window: Int,
      backend: Backend,
      precision: KevPrecision,
      gpuFailure: String?,
    ): KevDecider {
      val options = options(backend, precision)
      val start = System.nanoTime()
      val model = CompiledModel.create(file.absolutePath, options, KevRuntime.environment(context))
      val compileMs = KevPipeline.millis(System.nanoTime() - start)
      val buffers = ArrayList<TensorBuffer>()
      try {
        checkTensor(
          model.getInputTensorType(IDS, SIGNATURE),
          TensorType.ElementType.INT,
          listOf(1, window),
          IDS,
        )
        checkTensor(
          model.getInputTensorType(VALID, SIGNATURE),
          TensorType.ElementType.FLOAT,
          listOf(1, window),
          VALID,
        )
        checkTensor(
          model.getOutputTensorType(HIDDEN, SIGNATURE),
          TensorType.ElementType.FLOAT,
          listOf(1, window, KevPointerHead.HIDDEN_SIZE),
          HIDDEN,
        )
        val ids = model.createInputBuffer(IDS, SIGNATURE).also { buffers.add(it) }
        val valid = model.createInputBuffer(VALID, SIGNATURE).also { buffers.add(it) }
        val hidden = model.createOutputBuffer(HIDDEN, SIGNATURE).also { buffers.add(it) }
        return KevDecider(
          backend,
          precision,
          window,
          file,
          compileMs,
          gpuFailure,
          model,
          ids,
          valid,
          hidden,
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
 * The GPU precision of the graphs ([KevDecider], [KevPairDecider]); CPU ignores it. [wireName] is
 * the launch extra's value.
 */
enum class KevPrecision(val wireName: String) {
  /** float32 storage and arithmetic. */
  FP32("fp32"),

  /** float16 storage with float32 accumulation (LiteRT `FP16_WITH_FP32_ACCUM`). */
  FP16_FP32_ACCUM("fp16acc");

  companion object {
    /**
     * The graphs whose gate in this app passes the bar at [FP16_FP32_ACCUM] on the Galaxy S26 with
     * the model repository's files; the other graphs run at [FP32] by default.
     */
    val FP16_FP32_ACCUM_GRAPHS: Set<KevGraphKey> =
      (KevFiles.WINDOWS.map { KevGraphKey.Window(it) } +
          KevFiles.PAIRS.map { KevGraphKey.Pair(it) })
        .toSet()

    /**
     * The precision [graph] compiles with when the launch names none: [FP16_FP32_ACCUM] for the
     * graphs of [FP16_FP32_ACCUM_GRAPHS], [FP32] for the others and for a file of [fileBytes] that
     * is a pre-rewrite size ([KevFiles.PRE_REWRITE_BYTES]).
     */
    fun defaultFor(graph: KevGraphKey, fileBytes: Long): KevPrecision =
      if (graph in FP16_FP32_ACCUM_GRAPHS && KevFiles.PRE_REWRITE_BYTES[graph] != fileBytes) {
        FP16_FP32_ACCUM
      } else {
        FP32
      }

    /**
     * The one precision of [precisions] (the graphs a line names); for none, [forced] (the launch's
     * precision, or null); null when they differ.
     */
    fun common(precisions: List<KevPrecision>, forced: KevPrecision?): KevPrecision? =
      if (precisions.isEmpty()) forced else precisions.distinct().singleOrNull()
  }
}

/**
 * One LiteRT Environment and one thread for the whole process. Every CompiledModel call (create,
 * run, close) runs on [dispatcher]; the Environment outlives Activity and ViewModel instances.
 */
object KevRuntime {
  private val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "Kev-LiteRT").apply { isDaemon = true }
  }

  /** The single worker thread as a coroutine dispatcher. */
  val dispatcher: CoroutineDispatcher = executor.asCoroutineDispatcher()

  private var environment: Environment? = null

  /**
   * The process Environment; `Environment.create(context)` also gives it the app's cache directory.
   */
  fun environment(context: Context): Environment =
    environment ?: Environment.create(context.applicationContext).also { environment = it }
}
