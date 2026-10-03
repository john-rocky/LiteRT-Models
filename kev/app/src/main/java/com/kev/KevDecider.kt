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
 * One resident Kev-0.8B row-prefill graph on LiteRT `CompiledModel`: `ids` int32 `[1, L]` and
 * `valid` float32 `[1, L]` in, `hidden` float32 `[1, L, 1024]` (after the final RMSNorm) out,
 * signature `serving_default`. The three buffers are created once and reused for every row.
 *
 * GPU always requests FP32 precision: the default GPU precision runs the graph in float16, which
 * gives non-finite hidden states on part of the rows. CPU runs four threads. Create, run and close
 * only on [KevRuntime.dispatcher].
 */
class KevDecider
private constructor(
  val backend: Backend,
  override val length: Int,
  val file: File,
  /** Wall time of `CompiledModel.create` (graph load and compilation), in milliseconds. */
  val compileMs: Double,
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

  /** A created graph and, when GPU failed and CPU took over, GPU's error. */
  class Created(val decider: KevDecider, val gpuFailure: String?)

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
     * Compiles the [window] graph from `files/` on [backend]. With [cpuFallback], a GPU failure is
     * kept as [Created.gpuFailure] and the graph is compiled on CPU instead.
     */
    fun create(context: Context, window: Int, backend: Backend, cpuFallback: Boolean): Created {
      val file = File(context.filesDir, KevFiles.graph(window))
      check(file.isFile) { "Missing ${file.name}" }
      if (backend == Backend.CPU || !cpuFallback) {
        return Created(compile(context, file, window, backend), null)
      }
      return try {
        Created(compile(context, file, window, Backend.GPU), null)
      } catch (failure: Exception) {
        Created(compile(context, file, window, Backend.CPU), describe(failure))
      } catch (failure: LinkageError) {
        Created(compile(context, file, window, Backend.CPU), describe(failure))
      }
    }

    fun describe(failure: Throwable): String = "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}"

    private fun compile(context: Context, file: File, window: Int, backend: Backend): KevDecider {
      val options =
        CompiledModel.Options(backend.accelerator).apply {
          when (backend) {
            Backend.GPU ->
              gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
            Backend.CPU -> cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
          }
        }
      val start = System.nanoTime()
      val model = CompiledModel.create(file.absolutePath, options, KevRuntime.environment(context))
      val compileMs = KevPipeline.millis(System.nanoTime() - start)
      val buffers = ArrayList<TensorBuffer>()
      try {
        checkTensor(model.getInputTensorType(IDS, SIGNATURE), TensorType.ElementType.INT, listOf(1, window), IDS)
        checkTensor(model.getInputTensorType(VALID, SIGNATURE), TensorType.ElementType.FLOAT, listOf(1, window), VALID)
        checkTensor(
          model.getOutputTensorType(HIDDEN, SIGNATURE),
          TensorType.ElementType.FLOAT,
          listOf(1, window, KevPointerHead.HIDDEN_SIZE),
          HIDDEN,
        )
        val ids = model.createInputBuffer(IDS, SIGNATURE).also { buffers.add(it) }
        val valid = model.createInputBuffer(VALID, SIGNATURE).also { buffers.add(it) }
        val hidden = model.createOutputBuffer(HIDDEN, SIGNATURE).also { buffers.add(it) }
        return KevDecider(backend, window, file, compileMs, model, ids, valid, hidden)
      } catch (failure: Throwable) {
        buffers.forEach { it.close() }
        model.close()
        throw failure
      }
    }

    private fun checkTensor(type: TensorType, element: TensorType.ElementType, shape: List<Int>, name: String) {
      val dimensions = type.layout?.dimensions.orEmpty()
      check(type.elementType == element && dimensions == shape) {
        "Graph tensor $name is ${type.elementType} $dimensions, expected $element $shape"
      }
    }
  }
}

/**
 * One LiteRT Environment and one thread for the whole process. Every CompiledModel call (create,
 * run, close) runs on [dispatcher]; the Environment outlives Activity and ViewModel instances.
 */
object KevRuntime {
  private val executor =
    Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "Kev-LiteRT").apply { isDaemon = true } }

  /** The single worker thread as a coroutine dispatcher. */
  val dispatcher: CoroutineDispatcher = executor.asCoroutineDispatcher()

  private var environment: Environment? = null

  /** The process Environment; `Environment.create(context)` also gives it the app's cache directory. */
  fun environment(context: Context): Environment =
    environment ?: Environment.create(context.applicationContext).also { environment = it }
}
