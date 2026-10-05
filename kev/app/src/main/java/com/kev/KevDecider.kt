package com.kev

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

/**
 * One compiled Kev-0.8B row-prefill graph on LiteRT `CompiledModel`: `ids` int32 `[1, L]` and
 * `valid` float32 `[1, L]` in, `hidden` float32 `[1, L, 1024]` (after the final RMSNorm) out,
 * signature `serving_default`. The three buffers are created once and reused for every row.
 *
 * GPU always requests an explicit precision ([KevPrecision]): the default GPU precision computes in
 * float16, which gave non-finite hidden states on part of the rows with the earlier kernel and
 * missed the parity bar on a desktop GPU with the rewritten one. NPU runs the graph on the Qualcomm
 * HTP with the CPU for the one op the HTP does not take; CPU runs four threads. Create, run and
 * close only on [KevRuntime.dispatcher].
 */
class KevDecider
private constructor(
  /** Where the graph was compiled (after a fallback, the backend it fell back to). */
  val backend: Backend,
  /** The GPU precision the graph was compiled with (NPU and CPU ignore it). */
  val precision: KevPrecision,
  override val length: Int,
  val file: File,
  /** Wall time of `CompiledModel.create` (graph load and compilation), in milliseconds. */
  val compileMs: Double,
  /** GPU's error when GPU was requested and this graph was compiled on CPU instead. */
  val gpuFailure: String?,
  /** NPU's error when NPU was requested and this graph was compiled on GPU instead. */
  val npuFailure: String?,
  /**
   * What the NPU compile did (the NPU backend only): cache state, the log lines, the cache files.
   */
  val npu: KevNpuCompile?,
  private val model: CompiledModel,
  private val ids: TensorBuffer,
  private val valid: TensorBuffer,
  private val hidden: TensorBuffer,
) : RowRunner, Closeable {
  /**
   * Where the graph runs. GPU never runs at its default precision; NPU is the Qualcomm HTP together
   * with the CPU ([accelerators]).
   */
  enum class Backend(val wireName: String) {
    GPU("gpu"),
    NPU("npu"),
    CPU("cpu");

    /**
     * The accelerators of the graph's options. On the NPU the int8 embedding lookup is the one op
     * the HTP does not take, so it runs on the CPU; LiteRT 2.2.0 does not compile these graphs for
     * the NPU alone.
     */
    val accelerators: List<Accelerator>
      get() =
        when (this) {
          GPU -> listOf(Accelerator.GPU)
          NPU -> listOf(Accelerator.NPU, Accelerator.CPU)
          CPU -> listOf(Accelerator.CPU)
        }

    companion object {
      /** The `backend` extra: `gpu`, `npu` or `cpu`; null for anything else. */
      fun of(name: String): Backend? = entries.firstOrNull {
        it.wireName == name.trim().lowercase(Locale.ROOT)
      }

      /**
       * The backend [graph] compiles on when [requested] is chosen: with the NPU, the row windows
       * of [KevFiles.NPU_WINDOWS] run on it and every other graph on the GPU.
       */
      fun forGraph(requested: Backend, graph: KevGraphKey): Backend =
        if (
          requested == NPU && !(graph is KevGraphKey.Window && graph.window in KevFiles.NPU_WINDOWS)
        ) {
          GPU
        } else {
          requested
        }
    }
  }

  /**
   * Where the graph ran: [backend], except an NPU graph whose log shows that the HTP did not take
   * it (LiteRT then runs it on the CPU without an error).
   */
  val ranOn: Backend
    get() = if (backend == Backend.NPU && npu?.evidence?.applied == false) Backend.CPU else backend

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
     * Compiles the [window] graph from `files/` on [backend] ([precision] on GPU; [npuOptions] when
     * the APK carries the NPU libraries). With [fallback], an NPU failure is kept as [npuFailure]
     * and the graph compiled on GPU, a GPU failure as [gpuFailure] and the graph compiled on CPU.
     */
    fun create(
      context: Context,
      window: Int,
      backend: Backend,
      precision: KevPrecision,
      fallback: Boolean,
      npuOptions: KevNpuOptions?,
    ): KevDecider {
      val file = File(context.filesDir, KevFiles.graph(window))
      check(file.isFile) { "Missing ${file.name}" }
      fun on(target: Backend, gpuFailure: String?, npuFailure: String?) =
        compile(context, file, window, target, precision, gpuFailure, npuFailure, npuOptions)
      if (backend == Backend.CPU || !fallback) return on(backend, null, null)
      var npuFailure: String? = null
      if (backend == Backend.NPU) {
        try {
          return on(Backend.NPU, null, null)
        } catch (failure: Exception) {
          npuFailure = describe(failure)
        } catch (failure: LinkageError) {
          npuFailure = describe(failure)
        }
      }
      return try {
        on(Backend.GPU, null, npuFailure)
      } catch (failure: Exception) {
        on(Backend.CPU, describe(failure), npuFailure)
      } catch (failure: LinkageError) {
        on(Backend.CPU, describe(failure), npuFailure)
      }
    }

    fun describe(failure: Throwable): String =
      "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}"

    /**
     * The options of one graph: GPU with [precision] and, for a graph of several signatures,
     * [shareConstants] (true: the weights held once for all signatures; null: LiteRT's default),
     * NPU with the CPU, or CPU with [CPU_THREADS] threads; with [npu] (the APK carries the NPU
     * libraries) every graph also gets its Qualcomm options ([KevNpuOptions]).
     */
    fun options(
      backend: Backend,
      precision: KevPrecision,
      shareConstants: Boolean? = null,
      npu: KevNpuOptions? = null,
    ): CompiledModel.Options =
      CompiledModel.Options(*backend.accelerators.toTypedArray()).apply {
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
          // The HTP takes the Qualcomm options below; the CPU part runs one small op.
          Backend.NPU -> Unit
          Backend.CPU -> cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
        }
        npu?.qualcommOptions(npuGraph = backend == Backend.NPU)?.let { qualcommOptions = it }
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
      npuFailure: String?,
      npuOptions: KevNpuOptions?,
    ): KevDecider {
      val options = options(backend, precision, npu = npuOptions)
      val npuStart =
        if (backend == Backend.NPU) KevNpuCompiler.before(context, file, npuOptions) else null
      val start = System.nanoTime()
      val model = CompiledModel.create(file.absolutePath, options, KevRuntime.environment(context))
      val compileMs = KevPipeline.millis(System.nanoTime() - start)
      val npu = npuStart?.let { KevNpuCompiler.after(context, it, compileMs) }
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
          npuFailure,
          npu,
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
   * The process Environment. `Environment.create(context, …)` also gives it the app's cache
   * directory, where LiteRT keeps the NPU graphs it compiled (the JIT cache); with the NPU
   * libraries in the APK it also names their directory ([environmentOptions]).
   */
  fun environment(context: Context): Environment =
    environment
      ?: context.applicationContext
        .let { app ->
          Environment.create(
            app,
            environmentOptions(
              app.applicationInfo.nativeLibraryDir,
              KevNpu.librariesInstalled(app),
            ),
          )
        }
        .also { environment = it }

  /**
   * The Environment options: with the NPU libraries ([npuLibraries]) the dispatch library and the
   * JIT compiler plugin directories, both the APK's [nativeLibraryDir] (without the plugin a graph
   * asked for on the NPU is not compiled for it); without them none, as before the NPU backend.
   */
  fun environmentOptions(
    nativeLibraryDir: String,
    npuLibraries: Boolean,
  ): Map<Environment.Option, String> =
    if (npuLibraries) {
      mapOf(
        Environment.Option.DispatchLibraryDir to nativeLibraryDir,
        Environment.Option.CompilerPluginLibraryDir to nativeLibraryDir,
      )
    } else {
      emptyMap()
    }
}
