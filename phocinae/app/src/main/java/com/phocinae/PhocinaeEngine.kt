// SPDX-License-Identifier: Apache-2.0
package com.phocinae

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors

/**
 * Single-row Phocinae-Largha-150M execution (the laya contract on mmBERT-small). The GPU runs at the
 * precision the caller selects (FP32 by default; FP16 is the fastest form that passed the parity bar on a
 * Galaxy S26); the NPU (Qualcomm HTP) compiles the same fp16-safe graph on the device (JIT, cached after the
 * first launch).
 */
class PhocinaeEngine(context: Context, val storage: Storage = Storage.WFP16) : Closeable {
  /** The published graphs store the FULLY_CONNECTED weights in float16 (one storage); kept as a launch selector. */
  enum class Storage(val argument: String) {
    WFP16("wfp16");

    companion object {
      /** Parses a supported intent selector, rejecting unknown values. */
      fun fromArgument(value: String): Storage =
        when (value.lowercase()) {
          "wfp16" -> WFP16
          else -> error("Unknown graph storage: $value; expected wfp16")
        }
    }
  }

  /** Selects the accelerator explicitly; GPU creation never silently falls back. */
  enum class Backend(val accelerator: Accelerator) {
    GPU(Accelerator.GPU),
    GPU_FP16(Accelerator.GPU),
    NPU(Accelerator.NPU),
    CPU(Accelerator.CPU);

    /** The intent / preference spelling. */
    val argument: String
      get() = name.lowercase()

    companion object {
      /** Parses a supported intent selector, rejecting unknown values. */
      fun fromArgument(value: String): Backend =
        when (value.lowercase()) {
          "gpu" -> GPU
          "gpu_fp16", "gpu-fp16" -> GPU_FP16
          "npu" -> NPU
          "cpu" -> CPU
          else -> error("Unknown accelerator: $value; expected gpu, gpu_fp16, npu or cpu")
        }
    }
  }

  /** Includes output readback because GPU run() can enqueue work asynchronously. */
  data class GraphTiming(val writeMs: Double, val enqueueMs: Double, val readbackMs: Double) {
    /** Total write, enqueue, and readback duration in milliseconds. */
    val totalMs: Double
      get() = writeMs + enqueueMs + readbackMs

    /** Emits named timing components for machine-readable gate reports. */
    fun toMap(): Map<String, Double> =
      linkedMapOf(
        "write_ms" to writeMs,
        "run_enqueue_ms" to enqueueMs,
        "readback_ms" to readbackMs,
        "write_run_read_ms" to totalMs,
      )
  }

  /** Raw model tensors and host/graph timings for one question row. */
  data class RawResult(
    val tokenLogits: FloatArray,
    val pooledCls: FloatArray,
    val markerLogits: FloatArray,
    val actFeatures: FloatArray,
    val actLogits: FloatArray,
    val mainTiming: GraphTiming,
    val actTiming: GraphTiming?,
    val embeddingLookupMs: Double,
    val featureMs: Double,
    val callMs: Double,
  ) {
    /** Rejects any NaN or infinity before displaying a decoded answer. */
    val finite: Boolean
      get() =
        tokenLogits.all { it.isFinite() } &&
          pooledCls.all { it.isFinite() } &&
          markerLogits.all { it.isFinite() } &&
          actFeatures.all { it.isFinite() } &&
          actLogits.all { it.isFinite() }

    /** Main plus act graph durations, excluding the separate embedding lookup. */
    val graphMs: Double
      get() = mainTiming.totalMs + (actTiming?.totalMs ?: 0.0)
  }

  /** Calibrated answer with its built sequence and complete per-question timing. */
  data class AnswerResult(
    val sequence: PhocinaeSequence,
    val answer: Map<String, Any?>,
    val raw: RawResult,
    val prepareMs: Double,
    val decodeMs: Double,
    /** Total write, enqueue, and readback duration in milliseconds. */
    val totalMs: Double,
  )

  private data class Key(val window: Int, val backend: Backend, val storage: Storage)

  private class Graph(
    val model: CompiledModel,
    val inputs: Map<String, TensorBuffer>,
    val outputs: Map<String, TensorBuffer>,
  ) : Closeable {
    override fun close() {
      try {
        (inputs.values + outputs.values).forEach { it.close() }
      } finally {
        model.close()
      }
    }
  }

  private val appContext = context.applicationContext
  private val filesDir = appContext.filesDir
  private val npuInstalled = npuLibrariesInstalled(appContext)
  private val mainGraphs = linkedMapOf<Key, Graph>()
  private val actGraphs = linkedMapOf<Backend, Graph>()
  private val embeddingInputs = linkedMapOf<Int, FloatArray>()
  private val embeddings: PhocinaeEmbeddings
  private val tokenizer: PhocinaeTokenizer
  /** Installed temperatures used by the interactive calibrated decoder. */
  val calibration: PhocinaeCalibration
  /** Tokenizer initialization duration in milliseconds. */
  val tokenizerLoadMs: Double
  /** Metadata parsing and table mapping duration in milliseconds. */
  val embeddingLoadMs: Double
  /** Metadata hash recorded in each device gate report. */
  val embeddingTableSha256: String
    get() = embeddings.sha256

  private var closed = false

  init {
    REQUIRED_FILES.forEach { requireFile(it) }
    val started = System.nanoTime()
    tokenizer = PhocinaeTokenizer(requireFile("tokenizer.json"))
    tokenizerLoadMs = milliseconds(System.nanoTime() - started)
    calibration = PhocinaeCalibration.load(requireFile("rl_agent_config.json"))
    val embeddingStarted = System.nanoTime()
    embeddings =
      PhocinaeEmbeddings(requireFile("tok_embeddings_fp16.bin"), requireFile("tok_embeddings.json"))
    embeddingLoadMs = milliseconds(System.nanoTime() - embeddingStarted)
  }

  /**
   * Compile separately from measured calls so the first graph call can remain a cold observation.
   */
  fun initialize(backend: Backend, window: Int = 256) =
    PhocinaeProcessRuntime.call {
      checkOpen()
      requireWindow(window)
      check(backend != Backend.NPU || npuLibrariesInstalled(appContext)) {
        "This APK has no Qualcomm NPU libraries; see the NPU section of the README."
      }
      mainGraph(window, backend)
      actGraph(backend)
      Unit
    }

  /** The gate and interactive app use exactly the same tokenizer and prompt builder. */
  fun prepare(
    state: Any?,
    question: Map<String, Any?>,
    window: Int = 256,
    questionId: String = "",
  ): PhocinaeSequence {
    checkOpen()
    requireWindow(window)
    return PhocinaePromptBuilder(tokenizer, maxLen = window, headMaxLen = HEAD_MAX_LEN)
      .build(state, question, questionId)
  }

  /**
   * One main invocation and one action-head invocation, including both readbacks. Named signature
   * buffers avoid the different signature versus FlatBuffer tensor orders in these artifacts.
   * Nonfinite main output is preserved in the result, and is never sent through the action head.
   */
  fun runRaw(sequence: PhocinaeSequence, backend: Backend, window: Int = 256): RawResult =
    PhocinaeProcessRuntime.call {
      checkOpen()
      requireWindow(window)
      require(sequence.ids.size <= window) { "Sequence exceeds window $window" }
      require(sequence.markers.size == sequence.question.optionCount) { "Missing option marker" }
      val main = mainGraph(window, backend)
      val act = actGraph(backend)
      val attention = FloatArray(window) { if (it < sequence.ids.size) 1f else 0f }
      val qtype = FloatArray(3).also { it[sequence.question.qtype] = 1f }

      val lookupStarted = System.nanoTime()
      val embeds = embeddingInputs.getOrPut(window) { FloatArray(window * PhocinaeEmbeddings.WIDTH) }
      embeddings.gather(sequence.ids, window, embeds)
      val started = System.nanoTime()
      val embeddingLookupMs = milliseconds(started - lookupStarted)
      main.inputs.getValue("inputs_embeds").writeFloat(embeds)
      main.inputs.getValue("attention_mask").writeFloat(attention)
      main.inputs.getValue("qtype_onehot").writeFloat(qtype)
      val written = System.nanoTime()
      main.model.run(main.inputs, main.outputs, SIGNATURE)
      val enqueued = System.nanoTime()
      val tokens = main.outputs.getValue("token_logits").readFloat()
      val pooled = main.outputs.getValue("pooled_cls").readFloat()
      val read = System.nanoTime()
      require(tokens.size == window && pooled.size == PhocinaeEmbeddings.WIDTH) { "Unexpected main graph output shape" }
      val mainTiming = timing(started, written, enqueued, read)
      val markers = PhocinaeDecoder.gather(tokens, sequence.markers)
      if (!tokens.all { it.isFinite() } || !pooled.all { it.isFinite() }) {
        return@call RawResult(
          tokens,
          pooled,
          markers,
          FloatArray(4) { Float.NaN },
          FloatArray(2) { Float.NaN },
          mainTiming,
          null,
          embeddingLookupMs,
          milliseconds(System.nanoTime() - read),
          milliseconds(System.nanoTime() - lookupStarted),
        )
      }
      val features = PhocinaeDecoder.actFeatures(markers)
      val featured = System.nanoTime()
      act.inputs.getValue("pooled_cls").writeFloat(pooled)
      act.inputs.getValue("feats").writeFloat(features)
      val actWritten = System.nanoTime()
      act.model.run(act.inputs, act.outputs, SIGNATURE)
      val actEnqueued = System.nanoTime()
      val action = act.outputs.getValue("act_logits").readFloat()
      val finished = System.nanoTime()
      require(action.size == 2) { "Unexpected action graph output shape" }
      RawResult(
        tokens,
        pooled,
        markers,
        features,
        action,
        mainTiming,
        timing(featured, actWritten, actEnqueued, finished),
        embeddingLookupMs,
        milliseconds(featured - read),
        milliseconds(finished - lookupStarted),
      )
    }

  /** Runs builder, both graphs, and calibrated decoding with separate phase timings. */
  fun answer(
    state: Any?,
    question: Map<String, Any?>,
    backend: Backend,
    questionId: String = "",
    window: Int = 256,
  ): AnswerResult {
    val started = System.nanoTime()
    val sequence = prepare(state, question, window, questionId)
    val prepared = System.nanoTime()
    val raw = runRaw(sequence, backend, window)
    check(raw.finite) {
      "Nonfinite model output on ${backend.name}; inspect the device gate report"
    }
    val decodeStarted = System.nanoTime()
    val decoded =
      PhocinaeDecoder.decode(raw.markerLogits, raw.actLogits, sequence.question, calibration)
    val finished = System.nanoTime()
    return AnswerResult(
      sequence,
      decoded,
      raw,
      milliseconds(prepared - started),
      milliseconds(finished - decodeStarted),
      milliseconds(finished - started),
    )
  }

  private fun mainGraph(window: Int, backend: Backend): Graph =
    mainGraphs.getOrPut(Key(window, backend, storage)) {
      createGraph(
        mainFilename(window),
        backend,
        listOf("attention_mask", "inputs_embeds", "qtype_onehot"),
        listOf("pooled_cls", "token_logits"),
        main = true,
      )
    }

  private fun actGraph(backend: Backend): Graph =
    actGraphs.getOrPut(backend) {
      createGraph(
        ACT_HEAD_FILE,
        backend,
        listOf("feats", "pooled_cls"),
        listOf("act_logits"),
        main = false,
      )
    }

  private fun createGraph(
    filename: String,
    backend: Backend,
    inputNames: List<String>,
    outputNames: List<String>,
    main: Boolean,
  ): Graph {
    // The action head is tiny and its logits reach 2e3-5e3, so an NPU run keeps it on the CPU.
    // The act head (0.4 MB) stays on the CPU on an NPU run.
    val accelerator = if (backend == Backend.NPU && !main) Accelerator.CPU else backend.accelerator
    val options =
      CompiledModel.Options(accelerator).apply {
        when {
          backend == Backend.GPU ->
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
          backend == Backend.GPU_FP16 ->
            gpuOptions =
              CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16)
          backend == Backend.NPU && main -> Unit
          else -> cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
        }
        // LiteRT 2.2.0 starts the NPU runtime once per process, from the options of the first
        // graph it compiles on any accelerator, and the HTP performance mode is fixed then. So
        // every graph carries BURST, the mode the published NPU numbers were measured with;
        // without it a GPU or CPU graph compiled first leaves a later NPU graph about 5x slower.
        if (npuInstalled) {
          qualcommOptions =
            CompiledModel.QualcommOptions(
              htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
            )
        }
      }
    val model =
      CompiledModel.create(
        requireFile(filename).absolutePath,
        options,
        PhocinaeProcessRuntime.environment(appContext),
      )
    val inputs = linkedMapOf<String, TensorBuffer>()
    val outputs = linkedMapOf<String, TensorBuffer>()
    try {
      inputNames.forEach { inputs[it] = model.createInputBuffer(it, SIGNATURE) }
      outputNames.forEach { outputs[it] = model.createOutputBuffer(it, SIGNATURE) }
      return Graph(model, inputs, outputs)
    } catch (failure: Throwable) {
      (inputs.values + outputs.values).forEach { it.close() }
      model.close()
      throw failure
    }
  }

  private fun requireFile(name: String): File =
    File(filesDir, name).also {
      check(it.isFile) { "Missing $name. Run scripts/install_to_device.sh, then relaunch." }
    }

  private fun checkOpen() = check(!closed) { "PhocinaeEngine is closed" }

  /** Resolves the selected fixed-window storage variant without changing GPU precision. */
  fun mainFilename(window: Int): String = "phocinae_largha_150m_L${window}_fp16fc_embeds.tflite"

  /** Closes model buffers and releases table references on the native runtime thread. */
  override fun close() =
    PhocinaeProcessRuntime.call {
      if (!closed) {
        closed = true
        try {
          (mainGraphs.values + actGraphs.values).forEach { it.close() }
        } finally {
          mainGraphs.clear()
          actGraphs.clear()
          embeddingInputs.clear()
          embeddings.close()
        }
      }
    }

  companion object {
    /** Runtime pin shared with the version catalog and gate metadata. */
    const val LITERT_VERSION = "2.2.0"
    private const val SIGNATURE = "serving_default"
    /** Shared files required by either graph storage choice. */
    const val ACT_HEAD_FILE = "phocinae_act_head_fp32.tflite"
    /** The publisher's head budget (rl_agent_config.json head_max_len). */
    const val HEAD_MAX_LEN = 192
    /** Windows the published graphs cover; the editor uses 256 and 384. */
    val WINDOWS = listOf(256, 384, 512)
    val REQUIRED_FILES =
      listOf(
        ACT_HEAD_FILE,
        "rl_agent_config.json",
        "tokenizer.json",
        "tok_embeddings_fp16.bin",
        "tok_embeddings.json",
      )

    private fun requireWindow(window: Int) =
      require(window in WINDOWS) { "Only windows 256, 384 and 512 are published" }

    private fun milliseconds(nanos: Long) = nanos / 1_000_000.0

    /** The dispatch library and the JIT compiler plugin must both be packaged for the NPU. */
    fun npuLibrariesInstalled(context: Context): Boolean {
      val libDir = File(context.applicationInfo.nativeLibraryDir)
      return NPU_LIBRARIES.all { File(libDir, it).isFile }
    }

    private val NPU_LIBRARIES =
      listOf("libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so", "libQnnHtp.so")

    private fun timing(start: Long, written: Long, enqueued: Long, read: Long) =
      GraphTiming(
        milliseconds(written - start),
        milliseconds(enqueued - written),
        milliseconds(read - enqueued),
      )
  }
}

/** One native thread and one Environment for the lifetime of this application process. */
private object PhocinaeProcessRuntime {
  private val executor =
    Executors.newSingleThreadExecutor { runnable ->
      Thread(runnable, "Phocinae-LiteRT").apply { isDaemon = true }
    }
  private var sharedEnvironment: Environment? = null

  // Both directories point at the packaged vendor libraries. Without CompilerPluginLibraryDir a
  // model requested on the NPU is not compiled for it; harmless for GPU and CPU.
  fun environment(context: Context): Environment =
    sharedEnvironment
      ?: Environment.create(
          context,
          mapOf(
            Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
            Environment.Option.CompilerPluginLibraryDir to context.applicationInfo.nativeLibraryDir,
          ),
        )
        .also { sharedEnvironment = it }

  fun <T> call(block: () -> T): T =
    try {
      executor.submit(Callable { block() }).get()
    } catch (failure: ExecutionException) {
      throw failure.cause ?: failure
    }
}
