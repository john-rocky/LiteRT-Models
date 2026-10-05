package com.kev

import android.content.Context
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.Closeable
import java.io.File

/**
 * One compiled shared-state pair on LiteRT `CompiledModel` (see [KevPairContract]): one file, two
 * signatures. The buffers are created once: the state call's two inputs and 48 outputs, the
 * question call's `ids`, `valid`, `state_valid` and `hidden`. The state reaches the question call
 * without a copy: its input map holds the state call's output buffers themselves ([stateCopy]
 * false; a debug run can instead read each state tensor back and write it into an input buffer of
 * the question call's own).
 *
 * GPU compiles with an explicit [precision] and, when [shareConstants], with constant tensor
 * sharing: both signatures use one copy of the weights; without it the GPU holds the weights once
 * per signature, which is faster and takes more memory ([KevPairShare] decides). CPU runs four
 * threads. The pair file is not in the form the NPU takes; only a debug run compiles it there.
 * Create, run and close only on [KevRuntime.dispatcher].
 */
class KevPairDecider
private constructor(
  val backend: KevDecider.Backend,
  /** The GPU precision the pair was compiled with (CPU ignores it). */
  val precision: KevPrecision,
  /** Whether the GPU holds one copy of the weights for both signatures (CPU ignores it). */
  val shareConstants: Boolean,
  val shape: KevPairShape,
  val file: File,
  /** Wall time of `CompiledModel.create` (graph load and compilation), in milliseconds. */
  val compileMs: Double,
  /** GPU's error when GPU was requested and this pair was compiled on CPU instead. */
  val gpuFailure: String?,
  /** What the NPU compile did (a debug run on the NPU only). */
  val npu: KevNpuCompile?,
  /** The state goes to the question call through the host (read back, written again). */
  val stateCopy: Boolean,
  private val model: CompiledModel,
  private val buffers: List<TensorBuffer>,
  private val stateInputs: Map<String, TensorBuffer>,
  private val stateOutputs: Map<String, TensorBuffer>,
  private val questionInputs: Map<String, TensorBuffer>,
  private val questionOutputs: Map<String, TensorBuffer>,
) : PairRunner, Closeable {
  /** Where the pair ran: [backend], or the CPU when the log shows the HTP did not take it. */
  val ranOn: KevDecider.Backend
    get() =
      if (backend == KevDecider.Backend.NPU && npu?.evidence?.applied == false) {
        KevDecider.Backend.CPU
      } else {
        backend
      }

  private val stateSignature = KevPairContract.stateSignature(shape)
  private val questionSignature = KevPairContract.questionSignature(shape)
  private var stateReady = false
  private var closed = false

  override val stateLength: Int
    get() = shape.stateLength

  override val questionLength: Int
    get() = shape.questionLength

  /**
   * Writes the state inputs, runs `state_prefill` and writes `state_valid` for the question calls.
   * `run()` only enqueues the work on the GPU; on the Galaxy S26 the next input write waits for it.
   */
  override fun runState(ids: IntArray, valid: FloatArray) {
    check(!closed) { "The pair is closed" }
    require(ids.size == stateLength && valid.size == stateLength) {
      "State inputs must have $stateLength entries"
    }
    stateReady = false
    stateInputs.getValue(KevPairContract.IDS).writeInt(ids)
    stateInputs.getValue(KevPairContract.VALID).writeFloat(valid)
    model.run(stateInputs, stateOutputs, stateSignature)
    if (stateCopy) {
      for (name in KevPairContract.stateNames) {
        questionInputs.getValue(name).writeFloat(stateOutputs.getValue(name).readFloat())
      }
    }
    questionInputs.getValue(KevPairContract.STATE_VALID).writeFloat(valid)
    stateReady = true
  }

  /** Writes the branch inputs, runs `question_step` and reads `hidden` back (Lq × 1024 floats). */
  override fun runQuestion(ids: IntArray, valid: FloatArray): FloatArray {
    check(!closed) { "The pair is closed" }
    check(stateReady) { "runState must run before the questions" }
    require(ids.size == questionLength && valid.size == questionLength) {
      "Question inputs must have $questionLength entries"
    }
    questionInputs.getValue(KevPairContract.IDS).writeInt(ids)
    questionInputs.getValue(KevPairContract.VALID).writeFloat(valid)
    model.run(questionInputs, questionOutputs, questionSignature)
    // run() only enqueues the work on the GPU; reading the output waits for it.
    val values = questionOutputs.getValue(KevPairContract.HIDDEN).readFloat()
    check(values.size == questionLength * KevPointerHead.HIDDEN_SIZE) {
      "hidden has ${values.size} values, expected ${questionLength * KevPointerHead.HIDDEN_SIZE}"
    }
    return values
  }

  override fun close() {
    if (closed) return
    closed = true
    try {
      buffers.forEach { it.close() }
    } finally {
      model.close()
    }
  }

  companion object {
    /**
     * Compiles the [shape] pair from `files/` on [backend] ([precision] and [shareConstants] on
     * GPU; [npuOptions] when the APK carries the NPU libraries). With [cpuFallback], a GPU failure
     * is kept as [gpuFailure] and the pair is compiled on CPU instead.
     */
    fun create(
      context: Context,
      shape: KevPairShape,
      backend: KevDecider.Backend,
      precision: KevPrecision,
      shareConstants: Boolean,
      cpuFallback: Boolean,
      npuOptions: KevNpuOptions?,
      stateCopy: Boolean = false,
    ): KevPairDecider {
      val file = File(context.filesDir, KevFiles.pair(shape))
      check(file.isFile) { "Missing ${file.name}" }
      fun on(target: KevDecider.Backend, gpuFailure: String?) =
        compile(
          context,
          file,
          shape,
          target,
          precision,
          shareConstants,
          gpuFailure,
          npuOptions,
          stateCopy,
        )
      if (backend != KevDecider.Backend.GPU || !cpuFallback) return on(backend, null)
      return try {
        on(KevDecider.Backend.GPU, null)
      } catch (failure: Exception) {
        on(KevDecider.Backend.CPU, KevDecider.describe(failure))
      } catch (failure: LinkageError) {
        on(KevDecider.Backend.CPU, KevDecider.describe(failure))
      }
    }

    private fun compile(
      context: Context,
      file: File,
      shape: KevPairShape,
      backend: KevDecider.Backend,
      precision: KevPrecision,
      shareConstants: Boolean,
      gpuFailure: String?,
      npuOptions: KevNpuOptions?,
      stateCopy: Boolean,
    ): KevPairDecider {
      val options = KevDecider.options(backend, precision, shareConstants, npuOptions)
      val npuStart =
        if (backend == KevDecider.Backend.NPU) KevNpuCompiler.before(context, file, npuOptions)
        else null
      val start = System.nanoTime()
      val model = CompiledModel.create(file.absolutePath, options, KevRuntime.environment(context))
      val compileMs = KevPipeline.millis(System.nanoTime() - start)
      val npu = npuStart?.let { KevNpuCompiler.after(context, it, compileMs) }
      val buffers = ArrayList<TensorBuffer>()
      try {
        checkContract(model, shape)
        val state = KevPairContract.stateSignature(shape)
        val question = KevPairContract.questionSignature(shape)
        fun input(name: String, signature: String) =
          model.createInputBuffer(name, signature).also { buffers.add(it) }
        val stateInputs =
          listOf(KevPairContract.IDS, KevPairContract.VALID).associateWith { input(it, state) }
        val stateOutputs =
          KevPairContract.stateNames.associateWith {
            model.createOutputBuffer(it, state).also { buffer -> buffers.add(buffer) }
          }
        // The question call reads the state call's outputs directly: no input buffers of its own
        // (with stateCopy, its own buffers that runState fills from the outputs).
        val questionInputs =
          listOf(KevPairContract.IDS, KevPairContract.VALID, KevPairContract.STATE_VALID)
            .associateWith { input(it, question) } +
            if (stateCopy) KevPairContract.stateNames.associateWith { input(it, question) }
            else stateOutputs
        val hidden =
          model.createOutputBuffer(KevPairContract.HIDDEN, question).also { buffers.add(it) }
        return KevPairDecider(
          backend,
          precision,
          shareConstants,
          shape,
          file,
          compileMs,
          gpuFailure,
          npu,
          stateCopy,
          model,
          buffers,
          stateInputs,
          stateOutputs,
          questionInputs,
          mapOf(KevPairContract.HIDDEN to hidden),
        )
      } catch (failure: Throwable) {
        buffers.forEach { it.close() }
        model.close()
        throw failure
      }
    }

    /** Checks the type and shape of every tensor of both signatures against [KevPairContract]. */
    private fun checkContract(model: CompiledModel, shape: KevPairShape) {
      val state = KevPairContract.stateSignature(shape)
      val question = KevPairContract.questionSignature(shape)
      val int = TensorType.ElementType.INT
      val float = TensorType.ElementType.FLOAT
      val ls = shape.stateLength
      val lq = shape.questionLength
      fun input(name: String, signature: String, element: TensorType.ElementType, dims: List<Int>) =
        KevDecider.checkTensor(model.getInputTensorType(name, signature), element, dims, name)
      input(KevPairContract.IDS, state, int, listOf(1, ls))
      input(KevPairContract.VALID, state, float, listOf(1, ls))
      input(KevPairContract.IDS, question, int, listOf(1, lq))
      input(KevPairContract.VALID, question, float, listOf(1, lq))
      input(KevPairContract.STATE_VALID, question, float, listOf(1, ls))
      for (name in KevPairContract.stateNames) {
        val dims = KevPairContract.stateShape(name, ls)
        KevDecider.checkTensor(model.getOutputTensorType(name, state), float, dims, name)
        input(name, question, float, dims)
      }
      KevDecider.checkTensor(
        model.getOutputTensorType(KevPairContract.HIDDEN, question),
        float,
        listOf(1, lq, KevPointerHead.HIDDEN_SIZE),
        KevPairContract.HIDDEN,
      )
    }
  }
}
