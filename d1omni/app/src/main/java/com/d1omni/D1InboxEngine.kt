package com.d1omni

import android.content.Context
import java.io.Closeable
import java.io.File

/** One question's call on the inbox engine: its row's bucket, the call, its probabilities and times. */
class D1QuestionCall(
  val row: D1Row,
  val bucket: Int,
  val call: D1Call,
  val probabilities: DoubleArray,
  val answer: LinkedHashMap<String, Any?>,
  /** Inputs + call + read-out + answer, in nanoseconds. */
  val totalNanos: Long,
  val backend: D1Backend,
  val precision: D1Precision?,
)

/**
 * The inbox demo's engine: the tokenizer and the contract ([D1Engine]), and every graph the demo
 * needs compiled once at startup and kept: the decision graphs L256 and L128 (`D1Residency`: L128
 * is given up when the memory Android reports is under 2.5 GB before it compiles), the audio graph
 * of the demo's clip (T1001) and the vision tower with the projector ([D1VisionEngine]); each kind
 * at its own GPU precision ([D1Precisions]). Use only on [D1Runtime.dispatcher].
 */
class D1InboxEngine
private constructor(
  private val context: Context,
  val decide: D1Engine,
  val vision: D1VisionEngine,
  val precisions: D1Precisions,
  /** Contract + tokenizer + the position table + every compile, in milliseconds. */
  val loadMs: Double,
  /** `D1Device.memory` right after the last compile. */
  val memoryAtReady: Map<String, Any?>,
) : Closeable {
  /** The resident graphs with their files, sizes, backends, precisions and compile times. */
  fun graphs(): List<D1DemoGraph> {
    val out = ArrayList<D1DemoGraph>()
    for (compile in decide.compiles) {
      if (compile.graph == "decide" && compile.bucket !in decide.resident) continue
      if (compile.graph == D1AudioEngine.GRAPH && compile.bucket !in decide.audio.resident) continue
      val name =
        if (compile.graph == "decide") decide.fileOf(compile.bucket)
        else requireNotNull(decide.audio.bucketFiles[compile.bucket]).name
      out.add(
        D1DemoGraph(
          if (compile.graph == "decide") "decide_L${compile.bucket}" else "audio_T${compile.bucket}",
          name,
          File(context.filesDir, name).length(),
          compile.backend,
          if (compile.backend == D1Backend.GPU) compile.precision else null,
          compile.compileMs,
          compile.gpuFailure,
        )
      )
    }
    val files = mapOf("vision_tower" to vision.contract.tower.file, "projector" to vision.contract.projector.file)
    for (compile in vision.compiles) {
      val name = requireNotNull(files[compile.graph])
      out.add(
        D1DemoGraph(
          compile.graph,
          name,
          File(context.filesDir, name).length(),
          compile.backend,
          if (compile.backend == D1Backend.GPU) compile.precision else null,
          compile.compileMs,
          compile.gpuFailure,
        )
      )
    }
    // A graph compiled again after a GPU failure appears once, as it runs now.
    return out.reversed().distinctBy { it.graph }.reversed()
  }

  /** Where each resident graph runs and at which precision (for the accelerator label). */
  fun backends(): List<Pair<D1Backend, D1Precision>> =
    graphs().map { it.backend to (it.precision ?: D1Precision.FP32) }

  /** The rows of [item] after [prefixRows] media rows (the item's kind: text, image or audio). */
  fun rows(item: D1InboxItem, prefixRows: Int): List<D1Row> =
    D1Rows.rows(decide.tokenizer, decide.contract, item.state, item.questions.values.toList(), prefixRows, item.kind)

  /**
   * One question: its inputs on the smallest resident decision graph that holds P + n, one call,
   * the read-out and `answer()`.
   */
  fun question(row: D1Row, prefix: FloatArray?): D1QuestionCall {
    val start = System.nanoTime()
    val bucket =
      D1Contract.bucketFor(row.positions, decide.resident)
        ?: throw IllegalStateException("no compiled decision graph holds ${row.positions} positions")
    val inputs = D1Rows.buildInputs(row.ids, prefix, row.prefixRows, bucket, row.question.type)
    val call = decide.call(bucket, inputs)
    val probabilities =
      D1Readout.probabilities(call.scores, row.prefixRows, row.markers, row.question, row.calibrate, decide.contract)
    check(probabilities.all { it.isFinite() }) { "non-finite probabilities on L$bucket" }
    val answer = D1Prompt.answer(row.question, probabilities)
    val graph = decide.graph(bucket)
    return D1QuestionCall(
      row,
      bucket,
      call,
      probabilities,
      answer,
      System.nanoTime() - start,
      graph.backend,
      if (graph.backend == D1Backend.GPU) graph.precision else null,
    )
  }

  override fun close() {
    try {
      vision.close()
    } finally {
      decide.close()
    }
  }

  companion object {
    /** The decision buckets compiled at startup (largest first: the second one is the one memory can refuse). */
    val DECISION_BUCKETS = listOf(256, 128)

    /** The audio bucket of the demo's clip (up to 10 s). */
    const val AUDIO_BUCKET = 1001

    /**
     * Reads the contract and the tokenizer, then compiles L256, L128, the audio graph T1001, the
     * vision tower and the projector, reporting each step to [progress].
     */
    fun load(
      context: Context,
      backend: D1Backend,
      precisions: D1Precisions,
      progress: (String) -> Unit = {},
    ): D1InboxEngine {
      val start = System.nanoTime()
      progress("Loading the tokenizer…")
      val decide = D1Engine.load(context, backend, precisions.decide, precisions.audio)
      try {
        val missing = DECISION_BUCKETS.filter { it !in decide.installed }
        require(missing.isEmpty()) {
          "Missing ${missing.joinToString { decide.fileOf(it) }}. Run scripts/install_to_device.sh, then reopen the app."
        }
        progress("Compiling the decision graphs…")
        decide.ensure(DECISION_BUCKETS, emptyList())
        progress("Compiling the audio graph…")
        decide.audio.ensure(AUDIO_BUCKET)
        progress("Compiling the vision tower and the projector…")
        val vision = D1VisionEngine.open(context, backend, precisions.vision)
        val loadMs = (System.nanoTime() - start) / 1e6
        return D1InboxEngine(
          context.applicationContext,
          decide,
          vision,
          precisions,
          loadMs,
          D1Device.memory(context),
        )
      } catch (failure: Throwable) {
        decide.close()
        throw failure
      }
    }
  }
}
