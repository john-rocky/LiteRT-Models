package com.d1omni

import android.content.Context
import java.io.Closeable
import java.io.File

/** One question's call on the app's engine: its row's bucket, the call, its probabilities and times. */
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
 * The app's engine: the tokenizer and the contract ([D1Engine]), and every graph a voice note, a photo or a message
 * needs, compiled once at startup and kept: the decision graphs L256 and L128 (`D1Residency`: L128 is given up when the
 * memory Android reports is under 2.5 GB before it compiles), the audio graph for clips up to 10 s (T1001; a longer
 * recording compiles its own bucket in its place when that file is installed) and the vision tower with the projector
 * ([D1VisionEngine]); each kind at its own GPU precision ([D1Precisions]). Use only on [D1Runtime.dispatcher].
 */
class D1AppEngine
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
  fun graphs(): List<D1RunGraph> {
    val out = ArrayList<D1RunGraph>()
    for (compile in decide.compiles) {
      if (compile.graph == "decide" && compile.bucket !in decide.resident) continue
      if (compile.graph == D1AudioEngine.GRAPH && compile.bucket !in decide.audio.resident) continue
      val name =
        if (compile.graph == "decide") decide.fileOf(compile.bucket)
        else requireNotNull(decide.audio.bucketFiles[compile.bucket]).name
      out.add(
        D1RunGraph(
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
        D1RunGraph(
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

  /** One row per question of [questions] over [state] after [prefixRows] media rows (a request of [kind]). */
  fun rows(state: Any?, questions: Collection<D1Question>, prefixRows: Int, kind: D1Kind): List<D1Row> =
    D1Rows.rows(decide.tokenizer, decide.contract, state, questions.toList(), prefixRows, kind)

  /**
   * One question: its inputs on the smallest resident decision graph that holds P + n, one call,
   * the read-out and `answer()`.
   */
  fun question(row: D1Row, prefix: FloatArray?): D1QuestionCall {
    val start = System.nanoTime()
    val bucket =
      D1Contract.bucketFor(row.positions, decide.resident)
        ?: throw IllegalStateException(
          "This question needs ${row.positions} positions (the media's ${row.prefixRows} and ${row.ids.size} of " +
            "text); the app keeps the decision graphs for up to ${decide.resident.maxOrNull() ?: 0}. Shorten the text " +
            "or the options."
        )
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

    /** The audio bucket compiled at startup (clips up to 10 s). */
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
    ): D1AppEngine {
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
        return D1AppEngine(
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
