package com.kev

import android.content.Context
import android.util.Log
import java.io.File

/**
 * The demo recording's interface: one logcat tag with one line per event, and the run JSON in
 * `files/` (see [KevDemoRun]).
 */
object KevDemo {
  const val LOG_TAG = "KevDemo"

  /** The engine became ready; [loadMs] = tokenizer + head + graph compile. */
  fun engineReady(loadMs: Long) = Log.i(LOG_TAG, "ENGINE_READY load_ms=$loadMs")

  /** GPU could not compile the graph and the app runs on CPU instead (not a failure of the app). */
  fun gpuFallback(error: String) = Log.i(LOG_TAG, "GPU_FALLBACK ${oneLine(error)}")

  fun autoplayStart(fixture: String) = Log.i(LOG_TAG, "AUTOPLAY_START fixture=${oneLine(fixture)}")

  /** A question finished; [ms] is the number on its card (input writes + `run()` + read-back). */
  fun questionDone(qid: String, ms: Long) = Log.i(LOG_TAG, "Q_DONE qid=${oneLine(qid)} ms=$ms")

  fun autoplayDone(json: String) = Log.i(LOG_TAG, "AUTOPLAY_DONE json=$json")

  /**
   * How a request runs: the form of [plan] and the predicted ms of both forms ("-" when a form
   * cannot take the request), or `none` with the reason; then what the plan saw ([inputs]: the
   * available memory in bytes and the resident graphs).
   */
  fun plan(plan: KevPlan, inputs: KevPlanInputs?) =
    Log.i(
      LOG_TAG,
      when (plan) {
        is KevPlan.Ready ->
          "PLAN form=${plan.form.wireName} rows=${ms(plan.prediction.rowsMs)} " +
            "pair=${ms(plan.prediction.pairMs)}"
        is KevPlan.NoWindow ->
          "PLAN form=none row=${plan.missing.rowTokens} window=${plan.missing.window ?: "-"}"
        is KevPlan.NoPair -> "PLAN form=none pair=${plan.miss}"
      } +
        " avail_mem_bytes=${inputs?.availableBytes ?: "-"} " +
        "resident=${inputs?.resident?.joinToString(",") { it.label } ?: "-"}",
    )

  /**
   * The graphs of a request: each question's window (L, or the pair's Lq), the graphs compiled
   * (with the available memory read right before each compile, bytes) and closed for it, and the
   * graphs left [resident].
   */
  fun windows(graphs: KevRequestGraphs, resident: List<KevGraphKey>) =
    Log.i(
      LOG_TAG,
      "WINDOWS form=${graphs.plan.form.wireName} questions=${graphs.windows.joinToString(",")} " +
        "compiled=${graphs.compiled.joinToString(",") { it.label }} " +
        "avail_mem_bytes=${graphs.availableBytes.joinToString(",")} " +
        "closed=${graphs.closed.joinToString(",") { it.label }} " +
        "resident=${resident.joinToString(",") { it.label }}",
    )

  /** The pair's state call of an autoplay request: its tokens and its ms. */
  fun stateDone(tokens: Int, ms: Long) = Log.i(LOG_TAG, "STATE_DONE tokens=$tokens ms=$ms")

  fun failed(reason: String) = Log.i(LOG_TAG, "failed ${oneLine(reason)}")

  /** Writes [run] to `files/kev-demo-<epoch ms>.json` (through a temporary file) and returns it. */
  fun writeRun(context: Context, run: Map<String, Any?>): File {
    val file = File(context.filesDir, "kev-demo-${System.currentTimeMillis()}.json")
    val temporary = File(context.filesDir, "${file.name}.tmp")
    temporary.writeText(KevJson.writeIndented(run, 1) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
    return file
  }

  private fun oneLine(text: String) = text.replace('\n', ' ').replace('\r', ' ')

  private fun ms(value: Double?) = value?.let { "${Math.round(it)}ms" } ?: "-"
}
