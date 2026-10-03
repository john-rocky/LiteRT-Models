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
}
