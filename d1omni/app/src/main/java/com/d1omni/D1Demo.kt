package com.d1omni

import android.content.Context
import android.util.Log
import java.io.File

/**
 * The demo recording's interface: one logcat tag with one line per event, and the run JSON in
 * `files/` (see [D1DemoRun]).
 */
object D1Demo {
  const val LOG_TAG = "D1OmniDemo"

  /**
   * The engine became ready; [loadMs] = contract + tokenizer + every graph's compile; [warmupMs] =
   * the untimed pass over the bundled inbox that follows (not part of [loadMs]).
   */
  fun engineReady(loadMs: Long, warmupMs: Long) = Log.i(LOG_TAG, "ENGINE_READY load_ms=$loadMs warmup_ms=$warmupMs")

  /** The GPU could not compile or run a graph and it runs on the CPU instead. */
  fun gpuFallback(error: String) = Log.i(LOG_TAG, "GPU_FALLBACK ${oneLine(error)}")

  fun autoplayStart(fixture: String) = Log.i(LOG_TAG, "AUTOPLAY_START fixture=${oneLine(fixture)}")

  /**
   * The voice note's sound started: [headMs] after `play()` (the playback head moving, or the time
   * frame 0 left the output by the AudioTimestamp), at [wallMs] (epoch ms).
   */
  fun playing(item: String, headMs: Long, wallMs: Long) =
    Log.i(LOG_TAG, "PLAYING item=$item head_ms=$headMs wall_ms=$wallMs")

  /** An item's work started (after its playback for a voice note). */
  fun deciding(item: String) = Log.i(LOG_TAG, "DECIDING item=$item")

  /** A question finished; [ms] is the number on its row (input writes + `run()` + read-back). */
  fun questionDone(item: String, qid: String, ms: Long) = Log.i(LOG_TAG, "Q_DONE item=$item qid=$qid ms=$ms")

  /** An item finished; [ms] is its total line (host steps + media graphs + decision calls). */
  fun itemDone(item: String, ms: Long) = Log.i(LOG_TAG, "ITEM_DONE item=$item ms=$ms")

  fun autoplayDone(json: String) = Log.i(LOG_TAG, "AUTOPLAY_DONE json=$json")

  fun failed(reason: String) = Log.i(LOG_TAG, "failed ${oneLine(reason)}")

  /** Writes [run] to `files/d1omni-demo-<epoch ms>.json` (through a temporary file) and returns it. */
  fun writeRun(context: Context, run: Map<String, Any?>): File {
    val file = File(context.filesDir, "d1omni-demo-${System.currentTimeMillis()}.json")
    val temporary = File(context.filesDir, "${file.name}.tmp")
    temporary.writeText(D1Json.writeIndented(run, 1) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
    return file
  }

  private fun oneLine(text: String) = text.replace('\n', ' ').replace('\r', ' ')
}
