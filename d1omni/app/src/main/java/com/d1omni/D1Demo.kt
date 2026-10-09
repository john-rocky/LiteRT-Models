package com.d1omni

import android.content.Context
import android.util.Log
import java.io.File

/**
 * The app's milestones for a recording harness: one logcat tag with one line per event, and each Decide's run JSON in
 * `files/` (see [D1Run]). The buttons and the reproduction run write the same lines.
 */
object D1Demo {
  const val LOG_TAG = "D1OmniDemo"

  /**
   * The engine became ready; [loadMs] = contract + tokenizer + every graph's compile; [warmupMs] = the untimed pass over
   * the bundled sample that follows (not part of [loadMs]).
   */
  fun engineReady(loadMs: Long, warmupMs: Long) = Log.i(LOG_TAG, "ENGINE_READY load_ms=$loadMs warmup_ms=$warmupMs")

  /** The GPU could not compile or run a graph and it runs on the CPU instead. */
  fun gpuFallback(error: String) = Log.i(LOG_TAG, "GPU_FALLBACK ${oneLine(error)}")

  /** The microphone started (16 kHz mono PCM16, at most [maxSeconds]). */
  fun recordStart(maxSeconds: Double) = Log.i(LOG_TAG, "RECORD_START max_seconds=%.1f".format(java.util.Locale.ROOT, maxSeconds))

  /** The recording ended ([end]: stopped, limit or error) and was saved as [wav]. */
  fun recordStop(samples: Int, wav: String, sha256: String, end: String) =
    Log.i(
      LOG_TAG,
      "RECORD_STOP samples=$samples seconds=%.3f wav=$wav sha256=$sha256 end=$end".format(
        java.util.Locale.ROOT,
        samples / D1Audio.SAMPLE_RATE.toDouble(),
      ),
    )

  /** A file was picked (the voice screen's WAV or the photo screen's picture). */
  fun picked(kind: D1Kind, sha256: String, bytes: Int, name: String?) =
    Log.i(LOG_TAG, "PICKED kind=${kind.wireName} sha256=$sha256 bytes=$bytes name=${oneLine(name ?: "-")}")

  /** The Recent photos sheet opened with these pictures (newest added first). */
  fun recentPhotos(names: List<String>) =
    Log.i(LOG_TAG, "PHOTOS_SHOWN n=${names.size} names=${names.joinToString(",") { oneLine(it).replace(',', '_') }}")

  /** The message Decide sends was typed (not the sample's text): its length in characters. */
  fun typed(chars: Int) = Log.i(LOG_TAG, "TYPED chars=$chars")

  /** Load sample filled the three screens. */
  fun sampleLoaded(id: String) = Log.i(LOG_TAG, "SAMPLE_LOADED id=$id")

  fun decideStart(input: D1Input, source: D1Source) =
    Log.i(LOG_TAG, "DECIDE_START item=${input.wireName} source=${source.wireName}")

  /** A question finished; [ms] is its decision call (input writes + `run()` + read-back). */
  fun questionDone(input: D1Input, qid: String, ms: Long) =
    Log.i(LOG_TAG, "Q_DONE item=${input.wireName} qid=$qid ms=$ms")

  /** The input's answers are on screen; [ms] is the number under them, [json] its run JSON. */
  fun decideDone(input: D1Input, ms: Long, json: String) =
    Log.i(LOG_TAG, "DECIDE_DONE item=${input.wireName} ms=$ms json=$json")

  fun autoplayStart() = Log.i(LOG_TAG, "AUTOPLAY_START")

  fun autoplayDone(jsons: List<String>) = Log.i(LOG_TAG, "AUTOPLAY_DONE json=${jsons.joinToString(",")}")

  fun failed(reason: String) = Log.i(LOG_TAG, "failed ${oneLine(reason)}")

  /** Writes [run] to `files/d1omni-run-<epoch ms>.json` (through a temporary file) and returns it. */
  fun writeRun(context: Context, run: Map<String, Any?>): File {
    val file = File(context.filesDir, "d1omni-run-${System.currentTimeMillis()}.json")
    val temporary = File(context.filesDir, "${file.name}.tmp")
    temporary.writeText(D1Json.writeIndented(run, 1) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
    return file
  }

  private fun oneLine(text: String) = text.replace('\n', ' ').replace('\r', ' ')
}
