package com.d1omni

import android.util.Log

/**
 * The demo recording's interface: one logcat tag with one line per event. The autoplay lines and
 * the run JSON come with the demo screen.
 */
object D1Demo {
  const val LOG_TAG = "D1OmniDemo"

  /** The engine became ready; [loadMs] = tokenizer + contract + the startup graphs' compile. */
  fun engineReady(loadMs: Long) = Log.i(LOG_TAG, "ENGINE_READY load_ms=$loadMs")

  /** The GPU could not compile or run a graph and it runs on the CPU instead. */
  fun gpuFallback(error: String) = Log.i(LOG_TAG, "GPU_FALLBACK ${oneLine(error)}")

  fun failed(reason: String) = Log.i(LOG_TAG, "failed ${oneLine(reason)}")

  private fun oneLine(text: String) = text.replace('\n', ' ').replace('\r', ' ')
}
