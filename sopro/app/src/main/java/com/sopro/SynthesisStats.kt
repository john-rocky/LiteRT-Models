// SPDX-License-Identifier: Apache-2.0
package com.sopro

import org.json.JSONObject

/** Wall-clock stages include lazy graph creation; per-graph write/run/read exclude creation. */
data class SynthesisStats(
  val seed: Long,
  val referenceId: String,
  val referenceCacheHit: Boolean,
  val placement: String,
  val precision: String,
  val ttfaMs: Double?,
  val ttfaToOnsetMs: Double?,
  val referencePrepMs: Double,
  val prefillMs: Double,
  val arStepMs: List<Double>,
  val arStepsTotalMs: Double,
  val acousticConditionMs: Double,
  val velocityMs: List<Double>,
  val vocoderMs: List<Double>,
  val istftMs: Double,
  val playbackProcessingMs: Double,
  val postprocessMs: Double,
  val totalWallMs: Double,
  val finalSeconds: Double,
  val rtf: Double,
  val tokenCount: Int,
  val pcmChunkSamples: List<Int>,
  val graphTimings: List<SoproEngine.CallTiming>,
) {
  companion object {
    fun onsetMs(ttfaMs: Double?, leadCutSamples: Int): Double? = ttfaMs?.plus(leadCutSamples / 24.0)
  }

  fun toJson(): JSONObject =
    JSONObject(
      linkedMapOf<String, Any?>(
        "seed" to seed,
        "reference_id" to referenceId,
        "reference_cache_hit" to referenceCacheHit,
        "placement" to placement,
        "precision" to precision,
        "ttfa_ms" to ttfaMs,
        "ttfa_to_onset_ms" to ttfaToOnsetMs,
        "reference_prep_ms" to referencePrepMs,
        "prefill_ms" to prefillMs,
        "ar_steps_total_ms" to arStepsTotalMs,
        "ar_step_ms" to arStepMs,
        "ar_sampling_and_loop_ms" to maxOf(0.0, arStepsTotalMs - arStepMs.sum()),
        "acoustic_condition_ms" to acousticConditionMs,
        "velocity_ms" to velocityMs,
        "vocoder_calls_ms" to vocoderMs,
        "istft_ms" to istftMs,
        "playback_processing_ms" to playbackProcessingMs,
        "postprocess_ms" to postprocessMs,
        "total_wall_ms" to totalWallMs,
        "final_seconds" to finalSeconds,
        "rtf" to rtf,
        "token_count" to tokenCount,
        "pcm_chunk_samples" to pcmChunkSamples,
        "sample_rate_hz" to 24000,
        "ttfa_definition" to
          "tap to first PCM chunk handed to sink; does not measure AudioTrack hardware latency",
        "playback_policy" to "gain and soft limit; no trim or fade",
        "saved_wav_policy" to
          "offline gain, lead/trail trim, soft limit and 80 ms fade-out; PCM16 encoding",
        "graph_ms" to
          graphTimings.map { t ->
            mapOf(
              "graph" to t.graph,
              "signature" to t.signature,
              "backend" to t.backend,
              "write_ms" to t.writeMs,
              "run_ms" to t.runMs,
              "read_ms" to t.readbackMs,
              "total_ms" to t.totalMs,
            )
          },
      )
    )
}
