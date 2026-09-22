// SPDX-License-Identifier: Apache-2.0
package com.sopro

import java.util.concurrent.CancellationException

/** Pointwise playback policy. Saved audio still uses the complete offline postprocessor. */
object StreamingPlayback {
  fun process(raw: FloatArray, levelDb: Double): FloatArray {
    val gain = PostProcess.outputGain(levelDb).toFloat()
    return PostProcess.softLimit(FloatArray(raw.size) { raw[it] * gain })
  }
}

/** The same invoker boundary is used by CompiledModel and the recorded-output JVM replay. */
class StreamingPcmDecoder(
  private val window: FloatArray,
  private val contextSamples: Int,
  private val maximumSamples: Int,
  private val levelDb: Double,
  private val tapNanos: Long,
  private val sink: (FloatArray) -> Unit,
  private val playbackProcessing: Boolean = true,
  private val shouldStop: () -> Boolean = { false },
  private val clock: () -> Long = System::nanoTime,
) {
  data class Result(
    val raw: FloatArray,
    val features: FloatArray,
    val chunkLengths: List<Int>,
    val ttfaMs: Double?,
    val istftMs: Double,
    val playbackProcessingMs: Double,
  )

  fun run(
    mel: FloatArray,
    frames: Int,
    invoke: (String, List<FloatArray>) -> List<FloatArray>,
  ): Result {
    val istft = StreamingIstft(window)
    val rawChunks = mutableListOf<FloatArray>()
    val features = mutableListOf<FloatArray>()
    var skipRemaining = contextSamples
    var emitted = 0
    var firstPcm: Long? = null
    var istftNs = 0L
    var playbackNs = 0L
    fun checkActive() {
      if (shouldStop()) throw CancellationException("Synthesis cancelled")
    }
    StaticStreamFeatures.run(
      mel,
      frames,
      { mode, args ->
        checkActive()
        invoke(mode, args).also { checkActive() }
      },
    ) { chunkFeatures, flush ->
      checkActive()
      features += chunkFeatures
      val start = clock()
      val pcm = istft.process(chunkFeatures, flush)
      istftNs += clock() - start
      val skip = minOf(skipRemaining, pcm.size)
      skipRemaining -= skip
      val count = minOf(pcm.size - skip, maxOf(0, maximumSamples - emitted))
      if (count > 0) {
        val raw = pcm.copyOfRange(skip, skip + count)
        rawChunks += raw
        emitted += count
        val processingStart = clock()
        val played = if (playbackProcessing) StreamingPlayback.process(raw, levelDb) else raw
        playbackNs += clock() - processingStart
        checkActive()
        // Timestamp immediately before handing the first nonempty PCM chunk to the sink.
        if (firstPcm == null) firstPcm = clock()
        sink(played)
      }
    }
    return Result(
      concatenate(rawChunks),
      concatenate(features),
      rawChunks.map { it.size },
      firstPcm?.let { (it - tapNanos) / 1_000_000.0 },
      istftNs / 1_000_000.0,
      playbackNs / 1_000_000.0,
    )
  }

  private fun concatenate(parts: List<FloatArray>): FloatArray {
    val result = FloatArray(parts.sumOf { it.size })
    var offset = 0
    parts.forEach {
      it.copyInto(result, offset)
      offset += it.size
    }
    return result
  }
}
