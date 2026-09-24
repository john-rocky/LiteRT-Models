package com.nemotron3diar

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTimestamp
import android.media.AudioTrack

/**
 * Plays a mono float clip on an [AudioTrack] and reports the position being heard: the last [AudioTimestamp]
 * (a frame presented at the output at a known System.nanoTime) extrapolated to now, capped at [MAX_EXTRAPOLATION_NS]
 * past it. The streaming loop feeds the model up to this position, so it hears the clip as a microphone next to
 * the speaker would.
 */
class LivePlayer(private val samples: FloatArray, private val rate: Int) : AutoCloseable {

  private val track: AudioTrack
  private val stamp = AudioTimestamp()
  private var last = 0L
  private var writer: Thread? = null

  @Volatile private var closed = false
  private var released = false

  init {
    val min = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
    track =
      AudioTrack.Builder()
        .setAudioAttributes(
          AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        )
        .setAudioFormat(
          AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(rate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        )
        .setTransferMode(AudioTrack.MODE_STREAM)
        .setBufferSizeInBytes(maxOf(min, rate / 5 * 4))
        .build()
  }

  val numSamples: Int
    get() = samples.size

  /** Starts playback; a writer thread feeds the clip with blocking writes. */
  fun start() {
    track.play()
    writer =
      Thread(
          {
            var pos = 0
            while (pos < samples.size && !closed) {
              val n = track.write(samples, pos, minOf(WRITE_CHUNK, samples.size - pos), AudioTrack.WRITE_BLOCKING)
              if (n <= 0) break
              pos += n
            }
          },
          "n3d-player",
        )
        .apply { start() }
  }

  /** Samples heard so far, in [0, numSamples]; never decreases. */
  @Synchronized
  fun position(): Long {
    if (released) return last
    val p =
      if (track.getTimestamp(stamp)) {
        val ahead = (System.nanoTime() - stamp.nanoTime).coerceIn(0L, MAX_EXTRAPOLATION_NS)
        stamp.framePosition + ahead * rate / 1_000_000_000L
      } else {
        track.playbackHeadPosition.toLong()
      }
    last = maxOf(last, p.coerceIn(0L, samples.size.toLong()))
    return last
  }

  override fun close() {
    closed = true
    runCatching { track.pause() }
    runCatching { track.flush() }
    writer?.join(1000)
    synchronized(this) {
      released = true
      track.release()
    }
  }

  companion object {
    private const val WRITE_CHUNK = 4096
    private const val MAX_EXTRAPOLATION_NS = 200_000_000L
  }
}
