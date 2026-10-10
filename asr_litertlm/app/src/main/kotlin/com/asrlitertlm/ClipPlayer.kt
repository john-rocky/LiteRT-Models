package com.asrlitertlm

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Handler
import android.os.Looper
import java.io.File

/**
 * Plays one 16-bit wav through the phone's speaker. [onDone] runs on the main thread once the last sample has reached
 * the mixer plus the output latency (about 0.2 s on the Galaxy S26), or with a sentence when playback failed.
 */
class ClipPlayer {
  private val main = Handler(Looper.getMainLooper())
  private var track: AudioTrack? = null
  private var finish: Runnable? = null

  /** Starts [wav]; [onProgress] gets the played share (0..1) every 50 ms. Stops a clip that is still playing. */
  fun play(wav: File, onProgress: (Float) -> Unit, onDone: (failure: String?) -> Unit) {
    stop()
    val newTrack: AudioTrack
    val frames: Int
    val rate: Int
    try {
      val pcm = Wav.pcm16(wav)
      frames = pcm.frames
      rate = pcm.sampleRate
      newTrack =
        AudioTrack.Builder()
          .setAudioAttributes(
            AudioAttributes.Builder()
              .setUsage(AudioAttributes.USAGE_MEDIA)
              .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
              .build()
          )
          .setAudioFormat(
            AudioFormat.Builder()
              .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
              .setSampleRate(pcm.sampleRate)
              .setChannelMask(
                if (pcm.channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO
              )
              .build()
          )
          .setTransferMode(AudioTrack.MODE_STATIC)
          .setBufferSizeInBytes(pcm.data.size)
          .build()
      check(newTrack.write(pcm.data, 0, pcm.data.size) == pcm.data.size) { "AudioTrack took a partial buffer" }
    } catch (e: RuntimeException) {
      onDone("Could not play ${wav.name}: ${e.message ?: e}")
      return
    }
    val done = Runnable {
      if (track === newTrack) {
        stop()
        onDone(null)
      }
    }
    finish = done
    track = newTrack
    newTrack.positionNotificationPeriod = rate / PROGRESS_PER_SECOND
    newTrack.notificationMarkerPosition = maxOf(1, frames - rate / MARKER_LEAD_PER_SECOND)
    newTrack.setPlaybackPositionUpdateListener(
      object : AudioTrack.OnPlaybackPositionUpdateListener {
        override fun onMarkerReached(t: AudioTrack) {
          main.removeCallbacks(done)
          main.postDelayed(done, OUTPUT_LATENCY_MS)
        }

        override fun onPeriodicNotification(t: AudioTrack) {
          if (track === t) onProgress((t.playbackHeadPosition.toFloat() / frames).coerceIn(0f, 1f))
        }
      },
      main,
    )
    newTrack.play()
    // A marker that never fires (a muted or rerouted output) still ends the clip.
    main.postDelayed(done, frames * 1000L / rate + FALLBACK_SLACK_MS)
  }

  /** Stops and frees the current clip without calling its onDone. */
  fun stop() {
    finish?.let { main.removeCallbacks(it) }
    finish = null
    track?.let {
      runCatching { it.stop() }
      it.release()
    }
    track = null
  }

  private companion object {
    const val PROGRESS_PER_SECOND = 20
    const val MARKER_LEAD_PER_SECOND = 50 // the marker sits 20 ms before the last frame
    const val OUTPUT_LATENCY_MS = 250L
    const val FALLBACK_SLACK_MS = 3_000L
  }
}
