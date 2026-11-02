package com.d1_omni_2b

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import java.io.Closeable
import kotlin.math.max

/** One recording: 16 kHz mono 16-bit samples, and why it stopped ("stopped", "limit" or "error"). */
class Recording(val samples: ShortArray, val end: String, val error: String?)

/**
 * Records the microphone at the model's input format, 16 kHz mono 16-bit PCM (`AudioRecord`, source
 * VOICE_RECOGNITION: tuned for speech without the call path's processing), on its own thread until [stop] or
 * [maxSamples]. [onProgress] gets the samples so far after each 100 ms block. The caller holds the RECORD_AUDIO
 * permission. One recording at a time.
 */
class Recorder : Closeable {
  private val lock = Any()
  private var thread: Thread? = null
  @Volatile private var stopRequested = false

  val recording: Boolean
    get() = synchronized(lock) { thread?.isAlive == true }

  @SuppressLint("MissingPermission")
  fun start(maxSamples: Int, onProgress: (samples: Int) -> Unit, onDone: (Recording) -> Unit) {
    synchronized(lock) {
      check(thread?.isAlive != true) { "already recording" }
      stopRequested = false
      val minBytes = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
      val record =
        AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, max(minBytes, BLOCK * 2 * 4))
      if (record.state != AudioRecord.STATE_INITIALIZED) {
        val state = record.state
        record.release()
        throw IllegalStateException("the microphone could not be opened (AudioRecord state $state)")
      }
      thread = Thread({ run(record, maxSamples, onProgress, onDone) }, "d1-record").apply {
        isDaemon = true
        start()
      }
    }
  }

  fun stop() {
    stopRequested = true
  }

  override fun close() {
    stop()
    synchronized(lock) { thread }?.join(1000)
  }

  private fun run(record: AudioRecord, maxSamples: Int, onProgress: (Int) -> Unit, onDone: (Recording) -> Unit) {
    var samples = ShortArray(RATE * 4)
    var count = 0
    val block = ShortArray(BLOCK)
    var end = "stopped"
    var error: String? = null
    try {
      record.startRecording()
      while (!stopRequested && count < maxSamples) {
        val read = record.read(block, 0, minOf(BLOCK, maxSamples - count))
        if (read < 0) {
          end = "error"
          error = "AudioRecord.read returned $read"
          break
        }
        if (count + read > samples.size) samples = samples.copyOf(max(samples.size * 2, count + read))
        block.copyInto(samples, count, 0, read)
        count += read
        onProgress(count)
      }
      if (!stopRequested && count >= maxSamples && end != "error") end = "limit"
    } catch (failure: Exception) {
      end = "error"
      error = failure.message ?: failure.toString()
    } finally {
      runCatching { record.stop() }
      record.release()
    }
    onDone(Recording(samples.copyOf(count), end, error))
  }

  companion object {
    const val RATE = 16000
    /** 100 ms at 16 kHz. */
    const val BLOCK = 1600
  }
}

/**
 * Plays a clip through the speaker at the phone's media volume (`AudioTrack`, one static buffer of 16-bit PCM):
 * what the reader made of the clip, its channels averaged, at the clip's own rate. One clip at a time; [stop] ends it.
 */
class Player : Closeable {
  private val lock = Any()
  private var track: AudioTrack? = null
  private var watcher: Thread? = null

  fun play(clip: AudioClip, onDone: () -> Unit) {
    val samples = ShortArray(clip.mono.size) { (clip.mono[it] * 32768f).toInt().coerceIn(-32768, 32767).toShort() }
    require(samples.isNotEmpty()) { "nothing to play" }
    synchronized(lock) {
      stopLocked()
      val created =
        AudioTrack.Builder()
          .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
          .setAudioFormat(
            AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT).setSampleRate(clip.rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build()
          )
          .setTransferMode(AudioTrack.MODE_STATIC)
          .setBufferSizeInBytes(samples.size * 2)
          .build()
      try {
        check(created.write(samples, 0, samples.size) == samples.size) { "the player took only part of the clip" }
      } catch (failure: Throwable) {
        created.release()
        throw failure
      }
      track = created
      created.play()
      // The clip's length plus the output's latency, then the track is released.
      val millis = samples.size * 1000L / clip.rate + 300
      watcher = Thread({
        try {
          Thread.sleep(millis)
          synchronized(lock) { if (track === created) stopLocked() }
        } catch (_: InterruptedException) {
        }
        onDone()
      }, "d1-play").apply {
        isDaemon = true
        start()
      }
    }
  }

  fun stop() = synchronized(lock) { stopLocked() }

  override fun close() = stop()

  private fun stopLocked() {
    val running = watcher
    watcher = null
    if (running != null && running !== Thread.currentThread()) running.interrupt()
    track?.let {
      runCatching { it.stop() }
      it.release()
    }
    track = null
  }
}
