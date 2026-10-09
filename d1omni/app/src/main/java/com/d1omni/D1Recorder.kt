package com.d1omni

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.Closeable
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sqrt

/** One recording: its 16 kHz mono int16 samples, when it started and stopped (wall ms), why it stopped. */
class D1Recording(
  val samples: ShortArray,
  val startedWallMs: Long,
  val stoppedWallMs: Long,
  /** "stopped" (the Stop button), "limit" (the longest clip the installed audio graph holds), "error". */
  val end: String,
  val audioSource: String,
  val error: String?,
)

/**
 * Records the microphone at the model's input format, 16 kHz mono PCM16 (`AudioRecord`, source VOICE_RECOGNITION: the
 * source Android tunes for speech to a recogniser, without the call path's processing), on its own thread until [stop]
 * or [maxSamples]. [onLevel] gets the level of each 100 ms block (dBFS, -90 .. 0) and the samples so far. The caller
 * holds the RECORD_AUDIO permission. One recording at a time; the device's volume and recording settings are never
 * touched.
 */
class D1Recorder : Closeable {
  private val lock = Any()
  private var thread: Thread? = null
  @Volatile private var stopRequested = false

  val recording: Boolean
    get() = synchronized(lock) { thread?.isAlive == true }

  @SuppressLint("MissingPermission")
  fun start(maxSamples: Int, onLevel: (dbfs: Float, samples: Int) -> Unit, onDone: (D1Recording) -> Unit) {
    synchronized(lock) {
      check(thread?.isAlive != true) { "already recording" }
      stopRequested = false
      val minBytes =
        AudioRecord.getMinBufferSize(D1Audio.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
      val record =
        AudioRecord(
          MediaRecorder.AudioSource.VOICE_RECOGNITION,
          D1Audio.SAMPLE_RATE,
          AudioFormat.CHANNEL_IN_MONO,
          AudioFormat.ENCODING_PCM_16BIT,
          max(minBytes, BLOCK * 2 * 4),
        )
      if (record.state != AudioRecord.STATE_INITIALIZED) {
        val state = record.state
        record.release()
        throw IllegalStateException("the microphone could not be opened (AudioRecord state $state)")
      }
      thread =
        Thread({ run(record, maxSamples, onLevel, onDone) }, "D1Omni-Record").apply {
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
    val running = synchronized(lock) { thread }
    running?.join(JOIN_MILLIS)
  }

  private fun run(record: AudioRecord, maxSamples: Int, onLevel: (Float, Int) -> Unit, onDone: (D1Recording) -> Unit) {
    var samples = ShortArray(D1Audio.SAMPLE_RATE * 4)
    var count = 0
    val block = ShortArray(BLOCK)
    var end = "stopped"
    var error: String? = null
    var started = 0L
    try {
      record.startRecording()
      started = System.currentTimeMillis()
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
        onLevel(level(block, read), count)
      }
      if (!stopRequested && count >= maxSamples && end != "error") end = "limit"
    } catch (failure: Exception) {
      onDone(D1Recording(samples.copyOf(count), System.currentTimeMillis(), System.currentTimeMillis(), "error",
        SOURCE_NAME, D1Decider.describe(failure)))
      runCatching { record.release() }
      return
    }
    val stopped = System.currentTimeMillis()
    runCatching { record.stop() }
    record.release()
    onDone(D1Recording(samples.copyOf(count), started, stopped, end, SOURCE_NAME, error))
  }

  companion object {
    /** 100 ms at 16 kHz. */
    const val BLOCK = 1600
    const val SOURCE_NAME = "VOICE_RECOGNITION"
    private const val JOIN_MILLIS = 1000L

    /** dBFS of the first [count] samples of [block] (-90 for silence). */
    fun level(block: ShortArray, count: Int): Float {
      if (count <= 0) return -90f
      var sum = 0.0
      for (index in 0 until count) {
        val value = block[index] / 32768.0
        sum += value * value
      }
      val rms = sqrt(sum / count)
      return if (rms <= 0.0) -90f else max(-90.0, 20 * log10(rms)).toFloat()
    }
  }
}
