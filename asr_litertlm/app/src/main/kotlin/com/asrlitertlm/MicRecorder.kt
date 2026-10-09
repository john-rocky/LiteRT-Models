package com.asrlitertlm

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.concurrent.thread

/** The end of one recording. */
sealed interface MicResult {
  /** A wav of 16 kHz mono 16-bit audio. [stoppedAtLimit] is true when the 30 s limit ended it. */
  data class Recorded(val wav: File, val seconds: Double, val levels: Wav.Levels, val stoppedAtLimit: Boolean) :
    MicResult

  /** A sentence for the screen. */
  data class Failed(val message: String) : MicResult
}

/**
 * Records 16 kHz mono 16-bit audio (what the three models take) from the microphone, on its own thread, until
 * [stop] is called or [MAX_SECONDS] have been recorded.
 */
class MicRecorder(private val context: Context) {
  @Volatile private var stopRequested = false

  @Volatile
  var isRecording = false
    private set

  fun hasPermission(): Boolean =
    context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

  /**
   * Starts a recording into [output]. Returns a failure sentence at once (and records nothing) when the microphone
   * permission is missing or a recording is running; otherwise returns null, reports the seconds recorded so far to
   * [onProgress] and the result to [onDone], both on the recording thread.
   */
  fun start(output: File, onProgress: (Double) -> Unit, onDone: (MicResult) -> Unit): String? {
    if (!hasPermission()) return PERMISSION_MESSAGE
    if (isRecording) return "A recording is already running."
    stopRequested = false
    isRecording = true
    thread(name = "mic-recorder") {
      val result =
        try {
          record(output, onProgress)
        } catch (e: SecurityException) {
          MicResult.Failed(PERMISSION_MESSAGE)
        } catch (e: RuntimeException) {
          MicResult.Failed("Recording failed: ${e.message ?: e}")
        }
      isRecording = false
      onDone(result)
    }
    return null
  }

  /** Ends the recording; [start]'s onDone follows. */
  fun stop() {
    stopRequested = true
  }

  @SuppressLint("MissingPermission") // checked in start(); a revocation in between ends in SecurityException
  private fun record(output: File, onProgress: (Double) -> Unit): MicResult {
    val minBuffer =
      AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
    if (minBuffer <= 0) return MicResult.Failed("This phone cannot record 16 kHz mono audio.")
    val recorder =
      AudioRecord(
        MediaRecorder.AudioSource.VOICE_RECOGNITION,
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        maxOf(minBuffer, SAMPLE_RATE),
      )
    if (recorder.state != AudioRecord.STATE_INITIALIZED) {
      recorder.release()
      return MicResult.Failed("The microphone could not be opened. Another app may be using it.")
    }
    val pcm = ByteArrayOutputStream()
    val maxBytes = SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_SECONDS
    try {
      recorder.startRecording()
      if (recorder.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
        return MicResult.Failed("The microphone did not start. Another app may be using it.")
      }
      val buffer = ByteArray(SAMPLE_RATE / 10 * BYTES_PER_SAMPLE) // 100 ms
      while (!stopRequested && pcm.size() < maxBytes) {
        val count = recorder.read(buffer, 0, minOf(buffer.size, maxBytes - pcm.size()))
        if (count < 0) return MicResult.Failed("Recording failed (AudioRecord error $count).")
        pcm.write(buffer, 0, count)
        onProgress(pcm.size().toDouble() / (SAMPLE_RATE * BYTES_PER_SAMPLE))
      }
    } finally {
      runCatching { recorder.stop() }
      recorder.release()
    }
    val bytes = pcm.toByteArray()
    if (bytes.isEmpty()) return MicResult.Failed("No audio was recorded.")
    Wav.writeMono16(output, bytes, SAMPLE_RATE)
    return MicResult.Recorded(
      wav = output,
      seconds = bytes.size.toDouble() / (SAMPLE_RATE * BYTES_PER_SAMPLE),
      levels = Wav.levels(bytes),
      stoppedAtLimit = bytes.size >= maxBytes,
    )
  }

  companion object {
    const val SAMPLE_RATE = 16_000
    const val MAX_SECONDS = 30
    private const val BYTES_PER_SAMPLE = 2

    const val PERMISSION_MESSAGE =
      "Microphone permission is off, so the app cannot record. The three clips still work. To use the " +
        "microphone, allow it in Settings > Apps > ASR LiteRT-LM > Permissions."
  }
}
