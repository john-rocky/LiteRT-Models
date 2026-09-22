// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

object ReferenceRecorder {
  @SuppressLint("MissingPermission")
  fun recordTenSeconds(shouldStop: () -> Boolean, onProgress: (Double) -> Unit): FloatArray {
    var recorder: AudioRecord? = null
    var rate = 48000
    try {
      for (candidate in intArrayOf(24000, 48000)) {
        val minimum =
          AudioRecord.getMinBufferSize(
            candidate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
          )
        if (minimum <= 0) continue
        val trial =
          try {
            AudioRecord.Builder()
              .setAudioSource(MediaRecorder.AudioSource.MIC)
              .setAudioFormat(
                AudioFormat.Builder()
                  .setSampleRate(candidate)
                  .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                  .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                  .build()
              )
              .setBufferSizeInBytes(maxOf(minimum, candidate / 5 * 2))
              .build()
          } catch (_: IllegalArgumentException) {
            continue
          }
        if (trial.state == AudioRecord.STATE_INITIALIZED && trial.sampleRate == candidate) {
          recorder = trial
          rate = candidate
          break
        }
        trial.release()
      }
      val current = recorder ?: error("Neither 24 kHz nor 48 kHz mono recording is available")
      val samples = FloatArray(rate * 10)
      val block = ShortArray(1024)
      var offset = 0
      current.startRecording()
      check(current.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
        "Microphone could not start"
      }
      while (offset < samples.size) {
        if (shouldStop()) throw java.util.concurrent.CancellationException("Recording stopped")
        val count =
          current.read(
            block,
            0,
            minOf(block.size, samples.size - offset),
            AudioRecord.READ_BLOCKING,
          )
        check(count > 0) { "Microphone read failed ($count)" }
        for (i in 0 until count) samples[offset + i] = block[i] / 32768f
        offset += count
        onProgress(offset.toDouble() / rate)
      }
      return GenericResampler.resample(samples, rate)
    } finally {
      recorder?.let {
        try {
          it.stop()
        } catch (_: IllegalStateException) {}
        it.release()
      }
    }
  }
}
