package com.asrlitertlm

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max

/**
 * Minimal RIFF/WAVE helpers: the length and the samples of a wav (walks the chunks; the bundled clips carry a LIST
 * chunk), a 16-bit mono writer for the microphone, a trimmed copy for clips longer than one model window, and the
 * level of a recording.
 */
object Wav {
  /** 16-bit PCM samples of a wav, interleaved when [channels] > 1. */
  class Pcm(val data: ByteArray, val sampleRate: Int, val channels: Int) {
    /** Frames (samples per channel). */
    val frames: Int
      get() = data.size / (2 * channels)

    val seconds: Double
      get() = frames.toDouble() / sampleRate
  }

  /** Peak and RMS of a recording in dBFS (-120 for digital silence). */
  data class Levels(val peakDbfs: Double, val rmsDbfs: Double)

  private class Info(
    val sampleRate: Int,
    val channels: Int,
    val bits: Int,
    val byteRate: Int,
    val dataSize: Long,
    val data: ByteArray?,
  )

  private fun read(file: File, withData: Boolean): Info {
    RandomAccessFile(file, "r").use { f ->
      val head = ByteArray(12).also { f.readFully(it) }
      require(String(head, 0, 4) == "RIFF" && String(head, 8, 4) == "WAVE") { "not a wav: $file" }
      var fmt: ByteBuffer? = null
      while (f.filePointer + 8 <= f.length()) {
        val header = ByteArray(8).also { f.readFully(it) }
        val id = String(header, 0, 4)
        val size = ByteBuffer.wrap(header, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xffffffffL
        when (id) {
          "fmt " -> {
            fmt = ByteBuffer.wrap(ByteArray(size.toInt()).also { f.readFully(it) }).order(ByteOrder.LITTLE_ENDIAN)
            if (size % 2 == 1L) f.skipBytes(1)
          }
          "data" -> {
            val format = requireNotNull(fmt) { "data before fmt in $file" }
            // A recorder that never patched the header reports 0 or 0xFFFFFFFF: take what the file holds.
            val available = f.length() - f.filePointer
            val length = if (size == 0L || size > available) available else size
            val data = if (withData) ByteArray(length.toInt()).also { f.readFully(it) } else null
            return Info(
              sampleRate = format.getInt(4),
              channels = format.getShort(2).toInt(),
              bits = format.getShort(14).toInt(),
              byteRate = format.getInt(8),
              dataSize = length,
              data = data,
            )
          }
          else -> f.seek(f.filePointer + size + (size % 2))
        }
      }
    }
    error("no data chunk in $file")
  }

  /** The length of a wav in seconds, from its header. */
  fun seconds(file: File): Double = read(file, withData = false).let { it.dataSize.toDouble() / it.byteRate }

  /** The samples of a 16-bit PCM wav. */
  fun pcm16(file: File): Pcm =
    read(file, withData = true).let {
      require(it.bits == 16) { "not 16-bit PCM: $file" }
      Pcm(it.data!!, it.sampleRate, it.channels)
    }

  /** Writes 16-bit mono PCM as a canonical 44-byte-header wav. */
  fun writeMono16(file: File, pcm: ByteArray, sampleRate: Int) {
    val header =
      ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
        put("RIFF".toByteArray())
        putInt(36 + pcm.size)
        put("WAVE".toByteArray())
        put("fmt ".toByteArray())
        putInt(16)
        putShort(1)
        putShort(1)
        putInt(sampleRate)
        putInt(sampleRate * 2)
        putShort(2)
        putShort(16)
        put("data".toByteArray())
        putInt(pcm.size)
      }
    file.outputStream().use {
      it.write(header.array())
      it.write(pcm)
    }
  }

  /**
   * Returns [source] when it is at most [maxSeconds] long; otherwise writes its first [maxSeconds] (mono, 16-bit)
   * to [target] and returns that. The models read one window of about 30 s per message.
   */
  fun trimmed(source: File, maxSeconds: Double, target: File): File {
    if (seconds(source) <= maxSeconds) return source
    val pcm = pcm16(source)
    require(pcm.channels == 1) { "only mono clips are trimmed: $source" }
    val keepBytes = (maxSeconds * pcm.sampleRate).toInt() * 2
    writeMono16(target, pcm.data.copyOf(keepBytes), pcm.sampleRate)
    return target
  }

  /** Peak and RMS of 16-bit little-endian PCM. */
  fun levels(pcm: ByteArray): Levels {
    val samples = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
    var peak = 0
    var sumOfSquares = 0.0
    for (k in 0 until samples.limit()) {
      val value = samples.get(k).toInt()
      peak = max(peak, abs(value))
      sumOfSquares += value.toDouble() * value
    }
    val count = max(samples.limit(), 1)
    return Levels(peakDbfs = toDbfs(peak.toDouble()), rmsDbfs = toDbfs(Math.sqrt(sumOfSquares / count)))
  }

  private fun toDbfs(amplitude: Double): Double =
    if (amplitude <= 0.0) SILENCE_DBFS else max(SILENCE_DBFS, 20 * log10(amplitude / 32768.0))

  private const val SILENCE_DBFS = -120.0
}
