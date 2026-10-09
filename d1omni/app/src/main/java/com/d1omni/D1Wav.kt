package com.d1omni

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A RIFF / WAVE reader for the clips the model takes, Android-free: 16 kHz, mono, 16-bit PCM
 * (format 1, or WAVE_FORMAT_EXTENSIBLE with the PCM sub-format) -> the int16 samples, as the model
 * repository's `read_audio` gives them (`soundfile.read(path, dtype="int16")`). Another rate or more
 * than one channel is refused with the same reason as `read_audio` (resampling is not part of the
 * provider's code); another sample format is refused too (`soundfile` would convert it to int16 on
 * the way, this reader does not). Chunks other than `fmt ` and `data` (LIST, fact, …) are skipped,
 * odd-sized chunks keep their pad byte, and a `data` chunk longer than the file is read up to the
 * last whole sample.
 */
object D1Wav {
  const val SAMPLE_RATE = D1Audio.SAMPLE_RATE

  private const val PCM = 1
  private const val EXTENSIBLE = 0xFFFE
  private const val HEADER_BYTES = 12
  private const val CHUNK_HEADER_BYTES = 8
  private const val FMT_MIN_BYTES = 16
  private const val EXTENSIBLE_FMT_BYTES = 40
  private const val BITS = 16

  /** The PCM sub-format GUID's first two bytes (0x0001) and the 14 bytes every KSDATAFORMAT GUID shares. */
  private val GUID_TAIL =
    byteArrayOf(0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80.toByte(), 0x00, 0x00, 0xAA.toByte(), 0x00, 0x38, 0x9B.toByte(), 0x71)

  /** The int16 samples of [file] (see the class comment for what is refused). */
  fun read(file: File): ShortArray = parse(file.readBytes(), file.name)

  /** The int16 samples of a wav file's [bytes]; [name] goes into the error messages. */
  fun parse(bytes: ByteArray, name: String = "wav"): ShortArray {
    require(
      bytes.size >= HEADER_BYTES &&
        String(bytes, 0, 4, Charsets.ISO_8859_1) == "RIFF" &&
        String(bytes, 8, 4, Charsets.ISO_8859_1) == "WAVE"
    ) {
      "$name: not a RIFF / WAVE file"
    }
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    var position = HEADER_BYTES
    var format: Format? = null
    while (position + CHUNK_HEADER_BYTES <= bytes.size) {
      val id = String(bytes, position, 4, Charsets.ISO_8859_1)
      val size = buffer.getInt(position + 4).toLong() and 0xFFFFFFFFL
      val body = position + CHUNK_HEADER_BYTES
      when (id) {
        "fmt " -> {
          require(size >= FMT_MIN_BYTES && body + size <= bytes.size) { "$name: a fmt chunk of $size bytes" }
          format = Format.of(buffer, body, size.toInt(), name)
        }
        "data" -> {
          val fmt = requireNotNull(format) { "$name: the data chunk comes before the fmt chunk" }
          fmt.check(name)
          val available = minOf(size, (bytes.size - body).toLong()).toInt()
          val samples = ShortArray(available / 2)
          buffer.position(body)
          buffer.asShortBuffer().get(samples)
          return samples
        }
      }
      position = (body + size + (size and 1L)).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    }
    throw IllegalArgumentException("$name: no data chunk")
  }

  private class Format(
    val code: Int,
    val channels: Int,
    val sampleRate: Int,
    val blockAlign: Int,
    val bits: Int,
    val subFormatPcm: Boolean,
  ) {
    fun check(name: String) {
      require(code == PCM || (code == EXTENSIBLE && subFormatPcm)) {
        "$name: format $code is not 16-bit PCM; the model takes 16 kHz mono 16-bit PCM"
      }
      require(sampleRate == SAMPLE_RATE) {
        "$name: $sampleRate Hz; the model takes $SAMPLE_RATE Hz mono (resample it first)"
      }
      require(channels == 1) { "$name: $channels channels; the model takes one (mono)" }
      require(bits == BITS && blockAlign == 2) {
        "$name: $bits-bit samples; the model takes 16 kHz mono 16-bit PCM"
      }
    }

    companion object {
      fun of(buffer: ByteBuffer, body: Int, size: Int, name: String): Format {
        val code = buffer.getShort(body).toInt() and 0xFFFF
        var pcm = false
        if (code == EXTENSIBLE) {
          require(size >= EXTENSIBLE_FMT_BYTES) { "$name: an extensible fmt chunk of $size bytes" }
          // cbSize (2) + valid bits (2) + channel mask (4), then the 16-byte sub-format GUID.
          val guid = body + 24
          pcm =
            (buffer.getShort(guid).toInt() and 0xFFFF) == PCM &&
              (0 until GUID_TAIL.size).all { buffer.get(guid + 2 + it) == GUID_TAIL[it] }
        }
        return Format(
          code,
          buffer.getShort(body + 2).toInt() and 0xFFFF,
          buffer.getInt(body + 4),
          buffer.getShort(body + 12).toInt() and 0xFFFF,
          buffer.getShort(body + 14).toInt() and 0xFFFF,
          pcm,
        )
      }
    }
  }
}
