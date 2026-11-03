package com.reuse

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** RIFF / WAVE reader (PCM 8 / 16 / 24 / 32-bit, IEEE float 32 / 64, channels averaged to mono) and float32 writer. */
object WavFile {

  class Audio(val samples: FloatArray, val sampleRate: Int)

  fun isWav(bytes: ByteArray) =
    bytes.size >= 12 && String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
      String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE"

  fun read(file: File): Audio = read(file.readBytes())

  fun read(bytes: ByteArray): Audio {
    require(isWav(bytes)) { "Not a RIFF/WAVE file" }
    val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    var format = -1
    var channels = 0
    var rate = 0
    var bits = 0
    var dataOff = -1
    var dataLen = 0
    var pos = 12
    while (pos + 8 <= bytes.size && dataOff < 0) {
      val id = String(bytes, pos, 4, Charsets.US_ASCII)
      val len = b.getInt(pos + 4)
      val body = pos + 8
      if (id == "fmt ") {
        format = b.getShort(body).toInt() and 0xffff
        channels = b.getShort(body + 2).toInt()
        rate = b.getInt(body + 4)
        bits = b.getShort(body + 14).toInt()
        // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts with the format code.
        if (format == 0xFFFE && len >= 26) format = b.getShort(body + 24).toInt() and 0xffff
      } else if (id == "data") {
        dataOff = body
        dataLen = if (len < 0 || body + len > bytes.size) bytes.size - body else len
      }
      if (len < 0) break
      pos = body + len + (len and 1) // chunks are word aligned
    }
    require(format >= 0 && dataOff >= 0 && channels > 0 && rate > 0) { "WAV without a fmt or data chunk" }
    val width = bits / 8
    require((format == 1 && bits in intArrayOf(8, 16, 24, 32)) || (format == 3 && bits in intArrayOf(32, 64))) {
      "Unsupported WAV encoding: format $format, $bits bits"
    }
    val frames = dataLen / (width * channels)
    val out = FloatArray(frames)
    for (i in 0 until frames) {
      var acc = 0.0
      for (c in 0 until channels) {
        val o = dataOff + (i * channels + c) * width
        acc +=
          when {
            format == 3 && bits == 32 -> b.getFloat(o).toDouble()
            format == 3 -> b.getDouble(o)
            bits == 8 -> ((bytes[o].toInt() and 0xff) - 128) / 128.0
            bits == 16 -> b.getShort(o) / 32768.0
            bits == 24 ->
              (((bytes[o].toInt() and 0xff) or ((bytes[o + 1].toInt() and 0xff) shl 8) or
                (bytes[o + 2].toInt() shl 16))) / 8388608.0
            else -> b.getInt(o) / 2147483648.0
          }
      }
      out[i] = (acc / channels).toFloat()
    }
    return Audio(out, rate)
  }

  /** Writes mono float32 samples as a WAVE_FORMAT_IEEE_FLOAT file (fmt, fact and data chunks). */
  fun writeFloat(file: File, x: FloatArray, sampleRate: Int) {
    val data = x.size * 4
    val b = ByteBuffer.allocate(58 + data).order(ByteOrder.LITTLE_ENDIAN)
    b.put("RIFF".toByteArray(Charsets.US_ASCII)).putInt(50 + data).put("WAVE".toByteArray(Charsets.US_ASCII))
    b.put("fmt ".toByteArray(Charsets.US_ASCII)).putInt(18)
    b.putShort(3).putShort(1).putInt(sampleRate).putInt(sampleRate * 4).putShort(4).putShort(32).putShort(0)
    b.put("fact".toByteArray(Charsets.US_ASCII)).putInt(4).putInt(x.size)
    b.put("data".toByteArray(Charsets.US_ASCII)).putInt(data)
    x.forEach { b.putFloat(it) }
    file.writeBytes(b.array())
  }
}
