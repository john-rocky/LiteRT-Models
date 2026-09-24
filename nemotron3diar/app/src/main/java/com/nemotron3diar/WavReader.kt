package com.nemotron3diar

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Minimal RIFF/WAVE reader: 16-bit PCM or 32-bit float, mono (or the channel mean), any chunk layout. */
object WavReader {

  class Wav(val samples: FloatArray, val sampleRate: Int)

  /** int16 samples scale by 1/32768, as soundfile / libsndfile read them. */
  fun read(file: File): Wav {
    val bb = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    require(bb.remaining() >= 12 && tag(bb, 0) == "RIFF" && tag(bb, 8) == "WAVE") { "${file.name}: not a WAVE file" }
    var pos = 12
    var format = -1
    var channels = 0
    var rate = 0
    var bits = 0
    var dataPos = -1
    var dataLen = 0
    while (pos + 8 <= bb.limit()) {
      val id = tag(bb, pos)
      val len = bb.getInt(pos + 4)
      val body = pos + 8
      when (id) {
        "fmt " -> {
          format = bb.getShort(body).toInt() and 0xffff
          channels = bb.getShort(body + 2).toInt()
          rate = bb.getInt(body + 4)
          bits = bb.getShort(body + 14).toInt()
          if (format == 0xfffe && len >= 26) format = bb.getShort(body + 24).toInt() and 0xffff // extensible
        }
        "data" -> {
          dataPos = body
          dataLen = minOf(len, bb.limit() - body)
        }
      }
      pos = body + len + (len and 1)
    }
    require(dataPos >= 0 && channels > 0) { "${file.name}: no fmt / data chunk" }
    val frameBytes = channels * bits / 8
    val n = dataLen / frameBytes
    val out = FloatArray(n)
    for (i in 0 until n) {
      var acc = 0f
      for (c in 0 until channels) {
        val at = dataPos + i * frameBytes + c * bits / 8
        acc +=
          when {
            format == 1 && bits == 16 -> bb.getShort(at) / 32768f
            format == 3 && bits == 32 -> bb.getFloat(at)
            else -> error("${file.name}: unsupported WAVE format $format / $bits bit")
          }
      }
      out[i] = if (channels == 1) acc else acc / channels
    }
    return Wav(out, rate)
  }

  private fun tag(bb: ByteBuffer, at: Int) = String(ByteArray(4) { bb.get(at + it) }, Charsets.US_ASCII)
}
