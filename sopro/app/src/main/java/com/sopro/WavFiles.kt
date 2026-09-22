// SPDX-License-Identifier: Apache-2.0
package com.sopro

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt
import org.json.JSONObject

object WavFiles {
  data class Saved(val wav: File, val sidecar: File)

  fun write(
    directory: File,
    lang: String,
    referenceId: String,
    pcm: FloatArray,
    stats: JSONObject,
  ): Saved {
    require(lang in listOf("en", "pt", "fr", "de"))
    require(pcm.all { it.isFinite() })
    directory.mkdirs()
    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
    val base = "${timestamp}_$lang"
    val wav = File(directory, "$base.wav")
    val bytes = ByteBuffer.allocate(44 + pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    bytes.put("RIFF".toByteArray(Charsets.US_ASCII))
    bytes.putInt(36 + pcm.size * 2)
    bytes.put("WAVEfmt ".toByteArray(Charsets.US_ASCII))
    bytes.putInt(16)
    bytes.putShort(1)
    bytes.putShort(1)
    bytes.putInt(24000)
    bytes.putInt(48000)
    bytes.putShort(2)
    bytes.putShort(16)
    bytes.put("data".toByteArray(Charsets.US_ASCII))
    bytes.putInt(pcm.size * 2)
    pcm.forEach {
      bytes.putShort((it.coerceIn(-1f, 1f) * 32768f).roundToInt().coerceIn(-32768, 32767).toShort())
    }
    wav.writeBytes(bytes.array())
    val sidecar = File(directory, "$base.json")
    sidecar.writeText(
      JSONObject()
        .put("format", "PCM16")
        .put("sample_rate_hz", 24000)
        .put("channels", 1)
        .put("samples", pcm.size)
        .put("language", lang)
        .put("reference_id", referenceId)
        .put("wav", wav.name)
        .put("stats", stats)
        .toString(2) + "\n"
    )
    return Saved(wav, sidecar)
  }

  fun readPcm16(file: File): FloatArray = readPcm16(file.readBytes())

  fun readPcm16(bytes: ByteArray): FloatArray {
    require(bytes.size >= 44)
    val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    require(
      String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
        String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE"
    )
    var pos = 12
    var verified = false
    while (pos + 8 <= bytes.size) {
      val tag = String(bytes, pos, 4, Charsets.US_ASCII)
      val size = b.getInt(pos + 4)
      require(size >= 0 && pos + 8L + size <= bytes.size)
      if (tag == "fmt ") {
        require(
          size >= 16 &&
            b.getShort(pos + 8).toInt() == 1 &&
            b.getShort(pos + 10).toInt() == 1 &&
            b.getInt(pos + 12) == 24000 &&
            b.getShort(pos + 22).toInt() == 16
        ) {
          "Expected mono 24 kHz PCM16 WAV"
        }
        verified = true
      } else if (tag == "data") {
        require(verified && size % 2 == 0)
        return FloatArray(size / 2) { b.getShort(pos + 8 + it * 2) / 32768f }
      }
      pos += 8 + size + (size and 1)
    }
    error("WAV has no PCM data")
  }
}
