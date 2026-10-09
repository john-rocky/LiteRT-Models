package com.d1omni

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.zip.Inflater

/**
 * The JVM's stand-in for [D1Image]: a plain PNG reader (the chunks, zlib, the five row filters) that
 * returns the stored samples as RGB, with no colour management and no EXIF, so the tests compare
 * Pillow's decode with the file's own samples. The Android unit tests compile against android.jar,
 * which has no javax.imageio. 8-bit non-interlaced grey, RGB, palette, grey + alpha and RGBA;
 * alpha dropped as Pillow's `convert("RGB")` drops it.
 */
internal object D1ImageJvm {
  fun decode(bytes: ByteArray): D1Rgb {
    require(D1ImageOps.isPng(bytes)) { "not a PNG" }
    var position = 8
    var width = 0
    var height = 0
    var depth = 0
    var colorType = 0
    var interlace = 0
    var palette: ByteArray? = null
    val compressed = ByteArrayOutputStream()
    while (position + 12 <= bytes.size) {
      val length = ByteBuffer.wrap(bytes, position, 4).int
      val type = String(bytes, position + 4, 4, Charsets.ISO_8859_1)
      val data = bytes.copyOfRange(position + 8, position + 8 + length)
      when (type) {
        "IHDR" -> {
          width = ByteBuffer.wrap(data, 0, 4).int
          height = ByteBuffer.wrap(data, 4, 4).int
          depth = data[8].toInt()
          colorType = data[9].toInt()
          interlace = data[12].toInt()
        }
        "PLTE" -> palette = data
        "IDAT" -> compressed.write(data)
      }
      position += 12 + length
      if (type == "IEND") break
    }
    require(depth == 8 && interlace == 0) { "only 8-bit non-interlaced PNGs (depth $depth, interlace $interlace)" }
    val channels =
      when (colorType) {
        0, 3 -> 1
        2 -> 3
        4 -> 2
        6 -> 4
        else -> throw IllegalArgumentException("PNG colour type $colorType")
      }
    val stride = width * channels
    val raw = ByteArray((stride + 1) * height)
    val inflater = Inflater()
    inflater.setInput(compressed.toByteArray())
    var filled = 0
    while (filled < raw.size) {
      val count = inflater.inflate(raw, filled, raw.size - filled)
      if (count == 0 && (inflater.finished() || inflater.needsInput())) break
      filled += count
    }
    inflater.end()
    require(filled == raw.size) { "zlib gave $filled of ${raw.size} bytes" }
    val samples = ByteArray(stride * height)
    for (y in 0 until height) {
      val filter = raw[y * (stride + 1)].toInt()
      val source = y * (stride + 1) + 1
      val row = y * stride
      for (x in 0 until stride) {
        val value = raw[source + x].toInt() and 0xff
        val a = if (x >= channels) samples[row + x - channels].toInt() and 0xff else 0
        val b = if (y > 0) samples[row - stride + x].toInt() and 0xff else 0
        val c = if (y > 0 && x >= channels) samples[row - stride + x - channels].toInt() and 0xff else 0
        val predicted =
          when (filter) {
            0 -> 0
            1 -> a
            2 -> b
            3 -> (a + b) / 2
            4 -> {
              val p = a + b - c
              val pa = Math.abs(p - a)
              val pb = Math.abs(p - b)
              val pc = Math.abs(p - c)
              if (pa <= pb && pa <= pc) a else if (pb <= pc) b else c
            }
            else -> throw IllegalArgumentException("PNG row filter $filter")
          }
        samples[row + x] = (value + predicted).toByte()
      }
    }
    val out = ByteArray(width * height * 3)
    for (i in 0 until width * height) {
      for (k in 0 until 3) {
        out[i * 3 + k] =
          when (colorType) {
            0, 4 -> samples[i * channels]
            3 -> requireNotNull(palette) { "palette PNG without PLTE" }[(samples[i].toInt() and 0xff) * 3 + k]
            else -> samples[i * channels + k]
          }
      }
    }
    return D1Rgb(width, height, out)
  }
}
