package com.d1omni

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ExifInterface
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer

/** A decoded picture: its RGB values, the EXIF orientation applied, and what the decoder reported. */
class D1Decoded(val rgb: D1Rgb, val orientation: Int, val colorSpace: String?, val strippedChunks: List<String>)

/**
 * The parts of `load_image` (Pillow: open, `ImageOps.exif_transpose`, RGB) that do not need Android,
 * so that the JVM tests cover them: the EXIF orientation as Pillow applies it, and the PNG colour
 * chunks that Pillow ignores and Android's decoder would honour.
 */
object D1ImageOps {
  private val PNG_SIGNATURE =
    byteArrayOf(0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)

  /**
   * PNG chunks that describe colour (an ICC profile, the sRGB intent, gamma, primaries, coding-
   * independent code points, HDR metadata). Pillow returns the stored samples whatever they say;
   * BitmapFactory may convert the samples to another colour space.
   */
  val COLOR_CHUNKS = setOf("iCCP", "sRGB", "gAMA", "cHRM", "cICP", "mDCv", "cLLi")

  fun isPng(bytes: ByteArray): Boolean =
    bytes.size >= PNG_SIGNATURE.size && PNG_SIGNATURE.indices.all { bytes[it] == PNG_SIGNATURE[it] }

  /**
   * [bytes] without the [COLOR_CHUNKS] of a PNG (every other chunk unchanged, in order) and the
   * names removed; anything that is not a well-formed PNG chunk list comes back unchanged.
   */
  fun stripPngColorChunks(bytes: ByteArray): Pair<ByteArray, List<String>> {
    if (!isPng(bytes)) return bytes to emptyList()
    val out = java.io.ByteArrayOutputStream(bytes.size)
    out.write(bytes, 0, PNG_SIGNATURE.size)
    val removed = ArrayList<String>()
    var position = PNG_SIGNATURE.size
    while (position < bytes.size) {
      if (position + 12 > bytes.size) return bytes to emptyList()
      val length = ByteBuffer.wrap(bytes, position, 4).int
      if (length < 0 || position + 12L + length > bytes.size) return bytes to emptyList()
      val type = String(bytes, position + 4, 4, Charsets.ISO_8859_1)
      val total = 12 + length
      if (type in COLOR_CHUNKS) removed.add(type) else out.write(bytes, position, total)
      position += total
      if (type == "IEND") break
    }
    return out.toByteArray() to removed
  }

  /**
   * Pillow's `exif_transpose` for EXIF orientation [orientation]: 2 flip left-right, 3 rotate 180,
   * 4 flip top-bottom, 5 transpose, 6 rotate 270 (counter-clockwise), 7 transverse, 8 rotate 90;
   * 1 and anything else leave the picture as it is.
   */
  fun orient(rgb: D1Rgb, orientation: Int): D1Rgb {
    if (orientation !in 2..8) return rgb
    val w = rgb.width
    val h = rgb.height
    val swap = orientation >= 5
    val outWidth = if (swap) h else w
    val outHeight = if (swap) w else h
    val out = ByteArray(rgb.data.size)
    for (y in 0 until outHeight) {
      for (x in 0 until outWidth) {
        val (sx, sy) =
          when (orientation) {
            2 -> (w - 1 - x) to y
            3 -> (w - 1 - x) to (h - 1 - y)
            4 -> x to (h - 1 - y)
            5 -> y to x
            6 -> y to (h - 1 - x)
            7 -> (w - 1 - y) to (h - 1 - x)
            else -> (w - 1 - y) to x
          }
        System.arraycopy(rgb.data, (sy * w + sx) * 3, out, (y * outWidth + x) * 3, 3)
      }
    }
    return D1Rgb(outWidth, outHeight, out)
  }
}

/**
 * `load_image` on Android: the file's bytes (a PNG without its [D1ImageOps.COLOR_CHUNKS], so that
 * the decoder returns the stored samples as Pillow does) decoded by BitmapFactory into an
 * unpremultiplied ARGB_8888 bitmap whose raw bytes are read (`copyPixelsToBuffer`: no colour
 * conversion, unlike `getPixels`), alpha dropped, then the EXIF orientation read by ExifInterface
 * (JPEG, and PNG eXIf on Android 11+) applied as Pillow applies it. Android's `createScaledBitmap` or
 * a Canvas never resize here: the resize is [D1Vision.resizeFloat].
 */
object D1Image {
  fun decode(file: File): D1Decoded = decode(file.readBytes())

  fun decode(raw: ByteArray): D1Decoded {
    val (bytes, stripped) = D1ImageOps.stripPngColorChunks(raw)
    val options =
      BitmapFactory.Options().apply {
        inPreferredConfig = Bitmap.Config.ARGB_8888
        inPremultiplied = false
        inScaled = false
      }
    val bitmap =
      requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)) {
        "BitmapFactory could not decode the picture (${raw.size} bytes)"
      }
    val rgb: D1Rgb
    val colorSpace: String?
    try {
      check(bitmap.config == Bitmap.Config.ARGB_8888) { "decoded as ${bitmap.config}, not ARGB_8888" }
      colorSpace = if (android.os.Build.VERSION.SDK_INT >= 26) bitmap.colorSpace?.name else null
      val width = bitmap.width
      val height = bitmap.height
      val stride = bitmap.rowBytes
      val buffer = ByteBuffer.allocate(stride * height)
      bitmap.copyPixelsToBuffer(buffer)
      val rgba = buffer.array()
      val data = ByteArray(width * height * 3)
      for (y in 0 until height) {
        for (x in 0 until width) {
          val source = y * stride + x * 4
          val target = (y * width + x) * 3
          data[target] = rgba[source]
          data[target + 1] = rgba[source + 1]
          data[target + 2] = rgba[source + 2]
        }
      }
      rgb = D1Rgb(width, height, data)
    } finally {
      bitmap.recycle()
    }
    val orientation =
      runCatching {
          ExifInterface(ByteArrayInputStream(raw))
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        .getOrDefault(ExifInterface.ORIENTATION_NORMAL)
    return D1Decoded(D1ImageOps.orient(rgb, orientation), orientation, colorSpace, stripped)
  }
}
