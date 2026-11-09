package com.idol

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import java.io.File
import java.nio.ByteBuffer

/**
 * A picture as the pipeline reads it: the decoded bytes at full size, unpremultiplied, without colour conversion and
 * without the EXIF rotation (what PIL's Image.open gives the author's pipeline: idol/pipeline.py opens the file and
 * does not transpose it). RGBA when the file has an alpha channel, else RGB.
 */
class Photo(val name: String, val image: Image8) {
    val width: Int get() = image.width
    val height: Int get() = image.height

    /** A screen copy of at most [side] pixels a side (nearest sampling), opaque. */
    fun displayBitmap(side: Int = 1024): Bitmap = Photo.displayBitmap(image, side)

    companion object {
        /** Pictures above this many pixels are decoded at half size per side (and again) to bound the memory. */
        const val MAX_PIXELS = 40_000_000L

        fun fromUri(resolver: ContentResolver, uri: Uri, name: String): Photo {
            val bytes = requireNotNull(resolver.openInputStream(uri)) { "cannot open $uri" }.use { it.readBytes() }
            return Photo(name, decode(bytes))
        }

        fun fromFile(file: File): Photo = Photo(file.name, decode(file.readBytes()))

        private fun decode(bytes: ByteArray): Image8 {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
            require(bounds.outWidth > 0 && bounds.outHeight > 0) { "not an image the phone can decode" }
            var sample = 1
            while (bounds.outWidth.toLong() * bounds.outHeight / (sample.toLong() * sample) > MAX_PIXELS) sample *= 2
            val options = BitmapFactory.Options().apply {
                inPremultiplied = false
                inPreferredConfig = Bitmap.Config.ARGB_8888
                inSampleSize = sample
            }
            val decoded = requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)) {
                "not an image the phone can decode"
            }
            val bitmap = if (decoded.config == Bitmap.Config.ARGB_8888) decoded else decoded.copy(Bitmap.Config.ARGB_8888, false)
            val w = bitmap.width
            val h = bitmap.height
            // copyPixelsToBuffer keeps the stored values (getPixels would convert colour spaces); ARGB_8888 is R, G, B, A
            val rgba = ByteBuffer.allocateDirect(w * h * 4)
            bitmap.copyPixelsToBuffer(rgba)
            rgba.rewind()
            val hasAlpha = bitmap.hasAlpha()
            val ch = if (hasAlpha) 4 else 3
            val out = ByteArray(w * h * ch)
            val row = ByteArray(w * 4)
            for (y in 0 until h) {
                rgba.get(row)
                var o = y * w * ch
                for (x in 0 until w) {
                    out[o] = row[x * 4]
                    out[o + 1] = row[x * 4 + 1]
                    out[o + 2] = row[x * 4 + 2]
                    if (hasAlpha) out[o + 3] = row[x * 4 + 3]
                    o += ch
                }
            }
            if (bitmap !== decoded) bitmap.recycle()
            decoded.recycle()
            return Image8(w, h, ch, out)
        }

        fun displayBitmap(image: Image8, side: Int): Bitmap {
            val step = maxOf(1, (maxOf(image.width, image.height) + side - 1) / side)
            val dw = maxOf(1, image.width / step)
            val dh = maxOf(1, image.height / step)
            val ch = image.channels
            val s = image.data
            val px = IntArray(dw * dh)
            for (y in 0 until dh) for (x in 0 until dw) {
                val i = ((y * step) * image.width + x * step) * ch
                px[y * dw + x] = (0xFF shl 24) or ((s[i].toInt() and 0xFF) shl 16) or
                    ((s[i + 1].toInt() and 0xFF) shl 8) or (s[i + 2].toInt() and 0xFF)
            }
            return Bitmap.createBitmap(px, dw, dh, Bitmap.Config.ARGB_8888)
        }
    }
}
