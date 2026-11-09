package com.idol

import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.sin

/** An 8-bit image, channels interleaved, rows top to bottom: what a PIL image of mode RGB (3) or RGBA (4) holds. */
class Image8(val width: Int, val height: Int, val channels: Int, val data: ByteArray) {
    init {
        require(data.size == width * height * channels) { "${data.size} bytes for ${width}x${height}x$channels" }
    }

    fun copy() = Image8(width, height, channels, data.copyOf())

    /** One channel as a width x height plane. */
    fun channel(c: Int): ByteArray {
        val out = ByteArray(width * height)
        var p = c
        for (i in out.indices) {
            out[i] = data[p]
            p += channels
        }
        return out
    }

    /** PIL's convert("RGB"): the first three channels. */
    fun rgb(): Image8 {
        if (channels == 3) return this
        val out = ByteArray(width * height * 3)
        var p = 0
        var o = 0
        for (i in 0 until width * height) {
            out[o] = data[p]
            out[o + 1] = data[p + 1]
            out[o + 2] = data[p + 2]
            p += channels
            o += 3
        }
        return Image8(width, height, 3, out)
    }

    /** PIL's putalpha(L): an RGBA image with this image's colours and [alpha] as its alpha channel. */
    fun withAlpha(alpha: ByteArray): Image8 {
        require(alpha.size == width * height)
        val out = ByteArray(width * height * 4)
        var p = 0
        var o = 0
        for (i in 0 until width * height) {
            out[o] = data[p]
            out[o + 1] = data[p + 1]
            out[o + 2] = data[p + 2]
            out[o + 3] = alpha[i]
            p += channels
            o += 4
        }
        return Image8(width, height, 4, out)
    }
}

/**
 * The parts of Pillow 13.0.0 the author's preprocessing uses (idol/preprocess.py, scripts/idol_glue.py preprocess),
 * pixel for pixel, from Pillow's src/libImaging: Resample.c (Lanczos, 22-bit fixed-point coefficients, the pass order
 * of ImagingResampleInner), Convert.c (RGBA <-> RGBa around a resize), RankFilter.c + Filter.c (MinFilter(3) with edge
 * replication), Crop.c (zero fill outside the image) and Paste.c (paste with an L mask).
 * Pillow: Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh and contributors,
 * Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors, under the MIT-CMU License (licenses/Pillow-MIT-CMU.txt).
 */
object Pil {
    private const val PRECISION_BITS = 32 - 8 - 2

    private class Coeffs(val ksize: Int, val bounds: IntArray, val kk: IntArray)

    private fun sinc(x0: Double): Double {
        if (x0 == 0.0) return 1.0
        val x = x0 * PI
        return sin(x) / x
    }

    /** lanczos_filter: truncated sinc, support 3. */
    private fun lanczos(x: Double): Double = if (-3.0 <= x && x < 3.0) sinc(x) * sinc(x / 3) else 0.0

    /** precompute_coeffs for the box [0, inSize) -> outSize, then normalize_coeffs_8bpc. */
    private fun coeffs(inSize: Int, outSize: Int): Coeffs {
        val in0 = 0.0f
        val in1 = inSize.toFloat()
        val scale = (in1 - in0).toDouble() / outSize
        val filterscale = if (scale < 1.0) 1.0 else scale
        val support = 3.0 * filterscale
        val ksize = ceil(support).toInt() * 2 + 1
        val pre = DoubleArray(outSize * ksize)
        val bounds = IntArray(outSize * 2)
        val invFilterscale = 1.0 / filterscale
        for (xx in 0 until outSize) {
            val center = in0 + (xx + 0.5) * scale
            var ww = 0.0
            var xmin = (center - support + 0.5).toInt()
            if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt()
            if (xmax > inSize) xmax = inSize
            xmax -= xmin
            val base = xx * ksize
            for (x in 0 until xmax) {
                val w = lanczos((x + xmin - center + 0.5) * invFilterscale)
                pre[base + x] = w
                ww += w
            }
            if (ww != 0.0) for (x in 0 until xmax) pre[base + x] /= ww
            bounds[xx * 2] = xmin
            bounds[xx * 2 + 1] = xmax
        }
        val one = (1 shl PRECISION_BITS).toDouble()
        val kk = IntArray(pre.size) { i ->
            val v = pre[i]
            if (v < 0) (-0.5 + v * one).toInt() else (0.5 + v * one).toInt()
        }
        return Coeffs(ksize, bounds, kk)
    }

    private fun clip8(ss: Int): Byte {
        val v = ss shr PRECISION_BITS
        return (if (v < 0) 0 else if (v > 255) 255 else v).toByte()
    }

    /** _ImagingResampleHorizontal_8bpc: [rows] output rows from input rows offset.. */
    private fun horizontal(src: Image8, outWidth: Int, rows: Int, offset: Int, c: Coeffs): Image8 {
        val ch = src.channels
        val inW = src.width
        val s = src.data
        val out = ByteArray(outWidth * rows * ch)
        val half = 1 shl (PRECISION_BITS - 1)
        val acc = IntArray(ch)
        for (yy in 0 until rows) {
            val inRow = (yy + offset) * inW * ch
            val outRow = yy * outWidth * ch
            for (xx in 0 until outWidth) {
                val xmin = c.bounds[xx * 2]
                val xmax = c.bounds[xx * 2 + 1]
                val k = xx * c.ksize
                val o = outRow + xx * ch
                when (ch) {
                    3 -> {
                        var s0 = half
                        var s1 = half
                        var s2 = half
                        var p = inRow + xmin * 3
                        for (x in 0 until xmax) {
                            val w = c.kk[k + x]
                            s0 += (s[p].toInt() and 0xFF) * w
                            s1 += (s[p + 1].toInt() and 0xFF) * w
                            s2 += (s[p + 2].toInt() and 0xFF) * w
                            p += 3
                        }
                        out[o] = clip8(s0)
                        out[o + 1] = clip8(s1)
                        out[o + 2] = clip8(s2)
                    }
                    4 -> {
                        var s0 = half
                        var s1 = half
                        var s2 = half
                        var s3 = half
                        var p = inRow + xmin * 4
                        for (x in 0 until xmax) {
                            val w = c.kk[k + x]
                            s0 += (s[p].toInt() and 0xFF) * w
                            s1 += (s[p + 1].toInt() and 0xFF) * w
                            s2 += (s[p + 2].toInt() and 0xFF) * w
                            s3 += (s[p + 3].toInt() and 0xFF) * w
                            p += 4
                        }
                        out[o] = clip8(s0)
                        out[o + 1] = clip8(s1)
                        out[o + 2] = clip8(s2)
                        out[o + 3] = clip8(s3)
                    }
                    else -> {
                        acc.fill(half)
                        var p = inRow + xmin * ch
                        for (x in 0 until xmax) {
                            val w = c.kk[k + x]
                            for (b in 0 until ch) acc[b] += (s[p + b].toInt() and 0xFF) * w
                            p += ch
                        }
                        for (b in 0 until ch) out[o + b] = clip8(acc[b])
                    }
                }
            }
        }
        return Image8(outWidth, rows, ch, out)
    }

    /** _ImagingResampleVertical_8bpc over every column of [src]. */
    private fun vertical(src: Image8, outHeight: Int, c: Coeffs): Image8 {
        val ch = src.channels
        val w = src.width
        val stride = w * ch
        val s = src.data
        val out = ByteArray(w * outHeight * ch)
        val half = 1 shl (PRECISION_BITS - 1)
        val acc = IntArray(stride)
        for (yy in 0 until outHeight) {
            val ymin = c.bounds[yy * 2]
            val ymax = c.bounds[yy * 2 + 1]
            val k = yy * c.ksize
            // the same sums as Pillow's per-pixel loop (integer adds, so the order does not change them), row by row
            acc.fill(half)
            for (y in 0 until ymax) {
                val wgt = c.kk[k + y]
                val row = (y + ymin) * stride
                for (i in 0 until stride) acc[i] += (s[row + i].toInt() and 0xFF) * wgt
            }
            val o = yy * stride
            for (i in 0 until stride) out[o + i] = clip8(acc[i])
        }
        return Image8(w, outHeight, ch, out)
    }

    /** ImagingResample with LANCZOS on an 8-bit image (any channel count): no premultiplication here. */
    fun resizeLanczos(img: Image8, outWidth: Int, outHeight: Int): Image8 {
        val inW = img.width
        val inH = img.height
        val needHorizontal = outWidth != inW
        val needVertical = outHeight != inH
        if (!needHorizontal && !needVertical) return img.copy()
        // If height is being scaled down more than twice the amount the width is, run the vertical pass first.
        val horizontalFirst = !((inH - outHeight) > 0 && (inH - outHeight) > (inW - outWidth) * 2)
        var vert: Coeffs? = null
        if ((needHorizontal && horizontalFirst) || needVertical) vert = coeffs(inH, outHeight)
        var yFirst = 0
        var yLast = 0
        var horiz: Coeffs? = null
        if (needHorizontal) {
            if (horizontalFirst) {
                val v = vert!!
                yFirst = v.bounds[0]
                yLast = v.bounds[outHeight * 2 - 2] + v.bounds[outHeight * 2 - 1]
                if (yFirst != 0) for (i in 0 until outHeight) v.bounds[i * 2] -= yFirst
            }
            horiz = coeffs(inW, outWidth)
        }
        var cur = img
        if (horizontalFirst) {
            if (needHorizontal) cur = horizontal(cur, outWidth, yLast - yFirst, yFirst, horiz!!)
            if (needVertical) cur = vertical(cur, outHeight, vert!!)
        } else {
            if (needVertical) cur = vertical(cur, outHeight, vert!!)
            if (needHorizontal) cur = horizontal(cur, outWidth, outHeight, 0, horiz!!)
        }
        return cur
    }

    /**
     * Image.resize(size, LANCZOS): an RGBA image goes through RGBa (premultiplied) and back, as Pillow does; RGB and
     * single-channel images are resized directly; the same size returns a copy.
     */
    fun resize(img: Image8, outWidth: Int, outHeight: Int): Image8 {
        if (outWidth == img.width && outHeight == img.height) return img.copy()
        if (img.channels != 4) return resizeLanczos(img, outWidth, outHeight)
        return unpremultiply(resizeLanczos(premultiply(img), outWidth, outHeight))
    }

    /** Convert.c rgbA2rgba (RGBA -> RGBa): MULDIV255(c, a). */
    fun premultiply(img: Image8): Image8 {
        require(img.channels == 4)
        val s = img.data
        val out = ByteArray(s.size)
        var p = 0
        while (p < s.size) {
            val a = s[p + 3].toInt() and 0xFF
            for (b in 0 until 3) {
                val t = (s[p + b].toInt() and 0xFF) * a + 128
                out[p + b] = (((t shr 8) + t) shr 8).toByte()
            }
            out[p + 3] = s[p + 3]
            p += 4
        }
        return Image8(img.width, img.height, 4, out)
    }

    /** Convert.c rgba2rgbA (RGBa -> RGBA): CLIP8(255 * c / a), alpha 0 and 255 unchanged. */
    fun unpremultiply(img: Image8): Image8 {
        require(img.channels == 4)
        val s = img.data
        val out = ByteArray(s.size)
        var p = 0
        while (p < s.size) {
            val a = s[p + 3].toInt() and 0xFF
            if (a == 255 || a == 0) {
                out[p] = s[p]
                out[p + 1] = s[p + 1]
                out[p + 2] = s[p + 2]
            } else {
                for (b in 0 until 3) {
                    val v = (255 * (s[p + b].toInt() and 0xFF)) / a
                    out[p + b] = (if (v <= 0) 0 else if (v < 256) v else 255).toByte()
                }
            }
            out[p + 3] = s[p + 3]
            p += 4
        }
        return Image8(img.width, img.height, 4, out)
    }

    /** ImageFilter.MinFilter(3) on an L plane: the 3 x 3 minimum, edges replicated (ImagingExpand + RankFilter). */
    fun minFilter3(src: ByteArray, w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h)
        for (y in 0 until h) {
            val y0 = if (y > 0) y - 1 else 0
            val y1 = if (y < h - 1) y + 1 else h - 1
            for (x in 0 until w) {
                val x0 = if (x > 0) x - 1 else 0
                val x1 = if (x < w - 1) x + 1 else w - 1
                var m = 255
                for (yy in y0..y1) {
                    val row = yy * w
                    for (xx in x0..x1) {
                        val v = src[row + xx].toInt() and 0xFF
                        if (v < m) m = v
                    }
                }
                out[y * w + x] = m.toByte()
            }
        }
        return out
    }

    /** Image.crop((x0, y0, x1, y1)): the region, zeros where it leaves the image. */
    fun crop(img: Image8, x0: Int, y0: Int, x1: Int, y1: Int): Image8 {
        val w = maxOf(0, x1 - x0)
        val h = maxOf(0, y1 - y0)
        val ch = img.channels
        val out = ByteArray(w * h * ch)
        val xs = maxOf(0, x0)
        val xe = minOf(img.width, x1)
        if (xe > xs) {
            for (y in maxOf(0, y0) until minOf(img.height, y1)) {
                System.arraycopy(img.data, (y * img.width + xs) * ch, out, ((y - y0) * w + (xs - x0)) * ch, (xe - xs) * ch)
            }
        }
        return Image8(w, h, ch, out)
    }

    /**
     * Image.new("RGB", size, (0, 0, 0)).paste(rgba, mask=rgba.split()[3]): Paste.c paste_mask_L's BLEND on black,
     * DIV255(c * a).
     */
    fun compositeOnBlack(rgba: Image8): Image8 {
        require(rgba.channels == 4)
        val n = rgba.width * rgba.height
        val s = rgba.data
        val out = ByteArray(n * 3)
        for (i in 0 until n) {
            val a = s[i * 4 + 3].toInt() and 0xFF
            for (b in 0 until 3) {
                val t = (s[i * 4 + b].toInt() and 0xFF) * a + 128
                out[i * 3 + b] = (((t shr 8) + t) shr 8).toByte()
            }
        }
        return Image8(rgba.width, rgba.height, 3, out)
    }
}
