package com.idol

import kotlin.math.floor

/**
 * torch.nn.functional.interpolate(x, size, mode="bilinear", align_corners=...) on the CPU in float32, as ATen's
 * generic N-d kernel computes it (UpSampleKernel.cpp: compute_indices_weights_linear + Interpolate, the path a
 * 1- or 3-channel NCHW tensor takes): per output index the source index and weight from
 * area_pixel_compute_scale / area_pixel_compute_source_index / guard_index_and_lambda in float32, then
 * out = (row(y0) * wy0 + row(y1) * wy1) with row(y) = src(y, x0) * wx0 + src(y, x1) * wx1, each product and sum
 * rounded to float32.
 */
object TorchOps {
    private class Taps(val i0: IntArray, val i1: IntArray, val w0: FloatArray, val w1: FloatArray)

    private fun taps(inSize: Int, outSize: Int, alignCorners: Boolean): Taps {
        val i0 = IntArray(outSize)
        val i1 = IntArray(outSize)
        val w0 = FloatArray(outSize)
        val w1 = FloatArray(outSize)
        val scale: Float = if (alignCorners) {
            if (outSize > 1) (inSize - 1).toFloat() / (outSize - 1).toFloat() else 0f
        } else {
            inSize.toFloat() / outSize.toFloat()
        }
        for (o in 0 until outSize) {
            if (outSize == inSize) {
                // scale factor 1: a copy
                i0[o] = o
                i1[o] = o
                w0[o] = 1f
                w1[o] = 0f
                continue
            }
            val real: Float = if (alignCorners) {
                scale * o.toFloat()
            } else {
                val src = scale * (o.toFloat() + 0.5f) - 0.5f
                if (src < 0f) 0f else src
            }
            val idx = minOf(floor(real.toDouble()).toInt(), inSize - 1)
            val lambda = minOf(maxOf(real - idx.toFloat(), 0f), 1f)
            i0[o] = idx
            i1[o] = idx + if (idx < inSize - 1) 1 else 0
            w1[o] = lambda
            w0[o] = 1f - lambda
        }
        return Taps(i0, i1, w0, w1)
    }

    /** [src] = channels x inH x inW (planar, float32) -> channels x outH x outW. */
    fun bilinear(
        src: FloatArray,
        channels: Int,
        inH: Int,
        inW: Int,
        outH: Int,
        outW: Int,
        alignCorners: Boolean,
    ): FloatArray {
        require(src.size == channels * inH * inW)
        val ty = taps(inH, outH, alignCorners)
        val tx = taps(inW, outW, alignCorners)
        val out = FloatArray(channels * outH * outW)
        for (c in 0 until channels) {
            val sBase = c * inH * inW
            val oBase = c * outH * outW
            for (y in 0 until outH) {
                val r0 = sBase + ty.i0[y] * inW
                val r1 = sBase + ty.i1[y] * inW
                val wy0 = ty.w0[y]
                val wy1 = ty.w1[y]
                val o = oBase + y * outW
                for (x in 0 until outW) {
                    val x0 = tx.i0[x]
                    val x1 = tx.i1[x]
                    val wx0 = tx.w0[x]
                    val wx1 = tx.w1[x]
                    var t0 = src[r0 + x0] * wx0
                    t0 += src[r0 + x1] * wx1
                    var t1 = src[r1 + x0] * wx0
                    t1 += src[r1 + x1] * wx1
                    var v = t0 * wy0
                    v += t1 * wy1
                    out[o + x] = v
                }
            }
        }
        return out
    }
}
