package com.sam3

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Numeric primitives for the tracker host loop, ported 1:1 from
 * scripts/tracker_host_loop.py. Tensors are row-major float arrays with explicit
 * dims (NCHW like numpy). Binary masks use 64-bit packed bitsets (LSB-first bit
 * order, matching np.packbits(bitorder="little") read little-endian).
 */
internal object TM {

    fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

    // ---------------- fp16 / bf16
    fun halfToFloat(h: Int): Float {
        val s = (h ushr 15) and 0x1; val e = (h ushr 10) and 0x1F; val m = h and 0x3FF
        val bits = when {
            e == 0 -> if (m == 0) s shl 31 else {
                var mant = m; var exp2 = -1
                do { mant = mant shl 1; exp2++ } while (mant and 0x400 == 0)
                (s shl 31) or ((127 - 15 - exp2) shl 23) or ((mant and 0x3FF) shl 13)
            }
            e == 0x1F -> (s shl 31) or (0xFF shl 23) or (m shl 13)
            else -> (s shl 31) or ((e - 15 + 127) shl 23) or (m shl 13)
        }
        return Float.fromBits(bits)
    }

    /** float32 -> fp16 bits, round-to-nearest-even (like numpy .astype(float16)). */
    fun floatToHalf(f: Float): Int {
        val bits = f.toRawBits()
        val s = (bits ushr 16) and 0x8000
        val e = (bits ushr 23) and 0xFF
        var m = bits and 0x7FFFFF
        if (e == 0xFF) return s or 0x7C00 or (if (m != 0) 0x200 or (m ushr 13) else 0)
        val eH = e - 127 + 15
        if (eH >= 0x1F) return s or 0x7C00
        if (eH <= 0) {
            if (eH < -10) return s
            m = m or 0x800000
            val shift = 14 - eH
            val half = m ushr shift
            val rem = m and ((1 shl shift) - 1)
            val mid = 1 shl (shift - 1)
            var r = half
            if (rem > mid || (rem == mid && (half and 1) == 1)) r++
            return s or r
        }
        val half = m ushr 13
        val rem = m and 0x1FFF
        var r = (eH shl 10) or half
        if (rem > 0x1000 || (rem == 0x1000 && (r and 1) == 1)) r++
        return s or r
    }

    fun toHalfBits(v: Float): Short = floatToHalf(v).toShort()

    fun halfBitsToFloat(h: Short): Float = halfToFloat(h.toInt() and 0xFFFF)

    /** float32 -> bfloat16 -> float32 in place (round-to-nearest-even, like torch). */
    fun bf16InPlace(x: FloatArray) {
        for (i in x.indices) {
            val b = x[i].toRawBits()
            val r = ((b ushr 16) and 1) + 0x7FFF
            x[i] = Float.fromBits((b + r) and 0xFFFF0000.toInt())
        }
    }

    // ---------------- bilinear interpolation (align_corners=False, no antialias)
    private class AxisIdx(val i0: IntArray, val i1: IntArray, val lam: FloatArray)

    private fun axisIdx(iSz: Int, oSz: Int): AxisIdx {
        val i0 = IntArray(oSz); val i1 = IntArray(oSz); val lam = FloatArray(oSz)
        for (o in 0 until oSz) {
            var src = (o + 0.5) * (iSz.toDouble() / oSz) - 0.5
            if (src < 0.0) src = 0.0
            val f = min(floor(src).toInt(), iSz - 1)
            i0[o] = f; i1[o] = min(f + 1, iSz - 1)
            lam[o] = (src - f).toFloat()
        }
        return AxisIdx(i0, i1, lam)
    }

    /**
     * torch F.interpolate(mode="bilinear", align_corners=False), planes-major input
     * (n*c, ih, iw) -> (n*c, oh, ow). Rows first, then columns, f32 arithmetic —
     * the same evaluation order as the numpy reference.
     */
    fun interpBilinear(x: FloatArray, planes: Int, ih: Int, iw: Int, oh: Int, ow: Int): FloatArray {
        if (ih == oh && iw == ow) return x.copyOf()
        val ay = axisIdx(ih, oh)
        val ax = axisIdx(iw, ow)
        val out = FloatArray(planes * oh * ow)
        val top = FloatArray(iw)
        for (p in 0 until planes) {
            val base = p * ih * iw
            val obase = p * oh * ow
            for (y in 0 until oh) {
                val r0 = base + ay.i0[y] * iw
                val r1 = base + ay.i1[y] * iw
                val ly = ay.lam[y]
                for (xw in 0 until iw) top[xw] = x[r0 + xw] * (1f - ly) + x[r1 + xw] * ly
                val orow = obase + y * ow
                for (xo in 0 until ow) {
                    val lx = ax.lam[xo]
                    out[orow + xo] = top[ax.i0[xo]] * (1f - lx) + top[ax.i1[xo]] * lx
                }
            }
        }
        return out
    }

    // ---------------- antialiased bilinear (triangle filter, PIL-style)
    private class AAWeights(val xmin: IntArray, val xmax: IntArray, val ws: Array<FloatArray>)

    private val aaCache = HashMap<Long, AAWeights>()

    private fun aaWeights(iSz: Int, oSz: Int): AAWeights {
        val key = iSz.toLong() shl 32 or oSz.toLong()
        aaCache[key]?.let { return it }
        val scale = iSz.toDouble() / oSz
        val support = max(scale, 1.0)
        val xmin = IntArray(oSz); val xmax = IntArray(oSz)
        val ws = Array(oSz) { FloatArray(0) }
        for (o in 0 until oSz) {
            val center = (o + 0.5) * scale
            val lo = max(0, (center - support + 0.5).toInt())
            val hi = min(iSz, (center + support + 0.5).toInt())
            xmin[o] = lo; xmax[o] = hi
            val w = FloatArray(hi - lo)
            var s = 0.0
            for (k in lo until hi) {
                val v = max(1.0 - abs((k + 0.5 - center) / support), 0.0)
                w[k - lo] = v.toFloat(); s += v
            }
            if (s > 0) for (i in w.indices) w[i] = (w[i] / s).toFloat()
            ws[o] = w
        }
        val r = AAWeights(xmin, xmax, ws)
        aaCache[key] = r
        return r
    }

    /** torch F.interpolate(..., antialias=True), planes-major (n*c, ih, iw). */
    fun interpBilinearAA(x0: FloatArray, planes: Int, ih: Int, iw: Int, oh: Int, ow: Int): FloatArray {
        var x = x0
        var h = ih
        if (ih != oh) {
            val a = aaWeights(ih, oh)
            val out = FloatArray(planes * oh * iw)
            for (p in 0 until planes) {
                val base = p * ih * iw; val obase = p * oh * iw
                for (o in 0 until oh) {
                    val w = a.ws[o]; val lo = a.xmin[o]
                    val orow = obase + o * iw
                    for (k in w.indices) {
                        val wk = w[k]; val row = base + (lo + k) * iw
                        for (c in 0 until iw) out[orow + c] += wk * x[row + c]
                    }
                }
            }
            x = out; h = oh
        }
        if (iw != ow) {
            val a = aaWeights(iw, ow)
            val out = FloatArray(planes * h * ow)
            for (p in 0 until planes) {
                val base = p * h * iw; val obase = p * h * ow
                for (r in 0 until h) {
                    val row = base + r * iw; val orow = obase + r * ow
                    for (o in 0 until ow) {
                        val w = a.ws[o]; val lo = a.xmin[o]
                        var acc = 0f
                        for (k in w.indices) acc += w[k] * x[row + lo + k]
                        out[orow + o] = acc
                    }
                }
            }
            x = out
        }
        return x
    }

    // ---------------- erf / GELU
    /** erf via the Numerical Recipes erfc rational approximation (|err| < 1.2e-7). */
    fun erf(x: Double): Double {
        val z = abs(x)
        val t = 1.0 / (1.0 + 0.5 * z)
        val ans = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
            t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        val erfc = if (x >= 0) ans else 2.0 - ans
        return 1.0 - erfc
    }

    fun geluInPlace(x: FloatArray) {
        for (i in x.indices) {
            val v = x[i].toDouble()
            x[i] = (0.5 * v * (1.0 + erf(v / sqrt(2.0)))).toFloat()
        }
    }

    // ---------------- small NN ops (host-side conv chain, LN, linears)
    /** Conv2d with stride == kernel (non-overlapping blocks), NCHW. */
    fun convBlock(x: FloatArray, n: Int, ci: Int, h: Int, w: Int,
                  wgt: FloatArray, co: Int, k: Int, bias: FloatArray): FloatArray {
        val oh = h / k; val ow = w / k
        val out = FloatArray(n * co * oh * ow)
        for (b in 0 until n) {
            for (o in 0 until co) {
                val obase = (b * co + o) * oh * ow
                for (y in 0 until oh) for (xw in 0 until ow) {
                    var acc = 0f
                    for (c in 0 until ci) {
                        val ibase = (b * ci + c) * h * w + y * k * w + xw * k
                        val wbase = (o * ci + c) * k * k
                        for (ky in 0 until k) for (kx in 0 until k) {
                            acc += x[ibase + ky * w + kx] * wgt[wbase + ky * k + kx]
                        }
                    }
                    out[obase + y * ow + xw] = acc + bias[o]
                }
            }
        }
        return out
    }

    /** LayerNorm over the channel axis of NCHW (eps 1e-6), in place. */
    fun layerNorm2dInPlace(x: FloatArray, n: Int, c: Int, h: Int, w: Int,
                           wgt: FloatArray, bias: FloatArray) {
        val hw = h * w
        for (b in 0 until n) for (p in 0 until hw) {
            var mu = 0f
            for (ch in 0 until c) mu += x[(b * c + ch) * hw + p]
            mu /= c
            var va = 0f
            for (ch in 0 until c) { val d = x[(b * c + ch) * hw + p] - mu; va += d * d }
            va /= c
            val inv = (1.0 / sqrt(va.toDouble() + 1e-6)).toFloat()
            for (ch in 0 until c) {
                val i = (b * c + ch) * hw + p
                x[i] = (x[i] - mu) * inv * wgt[ch] + bias[ch]
            }
        }
    }

    /** x (m,inDim) @ w(out,inDim)^T + b -> (m,out). */
    fun linear(x: FloatArray, m: Int, inDim: Int, w: FloatArray, out: Int, b: FloatArray): FloatArray {
        val y = FloatArray(m * out)
        for (r in 0 until m) {
            val xb = r * inDim
            for (o in 0 until out) {
                var acc = 0f
                val wb = o * inDim
                for (i in 0 until inDim) acc += x[xb + i] * w[wb + i]
                y[r * out + o] = acc + b[o]
            }
        }
        return y
    }

    /** Sine PE of get_1d_sine_pe: pos (n) -> (n, dim), [sin | cos]. */
    fun sine1dPe(pos: FloatArray, dim: Int): FloatArray {
        val peDim = dim / 2
        val out = FloatArray(pos.size * dim)
        for (r in pos.indices) {
            for (i in 0 until peDim) {
                val dimT = Math.pow(10000.0, 2.0 * (i / 2) / peDim)
                val v = pos[r] / dimT
                out[r * dim + i] = Math.sin(v).toFloat()
                out[r * dim + peDim + i] = Math.cos(v).toFloat()
            }
        }
        return out
    }

    // ---------------- packed binary masks
    fun packedWords(px: Int): Int = (px + 63) ushr 6

    /** Pack (src[offset + i] > thresh) for i in 0 until px, LSB-first. */
    fun pack(src: FloatArray, offset: Int, px: Int, thresh: Float): LongArray {
        val w = LongArray(packedWords(px))
        for (i in 0 until px) {
            if (src[offset + i] > thresh) w[i ushr 6] = w[i ushr 6] or (1L shl (i and 63))
        }
        return w
    }

    fun popcount(a: LongArray): Int {
        var s = 0
        for (w in a) s += java.lang.Long.bitCount(w)
        return s
    }

    fun popcountAnd(a: LongArray, b: LongArray): Int {
        var s = 0
        for (i in a.indices) s += java.lang.Long.bitCount(a[i] and b[i])
        return s
    }

    fun popcountOr(a: LongArray, b: LongArray): Int {
        var s = 0
        for (i in a.indices) s += java.lang.Long.bitCount(a[i] or b[i])
        return s
    }

    fun testBit(a: LongArray, i: Int): Boolean = (a[i ushr 6] ushr (i and 63)) and 1L == 1L
}
