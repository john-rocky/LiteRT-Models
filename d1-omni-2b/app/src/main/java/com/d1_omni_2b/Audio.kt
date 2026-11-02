/*
 * The audio path of d1-Omni-2B in Kotlin: what the provider (LiquidAI/d1-Omni-2B, media.py and its vendored
 * Lfm3OmniFeatureExtractor) does to a clip before the audio encoder, ported from the model repository's host
 * (host/d1_audio_features.py), which reproduces the provider's float32 arithmetic step by step:
 * - the WAV reader: Python's `wave` module and the provider's `media.read_audio` (16-bit PCM only);
 * - the channel mean and the resample of `media.resampled` (torchaudio's default windowed sinc);
 * - the log-mel: transformers' `mel_filter_bank` (slaney), torch's periodic Hann window, pocketfft's real FFT
 *   (radix 4 and 5), SLEEF's float32 sine, cosine and base-10 logarithm, with the multiply-adds the provider's
 *   CPU fuses.
 * The provider's code is under the LFM Open License v1.0; the ported numerics come from transformers
 * (Apache-2.0), PyTorch and pocketfft (BSD-3-Clause) and SLEEF (Boost Software License 1.0). See NOTICE.
 */

package com.d1_omni_2b

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil

// ------------------------------------------------------------------------------------------------ float32

/** float32 arithmetic as the provider's CPU does it: one fused multiply-add, and SLEEF's sin, cos and log10. */
object F32 {
  /**
   * a * b + c rounded once to float32 (one FMA instruction). The double product of two floats is exact; the
   * double sum is rounded once (its error from TwoSum), and when it lands exactly on a float midpoint with a
   * nonzero error it moves one double step toward the true sum first. Exact in float32's normal range.
   */
  fun fma(a: Float, b: Float, c: Float): Float {
    val p = a.toDouble() * b.toDouble()
    val cd = c.toDouble()
    var s = p + cd
    val bv = s - p
    val err = (p - (s - bv)) + (cd - bv)
    if ((java.lang.Double.doubleToRawLongBits(s) and 0x1FFFFFFFL) == 0x10000000L && err != 0.0) {
      s = Math.nextAfter(s, if (err > 0) Double.POSITIVE_INFINITY else Double.NEGATIVE_INFINITY)
    }
    return s.toFloat()
  }

  // The constants as numpy writes them: a double literal rounded to float32.
  private val PI_A2 = 3.1414794921875.toFloat()
  private val PI_B2 = 0.00011315941810607910156.toFloat()
  private val PI_C2 = 1.9841872589410058936e-09.toFloat()
  private val M_1_PI = 0.318309886183790671537767526745028724.toFloat()
  private val TRIG_RANGE = 125.0f

  /** SLEEF's trig tail on (s = x^2 as a double-float, t = x as a double-float). */
  private fun trigTail(sx: Float, sy: Float, tx: Float, ty: Float): Float {
    var u = 2.6083159809786593541503e-06.toFloat()
    u = fma(u, sx, (-0.0001981069071916863322258).toFloat())
    u = fma(u, sx, 0.00833307858556509017944336.toFloat())
    // dfadd_ff(-0.1666..., u * sx)
    val c0 = (-0.166666597127914428710938).toFloat()
    val us = u * sx
    val ax = c0 + us
    val ay = (c0 - ax) + us
    // dfmul_f2f2(a, s)
    val mx = ax * sx
    val my = fma(ax, sy, fma(ay, sx, fma(ax, sx, -mx)))
    // dfadd_ff2(1, m)
    val xx = 1f + mx
    val xy = ((1f - xx) + mx) + my
    // dfmul (scalar result)
    return fma(tx, xx, fma(ty, xx, tx * xy))
  }

  private fun checkRange(d: Float) {
    require(kotlin.math.abs(d) < TRIG_RANGE) { "the SLEEF port covers |x| < 125" }
  }

  /** SLEEF xsinf_u1 for |d| < 125, with FMA: torch.sin of a float32 value on the provider's CPU. */
  fun sin(d: Float): Float {
    checkRange(d)
    val u = Math.rint((d * M_1_PI).toDouble()).toFloat()
    val q = u.toInt()
    val v = fma(u, -PI_A2, d)
    // dfadd2_ff(v, u * -PI_B2)
    val w = u * -PI_B2
    var sx = v + w
    val vv = sx - v
    var sy = (v - (sx - vv)) + (w - vv)
    // dfadd_f2f(s, u * -PI_C2)
    val w2 = u * -PI_C2
    val sx2 = sx + w2
    sy = ((sx - sx2) + w2) + sy
    sx = sx2
    val tx = sx
    val ty = sy
    // dfsqu(s)
    val qx = sx * sx
    val qy = fma(sx + sx, sy, fma(sx, sx, -qx))
    var r = trigTail(qx, qy, tx, ty)
    if ((q and 1) == 1) r = -r
    // sin(-0) is -0
    return if (d == 0f && 1f / d < 0) d else r
  }

  /** SLEEF xcosf_u1 for |d| < 125, with FMA: torch.cos of a float32 value on the provider's CPU. */
  fun cos(d: Float): Float {
    checkRange(d)
    val dq = fma(Math.rint(fma(d, M_1_PI, -0.5f).toDouble()).toFloat(), 2f, 1f)
    val q = dq.toInt()
    // dfadd2_ff(d, dq * (-PI_A2 * 0.5))
    val w = dq * (-PI_A2 * 0.5f)
    var sx = d + w
    var v = sx - d
    var sy = (d - (sx - v)) + (w - v)
    // dfadd2_f2f(s, dq * (-PI_B2 * 0.5))
    val y1 = dq * (-PI_B2 * 0.5f)
    var s = sx + y1
    v = s - sx
    sy = ((sx - (s - v)) + (y1 - v)) + sy
    sx = s
    // dfadd2_f2f(s, dq * (-PI_C2 * 0.5))
    val y2 = dq * (-PI_C2 * 0.5f)
    s = sx + y2
    v = s - sx
    sy = ((sx - (s - v)) + (y2 - v)) + sy
    sx = s
    val tx = sx
    val ty = sy
    val qx = sx * sx
    val qy = fma(sx + sx, sy, fma(sx, sx, -qx))
    val r = trigTail(qx, qy, tx, ty)
    return if ((q and 2) == 0) -r else r
  }

  /**
   * SLEEF xlog10f_u1 with FMA: torch.log10 of a float32 value on the provider's CPU (0 -> -inf, a negative
   * value or NaN -> NaN, +inf -> +inf; subnormals scaled by 2^64 first, as SLEEF does).
   */
  fun log10(d: Float): Float {
    val small = d < java.lang.Float.MIN_NORMAL
    val x = if (small) d * 1.8446744073709552E19.toFloat() else d
    var e = ((java.lang.Float.floatToRawIntBits(x * (1.0 / 0.75).toFloat()) shr 23) and 0xFF) - 0x7F
    val m = java.lang.Float.intBitsToFloat(java.lang.Float.floatToRawIntBits(x) + ((-e) shl 23))
    if (small) e -= 64
    // dfadd2_ff(-1, m), dfadd2_ff(1, m)
    var ax = -1f + m
    var v = ax - -1f
    val ay = (-1f - (ax - v)) + (m - v)
    val bx = 1f + m
    v = bx - 1f
    val by = (1f - (bx - v)) + (m - v)
    // dfdiv(a, b)
    val t = 1f / bx
    val xx = ax * t
    val u = fma(t, ax, -xx)
    val w = fma(-by, t, fma(-bx, t, 1f))
    val xy = fma(xx, w, fma(ay, t, u))
    val x2 = xx * xx
    var p = fma(0.1314289868e+0.toFloat(), x2, 0.1735493541e+0.toFloat())
    p = fma(p, x2, 0.2895309627e+0.toFloat())
    // dfmul_f2f((0.30103001, -1.432098889e-08), e)
    val ef = e.toFloat()
    val c1 = 0.30103001.toFloat()
    val c2 = (-1.432098889e-08).toFloat()
    var sx = c1 * ef
    var sy = fma(c2, ef, fma(c1, ef, -sx))
    // dfmul_f2f2(x, (0.868588984, -2.170757285e-08))
    val k1 = 0.868588984.toFloat()
    val k2 = (-2.170757285e-08).toFloat()
    val px = xx * k1
    val py = fma(xx, k2, fma(xy, k1, fma(xx, k1, -px)))
    // dfadd_f2f2(s, p)
    var s = sx + px
    sy = (((sx - s) + px) + sy) + py
    sx = s
    // dfadd_f2f(s, x2 * x * t)
    val tail = (x2 * xx) * p
    s = sx + tail
    sy = ((sx - s) + tail) + sy
    sx = s
    ax = sx + sy
    return when {
      d == 0f -> Float.NEGATIVE_INFINITY
      d < 0 || d.isNaN() -> Float.NaN
      d == Float.POSITIVE_INFINITY -> Float.POSITIVE_INFINITY
      else -> ax
    }
  }
}

// ------------------------------------------------------------------------------------------------ the FFT

/**
 * pocketfft_r<float>(n).exec(forward), as PyTorch runs it without MKL (rfftp), for lengths whose factors are
 * 4 and 5 (the extractor's 400 = 4 * 4 * 5 * 5); other plans are not ported.
 */
class RealFft(val n: Int) {
  private val fact: IntArray
  private val tw: Array<FloatArray?>

  init {
    require(n >= 1) { "an FFT length is at least 1" }
    val lpf = if (n < 50) 0 else largestPrimeFactor(n)
    fact = if (n > 1) factorize(n) else IntArray(0)
    if (lpf * lpf > n || fact.any { it != 4 && it != 5 }) {
      throw UnsupportedOperationException("pocketfft's plan for length $n is not ported (factors 4 and 5 only)")
    }
    val twid = Twiddles(n)
    tw = arrayOfNulls(fact.size)
    var l1 = 1
    for (k in fact.indices) {
      val ip = fact[k]
      val ido = n / (l1 * ip)
      if (k < fact.size - 1) { // the last factor needs no twiddles
        val t = FloatArray((ip - 1) * (ido - 1))
        for (j in 1 until ip) {
          for (i in 1..(ido - 1) / 2) {
            twid.at(j * l1 * i)
            t[(j - 1) * (ido - 1) + 2 * i - 2] = twid.re
            t[(j - 1) * (ido - 1) + 2 * i - 1] = twid.im
          }
        }
        tw[k] = t
      }
      l1 *= ip
    }
  }

  /** rfftp::exec(c, 1, true): r0, r1, i1, r2, i2, ... (the forward transform in pocketfft's packed order). */
  fun halfcomplex(x: FloatArray, scratch: FloatArray = FloatArray(n)): FloatArray {
    require(x.size == n && scratch.size == n) { "this plan transforms $n values" }
    var c = x
    var ch = scratch
    var l1 = n
    for (k1 in fact.indices) {
      val k = fact.size - k1 - 1
      val ip = fact[k]
      val ido = n / l1
      l1 /= ip
      if (ip == 4) radf4(ido, l1, c, ch, tw[k]) else radf5(ido, l1, c, ch, tw[k])
      val swap = c
      c = ch
      ch = swap
    }
    return c
  }

  private class Twiddles(private val n: Int) {
    private val ang = 0.25 * Math.PI / n
    private val shift: Int
    private val mask: Int
    private val v1: Array<DoubleArray>
    private val v2: Array<DoubleArray>
    var re = 0f
    var im = 0f

    init {
      val nval = (n + 2) / 2
      var s = 1
      while ((1 shl s) * (1 shl s) < nval) s++
      shift = s
      mask = (1 shl s) - 1
      v1 = Array(mask + 1) { if (it == 0) doubleArrayOf(1.0, 0.0) else calc(it) }
      v2 = Array((nval + mask) / (mask + 1)) { if (it == 0) doubleArrayOf(1.0, 0.0) else calc(it * (mask + 1)) }
    }

    /** sincos_2pibyn's table entry: (cos, sin) of 2 pi x / n by octant, libm's double cos and sin. */
    private fun calc(start: Int): DoubleArray {
      var x = start shl 3
      if (x < 4 * n) {
        if (x < 2 * n) {
          if (x < n) return doubleArrayOf(StrictMath.cos(x * ang), StrictMath.sin(x * ang))
          return doubleArrayOf(StrictMath.sin((2 * n - x) * ang), StrictMath.cos((2 * n - x) * ang))
        }
        x -= 2 * n
        if (x < n) return doubleArrayOf(-StrictMath.sin(x * ang), StrictMath.cos(x * ang))
        return doubleArrayOf(-StrictMath.cos((2 * n - x) * ang), StrictMath.sin((2 * n - x) * ang))
      }
      x = 8 * n - x
      if (x < 2 * n) {
        if (x < n) return doubleArrayOf(StrictMath.cos(x * ang), -StrictMath.sin(x * ang))
        return doubleArrayOf(StrictMath.sin((2 * n - x) * ang), -StrictMath.cos((2 * n - x) * ang))
      }
      x -= 2 * n
      if (x < n) return doubleArrayOf(-StrictMath.sin(x * ang), -StrictMath.cos(x * ang))
      return doubleArrayOf(-StrictMath.cos((2 * n - x) * ang), -StrictMath.sin((2 * n - x) * ang))
    }

    /** The twiddle of [index], combined in double and rounded to float32, into [re] and [im]. */
    fun at(index: Int) {
      var idx = index
      var sign = 1.0f
      if (2 * idx > n) {
        idx = n - idx
        sign = -1.0f
      }
      val x1 = v1[idx and mask]
      val x2 = v2[idx shr shift]
      re = (x1[0] * x2[0] - x1[1] * x2[1]).toFloat()
      im = sign * (x1[0] * x2[1] + x1[1] * x2[0]).toFloat()
    }
  }

  // Each pass reads CC(a, b, c) = cc[a + ido * (b + l1 * c)] and writes CH(a, b, c) = ch[a + ido * (b + ip * c)].
  // MULPM (as compiled): a = fma(wr, e, wi * f), b = fma(wr, f, -(wi * e)).

  private fun radf4(ido: Int, l1: Int, cc: FloatArray, ch: FloatArray, wa: FloatArray?) {
    fun ci(a: Int, b: Int, c: Int) = a + ido * (b + l1 * c)
    fun hi(a: Int, b: Int, c: Int) = a + ido * (b + 4 * c)
    for (k in 0 until l1) {
      val c0 = cc[ci(0, k, 0)]
      val c1 = cc[ci(0, k, 1)]
      val c2 = cc[ci(0, k, 2)]
      val c3 = cc[ci(0, k, 3)]
      val tr1 = c3 + c1
      val tr2 = c0 + c2
      ch[hi(0, 2, k)] = c3 - c1
      ch[hi(ido - 1, 1, k)] = c0 - c2
      ch[hi(0, 0, k)] = tr2 + tr1
      ch[hi(ido - 1, 3, k)] = tr2 - tr1
    }
    if (ido % 2 == 0) {
      for (k in 0 until l1) {
        val e0 = cc[ci(ido - 1, k, 0)]
        val e1 = cc[ci(ido - 1, k, 1)]
        val e2 = cc[ci(ido - 1, k, 2)]
        val e3 = cc[ci(ido - 1, k, 3)]
        val ti1 = (-HSQT2) * (e1 + e3)
        val tr1 = HSQT2 * (e1 - e3)
        ch[hi(ido - 1, 0, k)] = e0 + tr1
        ch[hi(ido - 1, 2, k)] = e0 - tr1
        ch[hi(0, 3, k)] = ti1 + e2
        ch[hi(0, 1, k)] = ti1 - e2
      }
    }
    if (ido <= 2) return
    val w = wa!!
    for (k in 0 until l1) {
      var i = 2
      while (i < ido) {
        val ic = ido - i
        val w1r = w[i - 2]
        val w1i = w[i - 1]
        val w2r = w[i - 2 + (ido - 1)]
        val w2i = w[i - 1 + (ido - 1)]
        val w3r = w[i - 2 + 2 * (ido - 1)]
        val w3i = w[i - 1 + 2 * (ido - 1)]
        val e1 = cc[ci(i - 1, k, 1)]
        val f1 = cc[ci(i, k, 1)]
        val cr2 = F32.fma(w1r, e1, w1i * f1)
        val ci2 = F32.fma(w1r, f1, -(w1i * e1))
        val e2 = cc[ci(i - 1, k, 2)]
        val f2 = cc[ci(i, k, 2)]
        val cr3 = F32.fma(w2r, e2, w2i * f2)
        val ci3 = F32.fma(w2r, f2, -(w2i * e2))
        val e3 = cc[ci(i - 1, k, 3)]
        val f3 = cc[ci(i, k, 3)]
        val cr4 = F32.fma(w3r, e3, w3i * f3)
        val ci4 = F32.fma(w3r, f3, -(w3i * e3))
        val tr1 = cr4 + cr2
        val tr4 = cr4 - cr2
        val ti1 = ci2 + ci4
        val ti4 = ci2 - ci4
        val a = cc[ci(i - 1, k, 0)]
        val b = cc[ci(i, k, 0)]
        val tr2 = a + cr3
        val tr3 = a - cr3
        val ti2 = b + ci3
        val ti3 = b - ci3
        ch[hi(i - 1, 0, k)] = tr2 + tr1
        ch[hi(i, 0, k)] = ti1 + ti2
        ch[hi(ic - 1, 3, k)] = tr2 - tr1
        ch[hi(ic, 3, k)] = ti1 - ti2
        ch[hi(i - 1, 2, k)] = tr3 + ti4
        ch[hi(i, 2, k)] = tr4 + ti3
        ch[hi(ic - 1, 1, k)] = tr3 - ti4
        ch[hi(ic, 1, k)] = tr4 - ti3
        i += 2
      }
    }
  }

  private fun radf5(ido: Int, l1: Int, cc: FloatArray, ch: FloatArray, wa: FloatArray?) {
    fun ci(a: Int, b: Int, c: Int) = a + ido * (b + l1 * c)
    fun hi(a: Int, b: Int, c: Int) = a + ido * (b + 5 * c)
    for (k in 0 until l1) {
      val c0 = cc[ci(0, k, 0)]
      val c1 = cc[ci(0, k, 1)]
      val c2 = cc[ci(0, k, 2)]
      val c3 = cc[ci(0, k, 3)]
      val c4 = cc[ci(0, k, 4)]
      val cr2 = c4 + c1
      val ci5 = c4 - c1
      val cr3 = c3 + c2
      val ci4 = c3 - c2
      ch[hi(0, 0, k)] = c0 + cr2 + cr3
      ch[hi(ido - 1, 1, k)] = F32.fma(TR12, cr3, F32.fma(TR11, cr2, c0))
      ch[hi(0, 2, k)] = F32.fma(TI11, ci5, TI12 * ci4)
      ch[hi(ido - 1, 3, k)] = F32.fma(TR11, cr3, F32.fma(TR12, cr2, c0))
      ch[hi(0, 4, k)] = F32.fma(TI12, ci5, -(TI11 * ci4))
    }
    if (ido == 1) return
    val w = wa!!
    for (k in 0 until l1) {
      var i = 2
      while (i < ido) {
        val ic = ido - i
        var dr2 = 0f
        var di2 = 0f
        var dr3 = 0f
        var di3 = 0f
        var dr4 = 0f
        var di4 = 0f
        var dr5 = 0f
        var di5 = 0f
        for (x in 0 until 4) {
          val wr = w[i - 2 + x * (ido - 1)]
          val wi = w[i - 1 + x * (ido - 1)]
          val e = cc[ci(i - 1, k, x + 1)]
          val f = cc[ci(i, k, x + 1)]
          val re = F32.fma(wr, e, wi * f)
          val im = F32.fma(wr, f, -(wi * e))
          when (x) {
            0 -> { dr2 = re; di2 = im }
            1 -> { dr3 = re; di3 = im }
            2 -> { dr4 = re; di4 = im }
            else -> { dr5 = re; di5 = im }
          }
        }
        val sr25 = dr2 + dr5
        val si25 = di2 + di5
        val ar25 = di2 - di5
        val ai25 = dr5 - dr2
        val sr34 = dr3 + dr4
        val si34 = di3 + di4
        val ar34 = di3 - di4
        val ai34 = dr4 - dr3
        // (dr2, di2, dr5, di5) = (sr25, si25, ar25, ai25); (dr3, di3, dr4, di4) = (sr34, si34, ar34, ai34)
        val a = cc[ci(i - 1, k, 0)]
        val b = cc[ci(i, k, 0)]
        ch[hi(i - 1, 0, k)] = a + sr25 + sr34
        ch[hi(i, 0, k)] = b + si25 + si34
        val tr2 = F32.fma(TR12, sr34, F32.fma(TR11, sr25, a))
        val ti2 = F32.fma(TR12, si34, F32.fma(TR11, si25, b))
        val tr3 = F32.fma(TR11, sr34, F32.fma(TR12, sr25, a))
        val ti3 = F32.fma(TR11, si34, F32.fma(TR12, si25, b))
        val tr5 = F32.fma(TI11, ar25, TI12 * ar34)
        val ti5 = F32.fma(TI11, ai25, TI12 * ai34)
        val tr4 = F32.fma(TI12, ar25, -(TI11 * ar34))
        val ti4 = F32.fma(TI12, ai25, -(TI11 * ai34))
        ch[hi(i - 1, 2, k)] = tr2 + tr5
        ch[hi(i, 2, k)] = ti5 + ti2
        ch[hi(ic - 1, 1, k)] = tr2 - tr5
        ch[hi(ic, 1, k)] = ti5 - ti2
        ch[hi(i - 1, 4, k)] = tr3 + tr4
        ch[hi(i, 4, k)] = ti4 + ti3
        ch[hi(ic - 1, 3, k)] = tr3 - tr4
        ch[hi(ic, 3, k)] = ti4 - ti3
        i += 2
      }
    }
  }

  companion object {
    private val HSQT2 = 0.707106781186547524400844362104849.toFloat()
    private val TR11 = 0.3090169943749474241022934171828191.toFloat()
    private val TI11 = 0.9510565162951535721164393333793821.toFloat()
    private val TR12 = (-0.8090169943749474241022934171828191).toFloat()
    private val TI12 = 0.5877852522924731291687059546390728.toFloat()

    /** rfftp::factorize: fours first, a two at the front, then odd factors. */
    fun factorize(start: Int): IntArray {
      var n = start
      val fact = ArrayList<Int>()
      while (n % 4 == 0) {
        fact.add(4)
        n /= 4
      }
      if (n % 2 == 0) {
        n /= 2
        fact.add(2)
        val t = fact[0]
        fact[0] = fact[fact.size - 1]
        fact[fact.size - 1] = t
      }
      var d = 3
      while (d * d <= n) {
        while (n % d == 0) {
          fact.add(d)
          n /= d
        }
        d += 2
      }
      if (n > 1) fact.add(n)
      return fact.toIntArray()
    }

    fun largestPrimeFactor(start: Int): Int {
      var n = start
      var res = 1
      while (n % 2 == 0) {
        res = 2
        n /= 2
      }
      var x = 3
      while (x * x <= n) {
        while (n % x == 0) {
          res = x
          n /= x
        }
        x += 2
      }
      return if (n > 1) n else res
    }
  }
}

// ------------------------------------------------------------------------------------------------ the mel filters

/** transformers.audio_utils.mel_filter_bank with the slaney scale and norm, in double. */
object MelFilters {
  private fun hertzToMel(freq: Double): Double {
    val logstep = 27.0 / StrictMath.log(6.4)
    var mels = 3.0 * freq / 200.0
    if (freq >= 1000.0) mels = 15.0 + StrictMath.log(freq / 1000.0) * logstep
    return mels
  }

  private fun melToHertz(mels: Double): Double {
    val logstep = StrictMath.log(6.4) / 27.0
    var freq = 200.0 * mels / 3.0
    if (mels >= 15.0) freq = 1000.0 * StrictMath.exp(logstep * (mels - 15.0))
    return freq
  }

  /** numpy.linspace: start + i * step, the last value exactly [stop]. */
  private fun linspace(start: Double, stop: Double, num: Int): DoubleArray {
    val step = (stop - start) / (num - 1)
    return DoubleArray(num) { if (it == num - 1) stop else it * step + start }
  }

  /** The filters [bins][mels] for [bins] frequency bins from 0 to the Nyquist frequency of [rate]. */
  fun bank(bins: Int, mels: Int, minFrequency: Double, maxFrequency: Double, rate: Int): Array<DoubleArray> {
    require(bins >= 2 && minFrequency <= maxFrequency) { "bad mel filter settings" }
    val melFreqs = linspace(hertzToMel(minFrequency), hertzToMel(maxFrequency), mels + 2)
    val filterFreqs = DoubleArray(melFreqs.size) { melToHertz(melFreqs[it]) }
    val fftFreqs = linspace(0.0, (rate / 2).toDouble(), bins)
    val diff = DoubleArray(filterFreqs.size - 1) { filterFreqs[it + 1] - filterFreqs[it] }
    val out = Array(bins) { DoubleArray(mels) }
    for (b in 0 until bins) {
      for (m in 0 until mels) {
        val down = -(filterFreqs[m] - fftFreqs[b]) / diff[m]
        val up = (filterFreqs[m + 2] - fftFreqs[b]) / diff[m + 1]
        val enorm = 2.0 / (filterFreqs[m + 2] - filterFreqs[m])
        out[b][m] = maxOf(0.0, minOf(down, up)) * enorm
      }
    }
    for (m in 0 until mels) {
      require((0 until bins).any { out[it][m] != 0.0 }) { "a mel filter has all zero values ($mels filters, $bins bins)" }
    }
    return out
  }
}

// ------------------------------------------------------------------------------------------------ the features

/** Lfm3OmniFeatureExtractor's settings (processor_config.json, `feature_extractor`). */
data class ExtractorSettings(
  val featureSize: Int = 128,
  val samplingRate: Int = 16000,
  val nFft: Int = 400,
  val hopLength: Int = 160,
  val winLength: Int = 400,
  val logFloor: Float = -12f,
  val normalizationOffset: Float = 4f,
  val normalizationDivisor: Float = 4f,
  val encoderStride: Int = 8,
  val paddingValue: Float = 0f,
) {
  companion object {
    /** The settings of processor_config.json's `feature_extractor`, refused when the port does not cover them. */
    fun of(processorConfig: Map<*, *>): ExtractorSettings {
      val doc = processorConfig["feature_extractor"] as? Map<*, *> ?: throw IllegalArgumentException("this checkpoint takes no audio")
      require(doc["feature_extractor_type"] == "Lfm3OmniFeatureExtractor") {
        "the app ports Lfm3OmniFeatureExtractor, not ${doc["feature_extractor_type"]}"
      }
      require((doc["padding_side"] ?: "right") == "right") { "the app ports right padding" }
      fun int(key: String, default: Int) = (doc[key] as JsonNumber?)?.toInt() ?: default
      fun float(key: String, default: Float) = (doc[key] as JsonNumber?)?.toDouble()?.toFloat() ?: default
      val d = ExtractorSettings()
      return ExtractorSettings(
        int("feature_size", d.featureSize),
        int("sampling_rate", d.samplingRate),
        int("n_fft", d.nFft),
        int("hop_length", d.hopLength),
        int("win_length", d.winLength),
        float("log_floor", d.logFloor),
        float("normalization_offset", d.normalizationOffset),
        float("normalization_divisor", d.normalizationDivisor),
        int("encoder_stride", d.encoderStride),
        float("padding_value", d.paddingValue),
      )
    }
  }
}

/** Lfm3OmniFeatureExtractor for one mono clip at the extractor's rate: its log-mel `input_features`. */
class FeatureExtractor(val settings: ExtractorSettings = ExtractorSettings()) {
  private val fft: RealFft
  private val window: FloatArray
  /** The filters as float32 [mels][bins] (the transposed matrix the provider multiplies with), and each band's bins. */
  private val filters: Array<FloatArray>
  private val firstBin: IntArray
  private val lastBin: IntArray
  private val bins = settings.nFft / 2 + 1

  init {
    val s = settings
    if (s.winLength != s.nFft) throw UnsupportedOperationException("the app ports a window as long as the FFT")
    fft = RealFft(s.nFft)
    window = hannWindow(s.winLength)
    val bank = MelFilters.bank(bins, s.featureSize, 0.0, s.samplingRate / 2.0, s.samplingRate)
    filters = Array(s.featureSize) { m -> FloatArray(bins) { b -> bank[b][m].toFloat() } }
    firstBin = IntArray(s.featureSize) { m -> filters[m].indexOfFirst { it != 0f } }
    lastBin = IntArray(s.featureSize) { m -> filters[m].indexOfLast { it != 0f } }
  }

  val samplingRate: Int
    get() = settings.samplingRate

  /** The clip right padded to whole frames (at least one FFT). */
  fun padLength(samples: Int): Int {
    val s = settings
    if (samples < s.nFft) return s.nFft
    return samples + Math.floorMod(-(samples - s.nFft), s.hopLength)
  }

  fun numFrames(samples: Int): Int = 1 + (padLength(samples) - settings.nFft) / settings.hopLength

  /** The audio encoder's tokens for [frames] mel frames: ceil(frames / encoder_stride). */
  fun numEncoderTokens(frames: Int): Int = Math.floorDiv(frames + settings.encoderStride - 1, settings.encoderStride)

  /** The longest clip of [frames] frames, in seconds at the extractor's rate. */
  fun seconds(frames: Int): Double = ((frames - 1) * settings.hopLength + settings.nFft).toDouble() / settings.samplingRate

  /**
   * The clip's `input_features` [featureSize, frames], row-major (one row per mel band): right padded with zeros
   * to whole frames, |STFT|^2 under the periodic Hann window, the slaney mel filters, log10 floored at log_floor,
   * then (x + offset) / divisor, every step in float32 as the provider computes it.
   */
  fun features(clip: FloatArray): FloatArray {
    val s = settings
    val padded = FloatArray(padLength(clip.size)) { s.paddingValue }
    System.arraycopy(clip, 0, padded, 0, clip.size)
    val frames = 1 + (padded.size - s.nFft) / s.hopLength
    val out = FloatArray(s.featureSize * frames)
    val frame = FloatArray(s.nFft)
    val scratch = FloatArray(s.nFft)
    val power = FloatArray(bins)
    val half = (s.nFft - 1) / 2
    for (f in 0 until frames) {
      val start = f * s.hopLength
      for (i in 0 until s.nFft) frame[i] = padded[start + i] * window[i]
      val r = fft.halfcomplex(frame, scratch) // overwrites frame, refilled for the next one
      power[0] = squaredHypot(r[0], 0f)
      for (k in 1..half) power[k] = squaredHypot(r[2 * k - 1], r[2 * k])
      if (s.nFft % 2 == 0) power[s.nFft / 2] = squaredHypot(r[s.nFft - 1], 0f)
      for (m in 0 until s.featureSize) {
        val w = filters[m]
        var sum = 0.0
        for (b in firstBin[m]..lastBin[m]) sum += w[b].toDouble() * power[b].toDouble()
        val logMel = maxOf(F32.log10(sum.toFloat()), s.logFloor)
        out[m * frames + f] = (logMel + s.normalizationOffset) / s.normalizationDivisor
      }
    }
    return out
  }

  /** torch.abs of a complex64 (hypotf), squared in float32. */
  private fun squaredHypot(re: Float, im: Float): Float {
    val h = Math.sqrt(re.toDouble() * re.toDouble() + im.toDouble() * im.toDouble()).toFloat()
    return h * h
  }

  companion object {
    /** torch.hann_window(n) (periodic, float32): cos(k * float32(2 pi / n)) * -0.5 + 0.5, SLEEF's cosine. */
    fun hannWindow(n: Int): FloatArray {
      val step = (Math.PI * 2.0 / n).toFloat()
      return FloatArray(n) { F32.cos(it.toFloat() * step) * -0.5f + 0.5f }
    }
  }
}

// ------------------------------------------------------------------------------------------------ the resample

/** `media.resampled`: torchaudio's default resample (Hann-windowed sinc, 6 zero crossings, rolloff 0.99). */
object Resample {
  /** The largest filter the app builds (phases x taps); every common rate needs well under it. */
  const val MAX_KERNEL = 1_000_000L

  private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

  /** The filter's phases and taps for [rate] -> [target], or null when the rates are equal. */
  fun shape(rate: Int, target: Int, zeros: Int = 6, rolloff: Double = 0.99): Pair<Int, Int>? {
    if (rate == target) return null
    val g = gcd(rate, target)
    val old = rate / g
    val new = target / g
    val cut = minOf(old, new) * rolloff
    val width = ceil(zeros * old / cut).toInt()
    return new to (2 * width + old)
  }

  /** How many samples a clip of [samples] at [rate] has at [target]. */
  fun length(samples: Int, rate: Int, target: Int): Int {
    if (rate == target) return samples
    val g = gcd(rate, target)
    return ceil((target / g).toDouble() * samples / (rate / g)).toInt()
  }

  /** The filter [new][2 * width + old], each torch step in float32 (the scalars as float32). */
  fun kernel(old: Int, new: Int, zeros: Int, cut: Double, width: Int): Array<FloatArray> {
    val size = 2 * width + old
    val cutF = cut.toFloat()
    val pi = Math.PI.toFloat()
    val scale = (cut / old).toFloat()
    return Array(new) { p ->
      val tp = (-p).toFloat() / new.toFloat()
      FloatArray(size) { j ->
        var t = (tp + (j - width).toFloat() / old.toFloat()) * cutF
        t = t.coerceIn((-zeros).toFloat(), zeros.toFloat())
        var window = F32.cos(t * pi / zeros.toFloat() / 2f)
        window *= window
        t *= pi
        val sinc = if (t == 0f) 1.0f else F32.sin(t) / t
        sinc * (window * scale)
      }
    }
  }

  /**
   * [wave] (mono) at [target]: the filter applied as PyTorch's strided convolution computes it (the kernel times
   * the input's columns, phases interleaved), each output summed in double and rounded to float32.
   */
  fun resampled(wave: FloatArray, rate: Int, target: Int, zeros: Int = 6, rolloff: Double = 0.99): FloatArray {
    if (rate == target) return wave
    require(rate > 0 && target > 0) { "a sampling rate of $rate Hz" }
    val g = gcd(rate, target)
    val old = rate / g
    val new = target / g
    val cut = minOf(old, new) * rolloff
    val width = ceil(zeros * old / cut).toInt()
    val size = 2 * width + old
    require(new.toLong() * size <= MAX_KERNEL) {
      "a clip at $rate Hz needs a resampling filter too large for this app; convert it to ${target} Hz first"
    }
    val kernel = kernel(old, new, zeros, cut, width)
    val x = FloatArray(wave.size + 2 * width + old)
    System.arraycopy(wave, 0, x, width, wave.size)
    val count = (x.size - size) / old + 1
    val keep = length(wave.size, rate, target)
    val out = FloatArray(keep)
    for (c in 0 until count) {
      val base = c * old
      for (p in 0 until new) {
        val at = c * new + p
        if (at >= keep) break
        val k = kernel[p]
        var sum = 0.0
        for (j in 0 until size) sum += k[j].toDouble() * x[base + j].toDouble()
        out[at] = sum.toFloat()
      }
    }
    return out
  }
}

// ------------------------------------------------------------------------------------------------ clips

/** One clip as read: its channels averaged into one (the first lines of `media.resampled`), at its own rate. */
class AudioClip(val name: String, val mono: FloatArray, val rate: Int, val channels: Int = 1) {
  val seconds: Double
    get() = mono.size.toDouble() / rate

  companion object {
    /** A clip from [pcm] (interleaved 16-bit samples, [channels] per frame): samples / 32768, channels averaged. */
    fun of(name: String, pcm: ShortArray, channels: Int, rate: Int): AudioClip {
      require(channels >= 1) { "$name: no channels" }
      val frames = pcm.size / channels
      require(frames > 0) { "$name: the clip has no samples" }
      // A clip shorter than its channel count would be averaged over time by the provider (it reads the longer
      // axis as the samples); refuse it instead.
      require(frames >= channels) { "$name: the clip has fewer samples than channels" }
      require(rate > 0) { "$name: a sampling rate of $rate Hz" }
      val mono = FloatArray(frames) { f ->
        if (channels == 1) {
          pcm[f].toFloat() / 32768f
        } else {
          var sum = 0f
          for (c in 0 until channels) sum += pcm[f * channels + c].toFloat() / 32768f
          (sum.toDouble() / channels).toFloat()
        }
      }
      return AudioClip(name, mono, rate, channels)
    }
  }
}

/** A WAV file's 16-bit PCM: interleaved samples, channels and rate. */
class WavPcm(val pcm: ShortArray, val channels: Int, val rate: Int)

/**
 * 16-bit PCM WAV files as Python's `wave` module and the provider's `media.read_audio` read them: a RIFF/WAVE file,
 * its chunks inside the RIFF size (each padded to an even size), a `fmt ` chunk of PCM (format 1, or 0xFFFE with
 * the PCM subformat) with 16-bit samples, then the `data` chunk; a data chunk that ends inside a sample frame is
 * refused. Every refusal is an [IllegalArgumentException] that says why.
 */
object Wav {
  private val PCM_SUBFORMAT = byteArrayOf(
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80.toByte(), 0x00, 0x00, 0xaa.toByte(), 0x00, 0x38, 0x9b.toByte(), 0x71,
  )

  /** A bounded reader over [bytes] from [offset]: at most [size] bytes (a chunk), as `wave._Chunk` reads. */
  private class Chunk(private val bytes: ByteArray, val offset: Int, val size: Long, private val parentEnd: Int) {
    var read = 0L

    /** At most [n] bytes, fewer at the chunk's or the file's end. */
    fun take(n: Int): ByteArray {
      if (read >= size) return ByteArray(0)
      val want = minOf(n.toLong(), size - read)
      val from = offset + read
      val have = maxOf(0L, minOf(want, parentEnd - from))
      read += have
      return bytes.copyOfRange(from.toInt(), (from + have).toInt())
    }
  }

  private fun u16(b: ByteArray, at: Int) = (b[at].toInt() and 0xff) or ((b[at + 1].toInt() and 0xff) shl 8)

  private fun u32(b: ByteArray, at: Int): Long = (u16(b, at).toLong()) or (u16(b, at + 2).toLong() shl 16)

  fun read(bytes: ByteArray, name: String = "the clip"): WavPcm {
    fun refuse(why: String): Nothing = throw IllegalArgumentException("$name: $why")
    if (bytes.size < 8) refuse("not a WAV file (too short)")
    if (String(bytes, 0, 4, Charsets.ISO_8859_1) != "RIFF") refuse("not a WAV file (it does not start with RIFF)")
    val riff = Chunk(bytes, 8, u32(bytes, 4), bytes.size)
    if (String(riff.take(4), Charsets.ISO_8859_1) != "WAVE") refuse("not a WAV file (no WAVE header)")
    var channels = 0
    var rate = 0L
    var width = 0
    var fmtRead = false
    while (true) {
      val head = riff.take(8)
      if (head.size < 8) break
      val chunkName = String(head, 0, 4, Charsets.ISO_8859_1)
      val chunk = Chunk(bytes, (8 + riff.read).toInt(), u32(head, 4), minOf(bytes.size.toLong(), 8 + riff.size).toInt())
      if (chunkName == "fmt ") {
        val fmt = chunk.take(14)
        if (fmt.size < 14) refuse("the fmt chunk is cut short")
        val tag = u16(fmt, 0)
        channels = u16(fmt, 2)
        rate = u32(fmt, 4)
        if (tag != 1 && tag != 0xFFFE) refuse("not PCM (format $tag): 16-bit PCM WAV only")
        val bits = chunk.take(2)
        if (bits.size < 2) refuse("the fmt chunk is cut short")
        if (tag == 0xFFFE) {
          val ext = chunk.take(8)
          val sub = chunk.take(16)
          if (ext.size < 8 || sub.size < 16) refuse("the fmt chunk is cut short")
          if (!sub.contentEquals(PCM_SUBFORMAT)) refuse("not PCM (an extended format): 16-bit PCM WAV only")
        }
        width = (u16(bits, 0) + 7) / 8
        if (width == 0) refuse("bad sample width")
        if (channels == 0) refuse("no channels")
        fmtRead = true
      } else if (chunkName == "data") {
        if (!fmtRead) refuse("the data chunk comes before the fmt chunk")
        if (width != 2) refuse("${width * 8}-bit samples: 16-bit PCM WAV only")
        val frameSize = channels * width
        val frames = chunk.size / frameSize
        val data = chunk.take(minOf(frames * frameSize, Int.MAX_VALUE.toLong()).toInt())
        if (data.size % 2 != 0) refuse("the data ends inside a sample")
        if ((data.size / 2) % channels != 0) refuse("the data ends inside a sample frame")
        val pcm = ShortArray(data.size / 2)
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(pcm)
        if (rate <= 0 || rate > Int.MAX_VALUE) refuse("a sampling rate of $rate Hz")
        return WavPcm(pcm, channels, rate.toInt())
      }
      // skip: the rest of the chunk and its pad byte, never past the RIFF size
      val skip = chunk.size + (chunk.size and 1L)
      if (riff.read + skip > riff.size) refuse("a chunk runs past the end of the RIFF data")
      riff.read += skip
    }
    refuse("no fmt or no data chunk")
  }

  /** A clip from a WAV file's bytes: refused with the reason when it is not 16-bit PCM or has no samples. */
  fun clip(bytes: ByteArray, name: String): AudioClip {
    val wav = read(bytes, name)
    return AudioClip.of(name, wav.pcm, wav.channels, wav.rate)
  }

  /** A 16-bit PCM mono WAV file of [pcm] at [rate]. */
  fun write(pcm: ShortArray, rate: Int): ByteArray {
    val out = ByteArrayOutputStream(44 + pcm.size * 2)
    val head = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
    head.put("RIFF".toByteArray(Charsets.ISO_8859_1)).putInt(36 + pcm.size * 2).put("WAVE".toByteArray(Charsets.ISO_8859_1))
    head.put("fmt ".toByteArray(Charsets.ISO_8859_1)).putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2)
    head.putShort(2).putShort(16)
    head.put("data".toByteArray(Charsets.ISO_8859_1)).putInt(pcm.size * 2)
    out.write(head.array())
    val body = ByteBuffer.allocate(pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    body.asShortBuffer().put(pcm)
    out.write(body.array())
    return out.toByteArray()
  }
}
