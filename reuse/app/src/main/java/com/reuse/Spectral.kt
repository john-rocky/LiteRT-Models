package com.reuse

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.expm1
import kotlin.math.ln1p
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The host side of the RE-USE graph at 16 kHz, as in NVIDIA's models/stfts.py and inference.py: an STFT with
 * n_fft = win = 640 (periodic Hann), hop 80, centered with reflect padding, one-sided (321 bins).
 *
 * Graph input [1, 2, T, 321]: log1p(|X|) and angle(X), time-major.
 * Graph output [1, 3, T, 321]: the compressed magnitude m, then x_r and x_i. The host restores |Y| = expm1(relu(m)),
 * zeroes every frame in which more than half of the bins are 0, takes the phase as (x_r, x_i) / hypot(x_r, x_i)
 * (the atan2 of the original model, kept out of the graph) and runs the inverse STFT.
 *
 * The transforms run in double precision; the graph sees float32.
 */
class Spectral {

  companion object {
    const val N_FFT = 640
    const val HOP = 80
    const val BINS = N_FFT / 2 + 1
  }

  private val window = DoubleArray(N_FFT) { 0.5 - 0.5 * cos(2.0 * PI * it / N_FFT) }
  private val fft = Fft(N_FFT)
  private val re = DoubleArray(N_FFT)
  private val im = DoubleArray(N_FFT)
  private var acc = DoubleArray(0)
  private var env = DoubleArray(0)

  /** STFT frames of a chunk of [samples] samples (center = true). */
  fun frames(samples: Int) = 1 + samples / HOP

  /** STFT of x[start until start + length] into the graph input spec[2 * T * BINS]. */
  fun analyze(x: FloatArray, start: Int, length: Int, spec: FloatArray) {
    val t = frames(length)
    val pad = N_FFT / 2
    for (frame in 0 until t) {
      for (i in 0 until N_FFT) {
        var j = frame * HOP + i - pad // reflect padding at both edges of the chunk
        if (j < 0) j = -j
        if (j >= length) j = 2 * (length - 1) - j
        re[i] = x[start + j] * window[i]
        im[i] = 0.0
      }
      fft.forward(re, im)
      im[0] = 0.0 // the DC and Nyquist bins of a real frame are real
      im[N_FFT / 2] = 0.0
      for (k in 0 until BINS) {
        val o = frame * BINS + k
        spec[o] = ln1p(sqrt(re[k] * re[k] + im[k] * im[k])).toFloat()
        spec[t * BINS + o] = atan2(im[k], re[k]).toFloat()
      }
    }
  }

  /** Inverse STFT of the graph output y[3 * T * BINS] into wav[(T - 1) * HOP] (one chunk). */
  fun synthesize(y: FloatArray, t: Int, wav: FloatArray) {
    val len = N_FFT + HOP * (t - 1)
    if (acc.size < len) {
      acc = DoubleArray(len)
      env = DoubleArray(len)
    } else {
      acc.fill(0.0, 0, len)
      env.fill(0.0, 0, len)
    }
    for (frame in 0 until t) {
      val m = frame * BINS
      var zeros = 0
      for (k in 0 until BINS) if (expm1(max(y[m + k], 0f)) == 0f) zeros++
      val keep = 2 * zeros <= BINS // a frame with more than half of its bins at 0 is zeroed
      for (k in 0 until BINS) {
        val mag = if (keep) expm1(max(y[m + k], 0f).toDouble()) else 0.0
        val xr = y[t * BINS + m + k].toDouble()
        val xi = y[2 * t * BINS + m + k].toDouble()
        val h = sqrt(xr * xr + xi * xi)
        re[k] = if (h > 0.0) mag * xr / h else mag
        im[k] = if (h > 0.0) mag * xi / h else 0.0
      }
      im[0] = 0.0 // the inverse real FFT ignores the imaginary parts of the DC and Nyquist bins
      im[N_FFT / 2] = 0.0
      for (k in 1 until N_FFT / 2) {
        re[N_FFT - k] = re[k]
        im[N_FFT - k] = -im[k]
      }
      fft.inverse(re, im)
      val s = frame * HOP
      for (i in 0 until N_FFT) {
        acc[s + i] += re[i] * window[i]
        env[s + i] += window[i] * window[i]
      }
    }
    val pad = N_FFT / 2
    for (i in 0 until HOP * (t - 1)) wav[i] = (acc[pad + i] / env[pad + i]).toFloat()
  }
}

/** Complex FFT of one fixed length: mixed radix (4, 2, odd factors), Stockham autosort, double precision. */
internal class Fft(private val n: Int) {

  private val radices: IntArray
  private val cosT = DoubleArray(n) { cos(2.0 * PI * it / n) }
  private val sinT = DoubleArray(n) { sin(2.0 * PI * it / n) }
  private val workRe = DoubleArray(n)
  private val workIm = DoubleArray(n)
  private val aRe = DoubleArray(n)
  private val aIm = DoubleArray(n)

  init {
    val r = ArrayList<Int>()
    var m = n
    while (m % 4 == 0) {
      r.add(4)
      m /= 4
    }
    while (m % 2 == 0) {
      r.add(2)
      m /= 2
    }
    var p = 3
    while (m > 1) {
      while (m % p == 0) {
        r.add(p)
        m /= p
      }
      p += 2
    }
    radices = r.toIntArray()
  }

  fun forward(re: DoubleArray, im: DoubleArray) = transform(re, im, -1.0)

  /** Inverse transform, scaled by 1 / n. */
  fun inverse(re: DoubleArray, im: DoubleArray) {
    transform(re, im, 1.0)
    for (i in 0 until n) {
      re[i] /= n
      im[i] /= n
    }
  }

  // One decimation-in-frequency pass per radix r: a sub-transform of length len = r * m and stride s turns into r
  // sub-transforms of length m and stride r * s; y[k] is multiplied by w_len^(p * k) (table index p * k * n / len).
  private fun transform(re: DoubleArray, im: DoubleArray, sign: Double) {
    var xr = re
    var xi = im
    var yr = workRe
    var yi = workIm
    var len = n
    var s = 1
    for (r in radices) {
      val m = len / r
      val tw = n / len
      for (p in 0 until m) {
        for (q in 0 until s) {
          when (r) {
            2 -> {
              val i0 = q + s * p
              val i1 = i0 + s * m
              val o0 = q + s * 2 * p
              val o1 = o0 + s
              val dr = xr[i0] - xr[i1]
              val di = xi[i0] - xi[i1]
              yr[o0] = xr[i0] + xr[i1]
              yi[o0] = xi[i0] + xi[i1]
              val e = p * tw
              val c = cosT[e]
              val sn = sign * sinT[e]
              yr[o1] = dr * c - di * sn
              yi[o1] = dr * sn + di * c
            }
            4 -> {
              val i0 = q + s * p
              val i1 = i0 + s * m
              val i2 = i1 + s * m
              val i3 = i2 + s * m
              val t0r = xr[i0] + xr[i2]
              val t0i = xi[i0] + xi[i2]
              val t1r = xr[i0] - xr[i2]
              val t1i = xi[i0] - xi[i2]
              val t2r = xr[i1] + xr[i3]
              val t2i = xi[i1] + xi[i3]
              // (a1 - a3) * w_4, w_4 = sign * i
              val t3r = -sign * (xi[i1] - xi[i3])
              val t3i = sign * (xr[i1] - xr[i3])
              val o = q + s * 4 * p
              yr[o] = t0r + t2r
              yi[o] = t0i + t2i
              rotate(t1r + t3r, t1i + t3i, p * tw, sign, yr, yi, o + s)
              rotate(t0r - t2r, t0i - t2i, 2 * p * tw, sign, yr, yi, o + 2 * s)
              rotate(t1r - t3r, t1i - t3i, 3 * p * tw, sign, yr, yi, o + 3 * s)
            }
            else -> {
              for (j in 0 until r) {
                aRe[j] = xr[q + s * (p + j * m)]
                aIm[j] = xi[q + s * (p + j * m)]
              }
              val rs = n / r
              for (k in 0 until r) {
                var sr = 0.0
                var si = 0.0
                for (j in 0 until r) {
                  val e = (j * k % r) * rs
                  val c = cosT[e]
                  val sn = sign * sinT[e]
                  sr += aRe[j] * c - aIm[j] * sn
                  si += aRe[j] * sn + aIm[j] * c
                }
                rotate(sr, si, p * k * tw, sign, yr, yi, q + s * (r * p + k))
              }
            }
          }
        }
      }
      val tr = xr
      xr = yr
      yr = tr
      val ti = xi
      xi = yi
      yi = ti
      len = m
      s *= r
    }
    if (xr !== re) {
      xr.copyInto(re)
      xi.copyInto(im)
    }
  }

  /** out[o] = (vr + i vi) * w^e, w = exp(sign * 2 pi i / n). */
  private fun rotate(vr: Double, vi: Double, e: Int, sign: Double, outR: DoubleArray, outI: DoubleArray, o: Int) {
    val c = cosT[e]
    val sn = sign * sinT[e]
    outR[o] = vr * c - vi * sn
    outI[o] = vr * sn + vi * c
  }
}
