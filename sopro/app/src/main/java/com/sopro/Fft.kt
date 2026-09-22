package com.sopro

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** Single-precision complex butterflies; only twiddle generation uses doubles. */
class Fft(val size: Int) {
  data class Spectrum(val real: FloatArray, val imag: FloatArray)

  private val twiddleReal = FloatArray(size) { cos(-2.0 * PI * it / size).toFloat() }
  private val twiddleImag = FloatArray(size) { sin(-2.0 * PI * it / size).toFloat() }

  init {
    require(size == 1024 || size == 400) { "Supported FFT lengths are 1024 and 400" }
  }

  fun realForward(input: FloatArray): Spectrum {
    require(input.size == size)
    val real = input.copyOf()
    val imag = FloatArray(size)
    transform(real, imag)
    return Spectrum(real.copyOf(size / 2 + 1), imag.copyOf(size / 2 + 1))
  }

  fun inverseReal(real: FloatArray, imag: FloatArray): FloatArray {
    require(real.size == size / 2 + 1 && imag.size == real.size)
    val r = FloatArray(size)
    val i = FloatArray(size)
    for (k in real.indices) {
      r[k] = real[k]
      i[k] = -imag[k]
    }
    for (k in 1 until size / 2) {
      r[size - k] = real[k]
      i[size - k] = imag[k]
    }
    // A real inverse ignores the imaginary components at DC and Nyquist.
    i[0] = 0f
    i[size / 2] = 0f
    transform(r, i)
    return FloatArray(size) { r[it] / size.toFloat() }
  }

  private fun transform(real: FloatArray, imag: FloatArray) {
    if (size == 1024) radix2(real, imag)
    else {
      val out = mixed(real, imag, 0, 1, size)
      out.real.copyInto(real)
      out.imag.copyInto(imag)
    }
  }

  private fun radix2(real: FloatArray, imag: FloatArray) {
    var reversed = 0
    for (n in 1 until size) {
      var bit = size shr 1
      while ((reversed and bit) != 0) {
        reversed = reversed xor bit
        bit = bit shr 1
      }
      reversed = reversed xor bit
      if (n < reversed) {
        val r = real[n]
        real[n] = real[reversed]
        real[reversed] = r
        val i = imag[n]
        imag[n] = imag[reversed]
        imag[reversed] = i
      }
    }
    var width = 2
    while (width <= size) {
      val half = width / 2
      val stride = size / width
      for (start in 0 until size step width) {
        for (j in 0 until half) {
          val a = start + j
          val b = a + half
          val t = j * stride
          val rr = real[b] * twiddleReal[t] - imag[b] * twiddleImag[t]
          val ii = real[b] * twiddleImag[t] + imag[b] * twiddleReal[t]
          val ar = real[a]
          val ai = imag[a]
          real[a] = ar + rr
          imag[a] = ai + ii
          real[b] = ar - rr
          imag[b] = ai - ii
        }
      }
      width *= 2
    }
  }

  private fun mixed(
    real: FloatArray,
    imag: FloatArray,
    offset: Int,
    stride: Int,
    n: Int,
  ): Spectrum {
    if (n == 1) return Spectrum(floatArrayOf(real[offset]), floatArrayOf(imag[offset]))
    val radix = if (n % 2 == 0) 2 else 5
    val subSize = n / radix
    val parts = Array(radix) { mixed(real, imag, offset + it * stride, stride * radix, subSize) }
    val r = FloatArray(n)
    val i = FloatArray(n)
    for (k in 0 until n) {
      val q = k % subSize
      var re = parts[0].real[q]
      var im = parts[0].imag[q]
      for (j in 1 until radix) {
        val t = ((j * k) % n) * (size / n)
        re += parts[j].real[q] * twiddleReal[t] - parts[j].imag[q] * twiddleImag[t]
        im += parts[j].real[q] * twiddleImag[t] + parts[j].imag[q] * twiddleReal[t]
      }
      r[k] = re
      i[k] = im
    }
    return Spectrum(r, i)
  }
}
