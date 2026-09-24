package com.nemotron3diar

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Log-mel front end of Nemotron-3-Diarization on a continuous 16 kHz stream, a 1:1 port of the transformers feature
 * extractor as mirrored in scripts/extract_frontend.py (`mel_mirror`) and scripts/host_loop.py.
 *
 * preemphasis y[n] = x[n] - 0.97 x[n-1] (y[0] = x[0]) -> frame i = y[160i-256, 160i+256), samples outside the stream
 * are zero -> hann(400, periodic=False) centered in the 512-sample frame (offset 56) -> fp32 real FFT (512)
 * -> |X|^2 as the processor rounds it (sqrt(re^2 + im^2), then squared) -> slaney mel filters [128, 257]
 * -> log(x + 2^-24). No normalization.
 *
 * The FFT is pocketfft's real FFTPACK plan in the processor's op order (torch.stft runs pocketfft): factors
 * [2, 4, 4, 4, 4], radf4 x 4 then radf2, twiddles cos / sin(2 pi k / 512) rounded to fp32, and the complex twiddle
 * products with one rounding (c * e + d * f as a fused multiply-add, as torch's build computes them). Bit-equal to
 * torch.fft.rfft on the Mac; any other fp32 FFT is within rounding of it but moves quiet mel bins (energy near
 * the 2^-24 guard) by up to ~3e-4 in the log domain.
 *
 * The processor computes the first streaming chunk with centered windows and the later ones uncentered, from audio
 * that starts 256 samples before their first frame; both give exactly the frames of the continuous stream (a chunk's
 * pre-emphasis restart lands on a zero-weight window sample), so every frame is computed once and cached. A frame is
 * final once the last sample under its non-zero window (160i + 198) has arrived, or when the stream is closed.
 */
class MelFrontend(private val melFilters: FloatArray, hann: FloatArray) {

  private val window = FloatArray(N_FFT)
  private val twiddles = arrayOfNulls<FloatArray>(FACTORS.size)
  private val filterStart = IntArray(N_MELS)
  private val filterEnd = IntArray(N_MELS)

  private var y = FloatArray(SAMPLE_RATE * 30)
  private var lastX = 0f

  /** Samples appended so far. */
  var numSamples = 0
    private set

  /** No more samples: frames reaching past the end are final with zeros there (the centered offline pass). */
  var closed = false
    private set

  private var cache = FloatArray(N_MELS * 3000)
  private var cachedFrames = 0

  // scratch
  private val buf = FloatArray(N_FFT)
  private val tmp = FloatArray(N_FFT)
  private val power = FloatArray(N_BINS)

  init {
    require(melFilters.size == N_MELS * N_BINS) { "mel filters: ${melFilters.size} floats" }
    require(hann.size == WIN_LENGTH) { "hann window: ${hann.size} floats" }
    val offset = (N_FFT - WIN_LENGTH) / 2
    hann.copyInto(window, offset)
    // rfftp::comp_twiddle: factor k (all but the last) gets (ip - 1) * (ido - 1) values
    var l1 = 1
    for ((k, ip) in FACTORS.withIndex()) {
      val ido = N_FFT / (l1 * ip)
      if (k < FACTORS.size - 1) {
        val tw = FloatArray((ip - 1) * (ido - 1))
        for (j in 1 until ip) for (i in 1..(ido - 1) / 2) {
          val a = 2.0 * PI * (j * l1 * i) / N_FFT
          tw[(j - 1) * (ido - 1) + 2 * i - 2] = cos(a).toFloat()
          tw[(j - 1) * (ido - 1) + 2 * i - 1] = sin(a).toFloat()
        }
        twiddles[k] = tw
      }
      l1 *= ip
    }
    // the non-zero span of every filter (summing only it is the same fp32 sum: the skipped terms are +0)
    for (m in 0 until N_MELS) {
      var a = 0
      while (a < N_BINS && melFilters[m * N_BINS + a] == 0f) a++
      var b = N_BINS
      while (b > a && melFilters[m * N_BINS + b - 1] == 0f) b--
      filterStart[m] = a
      filterEnd[m] = b
    }
  }

  /** Appends [count] samples of the stream (float PCM, int16 / 32768 scale). */
  fun append(x: FloatArray, offset: Int = 0, count: Int = x.size - offset) {
    check(!closed) { "stream closed" }
    if (numSamples + count > y.size) y = y.copyOf(maxOf(y.size * 2, numSamples + count))
    for (j in 0 until count) {
      val xi = x[offset + j]
      y[numSamples + j] = if (numSamples + j == 0) xi else xi - PREEMPHASIS * lastX
      lastX = xi
    }
    numSamples += count
  }

  /** Marks the end of the stream. */
  fun close() {
    closed = true
  }

  /** Whether frame [i] can be computed exactly now. */
  fun isFinal(i: Int): Boolean = closed || HOP * i + LAST_WEIGHTED_OFFSET < numSamples

  /** Writes log-mel frames [first, first + count) row-major into [out] ([count, 128]) starting at [outOffset]. */
  fun frames(first: Int, count: Int, out: FloatArray, outOffset: Int = 0) {
    for (j in 0 until count) {
      val i = first + j
      val dst = outOffset + j * N_MELS
      if (i < cachedFrames) {
        cache.copyInto(out, dst, i * N_MELS, (i + 1) * N_MELS)
        continue
      }
      check(isFinal(i)) { "frame $i needs samples up to ${HOP * i + LAST_WEIGHTED_OFFSET}, have $numSamples" }
      compute(i, out, dst)
      if (i == cachedFrames) {
        if ((i + 1) * N_MELS > cache.size) cache = cache.copyOf(maxOf(cache.size * 2, (i + 1) * N_MELS))
        out.copyInto(cache, i * N_MELS, dst, dst + N_MELS)
        cachedFrames++
      }
    }
  }

  private fun compute(i: Int, out: FloatArray, dst: Int) {
    val start = HOP * i - N_FFT / 2
    for (o in 0 until N_FFT) {
      val n = start + o
      buf[o] = (if (n in 0 until numSamples) y[n] else 0f) * window[o]
    }
    val hc = rfft(buf, tmp) // halfcomplex: r0, r1, i1, ..., r255, i255, r256
    power[0] = square(hc[0], 0f)
    for (k in 1 until N_FFT / 2) power[k] = square(hc[2 * k - 1], hc[2 * k])
    power[N_FFT / 2] = square(hc[N_FFT - 1], 0f)
    for (m in 0 until N_MELS) {
      var acc = 0f
      val row = m * N_BINS
      for (k in filterStart[m] until filterEnd[m]) acc += melFilters[row + k] * power[k]
      out[dst + m] = ln(acc + LOG_GUARD)
    }
  }

  /** |X|^2 as torch rounds it: sqrt(re^2 + im^2), then squared. */
  private fun square(re: Float, im: Float): Float {
    val mag = sqrt(re * re + im * im)
    return mag * mag
  }

  /** rfftp::exec (r2hc): the factors in reverse, ping-ponging between [c] and [ch]; returns the result array. */
  private fun rfft(c: FloatArray, ch: FloatArray): FloatArray {
    var p1 = c
    var p2 = ch
    var l1 = N_FFT
    for (k in FACTORS.indices.reversed()) {
      val ip = FACTORS[k]
      val ido = N_FFT / l1
      l1 /= ip
      if (ip == 4) radf4(ido, l1, p1, p2, twiddles[k]) else radf2(ido, l1, p1, p2, twiddles[k])
      val t = p1
      p1 = p2
      p2 = t
    }
    return p1
  }

  private fun radf2(ido: Int, l1: Int, input: FloatArray, ch: FloatArray, wa: FloatArray?) {
    fun cc(a: Int, b: Int, c: Int) = input[a + ido * (b + l1 * c)]
    fun at(a: Int, b: Int, c: Int) = a + ido * (b + 2 * c)
    for (k in 0 until l1) {
      ch[at(0, 0, k)] = cc(0, k, 0) + cc(0, k, 1)
      ch[at(ido - 1, 1, k)] = cc(0, k, 0) - cc(0, k, 1)
    }
    if (ido % 2 == 0) {
      for (k in 0 until l1) {
        ch[at(0, 1, k)] = -cc(ido - 1, k, 1)
        ch[at(ido - 1, 0, k)] = cc(ido - 1, k, 0)
      }
    }
    if (ido <= 2) return
    val w = wa!!
    for (k in 0 until l1) {
      var i = 2
      while (i < ido) {
        val ic = ido - i
        val wr = w[i - 2]
        val wi = w[i - 1]
        val tr2 = fma(wr, cc(i - 1, k, 1), wi * cc(i, k, 1))
        val ti2 = fma(wr, cc(i, k, 1), -(wi * cc(i - 1, k, 1)))
        ch[at(i - 1, 0, k)] = cc(i - 1, k, 0) + tr2
        ch[at(ic - 1, 1, k)] = cc(i - 1, k, 0) - tr2
        ch[at(i, 0, k)] = ti2 + cc(i, k, 0)
        ch[at(ic, 1, k)] = ti2 - cc(i, k, 0)
        i += 2
      }
    }
  }

  private fun radf4(ido: Int, l1: Int, input: FloatArray, ch: FloatArray, wa: FloatArray?) {
    fun cc(a: Int, b: Int, c: Int) = input[a + ido * (b + l1 * c)]
    fun at(a: Int, b: Int, c: Int) = a + ido * (b + 4 * c)
    for (k in 0 until l1) {
      val tr1 = cc(0, k, 3) + cc(0, k, 1)
      ch[at(0, 2, k)] = cc(0, k, 3) - cc(0, k, 1)
      val tr2 = cc(0, k, 0) + cc(0, k, 2)
      ch[at(ido - 1, 1, k)] = cc(0, k, 0) - cc(0, k, 2)
      ch[at(0, 0, k)] = tr2 + tr1
      ch[at(ido - 1, 3, k)] = tr2 - tr1
    }
    if (ido % 2 == 0) {
      for (k in 0 until l1) {
        val ti1 = -HSQT2 * (cc(ido - 1, k, 1) + cc(ido - 1, k, 3))
        val tr1 = HSQT2 * (cc(ido - 1, k, 1) - cc(ido - 1, k, 3))
        ch[at(ido - 1, 0, k)] = cc(ido - 1, k, 0) + tr1
        ch[at(ido - 1, 2, k)] = cc(ido - 1, k, 0) - tr1
        ch[at(0, 3, k)] = ti1 + cc(ido - 1, k, 2)
        ch[at(0, 1, k)] = ti1 - cc(ido - 1, k, 2)
      }
    }
    if (ido <= 2) return
    val w = wa!!
    val s = ido - 1
    for (k in 0 until l1) {
      var i = 2
      while (i < ido) {
        val ic = ido - i
        // MULPM: (a + ib) = conj(c + id) * (e + if) = (c e + d f) + i (c f - d e)
        val cr2 = fma(w[i - 2], cc(i - 1, k, 1), w[i - 1] * cc(i, k, 1))
        val ci2 = fma(w[i - 2], cc(i, k, 1), -(w[i - 1] * cc(i - 1, k, 1)))
        val cr3 = fma(w[s + i - 2], cc(i - 1, k, 2), w[s + i - 1] * cc(i, k, 2))
        val ci3 = fma(w[s + i - 2], cc(i, k, 2), -(w[s + i - 1] * cc(i - 1, k, 2)))
        val cr4 = fma(w[2 * s + i - 2], cc(i - 1, k, 3), w[2 * s + i - 1] * cc(i, k, 3))
        val ci4 = fma(w[2 * s + i - 2], cc(i, k, 3), -(w[2 * s + i - 1] * cc(i - 1, k, 3)))
        val tr1 = cr4 + cr2
        val tr4 = cr4 - cr2
        val ti1 = ci2 + ci4
        val ti4 = ci2 - ci4
        val tr2 = cc(i - 1, k, 0) + cr3
        val tr3 = cc(i - 1, k, 0) - cr3
        val ti2 = cc(i, k, 0) + ci3
        val ti3 = cc(i, k, 0) - ci3
        ch[at(i - 1, 0, k)] = tr2 + tr1
        ch[at(ic - 1, 3, k)] = tr2 - tr1
        ch[at(i, 0, k)] = ti1 + ti2
        ch[at(ic, 3, k)] = ti1 - ti2
        ch[at(i - 1, 2, k)] = tr3 + ti4
        ch[at(ic - 1, 1, k)] = tr3 - ti4
        ch[at(i, 2, k)] = tr4 + ti3
        ch[at(ic, 1, k)] = tr4 - ti3
        i += 2
      }
    }
  }

  companion object {
    const val SAMPLE_RATE = 16000
    const val N_FFT = 512
    const val HOP = 160
    const val WIN_LENGTH = 400
    const val N_MELS = 128
    const val N_BINS = N_FFT / 2 + 1
    const val PREEMPHASIS = 0.97f
    val LOG_GUARD = Math.scalb(1f, -24)
    private val FACTORS = intArrayOf(2, 4, 4, 4, 4) // rfftp::factorize(512)
    private const val HSQT2 = 0.70710677f

    /**
     * a * b + c with one rounding: the product is exact in double and the double sum rounds once more only when
     * it is not representable, which leaves the float result correct except at exact float midpoints.
     */
    private fun fma(a: Float, b: Float, c: Float): Float = (a.toDouble() * b + c).toFloat()

    /** Offset of the last non-zero window sample relative to frame center 160i (hann[398] at 56 + 398 - 256). */
    const val LAST_WEIGHTED_OFFSET = (N_FFT - WIN_LENGTH) / 2 + WIN_LENGTH - 2 - N_FFT / 2
  }
}
