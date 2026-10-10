package com.d1omni

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * One clip's sizes (`prepare()`'s info in the model repository's `d1_audio_host.py`): [samples] after
 * `waveform()`, the valid [frames] (n // 160), the STFT frames [stftFrames] (T = n // 160 + 1), the
 * graph's [bucket] T_b, its mask lengths [bucketDims] (T1, T2, T3) and the clip's subsampled
 * lengths [lengths] (L1, L2, L3); L3 = P, the prefix rows.
 */
class D1AudioInfo(
  val samples: Int,
  val frames: Int,
  val stftFrames: Int,
  val bucket: Int,
  val bucketDims: IntArray,
  val lengths: IntArray,
) {
  /** P: the audio prefix rows the decision graph reads (L3). */
  val prefixRows: Int
    get() = lengths[2]

  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "n" to samples,
      "frames" to frames,
      "T" to stftFrames,
      "T_b" to bucket,
      "T123" to bucketDims.toList(),
      "L123" to lengths.toList(),
      "P" to prefixRows,
    )
}

/** A normalized log-mel: [values] [128, T] row-major (mel bin major), [frames] valid of [stftFrames]. */
class D1Mel(val values: FloatArray, val frames: Int, val stftFrames: Int)

/**
 * Every stage of the front end for the JVM tests (`mel_stages`): [pre] the preemphasized samples
 * (float32), [power] [257, T] (only when asked for), [lin] / [log] [128, T], [mean] / [std] [128]
 * in float64, and [out] the float32 mel [128, T].
 */
class D1MelStages(
  val pre: FloatArray,
  val power: DoubleArray?,
  val lin: DoubleArray,
  val log: DoubleArray,
  val mean: DoubleArray,
  val std: DoubleArray,
  val out: FloatArray,
  val frames: Int,
  val stftFrames: Int,
)

/**
 * The five inputs of signature `audio_<T_b>` for one clip (`build_inputs`): [mel] [1, 128, T_b] (the
 * mel zero-padded on the right), [melValid] [1, T_b] = (t < frames), [v1] / [v2] / [v3] [1, T1..3] =
 * (t < L1..3), all float32 0 / 1 except the mel.
 */
class D1AudioInputs(
  val bucket: Int,
  val mel: FloatArray,
  val melValid: FloatArray,
  val v1: FloatArray,
  val v2: FloatArray,
  val v3: FloatArray,
) {
  /** The inputs by the graph's names, in its tensor order. */
  fun feeds(): LinkedHashMap<String, FloatArray> =
    linkedMapOf("mel" to mel, "mel_valid" to melValid, "v1" to v1, "v2" to v2, "v3" to v3)
}

/**
 * The audio host of the model repository (`host/d1_audio_host.py`, the provider's `audio.py` at
 * revision 414f8d64) in Kotlin, Android-free: one 16 kHz mono clip -> the audio graph's five inputs
 * -> the prefix rows. The names follow the Python file.
 *
 * The mel runs every step in float64 and rounds once at the end, as `mel(x, "float64")` does (the
 * Python host's float32 form uses numpy's float32 FFT and matrix product, which no JVM library
 * reproduces bit for bit): the preemphasis in float32 as the provider, the Hann window in float32
 * as `torch.hann_window(400, periodic=False)`, a 512-point real FFT (a 256-point complex radix-2
 * transform and its split), |X|² as `sqrt(re² + im²)` squared, the Slaney filter bank in float32
 * (computed in float64 as librosa computes it), `log(x + 2^-24)`, and the mean and standard
 * deviation of each mel bin over the valid frames only.
 */
object D1Audio {
  const val SAMPLE_RATE = 16000
  const val MIN_SAMPLES = 8000
  const val MAX_SECONDS = 30
  const val N_FFT = 512
  const val WIN = 400
  const val HOP = 160
  const val N_MELS = 128

  /** FFT bins of a real 512-sample frame: 0 … 256. */
  const val BINS = N_FFT / 2 + 1

  /** 2^-24, added before the log. */
  const val LOG_GUARD = 5.9604644775390625E-8

  const val STD_EPS = 1e-5

  /** STFT frames of a 5 / 10 / 20 / 30 s clip (n // 160 + 1): the graphs' buckets. */
  val T_BUCKETS = listOf(501, 1001, 2001, 3001)

  /** Width of a prefix row. */
  const val PREFIX_WIDTH = D1Rows.PREFIX_WIDTH

  private const val INT16_SCALE = 32768f
  private const val PREEMPHASIS = 0.97f

  /**
   * `waveform()` for int16 samples: / 32768 as float32, cut to 30 s, zero-padded to 8,000 samples
   * (0.5 s).
   */
  fun waveform(samples: ShortArray): FloatArray {
    val kept = minOf(samples.size, MAX_SECONDS * SAMPLE_RATE)
    val out = FloatArray(maxOf(kept, MIN_SAMPLES))
    for (i in 0 until kept) out[i] = samples[i] / INT16_SCALE
    return out
  }

  /** `waveform()` for float samples in [-1, 1]: cut to 30 s, zero-padded to 8,000 samples. */
  fun waveform(samples: FloatArray): FloatArray {
    val kept = minOf(samples.size, MAX_SECONDS * SAMPLE_RATE)
    return FloatArray(maxOf(kept, MIN_SAMPLES)).also { samples.copyInto(it, 0, 0, kept) }
  }

  /**
   * `slaney_filterbank()`: librosa's `filters.mel(sr, n_fft, n_mels, norm="slaney")` in float32,
   * computed as librosa (and the provider's copy) computes it — float64 mel points, each weight
   * rounded to float32, then the Slaney normalization applied in float64 to the float32 weight and
   * rounded again. [nMels] x (nFft / 2 + 1), row-major.
   */
  fun slaneyFilterbank(sr: Int = SAMPLE_RATE, nFft: Int = N_FFT, nMels: Int = N_MELS): FloatArray {
    val fSp = 200.0 / 3
    val minLogHz = 1000.0
    val logstep = ln(6.4) / 27.0
    val minLogMel = minLogHz / fSp
    fun hzToMel(f: Double): Double = if (f >= minLogHz) minLogMel + ln(f / minLogHz) / logstep else f / fSp
    fun melToHz(m: Double): Double = if (m >= minLogMel) minLogHz * exp(logstep * (m - minLogMel)) else fSp * m
    val bins = nFft / 2 + 1
    // np.linspace(start, stop, n_mels + 2): i * step + start, the last point set to stop.
    val count = nMels + 2
    val start = hzToMel(0.0)
    val stop = hzToMel(sr / 2.0)
    val step = (stop - start) / (count - 1)
    val melF = DoubleArray(count) { if (it == count - 1) melToHz(stop) else melToHz(it * step + start) }
    // np.fft.rfftfreq(n=n_fft, d=1/sr): i * (1 / (n * d)).
    val unit = 1.0 / (nFft * (1.0 / sr))
    val fftFreqs = DoubleArray(bins) { it * unit }
    val weights = FloatArray(nMels * bins)
    for (i in 0 until nMels) {
      val lowerWidth = melF[i + 1] - melF[i]
      val upperWidth = melF[i + 2] - melF[i + 1]
      // Slaney: each triangle scaled to unit area, 2 / (f[i + 2] - f[i]).
      val norm = 2.0 / (melF[i + 2] - melF[i])
      for (k in 0 until bins) {
        val lower = -(melF[i] - fftFreqs[k]) / lowerWidth
        val upper = (melF[i + 2] - fftFreqs[k]) / upperWidth
        val weight = maxOf(0.0, minOf(lower, upper)).toFloat()
        weights[i * bins + k] = (weight.toDouble() * norm).toFloat()
      }
    }
    return weights
  }

  /**
   * `hann_window_f32()`: `torch.hann_window(400, periodic=False)` in float32 — i * float32(2π / 399)
   * in float32, its cosine taken in float64 and rounded to float32, then * -0.5 + 0.5 in float32.
   */
  fun hannWindowF32(): FloatArray {
    val step = (PI * 2.0 / (WIN - 1)).toFloat()
    return FloatArray(WIN) { i ->
      val c = cos((i.toFloat() * step).toDouble()).toFloat()
      c * -0.5f + 0.5f
    }
  }

  /** `frame_count(n)`: (valid frames = n // 160, STFT frames T = n // 160 + 1). */
  fun frameCount(n: Int): IntArray = intArrayOf((n + N_FFT / 2 * 2 - N_FFT) / HOP, 1 + n / HOP)

  private val filterbank: FloatArray by lazy { slaneyFilterbank() }

  // Each mel filter's non-zero weights: [first, end) over the FFT bins (the sum skips exact zeros).
  private val filterRanges: IntArray by lazy {
    val ranges = IntArray(N_MELS * 2)
    for (m in 0 until N_MELS) {
      var first = BINS
      var end = 0
      for (k in 0 until BINS) {
        if (filterbank[m * BINS + k] != 0f) {
          if (first == BINS) first = k
          end = k + 1
        }
      }
      ranges[m * 2] = minOf(first, end)
      ranges[m * 2 + 1] = end
    }
    ranges
  }

  private val window: FloatArray by lazy { hannWindowF32() }

  /** `mel(x)`: the waveform's normalized log-mel [128, T] (the float64 form, see the class comment). */
  fun mel(x: FloatArray): D1Mel {
    val stages = melStages(x)
    return D1Mel(stages.out, stages.frames, stages.stftFrames)
  }

  /**
   * `mel_stages(x, "float64")`: every stage of the front end for [x] (`waveform()`'s output).
   * [keepPower] keeps the [257, T] power spectrum (tests only); [logInFloat32] takes the log of
   * `float32(lin) + float32(2^-24)` rounded to float32, the float32 form's log step, instead of the
   * float64 log (tests only).
   */
  fun melStages(x: FloatArray, keepPower: Boolean = false, logInFloat32: Boolean = false): D1MelStages {
    val n = x.size
    require(n >= 1) { "an empty waveform" }
    val (frames, stft) = frameCount(n).let { it[0] to it[1] }
    val pre = FloatArray(n)
    pre[0] = x[0]
    for (t in 1 until n) pre[t] = x[t] - PREEMPHASIS * x[t - 1]
    val power = if (keepPower) DoubleArray(BINS * stft) else null
    val lin = DoubleArray(N_MELS * stft)
    val fft = RealFft512()
    val frame = DoubleArray(N_FFT)
    val spectrum = DoubleArray(BINS)
    val offset = (N_FFT - WIN) / 2
    val pad = N_FFT / 2
    for (t in 0 until stft) {
      // center=True: 256 zeros on each side; the 400-sample window sits in the middle of the 512.
      frame.fill(0.0)
      val base = HOP * t - pad + offset
      for (j in 0 until WIN) {
        val index = base + j
        if (index in 0 until n) frame[offset + j] = pre[index].toDouble() * window[j].toDouble()
      }
      fft.power(frame, spectrum)
      if (power != null) for (k in 0 until BINS) power[k * stft + t] = spectrum[k]
      for (m in 0 until N_MELS) {
        var sum = 0.0
        for (k in filterRanges[m * 2] until filterRanges[m * 2 + 1]) {
          sum += filterbank[m * BINS + k].toDouble() * spectrum[k]
        }
        lin[m * stft + t] = sum
      }
    }
    val log = DoubleArray(N_MELS * stft)
    for (i in log.indices) {
      log[i] =
        if (logInFloat32) ln((lin[i].toFloat() + LOG_GUARD.toFloat()).toDouble()).toFloat().toDouble()
        else ln(lin[i] + LOG_GUARD)
    }
    val mean = DoubleArray(N_MELS)
    val std = DoubleArray(N_MELS)
    val out = FloatArray(N_MELS * stft)
    for (m in 0 until N_MELS) {
      val row = m * stft
      var sum = 0.0
      for (t in 0 until frames) sum += log[row + t]
      val average = sum / frames
      var squares = 0.0
      for (t in 0 until frames) {
        val d = log[row + t] - average
        squares += d * d
      }
      var deviation = sqrt(squares / (frames - 1.0))
      if (deviation.isNaN()) deviation = 0.0
      mean[m] = average
      std[m] = deviation
      for (t in 0 until frames) out[row + t] = ((log[row + t] - average) / (deviation + STD_EPS)).toFloat()
      // Frames t >= frames (the last STFT frame) stay 0.
    }
    return D1MelStages(pre, power, lin, log, mean, std, out, frames, stft)
  }

  /** Conv2d(k 3, stride 2, pad 1): `(n + 2 - 3) // 2 + 1` (the provider's lengths rule). */
  fun subLen(n: Int): Int = Math.floorDiv(n + 2 - 3, 2) + 1

  /** `lengths(frames)`: L1, L2, L3 (L3 = P). */
  fun lengths(frames: Int): IntArray {
    val l1 = subLen(frames)
    val l2 = subLen(l1)
    return intArrayOf(l1, l2, subLen(l2))
  }

  /** `dims(T_b)`: T1, T2, T3 of the bucket's mask inputs and output rows. */
  fun dims(bucket: Int): IntArray = lengths(bucket)

  /** `bucket_for(T)`: the smallest of [buckets] (ascending) holding T STFT frames. */
  fun bucketFor(stftFrames: Int, buckets: List<Int> = T_BUCKETS): Int =
    buckets.firstOrNull { stftFrames <= it }
      ?: throw IllegalArgumentException(
        "$stftFrames STFT frames exceed the largest bucket (${buckets.maxOrNull()}): waveform() caps a clip at 30 s"
      )

  /** `build_inputs(mel, frames, T_b)`: the five inputs of `audio_<T_b>`. */
  fun buildInputs(mel: D1Mel, bucket: Int): D1AudioInputs {
    val stft = mel.stftFrames
    require(stft <= bucket) { "$stft frames do not fit bucket $bucket" }
    require(mel.values.size == N_MELS * stft) { "mel holds ${mel.values.size} values, 128 x $stft expected" }
    val padded = FloatArray(N_MELS * bucket)
    for (m in 0 until N_MELS) mel.values.copyInto(padded, m * bucket, m * stft, m * stft + stft)
    val (t1, t2, t3) = dims(bucket).let { Triple(it[0], it[1], it[2]) }
    val (l1, l2, l3) = lengths(mel.frames).let { Triple(it[0], it[1], it[2]) }
    return D1AudioInputs(
      bucket,
      padded,
      FloatArray(bucket) { if (it < mel.frames) 1f else 0f },
      FloatArray(t1) { if (it < l1) 1f else 0f },
      FloatArray(t2) { if (it < l2) 1f else 0f },
      FloatArray(t3) { if (it < l3) 1f else 0f },
    )
  }

  /** The info of a clip of [samples] waveform samples on [bucket]. */
  fun info(samples: Int, frames: Int, stftFrames: Int, bucket: Int): D1AudioInfo =
    D1AudioInfo(samples, frames, stftFrames, bucket, dims(bucket), lengths(frames))

  /** The graph's inputs and output shapes for a bucket (batch 1): mel, mel_valid, v1, v2, v3 -> prefix. */
  fun inputShapes(bucket: Int): LinkedHashMap<String, List<Int>> {
    val (t1, t2, t3) = dims(bucket).let { Triple(it[0], it[1], it[2]) }
    return linkedMapOf(
      "mel" to listOf(1, N_MELS, bucket),
      "mel_valid" to listOf(1, bucket),
      "v1" to listOf(1, t1),
      "v2" to listOf(1, t2),
      "v3" to listOf(1, t3),
    )
  }

  fun outputShape(bucket: Int): Pair<String, List<Int>> = "prefix" to listOf(1, dims(bucket)[2], PREFIX_WIDTH)

  /** `prefix_rows(out, info)`: the first P rows of the graph's prefix [1, T3, 1024] -> [P, 1024]. */
  fun prefixRows(out: FloatArray, info: D1AudioInfo): FloatArray {
    require(out.size == info.bucketDims[2] * PREFIX_WIDTH) {
      "prefix holds ${out.size} values, ${info.bucketDims[2]} x $PREFIX_WIDTH expected for T${info.bucket}"
    }
    return out.copyOf(info.prefixRows * PREFIX_WIDTH)
  }

  /**
   * A 512-point FFT of a real frame in float64 (one 256-point complex radix-2 transform of the
   * even / odd samples, then the split), as |X|² of bins 0 … 256 the way the provider computes it:
   * `sqrt(re² + im²)` squared.
   */
  private class RealFft512 {
    private val half = N_FFT / 2
    // e^{-2πik/512} for k < 256: cos and sin of 2πk/512.
    private val cosine = DoubleArray(half) { cos(2.0 * PI * it / N_FFT) }
    private val sine = DoubleArray(half) { sin(2.0 * PI * it / N_FFT) }
    private val reversed =
      IntArray(half).also { table ->
        val bits = Integer.numberOfTrailingZeros(half)
        for (i in 0 until half) table[i] = Integer.reverse(i) ushr (Int.SIZE_BITS - bits)
      }
    private val re = DoubleArray(half)
    private val im = DoubleArray(half)

    fun power(frame: DoubleArray, out: DoubleArray) {
      for (m in 0 until half) {
        val j = reversed[m]
        re[j] = frame[2 * m]
        im[j] = frame[2 * m + 1]
      }
      var size = 2
      while (size <= half) {
        val span = size / 2
        val stride = N_FFT / size
        var start = 0
        while (start < half) {
          for (k in 0 until span) {
            val c = cosine[k * stride]
            val s = sine[k * stride]
            val a = start + k
            val b = a + span
            // (x + iy) e^{-iθ} = (x cos θ + y sin θ) + i (y cos θ - x sin θ)
            val tr = re[b] * c + im[b] * s
            val ti = im[b] * c - re[b] * s
            re[b] = re[a] - tr
            im[b] = im[a] - ti
            re[a] += tr
            im[a] += ti
          }
          start += size
        }
        size *= 2
      }
      // Bins 0 and 256: the even and odd sums of Z[0].
      out[0] = squared(re[0] + im[0], 0.0)
      out[half] = squared(re[0] - im[0], 0.0)
      for (k in 1 until half) {
        val ar = re[k]
        val ai = im[k]
        val br = re[half - k]
        val bi = -im[half - k]
        // even = (a + b) / 2, odd = (a - b) / 2i, X[k] = even + e^{-2πik/512} odd
        val er = (ar + br) * 0.5
        val ei = (ai + bi) * 0.5
        val or = (ai - bi) * 0.5
        val oi = -(ar - br) * 0.5
        val c = cosine[k]
        val s = sine[k]
        out[k] = squared(er + or * c + oi * s, ei + oi * c - or * s)
      }
    }

    private fun squared(x: Double, y: Double): Double {
      val magnitude = sqrt(x * x + y * y)
      return magnitude * magnitude
    }
  }
}
