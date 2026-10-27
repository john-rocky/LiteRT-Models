package com.ming3tts

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sin

/**
 * The VAE decoder's ISTFT head on the host (upstream audio_tokenizer/istft.py, the
 * host spec's host_mirror.Istft / IstftStream): a head row [3530] = 1765 log-magnitudes + 1765 phases ->
 * mag = min(exp(.), 100) -> complex spectrum -> 3528-point inverse real FFT (float64) -> times the
 * checkpoint's window (bf16-rounded Hann, NOT an exact Hann) -> overlap-add with hop 882 and
 * "same" padding (1323 samples trimmed at the start and at the end of the utterance).
 *
 * The stream emits every sample as soon as no later frame can touch it, with the same float32
 * adds in the same order as the whole-utterance fold, so streamed and offline audio are equal.
 */
class Istft(private val window: FloatArray) {
    val n = window.size              // 3528
    val hop = 882
    val pad = (n - hop) / 2          // 1323
    private val half = n / 2 + 1     // 1765
    private val fft = MixedRadixFft(n)
    private val re = DoubleArray(n)
    private val im = DoubleArray(n)
    private val w2 = FloatArray(n) { window[it] * window[it] }

    init {
        require(n == 3528) { "window length $n" }
    }

    /** One head row -> one windowed frame [3528] (float32). */
    fun frame(head: FloatArray, off: Int, out: FloatArray) {
        for (k in 0 until half) {
            val mag = min(exp(head[off + k].toDouble()).toFloat(), 100f)
            val p = head[off + half + k].toDouble()
            val r = mag * cos(p).toFloat()
            val i = mag * sin(p).toFloat()
            re[k] = r.toDouble()
            im[k] = i.toDouble()
        }
        // irfft: Hermitian extension; the imaginary parts of bins 0 and n/2 are ignored
        im[0] = 0.0
        im[half - 1] = 0.0
        for (k in 1 until half - 1) {
            re[n - k] = re[k]
            im[n - k] = -im[k]
        }
        fft.inverse(re, im)
        for (t in 0 until n) out[t] = (re[t] / n).toFloat() * window[t]
    }

    /** Streaming overlap-add of windowed frames; [push] returns the samples that became final. */
    inner class Stream {
        private var y = FloatArray(4 * n)
        private var env = FloatArray(4 * n)
        private var used = 0         // valid length of y / env
        private var frames = 0       // frames pushed
        private var done = 0         // fold-coordinate samples already emitted (incl. the pad)

        fun push(frame: FloatArray) {
            val need = frames * hop + n - done
            if (y.size < need) {
                y = y.copyOf(need * 2)
                env = env.copyOf(need * 2)
            }
            if (used < need) used = need
            val o = frames * hop - done
            for (t in 0 until n) {
                y[o + t] += frame[t]
                env[o + t] += w2[t]
            }
            frames++
        }

        /** Emit the samples before the next frame's start (call after a batch of [push]). */
        fun flush(sink: (FloatArray) -> Unit) = emit(frames * hop, sink)

        fun finish(sink: (FloatArray) -> Unit) {
            if (frames == 0) return
            emit((frames - 1) * hop + n - pad, sink)
        }

        private fun emit(upto: Int, sink: (FloatArray) -> Unit) {
            val k = upto - done
            if (k <= 0) return
            val lo = maxOf(0, pad - done)
            if (lo < k) sink(FloatArray(k - lo) { y[lo + it] / env[lo + it] })
            System.arraycopy(y, k, y, 0, used - k)
            System.arraycopy(env, k, env, 0, used - k)
            java.util.Arrays.fill(y, used - k, used, 0f)
            java.util.Arrays.fill(env, used - k, used, 0f)
            used -= k
            done = upto
        }
    }
}

/** In-place complex FFT for any n factored into small primes (here 3528 = 2^3 3^2 7^2). */
class MixedRadixFft(private val n: Int) {
    private val factors = IntList()
    private val cosT = DoubleArray(n) { cos(2.0 * PI * it / n) }
    private val sinT = DoubleArray(n) { sin(2.0 * PI * it / n) }
    private val tr = DoubleArray(n)
    private val ti = DoubleArray(n)
    private val sr = DoubleArray(64)
    private val si = DoubleArray(64)

    init {
        var m = n
        var p = 2
        while (m > 1) {
            while (m % p == 0) {
                factors.add(p)
                m /= p
            }
            p++
            require(p <= 64 || m == 1) { "n=$n has a large prime factor" }
        }
    }

    /** x <- sum_k x[k] exp(+2 pi i k t / n) (no 1/n). */
    fun inverse(re: DoubleArray, im: DoubleArray) {
        System.arraycopy(re, 0, tr, 0, n)
        System.arraycopy(im, 0, ti, 0, n)
        rec(tr, ti, 0, 1, re, im, 0, n, 0)
    }

    // Decimation in time: out[outOff .. outOff+len) = DFT of in[inOff + j*stride], j < len.
    private fun rec(
        inR: DoubleArray, inI: DoubleArray, inOff: Int, stride: Int,
        outR: DoubleArray, outI: DoubleArray, outOff: Int, len: Int, fi: Int,
    ) {
        if (len == 1) {
            outR[outOff] = inR[inOff]
            outI[outOff] = inI[inOff]
            return
        }
        val p = factors[fi]
        val m = len / p
        // p sub-transforms of length m, stored back to back in out
        for (r in 0 until p) {
            rec(inR, inI, inOff + r * stride, stride * p, outR, outI, outOff + r * m, m, fi + 1)
        }
        // combine: X[k + q m] = sum_r W_len^{r (k + q m)} Y_r[k]
        val tw = n / len
        for (k in 0 until m) {
            for (r in 0 until p) {
                sr[r] = outR[outOff + r * m + k]
                si[r] = outI[outOff + r * m + k]
            }
            for (q in 0 until p) {
                val idx = k + q * m
                var accR = 0.0
                var accI = 0.0
                for (r in 0 until p) {
                    val e = (r.toLong() * idx % len).toInt() * tw
                    val c = cosT[e]
                    val s = sinT[e]
                    accR += sr[r] * c - si[r] * s
                    accI += sr[r] * s + si[r] * c
                }
                outR[outOff + idx] = accR
                outI[outOff + idx] = accI
            }
        }
    }
}
