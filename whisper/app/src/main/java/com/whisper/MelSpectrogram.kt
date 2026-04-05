package com.whisper

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/**
 * Compute log-mel spectrogram from raw audio samples.
 * Matches OpenAI Whisper's preprocessing exactly.
 *
 * Parameters: 16kHz, n_fft=400, hop=160, n_mels=80, 30s chunks → [80, 3000]
 */
class MelSpectrogram(context: Context) {

    companion object {
        const val SAMPLE_RATE = 16000
        const val N_FFT = 400
        const val HOP_LENGTH = 160
        const val N_MELS = 80
        const val CHUNK_LENGTH = 30  // seconds
        const val N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH  // 480000
        const val N_FRAMES = N_SAMPLES / HOP_LENGTH  // 3000
        private const val FFT_SIZE = 512  // next power of 2 >= N_FFT
    }

    // Pre-loaded mel filterbank [80, 201] (201 = N_FFT/2 + 1)
    private val melFilters: FloatArray

    init {
        val stream = context.assets.open("mel_filters.bin")
        val bytes = stream.readBytes()
        stream.close()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        melFilters = FloatArray(bytes.size / 4)
        buf.asFloatBuffer().get(melFilters)
    }

    /**
     * Compute log-mel spectrogram from audio samples.
     * @param samples PCM float samples at 16kHz, mono
     * @return FloatArray of shape [N_MELS * N_FRAMES] in row-major order
     */
    fun compute(samples: FloatArray): FloatArray {
        // Pad or trim to exactly 30 seconds
        val padded = FloatArray(N_SAMPLES)
        val copyLen = minOf(samples.size, N_SAMPLES)
        System.arraycopy(samples, 0, padded, 0, copyLen)

        val nFreqs = N_FFT / 2 + 1  // 201

        // STFT: compute magnitude squared for each frame
        val magnitudes = FloatArray(N_FRAMES * nFreqs)
        val window = hannWindow(N_FFT)
        val fftReal = FloatArray(FFT_SIZE)
        val fftImag = FloatArray(FFT_SIZE)

        for (frame in 0 until N_FRAMES) {
            val offset = frame * HOP_LENGTH

            // Apply window
            fftReal.fill(0f)
            fftImag.fill(0f)
            for (i in 0 until N_FFT) {
                val idx = offset + i
                fftReal[i] = if (idx < padded.size) padded[idx] * window[i] else 0f
            }

            // FFT
            fft(fftReal, fftImag, FFT_SIZE)

            // Magnitude squared (power spectrum)
            for (i in 0 until nFreqs) {
                magnitudes[frame * nFreqs + i] = fftReal[i] * fftReal[i] + fftImag[i] * fftImag[i]
            }
        }

        // Apply mel filterbank: [80, 201] x [201, 3000] → [80, 3000]
        val melSpec = FloatArray(N_MELS * N_FRAMES)
        for (mel in 0 until N_MELS) {
            for (frame in 0 until N_FRAMES) {
                var sum = 0f
                for (freq in 0 until nFreqs) {
                    sum += melFilters[mel * nFreqs + freq] * magnitudes[frame * nFreqs + freq]
                }
                melSpec[mel * N_FRAMES + frame] = sum
            }
        }

        // Log scale and normalize (matching Whisper's log_mel_spectrogram)
        val logSpec = FloatArray(melSpec.size)
        var maxLog = -Float.MAX_VALUE
        for (i in melSpec.indices) {
            val v = ln(maxOf(melSpec[i], 1e-10f))  // natural log
            logSpec[i] = v
            if (v > maxLog) maxLog = v
        }

        // Clamp and normalize: matches Whisper's (log10 → clamp → scale)
        // Whisper uses log10, then clamp(max - 8.0), then (x + 4.0) / 4.0
        // Convert our ln to log10: log10(x) = ln(x) / ln(10)
        val ln10 = ln(10f)
        for (i in logSpec.indices) {
            var v = logSpec[i] / ln10  // convert to log10
            v = maxOf(v, maxLog / ln10 - 8.0f)
            logSpec[i] = (v + 4.0f) / 4.0f
        }

        return logSpec
    }

    private fun hannWindow(size: Int): FloatArray {
        return FloatArray(size) { i ->
            0.5f * (1f - cos(2f * PI.toFloat() * i / size))
        }
    }

    /**
     * Cooley-Tukey radix-2 FFT (in-place).
     */
    private fun fft(real: FloatArray, imag: FloatArray, n: Int) {
        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                var temp = real[i]; real[i] = real[j]; real[j] = temp
                temp = imag[i]; imag[i] = imag[j]; imag[j] = temp
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        // FFT butterfly
        var step = 2
        while (step <= n) {
            val halfStep = step / 2
            val angle = -2.0 * PI / step
            for (group in 0 until n step step) {
                for (pair in 0 until halfStep) {
                    val w = angle * pair
                    val wr = cos(w).toFloat()
                    val wi = sin(w).toFloat()

                    val i1 = group + pair
                    val i2 = i1 + halfStep

                    val tr = wr * real[i2] - wi * imag[i2]
                    val ti = wr * imag[i2] + wi * real[i2]

                    real[i2] = real[i1] - tr
                    imag[i2] = imag[i1] - ti
                    real[i1] = real[i1] + tr
                    imag[i1] = imag[i1] + ti
                }
            }
            step *= 2
        }
    }
}
