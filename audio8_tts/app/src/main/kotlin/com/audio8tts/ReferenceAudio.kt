package com.audio8tts

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Prepares a microphone recording for the codec encoder: the silence before and after the speech is cut to
 * [KEEP_S] (a 10 s take of a 5 s sentence would otherwise put 5 s of room noise into every prompt), then the peak goes
 * to -3 dBFS as in the demo's mic registration.
 */
object ReferenceAudio {
    /** -3 dBFS. */
    const val PEAK = 0.70794576f

    /** Silence kept before the first and after the last loud block, seconds. */
    const val KEEP_S = 0.25

    /** A 20 ms block counts as speech when its RMS is within this many dB of the loudest block. */
    const val FLOOR_DB = -35.0

    private const val BLOCK_S = 0.02

    fun normalize(x: FloatArray): FloatArray {
        var peak = 0f
        for (v in x) peak = max(peak, abs(v))
        val gain = if (peak > 0f) PEAK / peak else 1f
        return FloatArray(x.size) { x[it] * gain }
    }

    /** [x] from [KEEP_S] before its first speech block to [KEEP_S] after its last one; unchanged when all is quiet. */
    fun trimSilence(x: FloatArray, rate: Int): FloatArray {
        val block = max(1, (rate * BLOCK_S).toInt())
        val n = x.size / block
        if (n == 0) return x
        val rms = DoubleArray(n) { b ->
            var s = 0.0
            for (i in b * block until (b + 1) * block) s += x[i].toDouble() * x[i]
            sqrt(s / block)
        }
        val loudest = rms.max()
        if (loudest <= 0.0) return x
        val floor = loudest * 10.0.pow(FLOOR_DB / 20)
        val first = rms.indexOfFirst { it >= floor }
        val last = rms.indexOfLast { it >= floor }
        val keep = (rate * KEEP_S).toInt()
        val start = max(0, first * block - keep)
        val end = min(x.size, (last + 1) * block + keep)
        return x.copyOfRange(start, end)
    }
}
