package com.audio8tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

/** The recording clean-up: silence around the speech is cut to 0.25 s, the peak goes to -3 dBFS. */
class ReferenceAudioTest {
    private val rate = 44100

    /** [lead] s of faint noise, [tone] s of a 220 Hz tone, [tail] s of faint noise. */
    private fun take(lead: Double, tone: Double, tail: Double): FloatArray {
        val a = (lead * rate).toInt()
        val b = (tone * rate).toInt()
        val c = (tail * rate).toInt()
        return FloatArray(a + b + c) { i ->
            if (i in a until a + b) (0.3 * sin(2 * PI * 220 * i / rate)).toFloat()
            else if (i % 2 == 0) 0.0005f else -0.0005f
        }
    }

    @Test
    fun silenceAroundTheSpeechIsCutToAQuarterSecond() {
        val out = ReferenceAudio.trimSilence(take(lead = 2.0, tone = 3.0, tail = 4.0), rate)
        val seconds = out.size / rate.toDouble()
        assertEquals(3.5, seconds, 0.05)
    }

    @Test
    fun shortSilenceIsKept() {
        val x = take(lead = 0.1, tone = 2.0, tail = 0.1)
        assertEquals(x.size, ReferenceAudio.trimSilence(x, rate).size)
    }

    @Test
    fun allQuietIsReturnedUnchanged() {
        val x = FloatArray(rate)
        assertEquals(x.size, ReferenceAudio.trimSilence(x, rate).size)
    }

    @Test
    fun peakIsMinusThreeDbfs() {
        val y = ReferenceAudio.normalize(take(lead = 0.5, tone = 1.0, tail = 0.5))
        val peak = y.maxOf { abs(it) }
        assertEquals(ReferenceAudio.PEAK, peak, 1e-5f)
        assertTrue(ReferenceAudio.normalize(FloatArray(10)).all { it == 0f })
    }
}
