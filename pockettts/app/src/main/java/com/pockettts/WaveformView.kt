package com.pockettts

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.os.SystemClock
import android.view.View

/**
 * Minimal audio visualization: min/max bars of the generated waveform with a
 * playhead sweeping in real time while the clip plays. Purely cosmetic — the
 * synthesizer does not depend on it.
 */
class WaveformView(context: Context) : View(context) {

    private val barPaint = Paint().apply { color = 0xFFB9C0EE.toInt() }
    private val playedPaint = Paint().apply { color = 0xFF3F4FBF.toInt() }
    private val headPaint = Paint().apply {
        color = 0xFF1F2430.toInt()
        strokeWidth = 3f
    }

    private var mins = FloatArray(0)
    private var maxs = FloatArray(0)
    private var durationMs = 0L
    private var playStart = 0L

    /** Show `audio` and start the playhead sweep now. */
    fun start(audio: FloatArray, sampleRate: Int) {
        val buckets = 240
        mins = FloatArray(buckets)
        maxs = FloatArray(buckets)
        if (audio.isNotEmpty()) {
            val per = (audio.size + buckets - 1) / buckets
            for (b in 0 until buckets) {
                var lo = 0f
                var hi = 0f
                var i = b * per
                val end = minOf(i + per, audio.size)
                while (i < end) {
                    val v = audio[i]
                    if (v < lo) lo = v
                    if (v > hi) hi = v
                    i++
                }
                mins[b] = lo
                maxs[b] = hi
            }
        }
        durationMs = audio.size * 1000L / sampleRate
        playStart = SystemClock.uptimeMillis()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val n = mins.size
        if (n == 0) return
        val w = width.toFloat()
        val h = height.toFloat()
        val mid = h / 2f
        val bw = w / n
        val elapsed = SystemClock.uptimeMillis() - playStart
        val frac = if (durationMs > 0) (elapsed.toFloat() / durationMs).coerceIn(0f, 1f) else 1f
        val headX = frac * w
        for (b in 0 until n) {
            val x = b * bw
            val top = mid + mins[b] * mid * 0.9f
            val bot = mid + maxs[b] * mid * 0.9f
            canvas.drawRect(x, minOf(top, bot) - 1f, x + bw * 0.7f, maxOf(top, bot) + 1f,
                if (x <= headX) playedPaint else barPaint)
        }
        if (frac < 1f) {
            canvas.drawLine(headX, 0f, headX, h, headPaint)
            postInvalidateOnAnimation()
        }
    }
}
