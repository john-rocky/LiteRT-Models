package com.liveness

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/** Draws the liveness verdict (LIVE / SPOOF) + score, centered. */
class LivenessOverlayView(context: Context) : View(context) {
    private var isLive = false
    private var score = 0f
    private var has = false
    private val box = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 10f }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 90f; setShadowLayer(6f, 0f, 0f, Color.BLACK) }
    private val sub = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 44f; setShadowLayer(5f, 0f, 0f, Color.BLACK) }

    fun setResult(live: Boolean, s: Float) { isLive = live; score = s; has = true; postInvalidateOnAnimation() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!has) return
        val col = if (isLive) Color.rgb(50, 220, 100) else Color.rgb(240, 70, 70)
        // center square guide box (where to place the face)
        val s = minOf(width, height) * 0.6f; val l = (width - s) / 2; val tp = (height - s) / 2
        box.color = col; canvas.drawRect(l, tp, l + s, tp + s, box)
        text.color = col
        val label = if (isLive) "LIVE" else "SPOOF"
        canvas.drawText(label, l, tp - 30, text)
        canvas.drawText("live score ${(score * 100).toInt()}%", l, tp + s + 60, sub)
    }
}
