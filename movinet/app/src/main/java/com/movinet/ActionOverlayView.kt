package com.movinet

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View

/** Bottom panel that renders the current top-K Kinetics-600 predictions as bars. */
class ActionOverlayView(context: Context) : View(context) {

    private var preds: List<Prediction> = emptyList()

    private val panelPaint = Paint().apply { color = 0xAA000000.toInt() }
    private val barBgPaint = Paint().apply { color = 0x33FFFFFF }
    private val barPaint = Paint().apply { color = 0xFF4CAF50.toInt() }
    private val topBarPaint = Paint().apply { color = 0xFF00E5FF.toInt() }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 42f
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    private val scorePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFCCCCCC.toInt()
        textSize = 34f
        textAlign = Paint.Align.RIGHT
    }

    fun setPredictions(p: List<Prediction>) {
        preds = p
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (preds.isEmpty()) return

        val rowH = 96f
        val pad = 32f
        val panelH = pad * 2 + rowH * preds.size
        val top = height - panelH
        canvas.drawRect(0f, top, width.toFloat(), height.toFloat(), panelPaint)

        var y = top + pad
        preds.forEachIndexed { i, p ->
            val barLeft = pad
            val barRight = width - pad
            val barTop = y + rowH - 26f
            val barBottom = y + rowH - 8f
            canvas.drawRect(RectF(barLeft, barTop, barRight, barBottom), barBgPaint)
            val w = (barRight - barLeft) * p.score.coerceIn(0f, 1f)
            canvas.drawRect(RectF(barLeft, barTop, barLeft + w, barBottom),
                if (i == 0) topBarPaint else barPaint)
            canvas.drawText(p.label, barLeft, y + 42f, textPaint)
            canvas.drawText("${(p.score * 100).toInt()}%", barRight, y + 42f, scorePaint)
            y += rowH
        }
    }
}
