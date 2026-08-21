package com.edgeagent.lab

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * The screenshot with the model's answer drawn on it.
 *
 * Draws in screen coordinates and lets one matrix handle the fit-to-view scale,
 * so the marker cannot drift away from the pixel it names — the mapping from
 * normalized model output to screen pixels happens once, in [Framing.View].
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var screenshot: Bitmap? = null
    private var marks: List<Mark> = emptyList()

    /** A grounding already mapped into screen pixels. */
    data class Mark(val label: String, val at: PointF, val box: RectF?)

    private val destination = Rect()
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.parseColor("#2ECC71")
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#2ECC71")
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
    }
    private val labelBackground = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CC1B7F3B")
    }

    fun show(screenshot: Bitmap, marks: List<Mark>) {
        this.screenshot = screenshot
        this.marks = marks
        invalidate()
    }

    fun clear() {
        screenshot = null
        marks = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bitmap = screenshot ?: return

        // Letterbox the screenshot into the view, and reuse the same scale for
        // the marks.
        val scale = minOf(width.toFloat() / bitmap.width, height.toFloat() / bitmap.height)
        val drawWidth = bitmap.width * scale
        val drawHeight = bitmap.height * scale
        val left = (width - drawWidth) / 2f
        val top = (height - drawHeight) / 2f
        destination.set(
            left.toInt(),
            top.toInt(),
            (left + drawWidth).toInt(),
            (top + drawHeight).toInt(),
        )
        canvas.drawBitmap(bitmap, null, destination, null)

        val radius = 14f * resources.displayMetrics.density
        ringPaint.strokeWidth = 4f * resources.displayMetrics.density
        labelPaint.textSize = 13f * resources.displayMetrics.scaledDensity

        for (mark in marks) {
            val x = left + mark.at.x * scale
            val y = top + mark.at.y * scale

            mark.box?.let { box ->
                canvas.drawRect(
                    left + box.left * scale,
                    top + box.top * scale,
                    left + box.right * scale,
                    top + box.bottom * scale,
                    ringPaint,
                )
            }
            canvas.drawCircle(x, y, radius, ringPaint)
            canvas.drawCircle(x, y, radius * 0.22f, dotPaint)

            val padding = 6f * resources.displayMetrics.density
            val textWidth = labelPaint.measureText(mark.label)
            val metrics = labelPaint.fontMetrics
            val boxTop = y - radius - (metrics.bottom - metrics.top) - padding * 2
            val boxLeft = (x + radius).coerceAtMost(width - textWidth - padding * 2)
            canvas.drawRoundRect(
                boxLeft,
                boxTop,
                boxLeft + textWidth + padding * 2,
                boxTop + (metrics.bottom - metrics.top) + padding * 2,
                6f,
                6f,
                labelBackground,
            )
            canvas.drawText(
                mark.label,
                boxLeft + padding,
                boxTop + padding - metrics.top,
                labelPaint,
            )
        }
    }
}
