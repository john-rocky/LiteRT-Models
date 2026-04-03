package com.yolo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.view.View

class DetectionCanvasView(context: Context) : View(context) {

    private var detections: List<Detection> = emptyList()
    private var imageWidth = 1
    private var imageHeight = 1

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 40f
        setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
    }

    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
    }

    private val rect = RectF()

    fun setDetections(dets: List<Detection>, imgW: Int, imgH: Int) {
        detections = dets
        imageWidth = imgW
        imageHeight = imgH
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Match PreviewView's FILL_CENTER: uniform scale + center crop
        val scale = maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f

        for (det in detections) {
            val color = CocoLabels.color(det.classId)
            boxPaint.color = color

            rect.set(
                det.xMin * scale + offsetX,
                det.yMin * scale + offsetY,
                det.xMax * scale + offsetX,
                det.yMax * scale + offsetY,
            )
            canvas.drawRect(rect, boxPaint)

            val label = "${det.className} ${(det.score * 100).toInt()}%"
            val textW = textPaint.measureText(label)
            val textH = textPaint.textSize

            bgPaint.color = (color and 0x00FFFFFF) or 0xAA000000.toInt()
            canvas.drawRect(rect.left, rect.top - textH - 8, rect.left + textW + 16, rect.top, bgPaint)
            canvas.drawText(label, rect.left + 8, rect.top - 8, textPaint)
        }
    }
}
