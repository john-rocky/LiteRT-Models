package com.yolotracking

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.View

/**
 * Overlay view that draws tracked bounding boxes with IDs and motion trails.
 */
class TrackingCanvasView(context: Context) : View(context) {

    private var trackedDetections: List<Detection> = emptyList()
    private var tracks: List<Track> = emptyList()
    private var imageWidth = 1
    private var imageHeight = 1

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val idPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 44f
        isFakeBoldText = true
        setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 32f
        setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
    }

    private val bgPaint = Paint().apply { style = Paint.Style.FILL }

    private val trailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val rect = RectF()
    private val trailPath = Path()

    fun setResults(dets: List<Detection>, activeTracks: List<Track>, imgW: Int, imgH: Int) {
        trackedDetections = dets
        tracks = activeTracks
        imageWidth = imgW
        imageHeight = imgH
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scale = maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f

        // Draw trails
        for (track in tracks) {
            if (track.trail.size < 2) continue
            val color = CocoLabels.trackColor(track.trackId)
            trailPaint.color = (color and 0x00FFFFFF) or 0x99000000.toInt()

            trailPath.reset()
            var first = true
            for ((tx, ty) in track.trail) {
                val sx = tx * scale + offsetX
                val sy = ty * scale + offsetY
                if (first) { trailPath.moveTo(sx, sy); first = false }
                else trailPath.lineTo(sx, sy)
            }
            canvas.drawPath(trailPath, trailPaint)
        }

        // Draw boxes and labels
        for (det in trackedDetections) {
            val color = CocoLabels.trackColor(det.trackId)
            boxPaint.color = color

            rect.set(
                det.xMin * scale + offsetX,
                det.yMin * scale + offsetY,
                det.xMax * scale + offsetX,
                det.yMax * scale + offsetY,
            )
            canvas.drawRect(rect, boxPaint)

            // ID badge
            val idLabel = "#${det.trackId}"
            val idW = idPaint.measureText(idLabel)
            val idH = idPaint.textSize

            bgPaint.color = (color and 0x00FFFFFF) or 0xDD000000.toInt()
            canvas.drawRect(rect.left, rect.top - idH - 8, rect.left + idW + 16, rect.top, bgPaint)
            canvas.drawText(idLabel, rect.left + 8, rect.top - 8, idPaint)

            // Class label below ID
            val classLabel = det.className
            val clsW = labelPaint.measureText(classLabel)
            val clsH = labelPaint.textSize

            bgPaint.color = 0x88000000.toInt()
            canvas.drawRect(rect.left, rect.top, rect.left + clsW + 16, rect.top + clsH + 8, bgPaint)
            canvas.drawText(classLabel, rect.left + 8, rect.top + clsH, labelPaint)
        }
    }
}
