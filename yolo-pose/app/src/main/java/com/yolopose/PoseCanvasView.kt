package com.yolopose

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.View

/**
 * Overlay that draws pose keypoints + skeleton edges on top of a CameraX
 * PreviewView (FILL_CENTER) or a FIT_CENTER ImageView.
 */
class PoseCanvasView(context: Context) : View(context) {

    private var poses: List<Pose> = emptyList()
    private var imageWidth = 1
    private var imageHeight = 1

    /** Set true for ImageView (FIT_CENTER), false for PreviewView (FILL_CENTER). */
    var fitCenter: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    private val edgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
    }

    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val pointStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = 0xFF000000.toInt()
    }

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = 0x88FFFFFFu.toInt()
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFFu.toInt()
        textSize = 36f
        setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
    }

    private companion object {
        const val KP_CONF_THRESHOLD = 0.30f
        const val POINT_RADIUS = 8f
    }

    fun setPoses(items: List<Pose>, imgW: Int, imgH: Int) {
        poses = items
        imageWidth = imgW
        imageHeight = imgH
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (poses.isEmpty() || imageWidth <= 0 || imageHeight <= 0) return

        val sx = width.toFloat() / imageWidth
        val sy = height.toFloat() / imageHeight
        val scale = if (fitCenter) minOf(sx, sy) else maxOf(sx, sy)
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f

        for (pose in poses) {
            // Person bounding box (faint, just for context)
            canvas.drawRect(
                pose.xMin * scale + offsetX,
                pose.yMin * scale + offsetY,
                pose.xMax * scale + offsetX,
                pose.yMax * scale + offsetY,
                boxPaint,
            )

            // Skeleton edges
            for (edge in CocoKeypoints.EDGES) {
                val a = edge[0]
                val b = edge[1]
                val color = edge[2]
                if (pose.keypointConf(a) < KP_CONF_THRESHOLD) continue
                if (pose.keypointConf(b) < KP_CONF_THRESHOLD) continue
                edgePaint.color = color
                canvas.drawLine(
                    pose.keypointX(a) * scale + offsetX,
                    pose.keypointY(a) * scale + offsetY,
                    pose.keypointX(b) * scale + offsetX,
                    pose.keypointY(b) * scale + offsetY,
                    edgePaint,
                )
            }

            // Keypoint dots (drawn on top of edges)
            for (k in 0 until CocoKeypoints.NUM_KEYPOINTS) {
                if (pose.keypointConf(k) < KP_CONF_THRESHOLD) continue
                val cx = pose.keypointX(k) * scale + offsetX
                val cy = pose.keypointY(k) * scale + offsetY
                pointPaint.color = CocoKeypoints.POINT_COLORS[k]
                canvas.drawCircle(cx, cy, POINT_RADIUS, pointPaint)
                canvas.drawCircle(cx, cy, POINT_RADIUS, pointStrokePaint)
            }

            // Person score label
            val label = "${(pose.score * 100).toInt()}%"
            canvas.drawText(
                label,
                pose.xMin * scale + offsetX + 8f,
                pose.yMin * scale + offsetY - 8f,
                labelPaint,
            )
        }
    }
}
