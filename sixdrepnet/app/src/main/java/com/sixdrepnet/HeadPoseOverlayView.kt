package com.sixdrepnet

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View
import kotlin.math.cos
import kotlin.math.sin

/** Draws the 3D head-pose axes (from yaw/pitch/roll) centered in the view + a readout. */
class HeadPoseOverlayView(context: Context) : View(context) {

    private var pose: HeadPose? = null
    private val axis = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 12f; style = Paint.Style.STROKE }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 44f; setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    fun setPose(p: HeadPose) { pose = p; postInvalidateOnAnimation() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val hp = pose ?: return
        val cx = width / 2f; val cy = height / 2f; val size = minOf(width, height) * 0.28f
        val p = Math.toRadians(hp.pitch.toDouble())
        val ya = Math.toRadians(-hp.yaw.toDouble())
        val r = Math.toRadians(hp.roll.toDouble())
        // X axis (red — points to the subject's left)
        val x1 = size * (cos(ya) * cos(r)).toFloat() + cx
        val y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(ya)).toFloat() + cy
        // Y axis (green — points down)
        val x2 = size * (-cos(ya) * sin(r)).toFloat() + cx
        val y2 = size * (cos(p) * cos(r) - sin(p) * sin(ya) * sin(r)).toFloat() + cy
        // Z axis (blue — points out of the face)
        val x3 = size * sin(ya).toFloat() + cx
        val y3 = size * (-cos(ya) * sin(p)).toFloat() + cy
        axis.color = Color.rgb(255, 60, 60); canvas.drawLine(cx, cy, x1, y1, axis)
        axis.color = Color.rgb(60, 220, 90); canvas.drawLine(cx, cy, x2, y2, axis)
        axis.color = Color.rgb(70, 130, 255); canvas.drawLine(cx, cy, x3, y3, axis)
        canvas.drawText("yaw ${hp.yaw.toInt()}  pitch ${hp.pitch.toInt()}  roll ${hp.roll.toInt()}",
            40f, height - 60f, text)
    }
}
