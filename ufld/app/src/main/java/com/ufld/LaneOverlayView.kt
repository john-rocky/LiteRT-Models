package com.ufld

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/** Draws detected lane points, colored per lane, over the camera preview. */
class LaneOverlayView(context: Context) : View(context) {

    private var points: List<LanePoint> = emptyList()
    private var srcW = 1; private var srcH = 1
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val laneColors = intArrayOf(
        Color.rgb(255, 60, 60), Color.rgb(60, 255, 60),
        Color.rgb(60, 120, 255), Color.rgb(255, 220, 40))

    /** [sw]x[sh] = the camera frame dimensions the normalized points were computed in. */
    fun setPoints(p: List<LanePoint>, sw: Int, sh: Int) { points = p; srcW = sw; srcH = sh; postInvalidateOnAnimation() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // FILL_CENTER: map normalized (0..1) frame coords into the same rect PreviewView shows.
        val scale = maxOf(width.toFloat() / srcW, height.toFloat() / srcH)
        val rw = srcW * scale; val rh = srcH * scale
        val l = (width - rw) / 2f; val t = (height - rh) / 2f
        val r = width / 90f
        for (pt in points) {
            paint.color = laneColors[pt.lane % laneColors.size]
            canvas.drawCircle(l + pt.x * rw, t + pt.y * rh, r, paint)
        }
    }
}
