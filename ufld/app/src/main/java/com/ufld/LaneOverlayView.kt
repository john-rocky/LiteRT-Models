package com.ufld

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/** Draws detected lane points, colored per lane, over the camera preview. */
class LaneOverlayView(context: Context) : View(context) {

    private var points: List<LanePoint> = emptyList()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val laneColors = intArrayOf(
        Color.rgb(255, 60, 60), Color.rgb(60, 255, 60),
        Color.rgb(60, 120, 255), Color.rgb(255, 220, 40))

    fun setPoints(p: List<LanePoint>) { points = p; postInvalidateOnAnimation() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = width / 90f
        for (pt in points) {
            paint.color = laneColors[pt.lane % laneColors.size]
            canvas.drawCircle(pt.x * width, pt.y * height, r, paint)
        }
    }
}
