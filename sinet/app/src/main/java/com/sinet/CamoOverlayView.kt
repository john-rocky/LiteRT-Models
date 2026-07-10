package com.sinet

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.view.View

/**
 * Draws the camouflage heatmap over the camera preview, aligned to what PreviewView shows.
 *
 * PreviewView uses FILL_CENTER: the camera frame (aspect srcW:srcH) is scaled by
 * max(view/frame) and centered, cropping the overflow. We map the (square) result bitmap
 * into that same rect so the overlay lines up with the object instead of being stretched
 * to the full view.
 */
class CamoOverlayView(context: Context) : View(context) {
    private var bmp: Bitmap? = null
    private var srcW = 1
    private var srcH = 1
    private val src = Rect(); private val dst = Rect()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** [b] = square result map; [sw]x[sh] = the camera frame dimensions it was computed from. */
    fun setHeat(b: Bitmap, sw: Int, sh: Int) { bmp = b; srcW = sw; srcH = sh; postInvalidateOnAnimation() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = bmp ?: return
        src.set(0, 0, b.width, b.height)
        val scale = maxOf(width.toFloat() / srcW, height.toFloat() / srcH)   // FILL_CENTER
        val rw = srcW * scale; val rh = srcH * scale
        val l = ((width - rw) / 2f).toInt(); val t = ((height - rh) / 2f).toInt()
        dst.set(l, t, (l + rw).toInt(), (t + rh).toInt())
        canvas.drawBitmap(b, src, dst, paint)
    }
}
