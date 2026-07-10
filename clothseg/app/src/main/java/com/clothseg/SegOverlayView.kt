package com.clothseg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.view.View

/**
 * Draws the colored clothing-segmentation overlay over the camera preview, aligned to what
 * PreviewView shows (FILL_CENTER: frame scaled by max(view/frame), centered, overflow cropped).
 */
class SegOverlayView(context: Context) : View(context) {
    private var bmp: Bitmap? = null
    private var srcW = 1; private var srcH = 1
    private val src = Rect(); private val dst = Rect()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    fun setOverlay(b: Bitmap, sw: Int, sh: Int) { bmp = b; srcW = sw; srcH = sh; postInvalidateOnAnimation() }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = bmp ?: return
        src.set(0, 0, b.width, b.height)
        val scale = maxOf(width.toFloat() / srcW, height.toFloat() / srcH)
        val rw = srcW * scale; val rh = srcH * scale
        val l = ((width - rw) / 2f).toInt(); val t = ((height - rh) / 2f).toInt()
        dst.set(l, t, (l + rw).toInt(), (t + rh).toInt())
        canvas.drawBitmap(b, src, dst, paint)
    }
}
