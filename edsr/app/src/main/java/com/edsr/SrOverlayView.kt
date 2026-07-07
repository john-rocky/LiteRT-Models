package com.edsr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.view.View

/** Draws the super-resolved (512×512) bitmap, fit into the view. */
class SrOverlayView(context: Context) : View(context) {
    private var bmp: Bitmap? = null
    private val src = Rect(); private val dst = Rect()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val bg = Paint().apply { color = 0xFF101010.toInt() }
    fun setResult(b: Bitmap) { bmp = b; postInvalidateOnAnimation() }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = bmp ?: return
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), bg)
        val s = minOf(width, height); val left = (width - s) / 2; val top = (height - s) / 2
        src.set(0, 0, b.width, b.height); dst.set(left, top, left + s, top + s)
        canvas.drawBitmap(b, src, dst, paint)
    }
}
