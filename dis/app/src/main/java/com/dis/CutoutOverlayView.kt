package com.dis

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.view.View

/** Draws the cut-out composite (foreground on a replacement color) scaled to the view. */
class CutoutOverlayView(context: Context) : View(context) {
    private var bmp: Bitmap? = null
    private val src = Rect(); private val dst = Rect()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    fun setComposite(b: Bitmap) { bmp = b; postInvalidateOnAnimation() }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = bmp ?: return
        src.set(0, 0, b.width, b.height); dst.set(0, 0, width, height)
        canvas.drawBitmap(b, src, dst, paint)
    }
}
