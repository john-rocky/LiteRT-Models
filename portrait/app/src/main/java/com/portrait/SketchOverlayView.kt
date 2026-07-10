package com.portrait

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.view.View

/** Draws the generated pencil sketch, fit into the view. */
class SketchOverlayView(context: Context) : View(context) {
    private var bmp: Bitmap? = null
    private val src = Rect(); private val dst = Rect()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val bg = Paint().apply { color = 0xFFFFFFFF.toInt() }
    fun setSketch(b: Bitmap) { bmp = b; postInvalidateOnAnimation() }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = bmp ?: return
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), bg)
        val s = minOf(width, height); val l = (width - s) / 2; val tp = (height - s) / 2
        src.set(0, 0, b.width, b.height); dst.set(l, tp, l + s, tp + s)
        canvas.drawBitmap(b, src, dst, paint)
    }
}
