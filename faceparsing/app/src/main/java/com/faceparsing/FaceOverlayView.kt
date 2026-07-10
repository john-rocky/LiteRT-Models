package com.faceparsing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.view.View

/** Draws the Cityscapes-colored label map semi-transparently over the camera preview. */
class FaceOverlayView(context: Context) : View(context) {

    private var label: Bitmap? = null
    private val src = Rect()
    private val dst = Rect()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG).apply { alpha = 130 }

    fun setLabels(bmp: Bitmap) {
        label = bmp
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = label ?: return
        src.set(0, 0, b.width, b.height)
        dst.set(0, 0, width, height)
        canvas.drawBitmap(b, src, dst, paint)
    }
}
