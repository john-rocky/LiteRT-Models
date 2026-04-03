package com.realesrgan

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.view.MotionEvent
import android.view.View

/**
 * Before/after comparison slider view.
 * Left side shows the "before" image, right side shows the "after" image.
 * Drag the divider to compare.
 */
class CompareView(context: Context) : View(context) {

    private var beforeBitmap: Bitmap? = null
    private var afterBitmap: Bitmap? = null
    private var dividerRatio = 0.5f  // 0..1

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val handlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        style = Paint.Style.FILL
        setShadowLayer(8f, 0f, 0f, 0xFF000000.toInt())
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 32f
        setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
    }

    private val srcRect = Rect()
    private val dstRect = Rect()
    private val clipRect = RectF()

    fun setImages(before: Bitmap, after: Bitmap) {
        beforeBitmap = before
        afterBitmap = after
        dividerRatio = 0.5f
        invalidate()
    }

    private fun getImageRect(): RectF {
        val bmp = afterBitmap ?: beforeBitmap ?: return RectF()
        val scale = minOf(width.toFloat() / bmp.width, height.toFloat() / bmp.height)
        val imgW = bmp.width * scale
        val imgH = bmp.height * scale
        val left = (width - imgW) / 2f
        val top = (height - imgH) / 2f
        return RectF(left, top, left + imgW, top + imgH)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val after = afterBitmap ?: return
        val before = beforeBitmap ?: return

        val imgRect = getImageRect()
        val dividerX = imgRect.left + imgRect.width() * dividerRatio

        // Draw "after" (right side = full image, then clip left for "before")
        srcRect.set(0, 0, after.width, after.height)
        dstRect.set(imgRect.left.toInt(), imgRect.top.toInt(), imgRect.right.toInt(), imgRect.bottom.toInt())
        canvas.drawBitmap(after, srcRect, dstRect, paint)

        // Draw "before" clipped to left of divider
        canvas.save()
        clipRect.set(imgRect.left, imgRect.top, dividerX, imgRect.bottom)
        canvas.clipRect(clipRect)
        srcRect.set(0, 0, before.width, before.height)
        canvas.drawBitmap(before, srcRect, dstRect, paint)
        canvas.restore()

        // Divider line
        canvas.drawLine(dividerX, imgRect.top, dividerX, imgRect.bottom, dividerPaint)

        // Handle circle
        val cy = imgRect.centerY()
        canvas.drawCircle(dividerX, cy, 20f, handlePaint)

        // Arrow indicators
        val arrowPaint = Paint(handlePaint).apply { textSize = 28f; textAlign = Paint.Align.CENTER }
        canvas.drawText("\u25C0", dividerX - 6f, cy + 10f, arrowPaint)
        canvas.drawText("\u25B6", dividerX + 6f, cy + 10f, arrowPaint)

        // Labels
        labelPaint.textAlign = Paint.Align.LEFT
        canvas.drawText("Original", imgRect.left + 12f, imgRect.top + 40f, labelPaint)
        labelPaint.textAlign = Paint.Align.RIGHT
        canvas.drawText("ESRGAN 4x", imgRect.right - 12f, imgRect.top + 40f, labelPaint)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        val imgRect = getImageRect()
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                dividerRatio = ((event.x - imgRect.left) / imgRect.width()).coerceIn(0f, 1f)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
