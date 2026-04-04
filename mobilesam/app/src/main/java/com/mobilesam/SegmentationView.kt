package com.mobilesam

import android.content.Context
import android.graphics.*
import android.view.MotionEvent
import android.view.View

/**
 * Custom view that displays an image with mask overlay and handles tap input.
 * Reports tap coordinates in original image space.
 */
class SegmentationView(context: Context) : View(context) {

    var onTap: ((Float, Float) -> Unit)? = null

    private var imageBitmap: Bitmap? = null
    private var maskBitmap: Bitmap? = null
    private var points = mutableListOf<PointF>()

    private val imagePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val maskPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00E676.toInt()  // green
        style = Paint.Style.FILL
    }
    private val pointOutlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val drawMatrix = Matrix()
    private val inverseMatrix = Matrix()

    fun setImage(bitmap: Bitmap) {
        imageBitmap = bitmap
        maskBitmap = null
        points.clear()
        updateMatrix()
        invalidate()
    }

    fun setMask(bitmap: Bitmap) {
        maskBitmap = bitmap
        invalidate()
    }

    fun clearMask() {
        maskBitmap = null
        points.clear()
        invalidate()
    }

    private fun updateMatrix() {
        val bmp = imageBitmap ?: return
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw == 0f || vh == 0f) return

        val scale = minOf(vw / bmp.width, vh / bmp.height)
        val dx = (vw - bmp.width * scale) / 2f
        val dy = (vh - bmp.height * scale) / 2f

        drawMatrix.setScale(scale, scale)
        drawMatrix.postTranslate(dx, dy)
        drawMatrix.invert(inverseMatrix)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        updateMatrix()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bmp = imageBitmap ?: return

        canvas.drawBitmap(bmp, drawMatrix, imagePaint)
        maskBitmap?.let { canvas.drawBitmap(it, drawMatrix, maskPaint) }

        // Draw tap points
        for (pt in points) {
            val screenPts = floatArrayOf(pt.x, pt.y)
            drawMatrix.mapPoints(screenPts)
            canvas.drawCircle(screenPts[0], screenPts[1], 12f, pointPaint)
            canvas.drawCircle(screenPts[0], screenPts[1], 12f, pointOutlinePaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_UP && imageBitmap != null) {
            val imgPts = floatArrayOf(event.x, event.y)
            inverseMatrix.mapPoints(imgPts)

            val bmp = imageBitmap!!
            val ix = imgPts[0].coerceIn(0f, bmp.width.toFloat() - 1)
            val iy = imgPts[1].coerceIn(0f, bmp.height.toFloat() - 1)

            points.clear()
            points.add(PointF(ix, iy))
            invalidate()
            onTap?.invoke(ix, iy)
            return true
        }
        return true
    }
}
