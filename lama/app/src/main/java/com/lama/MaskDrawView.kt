package com.lama

import android.content.Context
import android.graphics.*
import android.view.MotionEvent
import android.view.View

/**
 * View for drawing inpainting mask over an image.
 * White strokes on transparent canvas = mask for inpainting.
 */
class MaskDrawView(context: Context) : View(context) {

    var onMaskChanged: (() -> Unit)? = null

    private var imageBitmap: Bitmap? = null
    private var maskBitmap: Bitmap? = null
    private var maskCanvas: Canvas? = null
    private var resultBitmap: Bitmap? = null
    private var showResult = false

    private val imagePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val maskOverlayPaint = Paint(Paint.FILTER_BITMAP_FLAG).apply { alpha = 120 }
    private val drawPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 40f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isAntiAlias = true
    }

    private val drawMatrix = Matrix()
    private val inverseMatrix = Matrix()
    private val path = Path()

    var brushSize: Float
        get() = drawPaint.strokeWidth
        set(value) { drawPaint.strokeWidth = value }

    fun setImage(bitmap: Bitmap) {
        imageBitmap = bitmap
        maskBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        maskCanvas = Canvas(maskBitmap!!)
        resultBitmap = null
        showResult = false
        path.reset()
        updateMatrix()
        invalidate()
    }

    fun getMask(): Bitmap? = maskBitmap

    fun setResult(bitmap: Bitmap) {
        resultBitmap = bitmap
        showResult = true
        invalidate()
    }

    fun toggleResult() {
        if (resultBitmap != null) {
            showResult = !showResult
            invalidate()
        }
    }

    fun clearMask() {
        maskBitmap?.eraseColor(Color.TRANSPARENT)
        path.reset()
        resultBitmap = null
        showResult = false
        invalidate()
    }

    private fun updateMatrix() {
        val bmp = imageBitmap ?: return
        val vw = width.toFloat(); val vh = height.toFloat()
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

        if (showResult && resultBitmap != null) {
            canvas.drawBitmap(resultBitmap!!, drawMatrix, imagePaint)
        } else {
            canvas.drawBitmap(bmp, drawMatrix, imagePaint)
            maskBitmap?.let { canvas.drawBitmap(it, drawMatrix, maskOverlayPaint) }
        }
    }

    private var lastX = 0f
    private var lastY = 0f

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (imageBitmap == null || showResult) return true

        val pts = floatArrayOf(event.x, event.y)
        inverseMatrix.mapPoints(pts)
        val x = pts[0]; val y = pts[1]

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                path.moveTo(x, y)
                lastX = x; lastY = y
            }
            MotionEvent.ACTION_MOVE -> {
                path.quadTo(lastX, lastY, (x + lastX) / 2, (y + lastY) / 2)
                maskCanvas?.drawPath(path, drawPaint)
                lastX = x; lastY = y
            }
            MotionEvent.ACTION_UP -> {
                maskCanvas?.drawPath(path, drawPaint)
                path.reset()
                onMaskChanged?.invoke()
            }
        }
        invalidate()
        return true
    }
}
