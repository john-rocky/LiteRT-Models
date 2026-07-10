package com.yolact

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.view.View

/** Draws the composited instance-mask bitmap (scaled to the view) + boxes + labels. */
class InstanceOverlayView(context: Context) : View(context) {

    private var masks: Bitmap? = null
    private var insts: List<Instance> = emptyList()
    private var srcW = 1; private var srcH = 1
    private val src = Rect()
    private val dst = Rect()
    private val maskPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 34f; setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    /** [sw]x[sh] = the camera frame dimensions the masks/boxes were computed in. */
    fun setResult(maskBitmap: Bitmap, instances: List<Instance>, sw: Int, sh: Int) {
        masks = maskBitmap; insts = instances; srcW = sw; srcH = sh
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = masks ?: return
        // FILL_CENTER: map the frame into the same rect PreviewView shows (scale=max, centered).
        val scale = maxOf(width.toFloat() / srcW, height.toFloat() / srcH)
        val rw = srcW * scale; val rh = srcH * scale
        val l = (width - rw) / 2f; val t = (height - rh) / 2f
        src.set(0, 0, b.width, b.height); dst.set(l.toInt(), t.toInt(), (l + rw).toInt(), (t + rh).toInt())
        canvas.drawBitmap(b, src, dst, maskPaint)
        for (ins in insts) {
            val col = Palette.color(ins.cls)
            boxPaint.color = col
            canvas.drawRect(l + ins.x1 * rw, t + ins.y1 * rh, l + ins.x2 * rw, t + ins.y2 * rh, boxPaint)
            val label = "${CocoLabels.NAMES[ins.cls]} ${(ins.score * 100).toInt()}%"
            canvas.drawText(label, l + ins.x1 * rw + 6, t + ins.y1 * rh + 34, textPaint)
        }
    }
}

/** Stable per-class colors. */
object Palette {
    private val C = IntArray(80) { i ->
        val h = (i * 47 % 360).toFloat()
        Color.HSVToColor(floatArrayOf(h, 0.75f, 1.0f))
    }
    fun color(cls: Int) = C[cls % 80]
}
