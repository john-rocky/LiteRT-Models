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
    private val src = Rect()
    private val dst = Rect()
    private val maskPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 34f; setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    fun setResult(maskBitmap: Bitmap, instances: List<Instance>) {
        masks = maskBitmap; insts = instances
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val b = masks ?: return
        src.set(0, 0, b.width, b.height); dst.set(0, 0, width, height)
        canvas.drawBitmap(b, src, dst, maskPaint)
        val sx = width.toFloat(); val sy = height.toFloat()
        for (ins in insts) {
            val col = Palette.color(ins.cls)
            boxPaint.color = col
            canvas.drawRect(ins.x1 * sx, ins.y1 * sy, ins.x2 * sx, ins.y2 * sy, boxPaint)
            val label = "${CocoLabels.NAMES[ins.cls]} ${(ins.score * 100).toInt()}%"
            canvas.drawText(label, ins.x1 * sx + 6, ins.y1 * sy + 34, textPaint)
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
