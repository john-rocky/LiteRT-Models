// Copyright 2026 Daisuke Majima. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// The square image area of the demo screen: the generated image, or a dark
// placeholder with one progress segment per sampling step (and room for a
// message: missing model files, errors). The square is as wide as the parent
// unless less height is left for it, so the whole screen fits without
// scrolling on any display.

package com.bonsai.imagegen

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.util.TypedValue
import android.view.View

class ImageArea(context: Context) : View(context) {

    /** The image to show; null shows the placeholder. */
    var bitmap: Bitmap? = null
        set(v) { field = v; invalidate() }

    /** Number of progress segments (one per step); 0 hides the bar. */
    var segments = 0
        set(v) { field = v; invalidate() }

    /** Segments whose step has finished. */
    var filled = 0
        set(v) { field = v; invalidate() }

    /** The segment whose step is running (0-based), or -1. */
    var active = -1
        set(v) { field = v; invalidate() }

    /** Text in the placeholder, or null. */
    var message: String? = null
        set(v) { field = v; messageLayout = null; invalidate() }

    /** Side of the square in px, from the last measure. */
    var side = 0
        private set

    private val density = resources.displayMetrics.density
    private val radius = 14 * density
    private val box = RectF()
    private val seg = RectF()
    private val clip = Path()
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = C_MESSAGE
        textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 14f, resources.displayMetrics)
    }
    private var messageLayout: StaticLayout? = null

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec)
        side = if (MeasureSpec.getMode(heightMeasureSpec) == MeasureSpec.UNSPECIFIED) w
        else minOf(w, MeasureSpec.getSize(heightMeasureSpec))
        setMeasuredDimension(w, side)
    }

    override fun onDraw(canvas: Canvas) {
        val left = (width - side) / 2f
        box.set(left, 0f, left + side, side.toFloat())
        clip.reset()
        clip.addRoundRect(box, radius, radius, Path.Direction.CW)
        canvas.save()
        canvas.clipPath(clip)
        val bmp = bitmap
        if (bmp != null) {
            canvas.drawBitmap(bmp, null, box, bitmapPaint)
        } else {
            fill.color = C_PLACEHOLDER
            canvas.drawRect(box, fill)
            if (segments > 0) drawSegments(canvas)
            message?.let { drawMessage(canvas, it) }
        }
        canvas.restore()
    }

    private fun drawSegments(canvas: Canvas) {
        val barWidth = side * 0.56f
        val gap = 6 * density
        val h = 6 * density
        val w = (barWidth - gap * (segments - 1)) / segments
        val x0 = box.centerX() - barWidth / 2
        val y0 = box.centerY() - h / 2
        for (i in 0 until segments) {
            fill.color = when {
                i < filled -> C_SEG_DONE
                i == active -> C_SEG_ACTIVE
                else -> C_SEG_TODO
            }
            val x = x0 + i * (w + gap)
            seg.set(x, y0, x + w, y0 + h)
            canvas.drawRoundRect(seg, h / 2, h / 2, fill)
        }
    }

    private fun drawMessage(canvas: Canvas, text: String) {
        val width = (side - 48 * density).toInt().coerceAtLeast(1)
        val layout = messageLayout?.takeIf { it.width == width }
            ?: StaticLayout.Builder.obtain(text, 0, text.length, textPaint, width)
                .setAlignment(Layout.Alignment.ALIGN_CENTER)
                .build().also { messageLayout = it }
        canvas.save()
        val y = box.centerY() - layout.height / 2f + if (segments > 0) 28 * density else 0f
        canvas.translate(box.centerX() - width / 2f, y)
        layout.draw(canvas)
        canvas.restore()
    }

    private companion object {
        const val C_PLACEHOLDER = 0xFF161B22.toInt()
        const val C_SEG_TODO = 0xFF262C35.toInt()
        const val C_SEG_ACTIVE = 0xFF27466F.toInt()
        const val C_SEG_DONE = 0xFF5B9BF8.toInt()
        const val C_MESSAGE = 0xFF8A919C.toInt()
    }
}
