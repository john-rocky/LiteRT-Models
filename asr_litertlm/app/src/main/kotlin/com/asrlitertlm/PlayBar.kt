package com.asrlitertlm

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.View

/** A thin rounded bar: the share of a clip played, or of the 30 s recording limit used. */
class PlayBar(context: Context, trackColor: Int) : View(context) {
  private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = trackColor }
  private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG)

  var fraction = 0f
    set(value) {
      if (value != field) {
        field = value
        invalidate()
      }
    }

  fun set(fraction: Float, fillColor: Int) {
    fillPaint.color = fillColor
    this.fraction = fraction
    invalidate()
  }

  override fun onDraw(canvas: Canvas) {
    val height = height.toFloat()
    val radius = height / 2
    canvas.drawRoundRect(0f, 0f, width.toFloat(), height, radius, radius, trackPaint)
    if (fraction > 0f) {
      canvas.drawRoundRect(0f, 0f, maxOf(height, width * fraction), height, radius, radius, fillPaint)
    }
  }
}
