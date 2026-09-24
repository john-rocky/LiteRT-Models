package com.nemotron3diar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/**
 * Per-speaker timeline that grows as streaming steps arrive: one row per speaker seen so far (speakers are numbered
 * by first arrival, up to 8), bars where its probability is above 0.5, one frame per 10 ms. Adapted from the
 * diarization sample's TimelineView (8 speakers, append-only).
 */
class TimelineView(ctx: Context) : View(ctx) {

  companion object {
    val COLORS =
      intArrayOf(
        Color.rgb(0x42, 0x85, 0xF4),
        Color.rgb(0xEA, 0x43, 0x35),
        Color.rgb(0xFB, 0xBC, 0x05),
        Color.rgb(0x34, 0xA8, 0x53),
        Color.rgb(0xAB, 0x47, 0xBC),
        Color.rgb(0x00, 0xAC, 0xC1),
        Color.rgb(0xFF, 0x70, 0x43),
        Color.rgb(0x9E, 0x9D, 0x24),
      )
    const val FRAMES_PER_SECOND = 100
    private const val MIN_SECONDS = 10
    private const val ROW_H = 64f
  }

  private var masks = ByteArray(FRAMES_PER_SECOND * 60)
  private var frames = 0
  private var seen = 0
  private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
  private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 30f }
  private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.GRAY
    textSize = 26f
  }

  /** Number of frames drawn so far. */
  val numFrames: Int
    get() = frames

  fun reset() {
    frames = 0
    seen = 0
    requestLayout()
    invalidate()
  }

  /** Appends [count] frames of logits [count x 8]. */
  fun append(logits: FloatArray, count: Int) {
    if (frames + count > masks.size) masks = masks.copyOf(maxOf(masks.size * 2, frames + count))
    for (f in 0 until count) {
      var m = 0
      for (s in 0 until SpeakerCache.NUM_SPEAKERS) {
        if (SpeakerCache.sigmoid(logits[f * SpeakerCache.NUM_SPEAKERS + s]) > 0.5f) m = m or (1 shl s)
      }
      masks[frames + f] = m.toByte()
      seen = seen or m
    }
    frames += count
    requestLayout()
    invalidate()
  }

  /** Speakers that have been active at least once. */
  fun speakers(): List<Int> = (0 until SpeakerCache.NUM_SPEAKERS).filter { seen and (1 shl it) != 0 }

  /** Seconds of activity per speaker. */
  fun speakerSeconds(): DoubleArray {
    val out = DoubleArray(SpeakerCache.NUM_SPEAKERS)
    for (f in 0 until frames) {
      val m = masks[f].toInt()
      for (s in 0 until SpeakerCache.NUM_SPEAKERS) if (m and (1 shl s) != 0) out[s] += 1.0 / FRAMES_PER_SECOND
    }
    return out
  }

  override fun onMeasure(w: Int, h: Int) {
    val rows = maxOf(1, speakers().size)
    setMeasuredDimension(MeasureSpec.getSize(w), rows * ROW_H.toInt() + 80)
  }

  override fun onDraw(c: Canvas) {
    val labelW = 130f
    val w = width - labelW - 8f
    val span = maxOf(frames, MIN_SECONDS * FRAMES_PER_SECOND).toFloat()
    val rows = speakers()
    for ((row, spk) in rows.withIndex()) {
      val y = row * ROW_H + 20
      textPaint.color = COLORS[spk]
      c.drawText("SPK ${spk + 1}", 8f, y + ROW_H / 2 + 10, textPaint)
      paint.color = Color.rgb(0xEE, 0xEE, 0xEE)
      c.drawRect(labelW, y, labelW + w, y + ROW_H - 12, paint)
      paint.color = COLORS[spk]
      var start = -1
      for (f in 0..frames) {
        val on = f < frames && masks[f].toInt() and (1 shl spk) != 0
        if (on && start < 0) start = f
        if (!on && start >= 0) {
          c.drawRect(labelW + start / span * w, y, labelW + f / span * w, y + ROW_H - 12, paint)
          start = -1
        }
      }
    }
    val base = rows.size.coerceAtLeast(1) * ROW_H + 50
    c.drawText("0 s", labelW, base, axisPaint)
    val end = "%.1f s".format(span / FRAMES_PER_SECOND)
    c.drawText(end, labelW + w - axisPaint.measureText(end), base, axisPaint)
  }

}
