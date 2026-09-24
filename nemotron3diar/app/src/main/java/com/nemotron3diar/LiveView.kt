package com.nemotron3diar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.view.View
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Full-screen "Play live" view: a clip plays while the streaming loop diarizes it at the audio's own pace.
 *
 * Top to bottom: state badge (READY / LIVE / DONE) with the playback clock (100 ms steps) and the number of speakers
 * seen so far; the latency line; the waveform up to the playhead, colored by the speaker the model has assigned so
 * far (grey until the step that covers it has run); eight speaker lanes (speakers are numbered by first arrival);
 * a small per-step line. The time axis spans the whole clip, so the lanes end as the full who-spoke-when of the
 * clip. The top and bottom bands are left empty for captions.
 */
class LiveView(ctx: Context) : View(ctx) {

  enum class State { READY, LIVE, DONE }

  private var clipFrames = 1
  private var framePeak = FloatArray(0)
  private var peakScale = 1f
  private var masks = ByteArray(0)
  private var dominant = ByteArray(0)
  private var frames = 0
  private var seen = 0
  private var playhead = 0L
  private var sampleRate = MelFrontend.SAMPLE_RATE

  /** Per speaker: runs [start, end) of active frames, extended as steps arrive. */
  private val runs = Array(NUM_SPEAKERS) { ArrayList<IntArray>() }

  var state = State.READY
    set(value) {
      field = value
      invalidate()
    }

  var latencyText = ""
    set(value) {
      field = value
      invalidate()
    }

  var stepText = ""
    set(value) {
      field = value
      invalidate()
    }

  private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
  private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { fontFeatureSettings = "tnum" }
  private val rect = RectF()

  init {
    setBackgroundColor(BACKGROUND)
  }

  /** Sets the clip: per-10 ms peaks for the waveform; resets the lanes. */
  fun setClip(samples: FloatArray, rate: Int) {
    sampleRate = rate
    val hop = rate / TimelineView.FRAMES_PER_SECOND
    clipFrames = max(1, (samples.size + hop - 1) / hop)
    framePeak = FloatArray(clipFrames)
    for (f in 0 until clipFrames) {
      var p = 0f
      for (i in f * hop until min(samples.size, (f + 1) * hop)) p = max(p, kotlin.math.abs(samples[i]))
      framePeak[f] = p
    }
    peakScale = max(1e-4f, framePeak.maxOrNull() ?: 1f)
    masks = ByteArray(clipFrames + FRAME_SLACK)
    dominant = ByteArray(clipFrames + FRAME_SLACK) { -1 }
    frames = 0
    seen = 0
    playhead = 0
    for (r in runs) r.clear()
    state = State.READY
    invalidate()
  }

  /** Playback position in samples; redraws only when the clock (0.1 s) or the playhead's pixel column changes. */
  fun setPlayhead(samples: Long) {
    if (samples == playhead) return
    val before = visibleKey(playhead)
    playhead = samples
    if (visibleKey(samples) != before) invalidate()
  }

  private fun visibleKey(samples: Long): Long {
    val tenths = samples * 10 / sampleRate
    val w = width * (1f - (172f + 44f) / 1080f)
    val col = (samples.toFloat() * TimelineView.FRAMES_PER_SECOND / sampleRate / clipFrames * w).toLong()
    return tenths * 100_000 + col
  }

  /** Appends [count] frames of logits [count x 8] (10 ms frames, contiguous with the previous ones). */
  fun append(logits: FloatArray, count: Int) {
    if (frames + count > masks.size) {
      masks = masks.copyOf(frames + count)
      dominant = dominant.copyOf(frames + count)
    }
    for (f in 0 until count) {
      var m = 0
      var best = -1
      var bestP = 0.5f
      for (s in 0 until NUM_SPEAKERS) {
        val p = SpeakerCache.sigmoid(logits[f * NUM_SPEAKERS + s])
        if (p > 0.5f) m = m or (1 shl s)
        if (p > bestP) {
          bestP = p
          best = s
        }
      }
      val g = frames + f
      masks[g] = m.toByte()
      dominant[g] = best.toByte()
      for (s in 0 until NUM_SPEAKERS) {
        if (m and (1 shl s) == 0) continue
        val r = runs[s]
        val last = r.lastOrNull()
        if (last != null && last[1] == g) last[1] = g + 1 else r += intArrayOf(g, g + 1)
      }
      seen = seen or m
    }
    frames += count
    invalidate()
  }

  /** Speakers active at least once so far. */
  val numSpeakers: Int
    get() = Integer.bitCount(seen)

  override fun onDraw(c: Canvas) {
    val u = width / 1080f
    val x0 = 172f * u
    val x1 = width - 44f * u
    val w = x1 - x0
    val span = clipFrames.toFloat()
    fun xOf(frame: Float) = x0 + frame / span * w

    // ---- header: state badge, clock, speakers
    val headerY = 300f * u
    val badge = when (state) {
      State.READY -> "READY"
      State.LIVE -> "● LIVE"
      State.DONE -> "DONE"
    }
    text.typeface = Typeface.DEFAULT_BOLD
    text.textSize = 36f * u
    val bw = text.measureText(badge) + 44f * u
    rect.set(44f * u, headerY - 44f * u, 44f * u + bw, headerY + 20f * u)
    paint.color = when (state) {
      State.READY -> Color.rgb(0x5F, 0x63, 0x68)
      State.LIVE -> Color.rgb(0xE5, 0x39, 0x35)
      State.DONE -> Color.rgb(0x2E, 0x7D, 0x32)
    }
    c.drawRoundRect(rect, 32f * u, 32f * u, paint)
    text.color = Color.WHITE
    c.drawText(badge, rect.left + 22f * u, headerY + 1f * u, text)

    val tenths = playhead * 10 / sampleRate
    val clock = if (state == State.READY) "–.– s" else "%d.%d s".format(tenths / 10, tenths % 10)
    text.typeface = Typeface.DEFAULT
    text.textSize = 72f * u
    c.drawText(clock, rect.right + 32f * u, headerY + 14f * u, text)

    val n = numSpeakers
    val spk = if (n == 1) "1 speaker" else "$n speakers"
    text.textSize = 44f * u
    text.typeface = Typeface.DEFAULT_BOLD
    c.drawText(spk, width - 44f * u - text.measureText(spk), headerY + 4f * u, text)

    text.typeface = Typeface.DEFAULT
    text.textSize = 36f * u
    text.color = Color.rgb(0xB8, 0xBE, 0xC6)
    c.drawText(latencyText, 44f * u, headerY + 96f * u, text)

    // ---- waveform, colored by the assigned speaker
    val waveTop = 470f * u
    val waveH = 300f * u
    val mid = waveTop + waveH / 2
    val headFrame = playhead.toFloat() * TimelineView.FRAMES_PER_SECOND / sampleRate
    val cols = w.toInt()
    val colW = w / cols
    for (col in 0 until cols) {
      val f0 = (col * span / cols).toInt()
      val f1 = max(f0 + 1, ((col + 1) * span / cols).toInt())
      if (f0 >= headFrame) break
      var p = 0f
      for (f in f0 until min(f1, clipFrames)) p = max(p, framePeak[f])
      val h = max(2f * u, sqrt(p / peakScale) * waveH / 2 * 0.96f)
      paint.color =
        if (f0 < frames) {
          var s = -1
          for (f in f0 until min(f1, frames)) if (dominant[f] >= 0) { s = dominant[f].toInt(); break }
          if (s >= 0) TimelineView.COLORS[s] else PENDING
        } else PENDING
      val x = x0 + col * colW
      c.drawRect(x, mid - h, x + max(1f, colW * 0.8f), mid + h, paint)
    }

    // ---- speaker lanes
    val laneTop = 850f * u
    val rowH = 130f * u
    val laneH = 108f * u
    for (s in 0 until NUM_SPEAKERS) {
      val y = laneTop + s * rowH
      val active = seen and (1 shl s) != 0
      paint.color = LANE
      rect.set(x0, y, x1, y + laneH)
      c.drawRoundRect(rect, 10f * u, 10f * u, paint)
      text.textSize = 34f * u
      text.typeface = Typeface.DEFAULT_BOLD
      text.color = if (active) TimelineView.COLORS[s] else DIM_LABEL
      c.drawText("SPK ${s + 1}", 44f * u, y + laneH / 2 + 12f * u, text)
      paint.color = TimelineView.COLORS[s]
      for (r in runs[s]) c.drawRect(xOf(r[0].toFloat()), y, xOf(r[1].toFloat()), y + laneH, paint)
    }
    val lanesBottom = laneTop + (NUM_SPEAKERS - 1) * rowH + laneH

    // ---- playhead across waveform and lanes
    if (state != State.READY) {
      val px = xOf(min(headFrame, span))
      paint.color = Color.WHITE
      c.drawRect(px - 1.5f * u, waveTop - 12f * u, px + 1.5f * u, lanesBottom + 12f * u, paint)
      c.drawCircle(px, waveTop - 12f * u, 9f * u, paint)
    }

    // ---- time axis: a tick every 10 s, the clip's end on the right
    val axisY = lanesBottom + 52f * u
    val seconds = span / TimelineView.FRAMES_PER_SECOND
    text.typeface = Typeface.DEFAULT
    text.textSize = 26f * u
    text.color = AXIS
    paint.color = AXIS
    var t = 0
    while (t < seconds - 4) {
      val x = xOf(t * TimelineView.FRAMES_PER_SECOND.toFloat())
      c.drawRect(x - 1f * u, lanesBottom + 14f * u, x + 1f * u, lanesBottom + 26f * u, paint)
      if (t == 0) c.drawText("0 s", x, axisY, text)
      t += 10
    }
    val end = "%.0f s".format(seconds)
    c.drawText(end, x1 - text.measureText(end), axisY, text)

    // ---- per-step line
    text.textSize = 28f * u
    c.drawText(stepText, 44f * u, axisY + 76f * u, text)
  }

  companion object {
    private const val NUM_SPEAKERS = SpeakerCache.NUM_SPEAKERS
    private const val FRAME_SLACK = 200 // the last chunk may score a few frames past the clip's own count
    private val BACKGROUND = Color.rgb(0x0E, 0x11, 0x16)
    private val LANE = Color.rgb(0x1B, 0x20, 0x28)
    private val PENDING = Color.rgb(0x6B, 0x72, 0x80)
    private val DIM_LABEL = Color.rgb(0x3C, 0x42, 0x4C)
    private val AXIS = Color.rgb(0x8A, 0x91, 0x9C)
  }
}
