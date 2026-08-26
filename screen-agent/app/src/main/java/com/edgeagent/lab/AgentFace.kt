package com.edgeagent.lab

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * The agent, as a robot's face.
 *
 * The app is a text field and some radio buttons; nothing about it says that
 * something is working on your behalf. A face fixes that: the eyes look around
 * while the screen is captured, narrow while the model thinks, and turn toward
 * the point it is about to press.
 *
 * It is drawn as a silhouette first — antenna, ears, square head — because at
 * 72dp the outline is all that survives. An earlier version was a single lens,
 * which read as a camera rather than as somebody.
 *
 * Drawn with primitives rather than a bitmap so it stays a few kilobytes, scales
 * to any density, and can be redrawn identically in the demo's build script.
 */
class AgentFace @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    enum class State { IDLE, LOOKING, THINKING, ACTING, DONE, STOPPED }

    private val body = Paint(Paint.ANTI_ALIAS_FLAG)
    private val eyeWhite = Paint(Paint.ANTI_ALIAS_FLAG)
    private val pupil = Paint(Paint.ANTI_ALIAS_FLAG)
    private val accent = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
    }

    private var phase = 0f
    private var blink = 1f          // 1 = open, 0 = shut
    private var nextBlinkAt = 2.2f

    /** Where the pupil is aiming, in view fractions from the centre. */
    private var aimX = 0f
    private var aimY = 0f

    var state: State = State.IDLE
        set(value) {
            if (field != value) {
                field = value
                invalidate()
            }
        }

    /** Point the eye at a screen coordinate, so the tap has a direction. */
    fun lookAt(xFraction: Float, yFraction: Float) {
        aimX = (xFraction - 0.5f).coerceIn(-0.5f, 0.5f)
        aimY = (yFraction - 0.5f).coerceIn(-0.5f, 0.5f)
        invalidate()
    }

    private val ticker = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 1200
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            phase += 1f / 60f
            if (phase > nextBlinkAt) {
                blink = 0f
                nextBlinkAt = phase + 2.4f + (phase % 1.3f)
            }
            blink = min(1f, blink + 0.12f)
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        ticker.start()
    }

    override fun onDetachedFromWindow() {
        ticker.cancel()
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        val s = min(width.toFloat(), height.toFloat())
        val cx = width / 2f
        val cy = height / 2f

        val tint = when (state) {
            State.IDLE -> 0xFF98A2B0.toInt()
            State.LOOKING, State.THINKING -> 0xFF6CA6FF.toInt()
            State.ACTING, State.DONE -> 0xFF3DDC84.toInt()
            State.STOPPED -> 0xFFFFB066.toInt()
        }
        val slate = 0xFF2A3444.toInt()
        val edge = 0xFF566680.toInt()
        val plate = 0xFF0A0E14.toInt()

        val headW = s * 0.74f
        val headH = s * 0.62f
        val hx0 = cx - headW / 2f
        val hy0 = cy - headH / 2f + s * 0.06f
        val hx1 = hx0 + headW
        val hy1 = hy0 + headH
        val radius = s * 0.20f

        // antenna: a stalk and a ball that pulses only while a call is in flight
        val pulse = if (state == State.THINKING) (sin(phase * 4.2f) * 0.5f + 0.5f) else 0f
        val ballR = s * 0.055f + s * 0.018f * pulse
        val top = hy0 - s * 0.10f
        stroke.color = edge
        stroke.strokeWidth = max(2f, s * 0.028f)
        canvas.drawLine(cx, top, cx, hy0 + 2f, stroke)
        accent.color = tint
        accent.alpha = 255
        canvas.drawCircle(cx, top, ballR, accent)

        // ears
        val earW = s * 0.07f
        val earH = s * 0.22f
        body.color = edge
        for (ex in floatArrayOf(hx0 - earW * 0.75f, hx1 - earW * 0.25f)) {
            canvas.drawRoundRect(
                RectF(ex, cy - earH / 2f + s * 0.05f, ex + earW, cy + earH / 2f + s * 0.05f),
                earW / 2f, earW / 2f, body,
            )
        }

        // head, then the darker face plate inside it
        body.color = slate
        val head = RectF(hx0, hy0, hx1, hy1)
        canvas.drawRoundRect(head, radius, radius, body)
        stroke.color = edge
        stroke.strokeWidth = max(2f, s * 0.02f)
        canvas.drawRoundRect(head, radius, radius, stroke)
        val inset = s * 0.06f
        body.color = plate
        canvas.drawRoundRect(
            RectF(hx0 + inset, hy0 + inset, hx1 - inset, hy1 - inset),
            radius * 0.72f, radius * 0.72f, body,
        )

        // eyes
        val eyeDx = headW * 0.20f
        val eyeY = hy0 + headH * 0.42f
        val eyeR = s * 0.082f
        val sweep = if (state == State.LOOKING) sin(phase * 2.6f) * 0.5f else aimX * 2f
        val rise = if (state == State.LOOKING) cos(phase * 1.7f) * 0.3f else aimY * 2f
        val ax = sweep.coerceIn(-1f, 1f) * eyeR * 0.42f
        val ay = rise.coerceIn(-1f, 1f) * eyeR * 0.42f

        for (ex in floatArrayOf(cx - eyeDx, cx + eyeDx)) {
            when {
                state == State.DONE -> {
                    stroke.color = tint
                    stroke.strokeWidth = max(3f, s * 0.035f)
                    canvas.drawArc(
                        RectF(ex - eyeR, eyeY - eyeR, ex + eyeR, eyeY + eyeR * 0.8f),
                        200f, 140f, false, stroke,
                    )
                }
                state == State.STOPPED || blink < 0.25f -> {
                    stroke.color = tint
                    stroke.strokeWidth = max(3f, s * 0.035f)
                    canvas.drawLine(ex - eyeR * 0.9f, eyeY, ex + eyeR * 0.9f, eyeY, stroke)
                }
                state == State.THINKING -> {
                    pupil.color = tint
                    canvas.drawRoundRect(
                        RectF(ex - eyeR, eyeY - eyeR * 0.42f, ex + eyeR, eyeY + eyeR * 0.42f),
                        eyeR * 0.42f, eyeR * 0.42f, pupil,
                    )
                }
                else -> {
                    pupil.color = tint
                    canvas.drawCircle(ex, eyeY, eyeR, pupil)
                    pupil.color = plate
                    canvas.drawCircle(ex + ax, eyeY + ay, eyeR * 0.42f, pupil)
                }
            }
        }

        // mouth
        val my = hy0 + headH * 0.74f
        when (state) {
            State.DONE -> {
                stroke.color = tint
                stroke.strokeWidth = max(2f, s * 0.025f)
                canvas.drawArc(
                    RectF(cx - s * 0.10f, my - s * 0.06f, cx + s * 0.10f, my + s * 0.06f),
                    20f, 140f, false, stroke,
                )
            }
            State.THINKING -> {
                for (i in 0..2) {
                    val on = (phase * 3f).toInt() % 3 == i
                    val rr = s * 0.018f * (if (on) 1.6f else 1.0f)
                    accent.color = if (on) tint else edge
                    canvas.drawCircle(cx - s * 0.07f + i * s * 0.07f, my, rr, accent)
                }
            }
            else -> {
                stroke.color = edge
                stroke.strokeWidth = max(2f, s * 0.022f)
                canvas.drawLine(cx - s * 0.075f, my, cx + s * 0.075f, my, stroke)
            }
        }

        // a ring leaving the head at the moment it presses
        if (state == State.ACTING) {
            val t = (phase * 1.6f) % 1f
            stroke.color = tint
            stroke.alpha = (255 * (1f - t)).toInt().coerceIn(0, 255)
            stroke.strokeWidth = max(2f, s * 0.03f * (1f - t)) + 1f
            canvas.drawCircle(cx, cy + s * 0.06f, headW * 0.6f + s * 0.28f * t, stroke)
            stroke.alpha = 255
        }
    }

    companion object {
        /** Suggested side length in dp for the floating badge. */
        const val BADGE_DP = 72
    }
}

/** Reads a state straight off an agent step note, so the caller stays dumb. */
fun faceStateFor(note: String): AgentFace.State = when {
    note.startsWith("capture") -> AgentFace.State.LOOKING
    note.startsWith("think") -> AgentFace.State.THINKING
    note.startsWith("done") -> AgentFace.State.DONE
    note.startsWith("stop") -> AgentFace.State.STOPPED
    else -> AgentFace.State.ACTING
}

/** Keeps the aim inside the eye when a target is off the visible screen. */
fun clampFraction(value: Float, size: Int): Float =
    if (size <= 0) 0.5f else max(0f, min(1f, value / size))
