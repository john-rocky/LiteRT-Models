package com.edgeagent.lab

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import kotlinx.coroutines.delay

/**
 * A badge that floats over whatever the agent is working on.
 *
 * Without it the demo is a phone operating itself with nobody home: the app
 * puts itself in the background, so the only thing on screen is the app being
 * driven. This puts the agent back on screen — the face from [AgentFace] plus a
 * line saying what it is doing right now.
 *
 * One thing it cannot do anywhere: some screens forbid it. A window that sets
 * `HIDE_NON_SYSTEM_OVERLAY_WINDOWS` forces every app overlay off screen while it
 * is showing, and Settings' own home page is one of them — an anti-tapjacking
 * measure, since that is where accessibility is granted. Over such a screen the
 * badge is added, draws, and is held at `READY_TO_SHOW` with
 * `mIsForceHiddenNonSystemOverlayWindow=true`. Nothing to fix; pick another app
 * to demonstrate on.
 *
 * Two things it must not do:
 *  - **take touches.** `FLAG_NOT_TOUCHABLE` matters more than usual here: the
 *    agent dispatches its own taps, and a badge that swallowed one would break
 *    the very thing it is illustrating.
 *  - **appear in what the model sees.** MediaProjection captures the whole
 *    display, badge included, so [duringCapture] takes it off screen for the
 *    frame and puts it back afterwards.
 */
object AgentOverlay {

    private const val TAG = "AgentOverlay"

    private var root: LinearLayout? = null
    private var face: AgentFace? = null
    private var label: TextView? = null
    private var windows: WindowManager? = null

    fun isAllowed(context: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.M || Settings.canDrawOverlays(context)

    /** Sends the user to the one screen where this can be granted. */
    fun requestPermission(context: Context) {
        context.startActivity(
            Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:${context.packageName}"),
            ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        )
    }

    fun show(context: Context, text: String = "starting") {
        if (!isAllowed(context) || root != null) {
            label?.text = text
            return
        }
        val app = context.applicationContext
        val dp = { v: Int ->
            TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), app.resources.displayMetrics
            ).toInt()
        }

        // Android clamps a NOT_TOUCHABLE system window to alpha 0.80 and there
        // is no way around it, so the badge is always slightly see-through. A
        // dark chip over a dark app therefore reads as a smear of the app's own
        // text; a near-white chip with dark type stays legible at 0.8 over
        // anything. That is why this is the one light surface in the project.
        val badge = LinearLayout(app).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(10), dp(8), dp(20), dp(8))
            background = GradientDrawable().apply {
                cornerRadius = dp(34).toFloat()
                setColor(0xFFFFFFFF.toInt())
                setStroke(dp(4), 0xFF2F6FEB.toInt())
            }
        }
        val eye = AgentFace(app).apply {
            layoutParams = LinearLayout.LayoutParams(dp(AgentFace.BADGE_DP), dp(AgentFace.BADGE_DP))
            state = AgentFace.State.LOOKING
        }
        val caption = TextView(app).apply {
            setTextColor(0xFF10151C.toInt())
            textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(dp(12), 0, 0, 0)
            this.text = text
        }
        badge.addView(eye)
        badge.addView(caption)

        // Explicit size, not WRAP_CONTENT: with wrap the window measured to
        // nothing until some later event (the IME appearing) forced a second
        // layout pass, so the badge was missing for the first half of a run.
        val params = WindowManager.LayoutParams(
            dp(320),
            dp(92),
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else
                @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = dp(12)
            y = dp(84)
        }

        val wm = app.getSystemService(WindowManager::class.java)
        runCatching { wm.addView(badge, params) }
            .onSuccess {
                windows = wm
                root = badge
                face = eye
                label = caption
                restX = params.x
                restY = params.y
                Log.i(TAG, "badge added")
                // Belt and braces for the same problem: a visibility round trip
                // once the first frame is up asks the window manager to look at
                // this window again, in case it was added while a transition
                // was still running.
                badge.postDelayed({
                    badge.visibility = View.GONE
                    badge.post { badge.visibility = View.VISIBLE }
                }, 400)
            }
            .onFailure { Log.w(TAG, "badge refused", it) }
    }

    fun update(state: AgentFace.State, text: String) {
        face?.state = state
        label?.text = text
        // The border carries the state as well as the face, because at a glance
        // on a moving screen the colour is what registers.
        val tint = when (state) {
            AgentFace.State.ACTING, AgentFace.State.DONE -> 0xFF12A150.toInt()
            AgentFace.State.STOPPED -> 0xFFCC6B14.toInt()
            else -> 0xFF2F6FEB.toInt()
        }
        (root?.background as? GradientDrawable)?.setStroke(
            (4 * (root?.resources?.displayMetrics?.density ?: 1f)).toInt(), tint
        )
    }

    /** Aim the eye at a point, given in screen pixels. */
    fun lookAt(x: Float, y: Float, screenWidth: Int, screenHeight: Int) {
        face?.lookAt(clampFraction(x, screenWidth), clampFraction(y, screenHeight))
    }

    /** Rest position, out of the way at the top-left. */
    private var restX = 0
    private var restY = 0
    private var mover: android.animation.ValueAnimator? = null

    private fun eyeOffset(): Int = ((AgentFace.BADGE_DP / 2 + 10) *
        (root?.resources?.displayMetrics?.density ?: 1f)).toInt()

    private suspend fun moveTo(targetX: Int, targetY: Int, millis: Long) {
        val view = root ?: return
        val wm = windows ?: return
        val params = view.layoutParams as? WindowManager.LayoutParams ?: return
        val fromX = params.x
        val fromY = params.y
        kotlinx.coroutines.suspendCancellableCoroutine<Unit> { cont ->
            mover?.cancel()
            mover = android.animation.ValueAnimator.ofFloat(0f, 1f).apply {
                duration = millis
                interpolator = android.view.animation.DecelerateInterpolator()
                addUpdateListener { a ->
                    val f = a.animatedValue as Float
                    params.x = (fromX + (targetX - fromX) * f).toInt()
                    params.y = (fromY + (targetY - fromY) * f).toInt()
                    runCatching { wm.updateViewLayout(view, params) }
                }
                addListener(object : android.animation.AnimatorListenerAdapter() {
                    override fun onAnimationEnd(animation: android.animation.Animator) {
                        if (cont.isActive) cont.resume(Unit) {}
                    }
                })
                start()
            }
            cont.invokeOnCancellation { mover?.cancel() }
        }
    }

    /**
     * Walk the badge to the point it is about to press, so the press is the
     * agent's and not the screen's. The label is dropped for the trip: a round
     * face reads as a finger, a pill with a sentence in it does not.
     */
    suspend fun pressAt(x: Float, y: Float) {
        val view = root ?: return
        label?.visibility = View.GONE
        collapse(true)
        face?.state = AgentFace.State.ACTING
        val off = eyeOffset()
        moveTo((x - off).toInt(), (y - off).toInt(), 420)
    }

    /** A round badge for travelling, a pill for standing still and talking. */
    private fun collapse(round: Boolean) {
        val view = root ?: return
        val wm = windows ?: return
        val params = view.layoutParams as? WindowManager.LayoutParams ?: return
        val d = view.resources.displayMetrics.density
        params.width = if (round) (92 * d).toInt() else (320 * d).toInt()
        runCatching { wm.updateViewLayout(view, params) }
    }

    /** Hold on the point for a beat after the gesture, then walk back. */
    suspend fun release() {
        if (root == null) return
        delay(420)
        moveTo(restX, restY, 380)
        collapse(false)
        label?.visibility = View.VISIBLE
    }

    /** Follow a swipe, so a scroll looks like the agent dragging the screen. */
    suspend fun drag(fromX: Float, fromY: Float, toX: Float, toY: Float, millis: Long) {
        if (root == null) return
        label?.visibility = View.GONE
        collapse(true)
        face?.state = AgentFace.State.ACTING
        val off = eyeOffset()
        moveTo((fromX - off).toInt(), (fromY - off).toInt(), 320)
        moveTo((toX - off).toInt(), (toY - off).toInt(), millis)
    }

    fun hide() {
        Log.i(TAG, "badge hide (root=${root != null})")
        root?.visibility = View.GONE
    }

    fun reveal() {
        Log.i(TAG, "badge reveal (root=${root != null})")
        root?.visibility = View.VISIBLE
    }

    fun dismiss() {
        val view = root ?: return
        runCatching { windows?.removeView(view) }
        root = null
        face = null
        label = null
    }

    /**
     * Runs [block] with the badge off screen, so it never lands in the frame the
     * model is shown. The pause is for the compositor: without it the next frame
     * can still carry the badge.
     */
    suspend fun <T> duringCapture(block: suspend () -> T): T {
        hide()
        delay(120)
        return try {
            block()
        } finally {
            reveal()
        }
    }
}
