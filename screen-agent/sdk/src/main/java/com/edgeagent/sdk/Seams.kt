package com.edgeagent.sdk

import android.graphics.Bitmap

/**
 * The three things a screen agent needs from its host.
 *
 * None of them mention MediaProjection or AccessibilityService. That is the
 * point: the library works against anything that can hand it a screen and
 * anything that can press one, so it can be driven by a test harness, by a
 * rooted `input tap`, or by a completely different capture route without
 * touching the loop.
 */

/** Produces the current screen. Returns null when a frame could not be had. */
fun interface ScreenSource {
    suspend fun capture(): Bitmap?
}

/**
 * Performs an action on the screen. Returns false if the host refused it.
 *
 * Only [tap] has no default: a host that can press a coordinate is the minimum
 * this library is useful with. Everything else declines by default, and
 * [Agent.operate] stops rather than pretending an unsupported action happened —
 * a silent no-op would look exactly like a screen that did not respond.
 */
interface ActionExecutor {
    suspend fun tap(x: Float, y: Float): Boolean

    /** Drag from one point to another. Used for scrolling. */
    suspend fun swipe(
        fromX: Float,
        fromY: Float,
        toX: Float,
        toY: Float,
        durationMs: Long = 300L,
    ): Boolean = false

    /** The system back gesture. */
    suspend fun back(): Boolean = false

    /** Put [text] into whatever field currently has input focus. */
    suspend fun typeText(text: String): Boolean = false

    /**
     * True when some field already holds input focus.
     *
     * The loop asks before deciding whether to press a text field: a screen that
     * opened with its field focused loses that focus to a tap landing a few
     * pixels off, and then there is nowhere for the characters to go.
     */
    fun hasTextFocus(): Boolean = false

    /** True when the host is actually able to act right now. */
    fun isReady(): Boolean
}

/**
 * Turns a screen plus an instruction into normalized targets.
 *
 * [LiteRtGrounder] is the shipped implementation; swap it to try another model
 * or to stub the model out in a test.
 */
interface Grounder {
    suspend fun locate(view: Framing.View, instruction: String): Located

    /**
     * Asks a plain question about the screen — no grounding prompt, no parsing.
     * The planner in [Planning] runs through here, and so does Ask mode.
     */
    suspend fun ask(view: Framing.View, question: String): String

    data class Located(
        val targets: List<Grounded>,
        /** The model's reply verbatim, for logs and debugging. */
        val raw: String,
        val ok: Boolean,
    )
}
