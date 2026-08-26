package com.edgeagent.lab

import com.edgeagent.sdk.ActionExecutor

/**
 * Wraps the real executor so the agent is *seen* doing what it does.
 *
 * The gesture itself is untouched — this only walks the badge to the point
 * first, and lets it linger a moment after. Which is the whole trick: the tap
 * is real, and now it visibly belongs to something.
 *
 * It sits in the app rather than the SDK because it is presentation. A host
 * with no badge passes [AccessibilityExecutor] straight through and the loop
 * behaves identically.
 */
class ShownExecutor(private val inner: ActionExecutor) : ActionExecutor {

    override fun isReady(): Boolean = inner.isReady()

    override fun hasTextFocus(): Boolean = inner.hasTextFocus()

    override suspend fun tap(x: Float, y: Float): Boolean {
        AgentOverlay.pressAt(x, y)
        val ok = inner.tap(x, y)
        AgentOverlay.release()
        return ok
    }

    override suspend fun swipe(
        fromX: Float,
        fromY: Float,
        toX: Float,
        toY: Float,
        durationMs: Long,
    ): Boolean {
        // The badge rides the swipe. Its own travel is a little slower than the
        // gesture so the movement reads at 30 fps in a recording.
        AgentOverlay.drag(fromX, fromY, toX, toY, durationMs + 200)
        val ok = inner.swipe(fromX, fromY, toX, toY, durationMs)
        AgentOverlay.release()
        return ok
    }

    override suspend fun back(): Boolean {
        AgentOverlay.update(AgentFace.State.ACTING, "going back")
        return inner.back()
    }

    override suspend fun typeText(text: String): Boolean {
        AgentOverlay.update(AgentFace.State.ACTING, "typing \"$text\"")
        return inner.typeText(text)
    }
}
