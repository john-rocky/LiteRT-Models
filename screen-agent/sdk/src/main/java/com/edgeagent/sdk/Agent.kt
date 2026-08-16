package com.edgeagent.sdk

import android.graphics.Bitmap
import android.graphics.PointF

/**
 * The multi-step loop: a goal in, a sequence of taps out.
 *
 * Everything it needs is passed in, so the loop itself has no Android
 * dependencies beyond `Bitmap` and can be lifted into the Phase 3 SDK as-is.
 *
 * Two stopping conditions matter more than the step cap, and both come from
 * measured model behaviour (see FINDINGS.md):
 *
 *  - The model does **not** reliably return `[]` for "nothing to do here" — it
 *    will invent a plausible target. So an empty reply is treated as done, but
 *    it cannot be relied on to arrive.
 *  - Therefore the real guard is the screen itself: if a tap changes nothing,
 *    the loop stops. A confabulated tap usually lands somewhere inert, and
 *    without this check the agent would keep tapping the same dead pixel until
 *    the cap.
 */
object Agent {

    data class Step(
        val index: Int,
        val label: String,
        /** Where it tapped, in screen pixels. Null when it stopped instead. */
        val at: PointF?,
        val note: String,
    )

    data class Outcome(val steps: List<Step>, val stoppedBecause: String)

    /** Wraps the goal in the vendor grounding contract. */
    fun promptFor(goal: String): String =
        "Goal: $goal\n" +
            "This is the current screen. Point to the single element I should tap " +
            "next to make progress toward the goal. Return [] if the goal is " +
            "already reached, or if nothing on this screen helps."

    /**
     * @param capture returns the current screen, or null if capture failed
     * @param ground  screen + instruction -> groundings, in normalized [0,1000]
     * @param tap     screen pixels -> true if the gesture was dispatched
     * @param settle  called after each tap so the UI can finish animating
     */
    suspend fun run(
        goal: String,
        maxSteps: Int,
        capture: suspend () -> Bitmap?,
        ground: suspend (Bitmap, String) -> Pair<List<Grounded>, Framing.View>,
        tap: suspend (Float, Float) -> Boolean,
        settle: suspend () -> Unit,
        onStep: suspend (Step, Bitmap, List<Grounded>, Framing.View) -> Unit,
    ): Outcome {
        val steps = mutableListOf<Step>()
        var frame = capture() ?: return Outcome(steps, "no frame from the screen")

        for (index in 1..maxSteps) {
            val (found, view) = ground(frame, promptFor(goal))
            if (found.isEmpty()) {
                val step = Step(index, "—", null, "model returned nothing to tap")
                steps.add(step)
                onStep(step, frame, found, view)
                return Outcome(steps, "the model reported the goal as reached")
            }

            val target = found.first()
            val at = view.toScreen(target.centre())
            val dispatched = tap(at.x, at.y)
            val step = Step(
                index,
                target.label,
                PointF(at.x, at.y),
                if (dispatched) "tapped" else "tap refused",
            )
            steps.add(step)
            onStep(step, frame, found, view)
            if (!dispatched) return Outcome(steps, "the system refused the gesture")

            settle()
            val after = capture() ?: return Outcome(steps, "no frame after the tap")
            if (!changed(frame, after)) {
                return Outcome(steps, "the screen did not change — stopping rather than repeating")
            }
            frame = after
        }
        return Outcome(steps, "reached the step limit")
    }

    /**
     * The wider loop: plan an action, then carry it out.
     *
     * [run] can only tap, which means it can only reach what is already on
     * screen. This one asks the model what to do first ([Planning]), so it can
     * scroll to something, go back out of a dead end, or type — and it pays one
     * extra model call per step for the privilege. Actions that do not need a
     * coordinate (back, scroll) skip the grounding call, so they are the fast
     * steps rather than the slow ones.
     *
     * @param frame  bitmap -> the view the model will be shown
     * @param ask    a plain question about a view; the planner runs through here
     * @param locate the grounding call, used only for [Act.Tap]
     */
    suspend fun operate(
        goal: String,
        maxSteps: Int,
        capture: suspend () -> Bitmap?,
        frame: suspend (Bitmap) -> Framing.View,
        ask: suspend (Framing.View, String) -> String,
        locate: suspend (Framing.View, String) -> List<Grounded>,
        executor: ActionExecutor,
        settle: suspend () -> Unit,
        onStep: suspend (Step, Bitmap, List<Grounded>, Framing.View) -> Unit,
    ): Outcome {
        val steps = mutableListOf<Step>()
        val history = mutableListOf<String>()
        var bitmap = capture() ?: return Outcome(steps, "no frame from the screen")

        for (index in 1..maxSteps) {
            val view = frame(bitmap)
            val reply = ask(view, Planning.promptFor(goal, history))
            val act = Planning.parse(reply)
            if (act == null) {
                val step = Step(index, "—", null, "no usable plan: ${reply.take(60)}")
                steps.add(step)
                onStep(step, bitmap, emptyList(), view)
                return Outcome(steps, "could not read an action out of the reply")
            }
            if (act is Act.Done) {
                val step = Step(index, "done", null, "model says the goal is reached")
                steps.add(step)
                onStep(step, bitmap, emptyList(), view)
                return Outcome(steps, "the model reported the goal as reached")
            }

            // Only actions that name a place need the grounding call. Back and
            // scroll are fixed gestures, which is why they cost one model call
            // and not two.
            val place = when (act) {
                is Act.Tap -> act.target
                // A field that is already focused must not be pressed: on a
                // screen that opens with its search box focused, a tap a few
                // pixels off takes the focus away and the characters land
                // nowhere. Skipping it also saves the grounding call.
                is Act.Type -> act.target?.takeUnless { executor.hasTextFocus() }
                else -> null
            }
            var found = emptyList<Grounded>()
            var at: PointF? = null
            if (place != null) {
                found = locate(view, place)
                val target = found.firstOrNull()
                if (target == null) {
                    val step = Step(index, place, null, "planned on something not found")
                    steps.add(step)
                    onStep(step, bitmap, found, view)
                    return Outcome(steps, "\"$place\" was not on the screen")
                }
                val point = view.toScreen(target.centre())
                at = PointF(point.x, point.y)
            }

            val done = perform(act, at, bitmap, executor, settle)
            val step = Step(index, Planning.describe(act), at, if (done) "ok" else "refused")
            steps.add(step)
            onStep(step, bitmap, found, view)
            if (!done) return Outcome(steps, "the host refused ${Planning.describe(act)}")
            history.add(Planning.describe(act))

            settle()
            val after = capture() ?: return Outcome(steps, "no frame after ${Planning.describe(act)}")
            // Typing moves a few characters inside one field, which is below the
            // thumbnail's resolution — checking it here would read as "nothing
            // happened". Every other action is supposed to move the screen.
            if (act !is Act.Type && !changed(bitmap, after)) {
                return Outcome(steps, "the screen did not change — stopping rather than repeating")
            }
            bitmap = after
        }
        return Outcome(steps, "reached the step limit")
    }

    private suspend fun perform(
        act: Act,
        at: PointF?,
        frame: Bitmap,
        executor: ActionExecutor,
        settle: suspend () -> Unit,
    ): Boolean = when (act) {
        is Act.Tap -> at != null && executor.tap(at.x, at.y)
        is Act.Back -> executor.back()
        is Act.Type -> {
            // Focus the field first when one was named. Android puts characters
            // into whatever holds input focus, so a type with nothing focused is
            // not a near miss — it goes nowhere.
            val focused = at == null || (executor.tap(at.x, at.y).also { settle() })
            focused && executor.typeText(act.text)
        }
        is Act.Scroll -> {
            // A drag down the middle of the screen: long enough to move a list
            // by most of a page, short of the edge gestures that mean "back"
            // and "home". When a field is focused the keyboard owns the bottom
            // half, and a swipe through it is glide typing — the first take of
            // the demo scrolled "TT" into the search box that way — so the
            // stroke moves up above it.
            val x = frame.width / 2f
            val keyboard = executor.hasTextFocus()
            val near = frame.height * (if (keyboard) 0.50f else 0.72f)
            val far = frame.height * (if (keyboard) 0.18f else 0.30f)
            if (act.down) executor.swipe(x, near, x, far)
            else executor.swipe(x, far, x, near)
        }
        is Act.Done -> true
    }

    /**
     * Cheap "did anything happen" test: shrink both frames to a thumbnail and
     * compare mean absolute luma. Tuned to ignore a blinking cursor or a clock
     * digit but catch a navigation.
     */
    private const val THUMB_W = 32
    private const val THUMB_H = 64
    private const val CHANGE_THRESHOLD = 6.0

    fun changed(a: Bitmap, b: Bitmap): Boolean {
        val ta = Bitmap.createScaledBitmap(a, THUMB_W, THUMB_H, true)
        val tb = Bitmap.createScaledBitmap(b, THUMB_W, THUMB_H, true)
        val pa = IntArray(THUMB_W * THUMB_H)
        val pb = IntArray(THUMB_W * THUMB_H)
        ta.getPixels(pa, 0, THUMB_W, 0, 0, THUMB_W, THUMB_H)
        tb.getPixels(pb, 0, THUMB_W, 0, 0, THUMB_W, THUMB_H)
        ta.recycle()
        tb.recycle()
        var sum = 0.0
        for (i in pa.indices) {
            sum += Math.abs(luma(pa[i]) - luma(pb[i]))
        }
        return sum / pa.size > CHANGE_THRESHOLD
    }

    private fun luma(c: Int): Double {
        val r = (c shr 16) and 0xFF
        val g = (c shr 8) and 0xFF
        val b = c and 0xFF
        return 0.299 * r + 0.587 * g + 0.114 * b
    }
}
