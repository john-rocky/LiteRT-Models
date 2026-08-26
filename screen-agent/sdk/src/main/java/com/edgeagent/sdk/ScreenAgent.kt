package com.edgeagent.sdk

import android.graphics.Bitmap
import android.graphics.PointF
import java.io.File

/**
 * The library's front door.
 *
 * ```kotlin
 * val agent = ScreenAgent(
 *     screen   = ScreenSource { captureService.grab() },
 *     executor = myAccessibilityExecutor,
 *     grounder = LiteRtGrounder.create(bundle, "cpu", cacheDir)!!,
 *     cacheDir = cacheDir,
 * )
 *
 * val where = agent.locate("the Notifications row")   // one shot, no action
 * agent.tapOnce("the Notifications row")              // find it and press it
 * agent.pursue("open the notification history")       // goal, several steps
 * ```
 *
 * Everything below `Framing` is shared by all three, so a caller who only wants
 * coordinates never drags in an executor.
 */
class ScreenAgent(
    private val screen: ScreenSource,
    private val executor: ActionExecutor,
    private val grounder: Grounder,
    private val cacheDir: File,
    private val settleMillis: Long = 1400L,
) {

    data class Sighting(
        val targets: List<Grounded>,
        val frame: Bitmap,
        val view: Framing.View,
        val raw: String,
    ) {
        /** Screen-pixel centre of the first target, or null if there was none. */
        fun firstPoint(): PointF? =
            targets.firstOrNull()?.let { val p = view.toScreen(it.centre()); PointF(p.x, p.y) }
    }

    /** Capture and ground once. No action is taken. */
    suspend fun locate(instruction: String): Sighting? {
        val frame = screen.capture() ?: return null
        val view = Framing.wholeScreen(frame, cacheDir)
        val found = grounder.locate(view, instruction)
        return Sighting(found.targets, frame, view, found.raw)
    }

    /** Locate, then press the first target. Returns the sighting and whether it pressed. */
    suspend fun tapOnce(instruction: String): Pair<Sighting?, Boolean> {
        val sighting = locate(instruction) ?: return null to false
        val point = sighting.firstPoint() ?: return sighting to false
        if (!executor.isReady()) return sighting to false
        return sighting to executor.tap(point.x, point.y)
    }

    /**
     * Drive toward a goal with the full action set — tap, scroll, back, type.
     *
     * Costs one extra model call per step than [pursue], because it asks what to
     * do before asking where. Use it when the goal may need something that is
     * not on screen yet.
     */
    suspend fun operate(
        goal: String,
        maxSteps: Int = 8,
        onStep: suspend (Agent.Step, Bitmap, List<Grounded>, Framing.View) -> Unit = { _, _, _, _ -> },
    ): Agent.Outcome = Agent.operate(
        goal = goal,
        maxSteps = maxSteps,
        capture = { screen.capture() },
        frame = { bitmap -> Framing.wholeScreen(bitmap, cacheDir) },
        ask = { view, question -> grounder.ask(view, question) },
        locate = { view, target -> grounder.locate(view, target).targets },
        executor = executor,
        settle = { kotlinx.coroutines.delay(settleMillis) },
        onStep = onStep,
    )

    /** Drive toward a goal by tapping only, stopping on the rules in [Agent]. */
    suspend fun pursue(
        goal: String,
        maxSteps: Int = 6,
        onStep: suspend (Agent.Step, Bitmap, List<Grounded>, Framing.View) -> Unit = { _, _, _, _ -> },
    ): Agent.Outcome = Agent.run(
        goal = goal,
        maxSteps = maxSteps,
        capture = { screen.capture() },
        ground = { bitmap, prompt ->
            val view = Framing.wholeScreen(bitmap, cacheDir)
            grounder.locate(view, prompt).targets to view
        },
        tap = { x, y -> if (executor.isReady()) executor.tap(x, y) else false },
        settle = { kotlinx.coroutines.delay(settleMillis) },
        onStep = onStep,
    )
}
