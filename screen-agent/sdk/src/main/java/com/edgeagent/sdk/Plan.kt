package com.edgeagent.sdk

import org.json.JSONException
import org.json.JSONObject

/**
 * One action the agent can take on a screen.
 *
 * [Act.Tap] is the only one that needs a coordinate, so it is the only one that
 * costs a second model call. Back and scroll are fixed gestures, which makes
 * them both cheaper and more reliable than anything grounded.
 */
sealed interface Act {
    /** Press an element the grounder still has to find. */
    data class Tap(val target: String) : Act

    /** Scroll the screen. [down] false scrolls back up. */
    data class Scroll(val down: Boolean) : Act

    /** The system back gesture. */
    object Back : Act

    /**
     * Put [text] into a field. [target] names the field to focus first; when it
     * is null the text goes to whatever already has input focus.
     *
     * Naming the field is what makes this reliable with a small planner. Asked
     * to sequence a tap and a type across two turns, a 3B model types first and
     * the text lands nowhere; asked for both in one action, it gets it right and
     * the loop does the focusing.
     */
    data class Type(val text: String, val target: String? = null) : Act

    /** The model's claim that the goal is already met. Verified, not trusted. */
    object Done : Act
}

/**
 * Deciding *what* to do, kept separate from deciding *where*.
 *
 * The grounding prompt in [Grounding] is a fixed vendor contract and must not be
 * paraphrased, so the planner does not extend it — it is a second, ordinary
 * question about the same screen, with its own prompt that is ours to tune. One
 * step is therefore at most two model calls: plan, then ground if the plan needs
 * a target.
 *
 * The vocabulary is deliberately tiny. Every verb here maps to something
 * [ActionExecutor] can actually do; a model that invents `swipe_left` or
 * `long_press` gets [Act.Tap] as the fallback rather than a silent no-op.
 */
object Planning {

    /**
     * History is fed back as plain past tense because that is what the model
     * reliably conditions on — a JSON transcript makes it echo the transcript
     * format instead of answering.
     */
    fun promptFor(goal: String, done: List<String>): String {
        val history = if (done.isEmpty()) "Nothing yet."
        else done.mapIndexed { i, s -> "${i + 1}. $s" }.joinToString("\n")
        return """
            You are operating an Android phone by looking at its screen.

            Goal: $goal

            What you have already done:
            $history

            Look at the screen and choose the single next action. Reply with one
            JSON object and nothing else. These are examples of the five actions,
            with made-up values — use your own, describing what is on this screen:

            {"action": "tap", "target": "the Notifications row"}
            {"action": "scroll", "direction": "down"}
            {"action": "back"}
            {"action": "type", "target": "the search box", "text": "battery"}
            {"action": "done"}

            Use "scroll" when the thing you need is probably below the visible
            area. Use "back" when this screen is a dead end. Use "done" only when
            the screen already shows the goal reached.

            For "type", name the field in "target" and put the words to type in
            "text". Both are required, and "text" must come from the goal above —
            never from the examples. The field gets focused for you, so do not
            spend a separate step tapping it first.
        """.trimIndent()
    }

    private val FENCE = Regex("```(?:json)?\\s*(.+?)```", RegexOption.DOT_MATCHES_ALL)

    // Both braces are escaped on purpose. Android's regex engine is ICU, not
    // the JDK's, and ICU rejects a bare closing brace with
    // "Syntax error in regexp pattern near index 9" — at class-init time, so it
    // takes the process down rather than failing the parse.
    private val OBJECT = Regex("\\{[^{}]*\\}", RegexOption.DOT_MATCHES_ALL)

    /**
     * Parses a planner reply. Returns null when nothing usable came back, which
     * the caller treats as a stop rather than guessing an action — a wrong guess
     * here presses something real.
     */
    fun parse(reply: String): Act? {
        var text = reply.trim()
        FENCE.find(text)?.let { text = it.groupValues[1].trim() }
        val json = objectIn(text) ?: return null

        val action = json.optString("action").trim().lowercase()
        val target = json.optString("target").trim()
        val typed = json.optString("text").trim()
        return when {
            action.startsWith("done") -> Act.Done
            action.startsWith("back") -> Act.Back
            action.startsWith("scroll") -> {
                val direction = json.optString("direction").trim().lowercase()
                Act.Scroll(down = !direction.startsWith("up"))
            }
            action.startsWith("type") ->
                // A "type" with only a target is the model naming the field and
                // forgetting the text; there is nothing to type.
                if (typed.isEmpty()) null else Act.Type(typed, target.ifEmpty { null })
            action.startsWith("tap") || action.startsWith("click") ->
                if (target.isEmpty()) null else Act.Tap(target)
            // Unknown verbs. The model invents them — `search` turns up often —
            // and the shape of the object says what it meant: text to put
            // somewhere is a type, a bare target is a press. Reading it as a
            // press either way silently drops the text, which is how a run ends
            // up pressing its way around a screen it meant to type into.
            typed.isNotEmpty() -> Act.Type(typed, target.ifEmpty { null })
            target.isNotEmpty() -> Act.Tap(target)
            else -> null
        }
    }

    /** The reply is usually bare JSON, sometimes JSON with a sentence around it. */
    private fun objectIn(text: String): JSONObject? {
        try {
            return JSONObject(text)
        } catch (e: JSONException) {
            // fall through to the embedded case
        }
        val match = OBJECT.find(text) ?: return null
        return try {
            JSONObject(match.value)
        } catch (e: JSONException) {
            null
        }
    }

    /** How a step reads in the log and in the history fed back to the model. */
    fun describe(act: Act): String = when (act) {
        is Act.Tap -> "tapped ${act.target}"
        is Act.Scroll -> if (act.down) "scrolled down" else "scrolled up"
        is Act.Type ->
            if (act.target == null) "typed \"${act.text}\""
            else "typed \"${act.text}\" into ${act.target}"
        Act.Back -> "went back"
        Act.Done -> "reported the goal as reached"
    }
}
