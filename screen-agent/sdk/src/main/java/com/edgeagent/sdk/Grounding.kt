package com.edgeagent.sdk

import org.json.JSONArray
import org.json.JSONException

/**
 * The data contract between the model and everything downstream.
 *
 * Coordinates stay in the model's own unit — normalized integers in [0, 1000] —
 * all the way to [Framing.Transform]. That is what survives tiling and
 * rescaling, and it keeps the one risky step (mapping back to screen pixels) a
 * single named function instead of arithmetic sprinkled through the UI.
 */
data class Grounded(
    val label: String,
    val imageId: Int,
    /** [x, y] in [0,1000], or null when this is a box. */
    val point: IntArray?,
    /** [xmin, ymin, xmax, ymax] in [0,1000], or null when this is a point. */
    val box: IntArray?,
) {
    /** Centre of whatever this is, in normalized [0,1000]. */
    fun centre(): IntArray = point ?: intArrayOf(
        (box!![0] + box[2]) / 2,
        (box[1] + box[3]) / 2,
    )
}

/**
 * Prompt + parser for LFM2.5-VL grounding.
 *
 * Both are a fixed vendor contract, copied verbatim from the official demo
 * (huggingface.co/spaces/LiquidAI/LFM2.5-VL-3B-WebGPU, src/main.js and
 * src/grounding.js). Do not paraphrase the prompt — the coordinate convention
 * and the JSON shape are what the model was tuned to emit.
 */
object Grounding {

    const val POINT_SYSTEM_PROMPT =
        "When asked for points corresponding to objects or regions, return a valid JSON array.\n" +
            "Each array item must be an object with:\n" +
            "- image_id: the 0-based index of the image\n" +
            "- point_2d: [x, y] normalized integer coordinates in [0, 1000]\n" +
            "- label: a concise label you choose for the predicted object or region\n" +
            "\n" +
            "Return one item per visible matching object or region. Return [] if none are visible."

    const val BOX_SYSTEM_PROMPT =
        "When asked for bounding boxes for objects, return a valid JSON array.\n" +
            "Each array item must be an object with:\n" +
            "- image_id: the 0-based index of the image\n" +
            "- bbox_2d: [xmin, ymin, xmax, ymax] normalized integer coordinates in [0, 1000]\n" +
            "- label: a concise label you choose for the predicted object or region\n" +
            "\n" +
            "Return one item per visible matching object or region. Return [] if none are visible."

    /** The CLI and the AAR both take a single user turn, so the two are joined. */
    fun promptFor(instruction: String): String =
        "$POINT_SYSTEM_PROMPT\n\n$instruction"

    private val FENCE = Regex("```(?:json)?\\s*(.+?)```", RegexOption.DOT_MATCHES_ALL)
    private val NUM = "(-?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)"
    private val BARE_BOX = Regex("\\[\\s*$NUM\\s*,\\s*$NUM\\s*,\\s*$NUM\\s*,\\s*$NUM\\s*]")
    private val BARE_POINT = Regex("\\[\\s*$NUM\\s*,\\s*$NUM\\s*]")

    /**
     * Accepts either 0..1 floats or 0..1000 integers, as the vendor parser does,
     * and normalizes both to 0..1000 integers. Returns null if the values are
     * neither.
     */
    private fun normalize(values: List<Double>, length: Int): IntArray? {
        if (values.size != length || values.any { !it.isFinite() }) return null
        if (values.all { it in 0.0..1.0 }) {
            return IntArray(length) { Math.round(values[it] * 1000).toInt() }
        }
        if (values.all { it == Math.floor(it) && it >= 0.0 && it <= 1000.0 }) {
            return IntArray(length) { values[it].toInt() }
        }
        return null
    }

    private fun validBox(c: IntArray) = c[2] > c[0] && c[3] > c[1]

    /** Parses a model reply into groundings. Empty list means "nothing usable". */
    fun parse(reply: String): List<Grounded> {
        var text = reply.trim()
        FENCE.find(text)?.let { text = it.groupValues[1].trim() }

        structured(text)?.let { if (it.isNotEmpty()) return it }

        // Fallback: bare arrays embedded in prose.
        val boxes = BARE_BOX.findAll(text).mapNotNull { m ->
            normalize(m.groupValues.drop(1).map { it.toDouble() }, 4)
                ?.takeIf(::validBox)
                ?.let { Grounded("box", 0, null, it) }
        }.toList()
        if (boxes.isNotEmpty()) return boxes

        return BARE_POINT.findAll(text).mapNotNull { m ->
            normalize(m.groupValues.drop(1).map { it.toDouble() }, 2)
                ?.let { Grounded("point", 0, it, null) }
        }.toList()
    }

    private fun structured(text: String): List<Grounded>? {
        val array = try {
            JSONArray(text)
        } catch (e: JSONException) {
            return null
        }
        val out = mutableListOf<Grounded>()
        for (i in 0 until array.length()) {
            val item = array.optJSONObject(i) ?: continue
            val label = item.optString("label").trim().ifEmpty { "?" }
            val imageId = item.optInt("image_id", 0)
            when {
                item.has("point_2d") -> {
                    val coords = normalize(doubles(item.optJSONArray("point_2d")), 2) ?: continue
                    out.add(Grounded(label, imageId, coords, null))
                }
                item.has("bbox_2d") -> {
                    val coords = normalize(doubles(item.optJSONArray("bbox_2d")), 4) ?: continue
                    if (!validBox(coords)) continue
                    out.add(Grounded(label, imageId, null, coords))
                }
            }
        }
        return out
    }

    private fun doubles(array: JSONArray?): List<Double> {
        if (array == null) return emptyList()
        return (0 until array.length()).map { array.optDouble(it, Double.NaN) }
    }
}
