package com.idol

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** One input or output tensor of a graph signature: its name in the flatbuffer and its shape. */
class TensorSpec(val name: String, val shape: IntArray) {
    val size: Int = shape.fold(1) { a, b -> a * b }
}

/** A signature of a graph: inputs in signature order, outputs sorted by their trailing index (output_0, output_1, ...). */
class SignatureSpec(val key: String, val inputs: List<TensorSpec>, val outputs: List<TensorSpec>)

class GraphSpec(
    val name: String,
    val file: String,
    val bytes: Long,
    val sha256: String?,
    /** "gpu" (ML Drift) or "cpu" (XNNPACK: BiRefNet, whose deformable convolutions have no GPU path). */
    val accelerator: String,
    /** "fp16_with_fp32_accum", "fp16" or "fp32". */
    val precision: String,
    val signatures: Map<String, SignatureSpec>,
    /**
     * A CPU graph's working memory: how far `MemAvailable` fell from just before its compile to the lowest point of
     * one call, measured on a phone (`run_peak_bytes`; BiRefNet's call takes about twice what its compile takes).
     * Null when the manifest does not say.
     */
    val runPeakBytes: Long? = null,
    /**
     * A graph that runs several blocks of one chain with each block's weights as runtime inputs (`shared_blocks`: the
     * blocks it stands for, in order; `block_weights`: the side file with their weights; `first_weight_input`: the
     * number of inputs before the weights, args_0 .. args_<n-1>). Empty for an ordinary graph.
     */
    val sharedBlocks: List<Int> = emptyList(),
    val blockWeights: String? = null,
    val firstWeightInput: Int = 3,
) {
    val onCpu: Boolean get() = accelerator == "cpu"

    /** The largest input or output tensor, in elements. */
    val maxIoElements: Long =
        signatures.values.flatMap { it.inputs + it.outputs }.maxOfOrNull { it.size.toLong() } ?: 0L

    fun signature(key: String? = null): SignatureSpec =
        if (key == null) signatures.values.first() else requireNotNull(signatures[key]) { "$name has no signature $key" }
}

class SideFile(val name: String, val shape: IntArray) {
    val elements: Long = shape.fold(1L) { a, b -> a * b }
    val bytes: Long = elements * 4
}

/**
 * `models/manifest.json` as `scripts/make_manifest.py` writes it: every graph's file, size, SHA-256, accelerator,
 * precision and signatures, the raw side files (little-endian float32, with their shapes) and the JSON files. The
 * app builds its file checks, its buffers and its run calls from it.
 */
class Manifest(
    val graphs: Map<String, GraphSpec>,
    val sideFiles: Map<String, SideFile>,
    val jsonFiles: List<String>,
    /** The matting graph the app uses unless the setting picks the alternative ("bir" or "dis"). */
    val defaultMatting: String,
) {
    fun graph(name: String): GraphSpec = requireNotNull(graphs[name]) { "manifest.json has no graph $name" }

    companion object {
        const val FILE = "manifest.json"

        fun read(file: File): Manifest = parse(JSONObject(file.readText()))

        fun parse(o: JSONObject): Manifest {
            val graphs = LinkedHashMap<String, GraphSpec>()
            val g = o.getJSONObject("graphs")
            for (name in g.keys()) {
                val e = g.getJSONObject(name)
                val sigs = LinkedHashMap<String, SignatureSpec>()
                val s = e.getJSONObject("signatures")
                for (key in s.keys()) {
                    val sig = s.getJSONObject(key)
                    val inputs = tensors(sig.getJSONArray("inputs"))
                    val outputs = tensors(sig.getJSONArray("outputs")).sortedBy { trailingIndex(it.name) }
                    sigs[key] = SignatureSpec(key, inputs, outputs)
                }
                graphs[name] = GraphSpec(
                    name = name,
                    file = e.getString("file"),
                    bytes = e.getLong("bytes"),
                    sha256 = e.optString("sha256", "").ifEmpty { null },
                    accelerator = e.optString("accelerator", "gpu"),
                    precision = e.optString("precision", "fp16_with_fp32_accum"),
                    signatures = sigs,
                    runPeakBytes = if (e.has("run_peak_bytes")) e.getLong("run_peak_bytes") else null,
                    sharedBlocks = e.optJSONArray("shared_blocks")?.let { a -> List(a.length()) { a.getInt(it) } }
                        ?: emptyList(),
                    blockWeights = e.optString("block_weights", "").ifEmpty { null },
                    firstWeightInput = e.optInt("first_weight_input", 3),
                )
            }
            val side = LinkedHashMap<String, SideFile>()
            o.optJSONObject("side_files")?.let { sf ->
                for (name in sf.keys()) side[name] = SideFile(name, ints(sf.getJSONObject(name).getJSONArray("shape")))
            }
            val json = ArrayList<String>()
            o.optJSONArray("json_files")?.let { a -> for (i in 0 until a.length()) json.add(a.getString(i)) }
            val matting = o.optJSONObject("matting")?.optString("default", "")?.ifEmpty { null }
                ?: if ("bir" in graphs) "bir" else "dis"
            return Manifest(graphs, side, json, matting)
        }

        private fun tensors(a: JSONArray): List<TensorSpec> = (0 until a.length()).map { i ->
            val t = a.getJSONArray(i)
            TensorSpec(t.getString(0), ints(t.getJSONArray(1)))
        }

        private fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }

        private fun trailingIndex(name: String): Int = name.substringAfterLast('_').toIntOrNull() ?: 0
    }
}
