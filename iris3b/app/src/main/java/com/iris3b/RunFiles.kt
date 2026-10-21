package com.iris3b

import java.io.File
import java.nio.file.Files
import java.security.MessageDigest

/**
 * The files of one run in the app's files dir, as the rows file names them (no Android API, so the
 * JVM check reads the very files the phone gets):
 *
 *  - rows file: `{"kind": "generic", "config": <config file>, "inputs": [{"file": ...}, ...]
 *    (every file the run needs; devicebench pushes them), "rows": [{"key", "prompt", "noise"}]}`;
 *  - config file: the residency policy (P1 / P2 / P4, with P4's `prefetch` rule), the dumps, the
 *    tokenizer and embedding files, the host tensor pack and its index, and per graph its file,
 *    accel, precision, signature, input names in feed order and output names with element counts;
 *  - tokenizer file: `{"files": {name: text}}`, written out to `tok/` for [PromptTokens], with
 *    `tok/embed_bf16.bin` a symbolic link to the embedding table.
 */
class RunFiles(val dir: File, rowsName: String) {

    class Row(val key: String, val prompt: String, val noise: String)

    class Tensor(val offset: Long, val count: Int, val shape: List<Int>)

    val rowsDoc: Map<String, Any?> = obj(Json.parse(File(dir, rowsName).readText()))
    val configName: String = rowsDoc.str("config")
    val config: Map<String, Any?> = obj(Json.parse(File(dir, configName).readText()))
    val rows: List<Row> = rowsDoc.list("rows").map {
        val r = obj(it)
        Row(r.str("key"), r.str("prompt"), (r["noise"] as? String) ?: "noise_pix")
    }
    val residency: Residency = Residency.valueOf(config.str("residency"))

    /**
     * Graph roles whose GPU program cache is serialized, as role prefixes: `"program_cache": true`
     * = every graph, a list (e.g. `["dit_"]`) = the roles starting with one of them, absent = none.
     */
    val programCacheRoles: List<String> = when (val pc = config["program_cache"]) {
        true -> listOf("")
        is List<*> -> pc.map { it as String }
        else -> emptyList()
    }
    val programCache: Boolean = programCacheRoles.isNotEmpty()
    val dumps: List<String> = (config["dumps"] as? List<*>)?.map { it as String } ?: emptyList()

    private val hostFile = File(dir, obj(config["host"]).str("file"))
    private val hostIndex: Map<String, Tensor> = obj(obj(config["host"])["tensors"]).mapValues { (_, v) ->
        val t = obj(v)
        Tensor(t.num("offset").toLong(), t.num("count").toInt(), (t["shape"] as List<*>).map { (it as Double).toInt() })
    }
    private val hostCache = HashMap<String, FloatArray>()

    /** Host tensors from the pack, each read once. */
    val host = HostTensors { name ->
        hostCache.getOrPut(name) {
            val t = requireNotNull(hostIndex[name]) { "host tensor $name is not in ${hostFile.name}" }
            HostOps.readFloats(hostFile, t.offset, t.count)
        }
    }

    fun hostShape(name: String): List<Int> = requireNotNull(hostIndex[name]) { "no host tensor $name" }.shape

    /** Every file the rows file declares, with a check that each is a regular file. */
    fun missingInputs(): List<String> =
        rowsDoc.list("inputs").map { obj(it).str("file") }.filter { !File(dir, it).isFile }

    /** The [EngineConfig] the config file describes. */
    fun engineConfig(): EngineConfig {
        val g = obj(config["graphs"])
        return EngineConfig(
            residency = residency,
            textEncoder = g.list("te").map { spec(it) },
            adapter = spec(g["adapter"]),
            dual = g.list("dual").map { spec(it) },
            dualModRows = g.list("dual_mod_rows").map { range(it) },
            single = g.list("single").map { spec(it) },
            singleModRows = g.list("single_mod_rows").map { range(it) },
            pixel = spec(g["pixel"]),
            dumps = dumps.toSet(),
            prefetch = prefetchRule(),
        )
    }

    /** P4's memory rule: `"prefetch": {"factor": x, "margin_bytes": n}`, each defaulting to [PrefetchRule]'s. */
    private fun prefetchRule(): PrefetchRule {
        val p = config["prefetch"] as? Map<*, *> ?: return PrefetchRule()
        val default = PrefetchRule()
        return PrefetchRule(
            factor = (p["factor"] as? Double) ?: default.factor,
            marginBytes = (p["margin_bytes"] as? Double)?.toLong() ?: default.marginBytes,
        )
    }

    /** Every graph file the config names, in pipeline order. */
    fun graphFiles(): List<String> {
        val c = engineConfig()
        return (c.textEncoder + c.adapter + c.dual + c.single + c.pixel).map { it.file }
    }

    /**
     * Writes the tokenizer files into `tok/` and links `tok/embed_bf16.bin` to the embedding
     * table; returns the sha256 of every file written.
     */
    fun materializeTokenizer(): Map<String, String> {
        val tok = File(dir, "tok")
        tok.mkdirs()
        val files = obj(obj(Json.parse(File(dir, config.str("tokenizer")).readText()))["files"])
        val sums = LinkedHashMap<String, String>()
        for ((name, text) in files) {
            val bytes = (text as String).toByteArray(Charsets.UTF_8)
            File(tok, name).writeBytes(bytes)
            sums[name] = sha256(bytes)
        }
        val link = File(tok, PromptTokens.EMBEDDING_FILE).toPath()
        Files.deleteIfExists(link)
        Files.createSymbolicLink(link, File(dir, config.str("embed")).absoluteFile.toPath())
        return sums
    }

    /** The output tensors of one row in out-file order: the image, then the dumps. */
    fun outputs(): List<Pair<String, Int>> {
        val out = arrayListOf("image" to HostOps.IMAGE_FLOATS)
        for (d in dumps) {
            when (d) {
                "te" -> out.add("te" to HostOps.TE_FLOATS)
                "y0" -> out.add("y0" to HostOps.WINDOW * HostOps.DIT_HIDDEN)
                "v" -> for (s in 0 until 4) out.add("v_s$s" to HostOps.PATCHES * HostOps.PIXELS * HostOps.CHANNELS)
                else -> throw IllegalArgumentException("unknown dump $d")
            }
        }
        return out
    }

    private fun spec(any: Any?): GraphSpec {
        val m = obj(any)
        val accel = (m["accel"] as? String) ?: "gpu"
        require(accel == "gpu" || accel == "cpu") { "${m["role"]}: accel $accel (gpu or cpu)" }
        val file = m.str("file")
        return GraphSpec(
            role = m.str("role"),
            file = file,
            precision = m.str("precision"),
            signature = m.str("signature"),
            inputs = m.list("inputs").map { it as String },
            outputs = m.list("outputs").map {
                val pair = it as List<*>
                (pair[0] as String) to (pair[1] as Double).toInt()
            },
            accel = accel,
            threads = ((m["threads"] ?: config["cpu_threads"]) as? Double)?.toInt() ?: 4,
            xnnpackCache = m["xnnpack_cache"] as? Boolean ?: false,
            bytes = File(dir, file).length(),
        )
    }

    private fun range(any: Any?): IntArray = (any as List<*>).map { (it as Double).toInt() }.toIntArray()

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun obj(any: Any?): Map<String, Any?> = requireNotNull(any as? Map<String, Any?>) { "expected a JSON object" }

        fun Map<String, Any?>.str(key: String): String = requireNotNull(this[key] as? String) { "missing string $key" }

        fun Map<String, Any?>.num(key: String): Double = requireNotNull(this[key] as? Double) { "missing number $key" }

        fun Map<String, Any?>.list(key: String): List<Any?> = requireNotNull(this[key] as? List<*>) { "missing list $key" }

        fun sha256(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }
}
