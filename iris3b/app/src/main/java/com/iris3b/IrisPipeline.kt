package com.iris3b

import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * One graph file of the pipeline and how to drive it: the accelerator (`gpu` = the GPU at
 * [precision]; `cpu` = XNNPACK with [threads] threads, its packed-weight cache on disk when
 * [xnnpackCache]), the signature, its input names in the order the pipeline feeds them, its
 * outputs with their element counts (outputs are picked by size, never by the signature's order),
 * and the graph file's size in [bytes] (for P4's memory rule).
 */
class GraphSpec(
    val role: String,
    val file: String,
    val precision: String,
    val signature: String,
    val inputs: List<String>,
    val outputs: List<Pair<String, Int>>,
    val accel: String = "gpu",
    val threads: Int = 4,
    val xnnpackCache: Boolean = false,
    val bytes: Long = 0,
)

/** Which graphs stay compiled. P1: every graph is loaded for its run and closed again. P2: the
 *  text encoder runs one graph at a time, then the eight DiT graphs stay compiled for the four
 *  steps and are closed at the end of the image. P4: P1's order, and while one graph runs the
 *  next one is compiled on a second thread (one ahead) when [PrefetchRule] allows it. */
enum class Residency { P1, P2, P4 }

/** P4 compiles the next graph ahead only while MemAvailable >= its bytes x [factor] + [marginBytes]. */
class PrefetchRule(val factor: Double = 4.0, val marginBytes: Long = 2_000_000_000L)

/** The form (graph files and precision) and the residency policy of one run. */
class EngineConfig(
    val residency: Residency,
    val textEncoder: List<GraphSpec>,
    val adapter: GraphSpec,
    val dual: List<GraphSpec>,
    val dualModRows: List<IntArray>,
    val single: List<GraphSpec>,
    val singleModRows: List<IntArray>,
    val pixel: GraphSpec,
    val dumps: Set<String>,
    val prefetch: PrefetchRule = PrefetchRule(),
)

/** A compiled graph. */
interface LoadedGraph {
    val spec: GraphSpec
}

/** One run's outputs (in the spec's output order) and its write / run / read times. */
class GraphRun(val outputs: List<FloatArray>, val writeNs: Long, val runNs: Long, val readNs: Long)

/** Compiles and runs graphs (LiteRT CompiledModel on the device, a fake in the JVM check). */
interface GraphExec {
    fun load(spec: GraphSpec): LoadedGraph

    /** Writes [inputs] (the spec's input order), runs, and reads every output back. */
    fun run(graph: LoadedGraph, inputs: List<FloatArray>): GraphRun

    fun close(graph: LoadedGraph)
}

/** Host tensors by name (rotary tables, modulation rows, the sigmas, the noise, ...). */
fun interface HostTensors {
    operator fun get(name: String): FloatArray
}

/** Thrown between graphs when files/STOP appeared. */
class StopRequested : RuntimeException("STOP")

/**
 * The timeline of one image: one event per host step and per graph load / run / close, each with
 * its milliseconds and the memory reading taken right after it. [listener] sees every event as it
 * happens (the app logs it, so a process that gets killed still leaves its timeline in logcat).
 */
class StageLog(
    private val clock: () -> Long,
    private val memory: () -> Map<String, Long>,
    private val listener: ((Event) -> Unit)? = null,
) {

    class Event(
        val stage: String,
        val graph: String?,
        val step: Int,
        val ms: Map<String, Double>,
        val memory: Map<String, Long>,
    )

    val events = ArrayList<Event>()

    fun now(): Long = clock()

    fun add(stage: String, graph: String?, step: Int, ms: Map<String, Double>) {
        val event = Event(stage, graph, step, ms, memory())
        events.add(event)
        listener?.invoke(event)
    }

    inline fun <T> host(stage: String, step: Int = -1, block: () -> T): T {
        val t0 = now()
        val result = block()
        add(stage, null, step, mapOf("ms" to (now() - t0) / 1e6))
        return result
    }
}

/** What one image produced: the clamped image [3, 768, 768] and the optional dumps by name. */
class PipelineResult(
    val image: FloatArray,
    val realLen: Int,
    val captionTokens: Int,
    val dumps: Map<String, FloatArray>,
)

/**
 * Prompt -> 768x768 image on LiteRT graphs:
 *
 *     tokenize -> embedding lookup (bf16 table) -> 12 text encoder graphs (one per tap)
 *     -> window [34:334] x mask, token-major -> adapter (once per prompt)
 *     -> 4 x (dual0..2 -> [y; x] -> single0..2 -> joint[:, 300:] -> pixel -> sampler)
 *     -> clamp -> patch-major to CHW
 *
 * Mirrors the Python reference pipeline (encoder chain, adapter, step, sampler, image); the graphs
 * are the files named in the [EngineConfig]. [availableBytes] is the
 * system's MemAvailable, read by P4's [PrefetchRule].
 */
class IrisPipeline(
    private val exec: GraphExec,
    private val config: EngineConfig,
    private val host: HostTensors,
    private val tokens: PromptTokens,
    private val availableBytes: () -> Long = { Long.MAX_VALUE },
    private val stop: () -> Boolean = { false },
) {
    private val patchIndex = HostOps.patchIndex()

    /** P4: every compile ahead that failed (that graph was compiled again on the pipeline thread). */
    val prefetchErrors = ArrayList<String>()

    fun generate(prompt: String, noise: FloatArray, log: StageLog): PipelineResult {
        val dumps = LinkedHashMap<String, FloatArray>()
        val ids = log.host("tokenize") { tokens.encode(prompt) }
        val embeds = log.host("embed") { tokens.embed(ids) }
        val teMask = log.host("te_mask") { HostOps.teMask(ids.realCount) }
        val windowMask = HostOps.windowMask(ids.realCount)
        val cos = host["te_rope_cos"]
        val sin = host["te_rope_sin"]

        val resident = LinkedHashMap<String, LoadedGraph>()
        val ahead = if (config.residency == Residency.P4) Prefetcher(planOrder(host["sigmas"].size), log) else null
        try {
            val te = FloatArray(HostOps.TE_FLOATS)
            var hidden = embeds
            for ((tap, spec) in config.textEncoder.withIndex()) {
                val feeds = listOf(hidden, teMask, cos, sin)
                hidden = (if (ahead != null) ahead.run(spec, -1, feeds) else once(spec, -1, log, feeds))[0]
                log.host("te_pack", tap) { HostOps.packTap(tap, hidden, windowMask, te) }
            }
            if ("te" in config.dumps) dumps["te"] = te

            val bias = log.host("adapter_bias") { HostOps.adapterMaskBias(windowMask) }
            val y0 = dit(config.adapter, -1, log, resident, ahead, listOf(te, bias, host["y_pos"]))[0]
            if ("y0" in config.dumps) dumps["y0"] = y0

            val sigmas = host["sigmas"]
            val cosJ = host["cos_j"]
            val sinJ = host["sin_j"]
            val cosI = host["cos_i"]
            val sinI = host["sin_i"]
            val pixPos = host["pix_pos"]
            val x = noise.copyOf()
            val clean = FloatArray(x.size)
            for (step in sigmas.indices) {
                val mods = host["mods_s$step"]
                val tEmb = host["t_emb_s$step"]
                var image = x
                var text = y0
                for ((d, spec) in config.dual.withIndex()) {
                    val r = config.dualModRows[d]
                    val modRows = log.host("mods", step) { HostOps.rows(mods, HostOps.DIT_HIDDEN, r[0], r[1]) }
                    val outs = dit(spec, step, log, resident, ahead, listOf(image, text, cosJ, sinJ, modRows))
                    image = bySize(outs, HostOps.PATCHES * HostOps.DIT_HIDDEN)
                    text = bySize(outs, HostOps.WINDOW * HostOps.DIT_HIDDEN)
                }
                var joint = log.host("concat", step) { HostOps.concatTextImage(text, image) }
                for ((s, spec) in config.single.withIndex()) {
                    val r = config.singleModRows[s]
                    val modRows = log.host("mods", step) { HostOps.rows(mods, HostOps.DIT_HIDDEN, r[0], r[1]) }
                    joint = dit(spec, step, log, resident, ahead, listOf(joint, cosJ, sinJ, modRows))[0]
                }
                val patchRows = log.host("slice", step) { HostOps.imageRows(joint) }
                val v = dit(config.pixel, step, log, resident, ahead, listOf(patchRows, tEmb, x, pixPos, cosI, sinI))[0]
                if ("v" in config.dumps) dumps["v_s$step"] = v
                val next = if (step + 1 < sigmas.size) sigmas[step + 1] else null
                log.host("sampler", step) { HostOps.samplerStep(x, v, sigmas[step], next, clean) }
            }
            closeAll(resident, log)
            log.host("clamp") { HostOps.clampUnit(clean) }
            val out = log.host("assemble") { HostOps.toImage(clean, patchIndex) }
            return PipelineResult(out, ids.realCount, ids.captionTokens, dumps)
        } finally {
            ahead?.close()
            for (graph in resident.values) {
                runCatching { exec.close(graph) }
            }
        }
    }

    /** Text encoder graphs under P1 and P2: loaded, run once and closed. */
    private fun once(spec: GraphSpec, step: Int, log: StageLog, inputs: List<FloatArray>): List<FloatArray> {
        val graph = load(spec, step, log)
        try {
            return run(graph, step, log, inputs)
        } finally {
            close(graph, step, log)
        }
    }

    /**
     * DiT graphs: as [once] under P1; under P2 compiled on first use and kept until the image ends;
     * under P4 through [ahead].
     */
    private fun dit(
        spec: GraphSpec,
        step: Int,
        log: StageLog,
        resident: MutableMap<String, LoadedGraph>,
        ahead: Prefetcher?,
        inputs: List<FloatArray>,
    ): List<FloatArray> {
        if (ahead != null) return ahead.run(spec, step, inputs)
        if (config.residency == Residency.P1) return once(spec, step, log, inputs)
        val graph = resident.getOrPut(spec.role) { load(spec, step, log) }
        return run(graph, step, log, inputs)
    }

    private fun load(spec: GraphSpec, step: Int, log: StageLog): LoadedGraph {
        if (stop()) throw StopRequested()
        val t0 = log.now()
        val graph = exec.load(spec)
        val ms = (log.now() - t0) / 1e6
        log.add("load", spec.role, step, mapOf("ms" to ms, "compile" to ms))
        return graph
    }

    private fun run(graph: LoadedGraph, step: Int, log: StageLog, inputs: List<FloatArray>): List<FloatArray> {
        if (stop()) throw StopRequested()
        val r = exec.run(graph, inputs)
        log.add(
            "run", graph.spec.role, step,
            mapOf("write" to r.writeNs / 1e6, "run" to r.runNs / 1e6, "read" to r.readNs / 1e6,
                "ms" to (r.writeNs + r.runNs + r.readNs) / 1e6),
        )
        return r.outputs
    }

    private fun close(graph: LoadedGraph, step: Int, log: StageLog) {
        val t0 = log.now()
        exec.close(graph)
        log.add("close", graph.spec.role, step, mapOf("ms" to (log.now() - t0) / 1e6))
    }

    private fun closeAll(resident: MutableMap<String, LoadedGraph>, log: StageLog) {
        for (graph in resident.values.toList()) {
            close(graph, -1, log)
        }
        resident.clear()
    }

    private fun bySize(outputs: List<FloatArray>, count: Int): FloatArray {
        val hits = outputs.filter { it.size == count }
        require(hits.size == 1) { "expected one output of $count floats, got ${outputs.map { it.size }}" }
        return hits[0]
    }

    /** P4: the image's graph calls in pipeline order, as (graph, step). */
    private fun planOrder(steps: Int): List<Pair<GraphSpec, Int>> =
        config.textEncoder.map { it to -1 } + (config.adapter to -1) +
            (0 until steps).flatMap { s -> (config.dual + config.single + config.pixel).map { it to s } }

    /**
     * P4 for one image: the graphs of [order] are compiled one ahead on a second thread. Before a
     * graph runs, the compile of the next one starts when MemAvailable >= its bytes x factor +
     * margin ([EngineConfig.prefetch]); the pipeline thread then waits only for what is left of it.
     * Each load event has `ms` (the pipeline thread's wait, or the whole compile when nothing was
     * ahead), `compile` (the compile itself), `prefetched` (1 / 0) and the rule's `avail_mb` /
     * `need_mb`. A compile ahead that fails is compiled again on the pipeline thread and ends P4 for
     * the image, as does a call out of [order] (P1 from there on).
     */
    private inner class Prefetcher(private val order: List<Pair<GraphSpec, Int>>, private val log: StageLog) {
        private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "iris3b-prefetch").apply { isDaemon = true } }
        private var next = 0
        private var pending: Future<Pair<LoadedGraph, Long>>? = null
        private var rule: Map<String, Double> = emptyMap()
        private var on = true

        fun run(spec: GraphSpec, step: Int, inputs: List<FloatArray>): List<FloatArray> {
            if (stop()) throw StopRequested()
            val inOrder = on && next < order.size && order[next].first === spec && order[next].second == step
            val graph = take(spec, step, inOrder)
            if (inOrder) next++ else on = false
            ahead()
            try {
                return this@IrisPipeline.run(graph, step, log, inputs)
            } finally {
                this@IrisPipeline.close(graph, step, log)
            }
        }

        /** Waits for [spec]'s compile ahead (or compiles it now) and logs its load event. */
        private fun take(spec: GraphSpec, step: Int, inOrder: Boolean): LoadedGraph {
            val job = pending
            pending = null
            val t0 = log.now()
            if (job != null && inOrder) {
                try {
                    val (graph, compileNs) = job.get()
                    log.add("load", spec.role, step,
                        rule + mapOf("ms" to (log.now() - t0) / 1e6, "compile" to compileNs / 1e6, "prefetched" to 1.0))
                    return graph
                } catch (failure: ExecutionException) {
                    prefetchErrors.add("${spec.role} s$step: ${failure.cause}")
                    on = false
                }
            } else if (job != null) {
                drain(job)
            }
            val t1 = log.now()
            val graph = exec.load(spec)
            val compile = (log.now() - t1) / 1e6
            log.add("load", spec.role, step,
                rule + mapOf("ms" to (log.now() - t0) / 1e6, "compile" to compile, "prefetched" to 0.0))
            return graph
        }

        /** Starts the next graph's compile when the memory rule allows it. */
        private fun ahead() {
            rule = emptyMap()
            if (!on || next >= order.size) return
            val spec = order[next].first
            val avail = availableBytes()
            val need = spec.bytes * config.prefetch.factor + config.prefetch.marginBytes
            rule = mapOf("avail_mb" to avail / MB, "need_mb" to need / MB)
            if (avail.toDouble() < need) return
            pending = worker.submit(Callable {
                val t0 = log.now()
                val graph = exec.load(spec)
                graph to (log.now() - t0)
            })
        }

        /** Waits for a compile ahead that will not be used and closes its graph. */
        private fun drain(job: Future<Pair<LoadedGraph, Long>>) {
            runCatching { job.get().first }.onSuccess { runCatching { exec.close(it) } }
        }

        /** Closes what is still ahead (an image that stopped or failed) and ends the worker. */
        fun close() {
            pending?.let { drain(it) }
            pending = null
            worker.shutdown()
        }
    }

    private companion object {
        const val MB = 1048576.0
    }
}
