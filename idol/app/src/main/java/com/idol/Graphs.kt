package com.idol

import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.File

/**
 * One compiled graph with its I/O buffers, made once per signature and reused. Inputs written with a tag are not
 * written again while the buffer still holds that tag (the RoPE tables, the position embedding, the LoRA gate, the
 * conditioning of the current phase...).
 */
class Graph(val spec: GraphSpec, private val model: CompiledModel) : AutoCloseable {
    val name: String get() = spec.name
    private val inputs = HashMap<String, List<TensorBuffer>>()
    private val outputs = HashMap<String, List<TensorBuffer>>()
    private val tags = HashMap<String, Array<Any?>>()
    var runs = 0
        private set

    init {
        try {
            for (sig in spec.signatures.values) {
                inputs[sig.key] = sig.inputs.map { model.createInputBuffer(it.name, sig.key) }
                outputs[sig.key] = sig.outputs.map { model.createOutputBuffer(it.name, sig.key) }
                tags[sig.key] = arrayOfNulls(sig.inputs.size)
            }
        } catch (e: Throwable) {
            close()
            throw e
        }
    }

    private fun sig(key: String?): SignatureSpec = spec.signature(key)

    /** Writes [data] to input [index] of signature [key], unless the buffer already holds [tag]. */
    fun write(index: Int, data: FloatArray, tag: Any? = null, key: String? = null) {
        val s = sig(key)
        val t = tags.getValue(s.key)
        if (tag != null && t[index] == tag) return
        val want = s.inputs[index].size
        require(data.size == want) { "$name ${s.key} input $index: ${data.size} values, expected $want" }
        inputs.getValue(s.key)[index].writeFloat(data)
        t[index] = tag
    }

    /**
     * Runs signature [key]. [overrides] replace input buffers by index (another graph's output buffer: the direct
     * handoff); the graph's own buffer at that index keeps its content and loses its tag.
     */
    fun run(key: String? = null, overrides: Map<Int, TensorBuffer> = emptyMap()) {
        val s = sig(key)
        val ins = inputs.getValue(s.key)
        val outs = outputs.getValue(s.key)
        val inMap = HashMap<String, TensorBuffer>()
        for ((i, t) in s.inputs.withIndex()) inMap[t.name] = overrides[i] ?: ins[i]
        val outMap = HashMap<String, TensorBuffer>()
        for ((i, t) in s.outputs.withIndex()) outMap[t.name] = outs[i]
        model.run(inMap, outMap, s.key)
        runs++
    }

    /** Output [index] of signature [key] as float32 (waits for the GPU). */
    fun read(index: Int, key: String? = null): FloatArray {
        val s = sig(key)
        val want = s.outputs[index].size
        val v = outputs.getValue(s.key)[index].readFloat()
        check(v.size >= want) { "$name ${s.key} output $index: ${v.size} values, expected $want" }
        return if (v.size == want) v else v.copyOf(want)
    }

    fun outputBuffer(index: Int, key: String? = null): TensorBuffer = outputs.getValue(sig(key).key)[index]

    /** Forget the tag of input [index] (its buffer was not written by us). */
    fun untag(index: Int, key: String? = null) {
        tags.getValue(sig(key).key)[index] = null
    }

    override fun close() {
        for (l in inputs.values) for (b in l) runCatching { b.close() }
        for (l in outputs.values) for (b in l) runCatching { b.close() }
        inputs.clear()
        outputs.clear()
        runCatching { model.close() }
    }
}

/** The memory guard found no graph left to close and the next one does not fit above the reserve. */
class NotEnoughMemoryException(message: String) : Exception(message)

/**
 * The compiled graphs the app keeps, all on the model thread. All 46 graphs do not fit in a phone's memory at once
 * (one denoiser block alone holds about 0.7 GB of GPU memory), so the pool keeps as many as fit above a reserve of
 * `MemAvailable` and, when the next graph does not fit, closes the resident graph whose next use in the run's
 * schedule is farthest away (Belady's rule over the fixed graph order of a run, read cyclically for the next run).
 * Every compile, close and memory shortfall is recorded.
 */
class GraphPool(
    private val env: Environment,
    private val dir: File,
    private val manifest: Manifest,
    var reserveKb: Long,
    var memoryCheck: Boolean,
    var cpuThreads: Int,
    var gpuPriority: CompiledModel.GpuOptions.Priority?,
) : AutoCloseable {
    class CompileRecord(
        val graph: String,
        val seconds: Double,
        val memAvailableBeforeKb: Long,
        val memAvailableAfterKb: Long,
        val phase: String,
    ) {
        fun json(): JSONObject = JSONObject().put("graph", graph).put("seconds", seconds)
            .put("mem_available_before_kb", memAvailableBeforeKb).put("mem_available_after_kb", memAvailableAfterKb)
            .put("phase", phase)
    }

    private val resident = LinkedHashMap<String, Graph>()
    private val lastUse = HashMap<String, Long>()
    private val pinned = HashSet<String>()
    private val footprintKb = HashMap<String, Long>()
    private var schedule: List<String> = emptyList()
    private var cursor = -1
    private var clock = 0L

    /** What the compile records are filed under ("preload", "run", "parity", ...). */
    var phase = "run"
    val compiles = ArrayList<CompileRecord>()
    val events = ArrayList<JSONObject>()

    val residentNames: List<String> get() = resident.keys.toList()

    fun isResident(name: String) = name in resident

    /** The order the next run will ask for graphs in (resets the position in it). */
    fun setSchedule(order: List<String>) {
        schedule = order
        cursor = -1
    }

    fun pin(name: String) {
        pinned.add(name)
    }

    fun unpin(name: String) {
        pinned.remove(name)
    }

    /** The graph [name], compiled now if it is not resident (closing others first when memory is short). */
    fun get(name: String): Graph {
        advance(name)
        lastUse[name] = clock++
        resident[name]?.let { return it }
        makeRoom(name, evict = true)
        return compile(name)
    }

    /** Compiles [name] if it fits without closing another graph; false when it does not fit. */
    fun preload(name: String): Boolean {
        if (name in resident) return true
        val need = estimateKb(name) + reserveKb
        val avail = DeviceInfo.memAvailableKb()
        if (avail in 0 until need) {
            event("preload_stop", name, avail, need)
            return false
        }
        lastUse[name] = clock++
        compile(name)
        return true
    }

    fun release(name: String) {
        resident.remove(name)?.let {
            it.close()
            event("close", name, DeviceInfo.memAvailableKb(), null)
        }
    }

    fun releaseIf(predicate: (String) -> Boolean) {
        for (n in resident.keys.filter(predicate)) release(n)
    }

    fun releaseAll() = releaseIf { true }

    override fun close() = releaseAll()

    private fun advance(name: String) {
        if (schedule.isEmpty()) return
        for (i in cursor + 1 until schedule.size) if (schedule[i] == name) {
            cursor = i
            return
        }
        for (i in 0..minOf(cursor, schedule.size - 1)) if (schedule[i] == name) {
            cursor = i
            return
        }
    }

    /** Distance in the schedule to the next use of [name] after the current position (cyclic); MAX when never. */
    private fun nextUse(name: String): Int {
        if (schedule.isEmpty()) return Int.MAX_VALUE
        for (i in cursor + 1 until schedule.size) if (schedule[i] == name) return i - cursor
        for (i in 0..minOf(cursor, schedule.size - 1)) if (schedule[i] == name) return schedule.size - cursor + i
        return Int.MAX_VALUE
    }

    private fun victim(exclude: String): String? {
        val candidates = resident.keys.filter { it != exclude && it !in pinned }
        if (candidates.isEmpty()) return null
        if (schedule.isEmpty()) return candidates.minByOrNull { lastUse[it] ?: 0L }
        return candidates.maxWithOrNull(compareBy<String>({ nextUse(it) }, { -(lastUse[it] ?: 0L) }))
    }

    /**
     * Closes graphs until `MemAvailable` covers the estimate for [name] plus the reserve. With [memoryCheck], fails
     * when nothing is left to close and even the reserve is not there (or the measured size of [name] is not).
     */
    private fun makeRoom(name: String, evict: Boolean) {
        val est = estimateKb(name)
        while (true) {
            val avail = DeviceInfo.memAvailableKb()
            if (avail < 0 || avail >= est + reserveKb) return
            val v = if (evict) victim(name) else null
            if (v == null) {
                event("short", name, avail, est + reserveKb)
                val measured = footprintKb[name]
                val floor = reserveKb + (measured ?: 0L)
                if (memoryCheck && avail < floor) {
                    val now = awaitAvailable(floor, MAX_WAIT_SHORT_MS)
                    event("waited", name, now, floor)
                    if (now in 0 until floor) {
                        throw NotEnoughMemoryException(
                            "Not enough memory for $name: ${now / 1024} MB available, ${floor / 1024} MB needed " +
                                "(reserve ${reserveKb / 1024} MB). Close other apps and try again.",
                        )
                    }
                }
                return
            }
            event("evict", v, avail, est + reserveKb, "for" to name)
            val freed = footprintKb[v]
            release(v)
            // A closed graph can return its GPU memory a few seconds late (Pixel 11 Pro, PowerVR: MemAvailable rose
            // from 1.6 to 8.9 GB within 8 s of the closes); without this wait the loop closed every resident graph.
            if (freed != null && freed > 0) awaitAvailable(avail + freed / 2, MAX_WAIT_EVICT_MS)
        }
    }

    /** Polls `MemAvailable` until it reaches [targetKb] or [maxMs] pass; returns the last value read. */
    private fun awaitAvailable(targetKb: Long, maxMs: Long): Long {
        val t0 = SystemClock.elapsedRealtime()
        var now = DeviceInfo.memAvailableKb()
        while (now in 0 until targetKb && SystemClock.elapsedRealtime() - t0 < maxMs) {
            Thread.sleep(POLL_MS)
            now = DeviceInfo.memAvailableKb()
        }
        return now
    }

    private fun compile(name: String): Graph {
        val spec = manifest.graph(name)
        val file = File(dir, spec.file)
        val options = if (spec.onCpu) {
            CompiledModel.Options(Accelerator.CPU).apply {
                cpuOptions = CompiledModel.CpuOptions(numThreads = cpuThreads)
            }
        } else {
            CompiledModel.Options(Accelerator.GPU).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    // one copy of the weights for a graph with several signatures (the octree decoder's buckets)
                    constantTensorSharing = if (spec.signatures.size > 1) true else null,
                    precision = precisionOf(spec.precision),
                    priority = gpuPriority,
                )
            }
        }
        val before = DeviceInfo.memAvailableKb()
        val t0 = SystemClock.elapsedRealtimeNanos()
        val model = CompiledModel.create(file.absolutePath, options, env)
        val graph = try {
            Graph(spec, model)
        } catch (e: Throwable) {
            runCatching { model.close() }
            throw e
        }
        val seconds = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9
        val after = DeviceInfo.memAvailableKb()
        if (before >= 0 && after >= 0) footprintKb[name] = maxOf(footprintKb[name] ?: 0L, before - after)
        resident[name] = graph
        compiles.add(CompileRecord(name, seconds, before, after, phase))
        event("compile", name, after, null, "seconds" to seconds)
        Log.i(TAG, "compiled $name in ${"%.2f".format(seconds)} s (${spec.accelerator}, ${spec.precision}), " +
            "MemAvailable ${before / 1024} -> ${after / 1024} MB, resident ${resident.size}")
        return graph
    }

    /**
     * The memory [name] is expected to take: what it took when it was last compiled, else an estimate from the
     * file (weights at the run precision) and its largest tensor (a transformer block keeps a few dozen of those).
     */
    fun estimateKb(name: String): Long {
        footprintKb[name]?.let { if (it > 0) return it }
        val spec = manifest.graph(name)
        val elementBytes = if (spec.precision == "fp32" || spec.onCpu) 4L else 2L
        val weights = if (elementBytes == 4L) spec.bytes else spec.bytes / 2
        val transformer = name.startsWith("dino_") || name.startsWith("den_") || name == "oct" ||
            name.startsWith("gs_") || name == "bir"
        val activations = spec.maxIoElements * elementBytes * (if (transformer) 26 else 8)
        return (weights + activations) / 1024
    }

    private fun event(kind: String, graph: String, availKb: Long, needKb: Long?, vararg extra: Pair<String, Any>) {
        val e = JSONObject().put("t_ms", SystemClock.elapsedRealtime()).put("event", kind).put("graph", graph)
            .put("mem_available_kb", availKb).put("resident", resident.size)
        if (needKb != null) e.put("need_kb", needKb)
        for ((k, v) in extra) e.put(k, v)
        events.add(e)
        if (kind != "compile") Log.i(TAG, "memory: $e")
    }

    companion object {
        const val TAG = "IdolGraphs"
        private const val POLL_MS = 100L
        private const val MAX_WAIT_EVICT_MS = 1_500L
        private const val MAX_WAIT_SHORT_MS = 8_000L

        fun precisionOf(p: String): CompiledModel.GpuOptions.Precision = when (p) {
            "fp32" -> CompiledModel.GpuOptions.Precision.FP32
            "fp16" -> CompiledModel.GpuOptions.Precision.FP16
            "fp16_with_fp32_accum" -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
            else -> CompiledModel.GpuOptions.Precision.DEFAULT
        }
    }
}
