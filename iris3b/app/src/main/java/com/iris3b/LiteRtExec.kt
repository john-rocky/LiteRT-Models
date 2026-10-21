package com.iris3b

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.Collections

/**
 * [GraphExec] on LiteRT CompiledModel, one accelerator per graph as its [GraphSpec] says: `gpu` =
 * the GPU only (no CPU fallback: an op the GPU cannot take fails the compile instead of hiding),
 * `cpu` = XNNPACK only.
 *
 * One [Environment] for every graph of the process (a null environment leaks an OpenCL context
 * per compile and aborts after ~20 of them), and every [TensorBuffer] is closed with its model
 * (leaked buffers accumulate until the process is OOM-killed).
 *
 * `run()` only enqueues the GPU work; the first `readFloat()` blocks until the graph finished, so
 * `readNs` holds the compute and `runNs` the enqueue.
 *
 * [programCache]: GpuOptions(serializationDir, modelCacheKey = the graph file, serializeProgramCache)
 * for the graphs whose role starts with one of [cacheRoles]. The cache holds the constant tensors
 * too (in their GPU form, up to [CACHE_FACTOR] times the graph file), so a graph that has no cache
 * file yet is compiled with the cache only while the cache dir's volume keeps
 * size x [CACHE_FACTOR] + [CACHE_MARGIN] free; [cacheDecisions] says what happened per graph.
 * The XNNPACK packed-weight cache of a `cpu` graph ([GraphSpec.xnnpackCache], one file next to the
 * graph, [xnnpackCacheFile]) follows the same free-space rule.
 */
class LiteRtExec(
    private val env: Environment,
    private val dir: File,
    private val programCache: File? = null,
    private val cacheRoles: List<String> = emptyList(),
) : GraphExec {

    /**
     * Per graph role (`<role> xnnpack` for the XNNPACK cache): the cache decision of its last
     * compile. Synchronized: under P4 graphs are compiled on the prefetch thread.
     */
    val cacheDecisions: MutableMap<String, String> = Collections.synchronizedMap(LinkedHashMap<String, String>())

    private class Loaded(
        override val spec: GraphSpec,
        val model: CompiledModel,
        val inputs: Map<String, TensorBuffer>,
        val outputs: Map<String, TensorBuffer>,
    ) : LoadedGraph

    override fun load(spec: GraphSpec): LoadedGraph {
        val graphFile = File(dir, spec.file)
        val options = if (spec.accel == "cpu") {
            val weightCache = xnnpackCacheFile(dir, spec).takeIf {
                spec.xnnpackCache && roomFor("${spec.role} xnnpack", it, graphFile)
            }
            CompiledModel.Options(Accelerator.CPU).apply {
                cpuOptions = CompiledModel.CpuOptions(
                    numThreads = spec.threads,
                    xnnPackWeightCachePath = weightCache?.absolutePath,
                )
            }
        } else {
            val cache = programCache?.takeIf { cacheAllowed(spec, graphFile, it) }
            CompiledModel.Options(Accelerator.GPU).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    precision = precisionOf(spec.precision),
                    serializationDir = cache?.absolutePath,
                    modelCacheKey = cache?.let { cacheKey(spec) },
                    serializeProgramCache = cache?.let { true },
                )
            }
        }
        val model = CompiledModel.create(graphFile.absolutePath, options, env)
        val buffers = ArrayList<TensorBuffer>()
        try {
            val inputs = spec.inputs.associateWith { model.createInputBuffer(it, spec.signature).also(buffers::add) }
            val outputs = spec.outputs.associate { (name, _) -> name to model.createOutputBuffer(name, spec.signature).also(buffers::add) }
            return Loaded(spec, model, inputs, outputs)
        } catch (failure: Throwable) {
            buffers.forEach { runCatching { it.close() } }
            runCatching { model.close() }
            throw failure
        }
    }

    override fun run(graph: LoadedGraph, inputs: List<FloatArray>): GraphRun {
        val g = graph as Loaded
        require(inputs.size == g.spec.inputs.size) { "${g.spec.role}: ${inputs.size} inputs for ${g.spec.inputs.size} names" }
        val t0 = System.nanoTime()
        for ((i, name) in g.spec.inputs.withIndex()) {
            g.inputs.getValue(name).writeFloat(inputs[i])
        }
        val t1 = System.nanoTime()
        g.model.run(g.inputs, g.outputs, g.spec.signature)
        val t2 = System.nanoTime()
        val outs = g.spec.outputs.map { (name, count) ->
            val values = g.outputs.getValue(name).readFloat()
            require(values.size == count) { "${g.spec.role} $name: ${values.size} floats, the config says $count" }
            values
        }
        val t3 = System.nanoTime()
        return GraphRun(outs, t1 - t0, t2 - t1, t3 - t2)
    }

    override fun close(graph: LoadedGraph) {
        val g = graph as Loaded
        g.inputs.values.forEach { runCatching { it.close() } }
        g.outputs.values.forEach { runCatching { it.close() } }
        g.model.close()
    }

    /** Whether [spec] compiles with the program cache (see the class comment); records why. */
    private fun cacheAllowed(spec: GraphSpec, graphFile: File, cacheDir: File): Boolean {
        if (cacheRoles.none { spec.role.startsWith(it) }) return false
        return roomFor(spec.role, cacheFile(cacheDir, spec), graphFile)
    }

    /**
     * Whether the cache file [cache] of [graphFile] may be used: it exists already, or its volume
     * keeps the graph size x [CACHE_FACTOR] + [CACHE_MARGIN] free; records why under [key].
     */
    private fun roomFor(key: String, cache: File, graphFile: File): Boolean {
        if (cache.isFile) {
            cacheDecisions[key] = "on: cache file present (${cache.length()} B)"
            return true
        }
        val free = cache.absoluteFile.parentFile!!.usableSpace
        val need = graphFile.length() * CACHE_FACTOR + CACHE_MARGIN
        val ok = free >= need
        cacheDecisions[key] = "${if (ok) "on" else "off"}: free $free B, need $need B"
        return ok
    }

    companion object {
        /** The cache file a graph's compile writes (ml_drift serialization_program_cache). */
        fun cacheFile(cacheDir: File, spec: GraphSpec): File =
            File(cacheDir, "${cacheKey(spec)}_mldrift_program_cache.bin")

        fun cacheKey(spec: GraphSpec): String = spec.file.removeSuffix(".tflite")

        /** The XNNPACK packed-weight cache file of a `cpu` graph, next to the graph file. */
        fun xnnpackCacheFile(dir: File, spec: GraphSpec): File = File(dir, spec.file + XNNPACK_CACHE_SUFFIX)

        const val XNNPACK_CACHE_SUFFIX = ".xnnpack_cache"

        /** Upper bound of a cache file over its graph file (int8 weights written as fp32). */
        const val CACHE_FACTOR = 4L

        /** Free space left on the volume after the planned cache writes. */
        const val CACHE_MARGIN = 4L shl 30

        fun precisionOf(name: String): CompiledModel.GpuOptions.Precision =
            when (name) {
                "fp32" -> CompiledModel.GpuOptions.Precision.FP32
                "fp16acc32" -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                "fp16" -> CompiledModel.GpuOptions.Precision.FP16
                "default" -> CompiledModel.GpuOptions.Precision.DEFAULT
                else -> throw IllegalArgumentException("unknown precision $name")
            }
    }
}
