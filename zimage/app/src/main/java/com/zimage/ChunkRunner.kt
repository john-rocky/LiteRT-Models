package com.zimage

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/** Runs one chunked-DiT int8 graph on the LiteRT GPU delegate. */
object ChunkRunner {

    /**
     * Loads one graph on the GPU, feeds [inputs] in convert-arg order, runs it, and
     * returns the first output, freeing the native buffers and graph before returning.
     *
     * FP32 compute is forced because the S3-DiT adaLN/attention path overflows fp16 to
     * NaN on the GPU delegate. A single shared [env] is reused across every call: a
     * per-call (null) environment leaks the ML Drift OpenCL context and aborts the
     * process after ~20 compiles.
     *
     * @param cacheDir when non-null, enables the GPU program cache; left null here
     *     because `serializeProgramCache` aborts the delegate on this runtime.
     */
    fun gpu(env: Environment, name: String, dir: File, inputs: List<FloatArray>,
            log: (String) -> Unit, cacheDir: File? = null): FloatArray {
        val options = CompiledModel.Options(Accelerator.GPU)
        options.gpuOptions = if (cacheDir != null) {
            CompiledModel.GpuOptions(
                precision = CompiledModel.GpuOptions.Precision.FP32,
                serializationDir = cacheDir.absolutePath,
                modelCacheKey = name,
                serializeProgramCache = true)
        } else {
            CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
        }
        val startMs = System.currentTimeMillis()
        val model = CompiledModel.create(File(dir, name).absolutePath, options, env)
        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()
        inputs.forEachIndexed { index, values -> inputBuffers[index].writeFloat(values) }
        model.run(inputBuffers, outputBuffers)
        val output = outputBuffers[0].readFloat()
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
        log(String.format("  %s %.1fs", name, (System.currentTimeMillis() - startMs) / 1000f))
        return output
    }
}
