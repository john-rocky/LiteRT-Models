package com.klein

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * Loads one LiteRT graph, runs it on the Mali GPU, and frees it again.
 *
 * FLUX.2-klein is 6.2 GB of int8 graphs — far past both the 2 GB flatbuffer load
 * limit and the GPU memory budget — so the twelve chunks are held one at a time
 * (sequential residency). Every native handle is released before returning:
 * leaked [com.google.ai.edge.litert.TensorBuffer]s accumulate until the process
 * is OOM-killed part-way through a run.
 *
 * Compute is forced to FP32: the modulated (adaLN) blocks overflow fp16 on ML
 * Drift and the whole image comes back NaN. The [Environment] is created once by
 * the caller and shared — a null environment leaks the OpenCL context and aborts
 * the process after roughly twenty compiles.
 */
object ChunkRunner {

    /**
     * Runs [name] with [inputs] bound in signature order.
     *
     * @param env shared LiteRT environment; must outlive the call.
     * @param name graph file name, resolved inside [dir].
     * @param dir directory holding the pushed .tflite graphs.
     * @param inputs one flat array per graph input, in signature order.
     * @return one flat array per graph output, in signature order.
     */
    fun gpu(env: Environment, name: String, dir: File,
            inputs: List<FloatArray>): List<FloatArray> {
        val options = CompiledModel.Options(Accelerator.GPU)
        options.gpuOptions = CompiledModel.GpuOptions(
            precision = CompiledModel.GpuOptions.Precision.FP32)
        val model = CompiledModel.create(File(dir, name).absolutePath, options, env)
        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()
        inputs.forEachIndexed { index, values -> inputBuffers[index].writeFloat(values) }
        model.run(inputBuffers, outputBuffers)
        val outputs = outputBuffers.map { it.readFloat() }
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
        return outputs
    }
}
