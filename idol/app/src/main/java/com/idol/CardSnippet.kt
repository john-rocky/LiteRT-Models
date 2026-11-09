package com.idol

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * The model card's Kotlin block, run on the phone as written (autorun plan plans/snippet.json). [dir] is the app's
 * model folder; [image] is the card's `image`: Matting.birefnetInput(the picture resized to a short side of 1024),
 * [1, 3, 1024, 1024]. Returns one line about the alpha it produced.
 */
fun cardSnippet(dir: File, image: FloatArray): String {
    // ---- the card's block, verbatim
    val env = Environment.create()
    val cpu = CompiledModel.Options(Accelerator.CPU)            // bir.tflite: BiRefNet, the one CPU graph
    val bir = CompiledModel.create(File(dir, "bir.tflite").absolutePath, cpu, env)
    val inputs = bir.createInputBuffers()
    val outputs = bir.createOutputBuffers()
    inputs[0].writeFloat(image)        // [1, 3, 1024, 1024]: RGB / 255, ImageNet mean / std
    bir.run(inputs, outputs)
    val alpha = outputs[0].readFloat() // [1024 * 1024] sigmoid map, 1 = the object
    for (b in inputs + outputs) b.close()
    bir.close()
    val gpu = CompiledModel.Options(Accelerator.GPU).apply {   // every other graph
        gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16)
    }
    // ---- end of the card's block
    runCatching { env.close() }
    var mn = Float.POSITIVE_INFINITY
    var mx = Float.NEGATIVE_INFINITY
    var sum = 0.0
    var above = 0
    for (v in alpha) {
        if (v < mn) mn = v
        if (v > mx) mx = v
        sum += v
        if (v > 0.5f) above++
    }
    return "card snippet: alpha ${alpha.size} values, min %.4f max %.4f mean %.4f, fraction > 0.5 %.4f; gpu options built (%s)"
        .format(mn, mx, sum / alpha.size, above.toDouble() / alpha.size, gpu.gpuOptions?.precision)
}

/**
 * The DINOv3 repo card's Kotlin block, run on the phone as written (autorun plan plans/snippet.json, after
 * [cardSnippet]). [dir] holds dino_0..3.tflite and the RoPE tables; [pixels] is the card's `pixels`: the picture Idol
 * feeds, [1, 3, 1024, 1024] RGB in [0, 1]. Returns the feature (for the check against the fixture's feature1).
 */
fun dinoCardSnippet(dir: File, pixels: FloatArray): FloatArray {
    // ---- the DINOv3 card's block, verbatim
    fun floats(f: File): FloatArray {
        val b = java.nio.ByteBuffer.wrap(f.readBytes()).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.remaining()).also { b.get(it) }
    }
    val env = Environment.create()
    val gpu = CompiledModel.Options(Accelerator.GPU).apply {
        gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM)
    }
    val rope = mapOf("args_1" to floats(File(dir, "dino_rope_cos.f32")), "args_2" to floats(File(dir, "dino_rope_sin.f32")))
    var x = pixels                                         // [1, 3, 1024, 1024] RGB in [0, 1]
    for (i in 0..3) {                                      // 8 layers per graph
        val model = CompiledModel.create(File(dir, "dino_$i.tflite").absolutePath, gpu, env)
        val inputs = listOf("args_0", "args_1", "args_2").associateWith { model.createInputBuffer(it, "serving_default") }
        val output = model.createOutputBuffer("output_0", "serving_default")
        for ((name, buffer) in inputs) buffer.writeFloat(if (name == "args_0") x else rope.getValue(name))
        model.run(inputs, mapOf("output_0" to output), "serving_default")
        x = output.readFloat()                             // [1, 4101, 1280]
        for (b in inputs.values + output) b.close()
        model.close()
    }
    env.close()
    // x: the image feature Idol's denoiser reads (1 class, 4 register and 64 x 64 patch tokens)
    // ---- end of the DINOv3 card's block
    return x
}
