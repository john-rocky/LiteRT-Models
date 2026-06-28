package com.mnv4

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import org.json.JSONArray
import java.io.Closeable
import java.io.File

/**
 * MobileNetV4-Conv-Medium ImageNet classifier on the LiteRT CompiledModel GPU.
 *   image[1,3,224,224] (ImageNet-normalized) -> logits[1,1000]
 *
 * Pure CNN. The only GPU re-authoring is the 9 `AdaptiveAvgPool2d(1)` global pools (Squeeze-Excite blocks
 * + the final classifier pool) -> two single-axis means; MobileNetV3's Hardswish maps to the native
 * HARD_SWISH builtin. ~4 ms / 224x224 on a Pixel 8a; device output == PyTorch (corr 1.0, top-1 match).
 */
class ImagenetClassifier(private val ctx: Context) : Closeable {

    companion object {
        const val SIZE = 256
        const val MODEL = "mnv4_fp16.tflite"
        val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    data class Pred(val label: String, val score: Float)

    private val labels: List<String> = run {
        val txt = ctx.assets.open("imagenet_classes.json").bufferedReader().use { it.readText() }
        val arr = JSONArray(txt)
        (0 until arr.length()).map { arr.getString(it) }
    }
    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, MODEL)
        check(f.exists()) { "Model not found: $MODEL. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns the top-k (label, softmax score). */
    fun classify(rgb: FloatArray, topK: Int = 5): List<Pred> {
        val hw = SIZE * SIZE
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            chw[i] = (rgb[i * 3] / 255f - MEAN[0]) / STD[0]
            chw[hw + i] = (rgb[i * 3 + 1] / 255f - MEAN[1]) / STD[1]
            chw[2 * hw + i] = (rgb[i * 3 + 2] / 255f - MEAN[2]) / STD[2]
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val logits = outBuf[0].readFloat()                     // [1000]
        val mx = logits.max()
        var sum = 0.0; val exp = DoubleArray(logits.size) { Math.exp((logits[it] - mx).toDouble()).also { e -> sum += e } }
        return logits.indices.sortedByDescending { logits[it] }.take(topK)
            .map { Pred(labels[it], (exp[it] / sum).toFloat()) }
    }

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
