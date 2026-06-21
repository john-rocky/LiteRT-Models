package com.da3

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.abs

/**
 * Depth Anything 3 (DA3-SMALL mono) depth predictor: single CompiledModel GPU inference.
 * The model is converted at the image's NATIVE aspect (896x504, no square padding) so the depth
 * matches the official DA3 pipeline (corr 0.9994). Input [1,3,MODEL_H,MODEL_W] NCHW, ImageNet-normalized.
 */
class DA3Predictor(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "DA3"
        private const val MODEL_FILE = "da3_small_gpu_fp16.tflite"
        const val MODEL_H = 896
        const val MODEL_W = 504
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>
    private val inputFloats = FloatArray(3 * MODEL_H * MODEL_W)
    private val pixels = IntArray(MODEL_H * MODEL_W)

    var acceleratorName = ""; private set

    init {
        compiledModel = try {
            val m = CompiledModel.create(context.assets, MODEL_FILE, CompiledModel.Options(Accelerator.GPU), null)
            acceleratorName = "GPU"; Log.i(TAG, "Model GPU ready"); m
        } catch (e: Exception) {
            Log.w(TAG, "GPU compile failed: ${e.message}, falling back to CPU")
            val m = CompiledModel.create(context.assets, MODEL_FILE, CompiledModel.Options(Accelerator.CPU), null)
            acceleratorName = "CPU"; Log.i(TAG, "Model CPU ready"); m
        }
        inputBuffers = compiledModel.createInputBuffers()
    }

    fun predict(src: Bitmap): DA3Result {
        val t = System.nanoTime()
        // resize to the model's native aspect (no letterbox padding) — matches the official pipeline
        val resized = Bitmap.createScaledBitmap(src, MODEL_W, MODEL_H, true)
        resized.getPixels(pixels, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)

        val plane = MODEL_H * MODEL_W
        for (i in pixels.indices) {
            val p = pixels[i]
            inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        inputBuffers[0].writeFloat(inputFloats)

        val out = compiledModel.run(inputBuffers)
        val depth = out[0].readFloat()

        val ms = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Inference: ${ms}ms ($acceleratorName)")
        return DA3Result(depth, MODEL_W, MODEL_H, ms, acceleratorName)
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
    }
}

data class DA3Result(
    val depth: FloatArray, val width: Int, val height: Int,
    val inferenceMs: Long, val accelerator: String
) {
    fun depthBitmap(): Bitmap {
        var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE
        for (d in depth) { if (d < mn) mn = d; if (d > mx) mx = d }
        val range = if (mx > mn) mx - mn else 1f
        val px = IntArray(width * height)
        for (i in px.indices) {
            val tt = ((depth[i] - mn) / range).coerceIn(0f, 1f)
            val r = (255 * (1.5f - abs(4 * tt - 3)).coerceIn(0f, 1f)).toInt()
            val g = (255 * (1.5f - abs(4 * tt - 2)).coerceIn(0f, 1f)).toInt()
            val b = (255 * (1.5f - abs(4 * tt - 1)).coerceIn(0f, 1f)).toInt()
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            it.setPixels(px, 0, width, 0, 0, width, height)
        }
    }
}
