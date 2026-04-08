package com.moge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * MoGe-2 geometry predictor: single CompiledModel GPU inference.
 */
class MoGePredictor(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "MoGe"
        private const val MODEL_FILE = "moge.tflite"
        const val MODEL_H = 448
        const val MODEL_W = 448
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    private val inputFloats = FloatArray(3 * MODEL_H * MODEL_W)
    private val resizedBitmap = Bitmap.createBitmap(MODEL_W, MODEL_H, Bitmap.Config.ARGB_8888)
    private val inputPixels = IntArray(MODEL_H * MODEL_W)
    private val scaleMatrix = Matrix()
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    var lastInferenceMs = 0L; private set
    var acceleratorName = ""; private set

    init {
        val modelFile = java.io.File(context.filesDir, MODEL_FILE)
        if (!modelFile.exists()) {
            throw IllegalStateException(
                "Model not found. Push via adb:\n" +
                "  adb push moge.tflite /data/local/tmp/ && " +
                "adb shell run-as com.moge cp /data/local/tmp/$MODEL_FILE /data/data/com.moge/files/"
            )
        }
        Log.i(TAG, "Loading model: ${modelFile.absolutePath} (${modelFile.length() / 1_000_000} MB)")

        compiledModel = try {
            val opts = CompiledModel.Options(Accelerator.GPU)
            val m = CompiledModel.create(modelFile.absolutePath, opts, null)
            acceleratorName = "GPU"
            Log.i(TAG, "Model GPU ready")
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, falling back to CPU")
            val m = CompiledModel.create(modelFile.absolutePath, CompiledModel.Options(Accelerator.CPU), null)
            acceleratorName = "CPU"
            Log.i(TAG, "Model CPU ready")
            m
        }
        inputBuffers = compiledModel.createInputBuffers()
    }

    fun predict(bitmap: Bitmap): MoGeResult {
        val t = System.nanoTime()

        // Letterbox
        val canvas = Canvas(resizedBitmap)
        canvas.drawColor(0xFF000000.toInt())
        val srcAspect = bitmap.width.toFloat() / bitmap.height
        val dstAspect = MODEL_W.toFloat() / MODEL_H
        val dstRect = if (srcAspect > dstAspect) {
            val fitH = MODEL_W / srcAspect
            val y = (MODEL_H - fitH) * 0.5f
            RectF(0f, y, MODEL_W.toFloat(), y + fitH)
        } else {
            val fitW = MODEL_H * srcAspect
            val x = (MODEL_W - fitW) * 0.5f
            RectF(x, 0f, x + fitW, MODEL_H.toFloat())
        }
        scaleMatrix.setRectToRect(
            RectF(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat()),
            dstRect, Matrix.ScaleToFit.FILL
        )
        canvas.drawBitmap(bitmap, scaleMatrix, scalePaint)
        resizedBitmap.getPixels(inputPixels, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)

        // NCHW [0,1]
        val planeSize = MODEL_H * MODEL_W
        for (i in inputPixels.indices) {
            val pixel = inputPixels[i]
            inputFloats[i] = ((pixel shr 16) and 0xFF) / 255f
            inputFloats[planeSize + i] = ((pixel shr 8) and 0xFF) / 255f
            inputFloats[2 * planeSize + i] = (pixel and 0xFF) / 255f
        }
        inputBuffers[0].writeFloat(inputFloats)

        val resultBuffers = compiledModel.run(inputBuffers)

        // Parse outputs — identify by element count and value range
        val bufs = (0 until resultBuffers.size).map { resultBuffers[it].readFloat() }
        val hwc3 = MODEL_H * MODEL_W * 3
        val hw1 = MODEL_H * MODEL_W

        val big = bufs.filter { it.size == hwc3 }
        val points: FloatArray
        val normal: FloatArray
        if (big.size == 2) {
            if (big[0].max() > 2f) { points = big[0]; normal = big[1] }
            else if (big[1].max() > 2f) { points = big[1]; normal = big[0] }
            else { points = big[1]; normal = big[0] }  // fallback: second is usually points
        } else {
            points = bufs[0]; normal = bufs[1]
        }
        val maskRaw = bufs.first { it.size == hw1 }
        val scaleRaw = bufs.first { it.size == 1 }

        val mask = FloatArray(planeSize)
        for (i in 0 until planeSize) mask[i] = maskRaw[i]

        lastInferenceMs = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Inference: ${lastInferenceMs}ms ($acceleratorName)")

        return MoGeResult(
            points = points, normal = normal, mask = mask,
            metricScale = scaleRaw[0], width = MODEL_W, height = MODEL_H,
            inferenceMs = lastInferenceMs
        )
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
    }
}

data class MoGeResult(
    val points: FloatArray, val normal: FloatArray, val mask: FloatArray,
    val metricScale: Float, val width: Int, val height: Int, val inferenceMs: Long
) {
    fun normalBitmap(): Bitmap {
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = ((normal[i * 3] + 1f) * 0.5f * 255f).toInt().coerceIn(0, 255)
            val g = ((normal[i * 3 + 1] + 1f) * 0.5f * 255f).toInt().coerceIn(0, 255)
            val b = ((normal[i * 3 + 2] + 1f) * 0.5f * 255f).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            it.setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }

    fun depthBitmap(): Bitmap {
        val depth = FloatArray(width * height)
        var minD = Float.MAX_VALUE; var maxD = Float.MIN_VALUE
        for (i in depth.indices) {
            val z = points[i * 3 + 2]
            if (mask[i] > 0.5f) { depth[i] = z; if (z < minD) minD = z; if (z > maxD) maxD = z }
        }
        val range = if (maxD > minD) maxD - minD else 1f
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            if (mask[i] <= 0.5f) { pixels[i] = 0xFF000000.toInt(); continue }
            val t = ((depth[i] - minD) / range).coerceIn(0f, 1f)
            val r: Float; val g: Float; val b: Float
            when {
                t < 0.25f -> { val s = t / 0.25f; r = 0.5f + 0.5f * s; g = s * 0.3f; b = 0.8f - 0.8f * s }
                t < 0.5f -> { val s = (t - 0.25f) / 0.25f; r = 1f; g = 0.3f + 0.7f * s; b = 0f }
                t < 0.75f -> { val s = (t - 0.5f) / 0.25f; r = 1f - s; g = 1f; b = 0f }
                else -> { val s = (t - 0.75f) / 0.25f; r = 0f; g = 1f - 0.5f * s; b = s }
            }
            pixels[i] = (0xFF shl 24) or ((r * 255).toInt().coerceIn(0, 255) shl 16) or
                    ((g * 255).toInt().coerceIn(0, 255) shl 8) or (b * 255).toInt().coerceIn(0, 255)
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            it.setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }
}
