package com.dsine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * DSINE surface normal estimator: TFLite GPU inference.
 *
 * Input:  image bitmap (any size, center-cropped and resized internally)
 * Output: normal map bitmap (MODEL_W x MODEL_H, RGB = XYZ normals)
 *
 * Push model via adb:
 *   adb push dsine.tflite /data/local/tmp/
 *   adb shell run-as com.dsine cp /data/local/tmp/dsine.tflite /data/data/com.dsine/files/
 */
class NormalEstimator(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "DSINE"
        private const val MODEL_FILE = "dsine.tflite"
        private const val MODEL_H = 480
        private const val MODEL_W = 640

        // ImageNet normalization (pixel values 0-255)
        private val MEAN = floatArrayOf(0.485f * 255f, 0.456f * 255f, 0.406f * 255f)
        private val STD = floatArrayOf(0.229f * 255f, 0.224f * 255f, 0.225f * 255f)
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    // Pre-allocated buffers
    private val inputFloats = FloatArray(3 * MODEL_H * MODEL_W)
    private val resizedBitmap = Bitmap.createBitmap(MODEL_W, MODEL_H, Bitmap.Config.ARGB_8888)
    private val inputPixels = IntArray(MODEL_H * MODEL_W)
    private val scaleMatrix = Matrix()
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    var lastInferenceMs = 0L; private set
    var acceleratorName = ""; private set

    init {
        // Load model from files dir (too large for APK assets)
        val modelFile = java.io.File(context.filesDir, MODEL_FILE)
        if (!modelFile.exists()) {
            try {
                context.assets.open(MODEL_FILE).use { input ->
                    modelFile.outputStream().use { output -> input.copyTo(output) }
                }
                Log.i(TAG, "Copied model from assets to filesDir")
            } catch (_: Exception) {
                throw IllegalStateException(
                    "Model not found. Push via adb:\n" +
                    "adb push dsine.tflite /data/local/tmp/ && " +
                    "adb shell run-as com.dsine cp /data/local/tmp/$MODEL_FILE " +
                    "/data/data/com.dsine/files/"
                )
            }
        }
        val modelPath = modelFile.absolutePath
        Log.i(TAG, "Loading model: $modelPath (${modelFile.length() / 1_000_000} MB)")

        compiledModel = try {
            val gpuOpts = CompiledModel.Options(Accelerator.GPU)
            try {
                gpuOpts.gpuOptions = CompiledModel.GpuOptions(
                    null, null, null,
                    CompiledModel.GpuOptions.Precision.FP32,
                    null, null, null, null, null, null, null, null, null, null, null
                )
            } catch (_: Exception) {}
            val m = CompiledModel.create(modelPath, gpuOpts, null)
            acceleratorName = "GPU"
            Log.i(TAG, "Model GPU ready")
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, falling back to CPU")
            val cpuOpts = CompiledModel.Options(Accelerator.CPU)
            val m = CompiledModel.create(modelPath, cpuOpts, null)
            acceleratorName = "CPU"
            Log.i(TAG, "Model CPU ready")
            m
        }
        inputBuffers = compiledModel.createInputBuffers()
    }

    /**
     * Estimate surface normals from an image.
     * Returns a bitmap where RGB channels encode XYZ normal directions.
     */
    fun estimate(bitmap: Bitmap): Bitmap {
        val t = System.nanoTime()

        // Resize to MODEL_W x MODEL_H (landscape) with center crop
        val canvas = Canvas(resizedBitmap)
        val srcAspect = bitmap.width.toFloat() / bitmap.height
        val dstAspect = MODEL_W.toFloat() / MODEL_H

        val srcRect = if (srcAspect > dstAspect) {
            // Source is wider — crop sides
            val cropW = (bitmap.height * dstAspect).toInt()
            val x = (bitmap.width - cropW) / 2
            android.graphics.RectF(x.toFloat(), 0f, (x + cropW).toFloat(), bitmap.height.toFloat())
        } else {
            // Source is taller — crop top/bottom
            val cropH = (bitmap.width / dstAspect).toInt()
            val y = (bitmap.height - cropH) / 2
            android.graphics.RectF(0f, y.toFloat(), bitmap.width.toFloat(), (y + cropH).toFloat())
        }

        scaleMatrix.setRectToRect(
            srcRect,
            android.graphics.RectF(0f, 0f, MODEL_W.toFloat(), MODEL_H.toFloat()),
            Matrix.ScaleToFit.FILL
        )
        canvas.drawBitmap(bitmap, scaleMatrix, scalePaint)
        resizedBitmap.getPixels(inputPixels, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)

        // Normalize: (pixel - mean) / std, NCHW layout [1, 3, H, W]
        val planeSize = MODEL_H * MODEL_W
        for (i in inputPixels.indices) {
            val pixel = inputPixels[i]
            inputFloats[i] = (((pixel shr 16) and 0xFF).toFloat() - MEAN[0]) / STD[0]
            inputFloats[planeSize + i] = (((pixel shr 8) and 0xFF).toFloat() - MEAN[1]) / STD[1]
            inputFloats[2 * planeSize + i] = ((pixel and 0xFF).toFloat() - MEAN[2]) / STD[2]
        }
        inputBuffers[0].writeFloat(inputFloats)

        // Run model
        val resultBuffers = compiledModel.run(inputBuffers)
        val output = resultBuffers[0].readFloat()  // [1, 3, H, W] normals

        // Convert normals to RGB bitmap: (normal + 1) / 2 * 255
        val normalBitmap = Bitmap.createBitmap(MODEL_W, MODEL_H, Bitmap.Config.ARGB_8888)
        val normalPixels = IntArray(planeSize)

        for (i in 0 until planeSize) {
            val nx = output[i]
            val ny = output[planeSize + i]
            val nz = output[2 * planeSize + i]

            val r = ((nx + 1f) * 0.5f * 255f).toInt().coerceIn(0, 255)
            val g = ((ny + 1f) * 0.5f * 255f).toInt().coerceIn(0, 255)
            val b = ((nz + 1f) * 0.5f * 255f).toInt().coerceIn(0, 255)

            normalPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        normalBitmap.setPixels(normalPixels, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)

        lastInferenceMs = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Inference: ${lastInferenceMs}ms")

        return normalBitmap
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
    }
}
