package com.rmbg

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
 * RMBG-1.4 (ISNet) background remover using LiteRT CompiledModel GPU.
 * Input: [1, 3, 1024, 1024] NCHW — Output: [1, 1, 1024, 1024] sigmoid mask.
 */
class BackgroundRemover(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "RMBG"
        private const val MODEL = "rmbg14.tflite"
        private const val INPUT_SIZE = 1024

        // RMBG-1.4 normalization: (x/255 - 0.5) / 1.0
        private val MEAN = floatArrayOf(0.5f, 0.5f, 0.5f)
        private val STD = floatArrayOf(1.0f, 1.0f, 1.0f)
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    private val inputFloats = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
    private val resizedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val inputPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val scaleMatrix = Matrix()
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    var acceleratorName = ""; private set
    var lastInferenceMs = 0L; private set

    init {
        Log.i(TAG, "Loading model: $MODEL")
        compiledModel = try {
            val gpuOpts = CompiledModel.Options(Accelerator.GPU)
            try {
                gpuOpts.gpuOptions = CompiledModel.GpuOptions(
                    null, null, null,
                    CompiledModel.GpuOptions.Precision.FP32,
                    null, null, null, null, null, null, null, null, null, null, null
                )
            } catch (_: Exception) {}
            val m = CompiledModel.create(context.assets, MODEL, gpuOpts, null)
            acceleratorName = "GPU"
            Log.i(TAG, "GPU ready")
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, falling back to CPU")
            val cpuOpts = CompiledModel.Options(Accelerator.CPU)
            val m = CompiledModel.create(context.assets, MODEL, cpuOpts, null)
            acceleratorName = "CPU"
            Log.i(TAG, "CPU ready")
            m
        }
        inputBuffers = compiledModel.createInputBuffers()
    }

    /**
     * Remove background from image. Returns ARGB bitmap with transparent background.
     */
    fun removeBackground(bitmap: Bitmap): Bitmap {
        val t = System.nanoTime()

        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(
            INPUT_SIZE.toFloat() / bitmap.width,
            INPUT_SIZE.toFloat() / bitmap.height
        )
        canvas.drawBitmap(bitmap, scaleMatrix, scalePaint)
        resizedBitmap.getPixels(inputPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Normalize: (pixel/255 - 0.5) / 1.0, NCHW planar layout
        val planeSize = INPUT_SIZE * INPUT_SIZE
        for (i in inputPixels.indices) {
            val pixel = inputPixels[i]
            inputFloats[i] = ((pixel shr 16) and 0xFF) / 255f - 0.5f
            inputFloats[planeSize + i] = ((pixel shr 8) and 0xFF) / 255f - 0.5f
            inputFloats[2 * planeSize + i] = (pixel and 0xFF) / 255f - 0.5f
        }
        inputBuffers[0].writeFloat(inputFloats)

        val resultBuffers = compiledModel.run(inputBuffers)
        val mask = resultBuffers[0].readFloat()

        // Build mask — output is sigmoid (0-1)
        val maskPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        for (i in mask.indices) {
            val a = (mask[i] * 255).toInt().coerceIn(0, 255)
            maskPixels[i] = (a shl 24) or 0x00FFFFFF
        }
        val maskBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        maskBitmap.setPixels(maskPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val scaledMask = Bitmap.createScaledBitmap(maskBitmap, bitmap.width, bitmap.height, true)
        maskBitmap.recycle()

        // Apply mask alpha to original
        val srcPixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(srcPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val mskPixels = IntArray(bitmap.width * bitmap.height)
        scaledMask.getPixels(mskPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val result = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        for (i in srcPixels.indices) {
            val alpha = (mskPixels[i] ushr 24) and 0xFF
            srcPixels[i] = (alpha shl 24) or (srcPixels[i] and 0x00FFFFFF)
        }
        result.setPixels(srcPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        scaledMask.recycle()

        lastInferenceMs = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Inference: ${lastInferenceMs}ms")
        return result
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
    }
}
