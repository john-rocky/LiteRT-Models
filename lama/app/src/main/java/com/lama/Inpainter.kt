package com.lama

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
 * LaMa-Dilated inpainter using LiteRT CompiledModel GPU.
 * Input: image [1,512,512,3] NHWC + mask [1,512,512,1] NHWC
 * Output: inpainted image [1,512,512,3] NHWC
 */
class Inpainter(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "LaMa"
        private const val MODEL = "lama_dilated.tflite"
        const val INPUT_SIZE = 512
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    private val imageFloats = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)
    private val maskFloats = FloatArray(INPUT_SIZE * INPUT_SIZE)
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
     * Inpaint masked regions.
     * @param image Source image (any size)
     * @param mask  Mask bitmap (same size as image) — white = area to inpaint
     * @return Inpainted image (same size as input)
     */
    fun inpaint(image: Bitmap, mask: Bitmap): Bitmap {
        val t = System.nanoTime()

        // Resize image to 512x512
        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(INPUT_SIZE.toFloat() / image.width, INPUT_SIZE.toFloat() / image.height)
        canvas.drawBitmap(image, scaleMatrix, scalePaint)
        resizedBitmap.getPixels(inputPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Image: RGB 0-1, NHWC
        var idx = 0
        for (pixel in inputPixels) {
            imageFloats[idx++] = ((pixel shr 16) and 0xFF) / 255f
            imageFloats[idx++] = ((pixel shr 8) and 0xFF) / 255f
            imageFloats[idx++] = (pixel and 0xFF) / 255f
        }

        // Resize mask to 512x512
        val resizedMask = Bitmap.createScaledBitmap(mask, INPUT_SIZE, INPUT_SIZE, true)
        val maskPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedMask.getPixels(maskPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        resizedMask.recycle()

        // Mask: single channel 0-1 (white = 1 = inpaint)
        for (i in maskPixels.indices) {
            maskFloats[i] = ((maskPixels[i] ushr 24) and 0xFF) / 255f
        }

        // Write inputs
        inputBuffers[0].writeFloat(imageFloats)
        inputBuffers[1].writeFloat(maskFloats)

        // Run
        val resultBuffers = compiledModel.run(inputBuffers)
        val output = resultBuffers[0].readFloat()

        // Build result bitmap at 512x512
        val resultPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        idx = 0
        for (i in resultPixels.indices) {
            val r = (output[idx++].coerceIn(0f, 1f) * 255).toInt()
            val g = (output[idx++].coerceIn(0f, 1f) * 255).toInt()
            val b = (output[idx++].coerceIn(0f, 1f) * 255).toInt()
            resultPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val resultBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        resultBitmap.setPixels(resultPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Scale back to original size
        val finalResult = Bitmap.createScaledBitmap(resultBitmap, image.width, image.height, true)
        resultBitmap.recycle()

        lastInferenceMs = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Inpaint: ${lastInferenceMs}ms")
        return finalResult
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
    }
}
