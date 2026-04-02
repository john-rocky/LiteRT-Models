package com.depthanything.sample

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.nio.FloatBuffer

/**
 * Real-time depth estimator using LiteRT CompiledModel (ML Drift GPU).
 *
 * Model: DepthAnything V2 Small, native Keras NHWC TFLite.
 * Input: [1, H, W, 3] NHWC float32 with ImageNet normalization.
 * Output: [1, H, W] float32 depth map.
 */
class DepthEstimator(context: Context, modelFileName: String) : AutoCloseable {

    companion object {
        private const val TAG = "DepthAnything"
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<com.google.ai.edge.litert.TensorBuffer>
    private val outputBuffers: List<com.google.ai.edge.litert.TensorBuffer>

    val inputWidth: Int
    val inputHeight: Int
    private val outputWidth: Int
    private val outputHeight: Int

    init {
        Log.i(TAG, "Loading model: $modelFileName")
        val options = CompiledModel.Options(Accelerator.GPU)
        // FP32 precision: same speed as FP16, avoids ViT attention overflow
        try {
            options.gpuOptions = CompiledModel.GpuOptions(
                null, null, null,
                CompiledModel.GpuOptions.Precision.FP32,
                null, null, null, null, null, null, null, null, null, null, null
            )
        } catch (_: Exception) {}

        compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        inputBuffers = compiledModel.createInputBuffers()
        outputBuffers = compiledModel.createOutputBuffers()

        // Detect shape from model
        val fd = context.assets.openFd(modelFileName)
        val channel = java.io.FileInputStream(fd.fileDescriptor).channel
        val buf = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val interp = org.tensorflow.lite.Interpreter(buf)
        val inShape = interp.getInputTensor(0).shape()   // [1, H, W, 3]
        val outShape = interp.getOutputTensor(0).shape()  // [1, H, W]
        interp.close()

        inputHeight = inShape[1]; inputWidth = inShape[2]
        outputHeight = outShape[1]; outputWidth = outShape[2]
        Log.i(TAG, "Model ready: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")
    }

    /**
     * Run depth estimation. Returns inference time in ms.
     * Writes colored depth map into [outputBitmap].
     */
    fun predict(inputBitmap: Bitmap, outputBitmap: Bitmap): Long {
        // Preprocess: resize + ImageNet normalize + NHWC layout
        val resized = Bitmap.createScaledBitmap(inputBitmap, inputWidth, inputHeight, true)
        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (resized !== inputBitmap) resized.recycle()

        val floatArray = FloatArray(1 * inputHeight * inputWidth * 3)
        var idx = 0
        for (pixel in pixels) {
            floatArray[idx++] = (Color.red(pixel) / 255f - MEAN[0]) / STD[0]
            floatArray[idx++] = (Color.green(pixel) / 255f - MEAN[1]) / STD[1]
            floatArray[idx++] = (Color.blue(pixel) / 255f - MEAN[2]) / STD[2]
        }

        inputBuffers[0].writeFloat(floatArray)

        val t0 = System.nanoTime()
        compiledModel.run(inputBuffers, outputBuffers)
        val ms = (System.nanoTime() - t0) / 1_000_000

        // Postprocess: min-max normalize + Inferno colormap
        val depth = outputBuffers[0].readFloat()
        var min = Float.MAX_VALUE; var max = Float.MIN_VALUE
        for (v in depth) { if (v < min) min = v; if (v > max) max = v }
        val range = (max - min).coerceAtLeast(1e-6f)

        val outPixels = IntArray(outputWidth * outputHeight)
        for (i in depth.indices) {
            val norm = ((depth[i] - min) / range * 255f).toInt().coerceIn(0, 255)
            val inv = 255 - norm  // invert: near=bright
            outPixels[i] = Colormap.inferno(inv)
        }

        val depthBmp = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
        depthBmp.setPixels(outPixels, 0, outputWidth, 0, 0, outputWidth, outputHeight)
        val scaled = Bitmap.createScaledBitmap(depthBmp, outputBitmap.width, outputBitmap.height, true)

        val canvas = android.graphics.Canvas(outputBitmap)
        canvas.drawBitmap(scaled, 0f, 0f, null)
        if (scaled !== depthBmp) depthBmp.recycle()
        scaled.recycle()

        return ms
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        compiledModel.close()
    }
}
