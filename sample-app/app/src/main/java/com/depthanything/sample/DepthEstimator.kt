package com.depthanything.sample

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * Real-time depth estimator using LiteRT CompiledModel (ML Drift GPU).
 * All buffers are pre-allocated — zero allocation per frame.
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

    // Pre-allocated buffers (zero per-frame allocation)
    private val inputFloats: FloatArray
    private val inputPixels: IntArray
    private val outputPixels: IntArray
    private val resizedBitmap: Bitmap
    private val depthBitmap: Bitmap
    private val scaleMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        Log.i(TAG, "Loading model: $modelFileName")
        val options = CompiledModel.Options(Accelerator.GPU)
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

        // Detect shape
        val fd = context.assets.openFd(modelFileName)
        val channel = java.io.FileInputStream(fd.fileDescriptor).channel
        val buf = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val interp = org.tensorflow.lite.Interpreter(buf)
        val inShape = interp.getInputTensor(0).shape()
        val outShape = interp.getOutputTensor(0).shape()
        interp.close()

        inputHeight = inShape[1]; inputWidth = inShape[2]
        outputHeight = outShape[1]; outputWidth = outShape[2]

        // Pre-allocate all buffers
        inputFloats = FloatArray(inputHeight * inputWidth * 3)
        inputPixels = IntArray(inputWidth * inputHeight)
        outputPixels = IntArray(outputWidth * outputHeight)
        resizedBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        depthBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)

        Log.i(TAG, "Model ready: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")
    }

    /**
     * Run depth estimation. Writes colored depth into [outputBitmap].
     * Returns total time (preprocess + inference + postprocess) in ms.
     */
    fun predict(inputBitmap: Bitmap, outputBitmap: Bitmap): Long {
        val t0 = System.nanoTime()

        // Preprocess: scale into pre-allocated bitmap
        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(
            inputWidth.toFloat() / inputBitmap.width,
            inputHeight.toFloat() / inputBitmap.height
        )
        canvas.drawBitmap(inputBitmap, scaleMatrix, paint)
        resizedBitmap.getPixels(inputPixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // ImageNet normalize NHWC
        var idx = 0
        for (pixel in inputPixels) {
            inputFloats[idx++] = (Color.red(pixel) / 255f - MEAN[0]) / STD[0]
            inputFloats[idx++] = (Color.green(pixel) / 255f - MEAN[1]) / STD[1]
            inputFloats[idx++] = (Color.blue(pixel) / 255f - MEAN[2]) / STD[2]
        }

        inputBuffers[0].writeFloat(inputFloats)

        // Inference
        compiledModel.run(inputBuffers, outputBuffers)

        // Postprocess: min-max + colormap into pre-allocated arrays
        val depth = outputBuffers[0].readFloat()
        var min = Float.MAX_VALUE; var max = Float.MIN_VALUE
        for (v in depth) { if (v < min) min = v; if (v > max) max = v }
        val range = (max - min).coerceAtLeast(1e-6f)

        for (i in depth.indices) {
            val inv = 255 - ((depth[i] - min) / range * 255f).toInt().coerceIn(0, 255)
            outputPixels[i] = Colormap.inferno(inv)
        }

        depthBitmap.setPixels(outputPixels, 0, outputWidth, 0, 0, outputWidth, outputHeight)

        // Scale depth to output size
        val outCanvas = Canvas(outputBitmap)
        scaleMatrix.setScale(
            outputBitmap.width.toFloat() / outputWidth,
            outputBitmap.height.toFloat() / outputHeight
        )
        outCanvas.drawBitmap(depthBitmap, scaleMatrix, paint)

        return (System.nanoTime() - t0) / 1_000_000
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
        depthBitmap.recycle()
    }
}
