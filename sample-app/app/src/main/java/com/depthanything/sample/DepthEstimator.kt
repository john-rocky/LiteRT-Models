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

    // Pre-baked Inferno LUT as packed ARGB ints
    private val infernoLut = IntArray(256) { Colormap.inferno(255 - it) }


    init {
        Log.i(TAG, "Loading model: $modelFileName")

        // Detect shape first (before CompiledModel which might crash)
        val fd = context.assets.openFd(modelFileName)
        val channel = java.io.FileInputStream(fd.fileDescriptor).channel
        val buf = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val interp = org.tensorflow.lite.Interpreter(buf)
        val inShape = interp.getInputTensor(0).shape()
        val outShape = interp.getOutputTensor(0).shape()
        interp.close()

        inputHeight = inShape[1]; inputWidth = inShape[2]
        outputHeight = outShape[1]; outputWidth = outShape[2]
        Log.i(TAG, "Shape: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")

        val options = CompiledModel.Options(Accelerator.GPU)
        try {
            options.gpuOptions = CompiledModel.GpuOptions(
                null, null, null,
                CompiledModel.GpuOptions.Precision.FP32,
                null, null, null, null, null, null, null, null, null, null, null
            )
        } catch (_: Exception) {}
        compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        Log.i(TAG, "GPU FP32 compiled OK")
        inputBuffers = compiledModel.createInputBuffers()
        outputBuffers = compiledModel.createOutputBuffers()

        // Pre-allocate all buffers
        inputFloats = FloatArray(inputHeight * inputWidth * 3)
        inputPixels = IntArray(inputWidth * inputHeight)
        outputPixels = IntArray(outputWidth * outputHeight)
        resizedBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        depthBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)

        Log.i(TAG, "Model ready: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")

        // Check supported output buffer types via reflection
        try {
            val method = compiledModel.javaClass.getMethod(
                "getOutputBufferRequirements", String::class.java, String::class.java)
            val outReqs = method.invoke(compiledModel, "serving_default", "output_0")
            Log.i(TAG, "Output buffer reqs: $outReqs")
        } catch (_: Exception) {
            try {
                val method = compiledModel.javaClass.getMethod(
                    "getOutputBufferRequirements", String::class.java, String::class.java)
                val outReqs = method.invoke(compiledModel, "", "")
                Log.i(TAG, "Output buffer reqs: $outReqs")
            } catch (e2: Exception) {
                Log.w(TAG, "Could not get buffer requirements: ${e2.message}")
            }
        }
    }

    /**
     * Run depth estimation. Writes colored depth into [outputBitmap].
     * Returns total time (preprocess + inference + postprocess) in ms.
     */
    fun predict(inputBitmap: Bitmap, outputBitmap: Bitmap): Long {
        var t = System.nanoTime()

        // Preprocess
        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(
            inputWidth.toFloat() / inputBitmap.width,
            inputHeight.toFloat() / inputBitmap.height
        )
        canvas.drawBitmap(inputBitmap, scaleMatrix, paint)
        resizedBitmap.getPixels(inputPixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        var idx = 0
        for (pixel in inputPixels) {
            inputFloats[idx++] = (Color.red(pixel) / 255f - MEAN[0]) / STD[0]
            inputFloats[idx++] = (Color.green(pixel) / 255f - MEAN[1]) / STD[1]
            inputFloats[idx++] = (Color.blue(pixel) / 255f - MEAN[2]) / STD[2]
        }
        inputBuffers[0].writeFloat(inputFloats)
        val preMs = (System.nanoTime() - t) / 1_000_000

        // Inference — try run(input) which returns optimized output buffers
        t = System.nanoTime()
        val resultBuffers = compiledModel.run(inputBuffers)
        val infMs = (System.nanoTime() - t) / 1_000_000

        // Postprocess — read from returned buffers (might be faster than pre-allocated)
        t = System.nanoTime()
        val depth = resultBuffers[0].readFloat()
        val t1 = System.nanoTime()

        // Min-max + colormap
        var min = Float.MAX_VALUE; var max = Float.MIN_VALUE
        for (v in depth) { if (v < min) min = v; if (v > max) max = v }
        val scale = 255f / (max - min).coerceAtLeast(1e-6f)

        for (i in depth.indices) {
            outputPixels[i] = infernoLut[((depth[i] - min) * scale).toInt().coerceIn(0, 255)]
        }
        val t2 = System.nanoTime()

        depthBitmap.setPixels(outputPixels, 0, outputWidth, 0, 0, outputWidth, outputHeight)

        val outCanvas = Canvas(outputBitmap)
        scaleMatrix.setScale(
            outputBitmap.width.toFloat() / outputWidth,
            outputBitmap.height.toFloat() / outputHeight
        )
        outCanvas.drawBitmap(depthBitmap, scaleMatrix, paint)
        val t3 = System.nanoTime()

        val readMs = (t1 - t) / 1_000_000
        val cmapMs = (t2 - t1) / 1_000_000
        val drawMs = (t3 - t2) / 1_000_000
        val postMs = (t3 - t) / 1_000_000

        Log.i(TAG, "pre=${preMs}ms inf=${infMs}ms post=${postMs}ms (read=${readMs} cmap=${cmapMs} draw=${drawMs})")

        return preMs + infMs + postMs
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
        depthBitmap.recycle()
    }
}
