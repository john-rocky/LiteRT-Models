package com.depthanything.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TFLiteDepthEstimator(
    private val context: Context,
    override val mode: InferenceMode,
    private val modelFileName: String
) : DepthEstimator {

    companion object {
        private const val TAG = "DepthAnything"
    }

    private var compiledModel: CompiledModel? = null
    private var inputBuffers: List<TensorBuffer>? = null
    private var outputBuffers: List<TensorBuffer>? = null

    private var inputWidth = 0
    private var inputHeight = 0
    private var outputWidth = 0
    private var outputHeight = 0
    private var isNchw = false

    init {
        initialize()
    }

    private fun initialize() {
        Log.i(TAG, "[LiteRT] Loading: $modelFileName (${mode.label})")

        val options = CompiledModel.Options(Accelerator.GPU)
        try {
            compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        } catch (e: Exception) {
            Log.e(TAG, "[LiteRT] CompiledModel.create failed: ${e.message}", e)
            throw e
        }

        inputBuffers = compiledModel!!.createInputBuffers()
        outputBuffers = compiledModel!!.createOutputBuffers()

        // Detect shape from model file (ONNX=NCHW, TFLite=NHWC)
        if (modelFileName.endsWith(".onnx")) {
            isNchw = true
            // ONNX model: [1, 3, 392, 518] -> [1, 392, 518]
            inputHeight = 392; inputWidth = 518
            outputHeight = 392; outputWidth = 518
        } else {
            // TFLite: use legacy Interpreter for shape detection
            val fd = context.assets.openFd(modelFileName)
            val inputStream = java.io.FileInputStream(fd.fileDescriptor)
            val channel = inputStream.channel
            val modelBuffer = channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                fd.startOffset, fd.declaredLength
            )
            val legacyInterp = org.tensorflow.lite.Interpreter(modelBuffer)
            val inputShape = legacyInterp.getInputTensor(0).shape()
            val outputShape = legacyInterp.getOutputTensor(0).shape()
            legacyInterp.close()

            if (inputShape.size == 4 && inputShape[1] == 3) {
                isNchw = true; inputHeight = inputShape[2]; inputWidth = inputShape[3]
            } else {
                isNchw = false; inputHeight = inputShape[1]; inputWidth = inputShape[2]
            }
            when {
                outputShape.size == 3 -> { outputHeight = outputShape[1]; outputWidth = outputShape[2] }
                outputShape.size == 4 && outputShape[3] == 1 -> { outputHeight = outputShape[1]; outputWidth = outputShape[2] }
                outputShape.size == 4 && outputShape[1] == 1 -> { outputHeight = outputShape[2]; outputWidth = outputShape[3] }
                else -> { outputHeight = outputShape[1]; outputWidth = outputShape[2] }
            }
        }

        Log.i(TAG, "[LiteRT] ${if (isNchw) "NCHW" else "NHWC"} ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")
    }

    override fun predict(bitmap: Bitmap): DepthResult {
        val model = compiledModel ?: throw IllegalStateException("Model not initialized")
        val inBufs = inputBuffers ?: throw IllegalStateException("Buffers not initialized")
        val outBufs = outputBuffers ?: throw IllegalStateException("Buffers not initialized")

        // Preprocess
        val floatBuf = ImageUtils.bitmapToFloatBuffer(
            bitmap, inputWidth, inputHeight, nchw = isNchw, normalize = true
        )
        val floatArray = FloatArray(floatBuf.remaining())
        floatBuf.get(floatArray)

        // Write input
        inBufs[0].writeFloat(floatArray)

        // Run inference
        val startTime = System.nanoTime()
        model.run(inBufs, outBufs)
        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

        // Read output
        val outputArray = outBufs[0].readFloat()
        val grayscale = ImageUtils.depthFloatsToGrayscale(outputArray, outputWidth, outputHeight)

        val scaled = Bitmap.createScaledBitmap(grayscale, bitmap.width, bitmap.height, true)
        if (scaled !== grayscale) grayscale.recycle()

        return DepthResult(scaled, elapsedMs, mode)
    }

    override fun close() {
        inputBuffers?.forEach { it.close() }
        outputBuffers?.forEach { it.close() }
        compiledModel?.close()
        inputBuffers = null
        outputBuffers = null
        compiledModel = null
    }
}
