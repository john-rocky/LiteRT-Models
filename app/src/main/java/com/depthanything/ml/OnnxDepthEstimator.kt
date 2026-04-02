package com.depthanything.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

class OnnxDepthEstimator(
    private val context: Context,
    private val modelFileName: String,
    private val optimized: Boolean = false
) : DepthEstimator {

    companion object {
        private const val TAG = "DepthAnything"
    }

    override val mode = InferenceMode.ONNX_CPU

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    private var inputName = ""
    private var inputWidth = 0
    private var inputHeight = 0
    private var outputWidth = 0
    private var outputHeight = 0

    init {
        initialize()
    }

    private fun initialize() {
        Log.i(TAG, "[ONNX] Loading model: $modelFileName (optimized=$optimized)")
        val cacheFile = java.io.File(context.cacheDir, modelFileName)
        if (!cacheFile.exists()) {
            Log.i(TAG, "[ONNX] Copying model to cache...")
            context.assets.open(modelFileName).use { input ->
                cacheFile.outputStream().use { output -> input.copyTo(output) }
            }
        }

        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

        val numCores = Runtime.getRuntime().availableProcessors()
        Log.i(TAG, "[ONNX] CPU cores: $numCores, optimized=$optimized")

        if (optimized) {
            // More threads on big cores
            val threads = numCores.coerceAtMost(6)
            opts.setIntraOpNumThreads(threads)
            Log.i(TAG, "[ONNX] Threads: $threads")
        } else {
            opts.setIntraOpNumThreads(4)
            Log.i(TAG, "[ONNX] Threads: 4")
        }

        session = env.createSession(cacheFile.absolutePath, opts)

        session?.let { sess ->
            val inputInfo = sess.inputInfo.entries.first()
            inputName = inputInfo.key
            val inputTensorInfo = inputInfo.value.info as TensorInfo
            val inputShape = inputTensorInfo.shape
            inputHeight = inputShape[2].toInt()
            inputWidth = inputShape[3].toInt()

            val outputInfo = sess.outputInfo.entries.first()
            val outputTensorInfo = outputInfo.value.info as TensorInfo
            val outputShape = outputTensorInfo.shape
            when (outputShape.size) {
                4 -> {
                    outputHeight = outputShape[2].toInt()
                    outputWidth = outputShape[3].toInt()
                }
                3 -> {
                    outputHeight = outputShape[1].toInt()
                    outputWidth = outputShape[2].toInt()
                }
            }

            Log.i(TAG, "[ONNX] Input: $inputName shape=${inputShape.contentToString()}")
            Log.i(TAG, "[ONNX] Output: shape=${outputShape.contentToString()}")
            Log.i(TAG, "[ONNX] Resolved: input=${inputWidth}x${inputHeight}, " +
                    "output=${outputWidth}x${outputHeight}")
        }
    }

    override fun predict(bitmap: Bitmap): DepthResult {
        val sess = session ?: throw IllegalStateException("Session not initialized")

        val floatBuf = ImageUtils.bitmapToFloatBuffer(bitmap, inputWidth, inputHeight, nchw = true)
        val shape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        val inputTensor = OnnxTensor.createTensor(env, floatBuf, shape)

        val startTime = System.nanoTime()
        val results = sess.run(mapOf(inputName to inputTensor))
        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

        val outputTensor = results[0] as OnnxTensor
        val rawOutput = outputTensor.floatBuffer
        val floatOutput = FloatArray(outputWidth * outputHeight)
        rawOutput.rewind()
        rawOutput.get(floatOutput)

        val grayscale = ImageUtils.depthFloatsToGrayscale(floatOutput, outputWidth, outputHeight)

        inputTensor.close()
        results.close()

        val scaled = Bitmap.createScaledBitmap(grayscale, bitmap.width, bitmap.height, true)
        if (scaled !== grayscale) grayscale.recycle()

        return DepthResult(scaled, elapsedMs, mode)
    }

    override fun close() {
        session?.close()
        session = null
    }
}
