package com.depthanything.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteDepthEstimator(
    private val context: Context,
    override val mode: InferenceMode,
    private val modelFileName: String
) : DepthEstimator {

    companion object {
        private const val TAG = "DepthAnything"
        private const val NUM_THREADS = 4
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private var inputWidth = 0
    private var inputHeight = 0
    private var outputWidth = 0
    private var outputHeight = 0
    private var isNchw = false
    private var inputDataType: DataType = DataType.FLOAT32

    init {
        initialize()
    }

    private fun initialize() {
        val modelBuffer = loadModelFile(modelFileName)
        val options = Interpreter.Options().apply {
            numThreads = NUM_THREADS
        }

        when (mode) {
            InferenceMode.NHWC_GPU_FP32, InferenceMode.FP16W_GPU_FP32 ->
                setupGpu(options, fp16 = false)

            InferenceMode.NHWC_GPU_FP16, InferenceMode.CLAMPED_GPU_FP16,
            InferenceMode.CLAMPED_FP16W_GPU_FP16 ->
                setupGpu(options, fp16 = true)

            else -> throw IllegalArgumentException("Unsupported TFLite mode: $mode")
        }

        Log.i(TAG, "[TFLite] Loading: $modelFileName (${mode.label})")
        interpreter = Interpreter(modelBuffer, options)
        interpreter?.let { interp ->
            val inputShape = interp.getInputTensor(0).shape()
            inputDataType = interp.getInputTensor(0).dataType()

            // Detect NCHW [1,3,H,W] vs NHWC [1,H,W,3]
            if (inputShape.size == 4 && inputShape[1] == 3) {
                isNchw = true
                inputHeight = inputShape[2]
                inputWidth = inputShape[3]
            } else {
                isNchw = false
                inputHeight = inputShape[1]
                inputWidth = inputShape[2]
            }

            val outputShape = interp.getOutputTensor(0).shape()
            when {
                outputShape.size == 3 -> {
                    outputHeight = outputShape[1]
                    outputWidth = outputShape[2]
                }
                outputShape.size == 4 && outputShape[3] == 1 -> {
                    // [1, H, W, 1]
                    outputHeight = outputShape[1]
                    outputWidth = outputShape[2]
                }
                outputShape.size == 4 && outputShape[1] == 1 -> {
                    // [1, 1, H, W]
                    outputHeight = outputShape[2]
                    outputWidth = outputShape[3]
                }
                else -> {
                    outputHeight = outputShape[1]
                    outputWidth = outputShape[2]
                }
            }

            Log.i(TAG, "[TFLite] Input: ${inputShape.contentToString()} $inputDataType " +
                    "(${if (isNchw) "NCHW" else "NHWC"})")
            Log.i(TAG, "[TFLite] Output: ${outputShape.contentToString()}")
            Log.i(TAG, "[TFLite] -> ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")
        }
    }

    private fun setupGpu(options: Interpreter.Options, fp16: Boolean) {
        val compat = CompatibilityList()
        Log.i(TAG, "[TFLite] GPU supported: ${compat.isDelegateSupportedOnThisDevice}")

        try {
            val gpuOptions = GpuDelegate.Options().apply {
                setPrecisionLossAllowed(fp16)
                setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER)
            }
            gpuDelegate = GpuDelegate(gpuOptions)
            options.addDelegate(gpuDelegate)
            Log.i(TAG, "[TFLite] GPU delegate added (FP16=$fp16)")
        } catch (e: Exception) {
            Log.e(TAG, "[TFLite] GPU delegate failed: ${e.message}")
            options.setUseXNNPACK(true)
            Log.i(TAG, "[TFLite] Falling back to XNNPACK")
        }
    }

    override fun predict(bitmap: Bitmap): DepthResult {
        val interp = interpreter ?: throw IllegalStateException("Interpreter not initialized")

        val needsNormalization = true

        val inputBuffer: ByteBuffer = if (inputDataType == DataType.UINT8) {
            bitmapToUint8Buffer(bitmap)
        } else {
            val floatBuf = ImageUtils.bitmapToFloatBuffer(
                bitmap, inputWidth, inputHeight,
                nchw = isNchw,
                normalize = needsNormalization
            )
            ImageUtils.floatBufferToByteBuffer(floatBuf)
        }

        val outputBuffer = ByteBuffer.allocateDirect(outputWidth * outputHeight * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        val startTime = System.nanoTime()
        interp.run(inputBuffer, outputBuffer)
        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

        outputBuffer.rewind()
        val floatOutput = FloatArray(outputWidth * outputHeight)
        outputBuffer.asFloatBuffer().get(floatOutput)
        val grayscale = ImageUtils.depthFloatsToGrayscale(floatOutput, outputWidth, outputHeight)

        val scaled = Bitmap.createScaledBitmap(grayscale, bitmap.width, bitmap.height, true)
        if (scaled !== grayscale) grayscale.recycle()

        return DepthResult(scaled, elapsedMs, mode)
    }

    private fun bitmapToUint8Buffer(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        if (resized !== bitmap) resized.recycle()

        val buffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3)
        buffer.order(ByteOrder.nativeOrder())
        for (pixel in pixels) {
            buffer.put(((pixel shr 16) and 0xFF).toByte())
            buffer.put(((pixel shr 8) and 0xFF).toByte())
            buffer.put((pixel and 0xFF).toByte())
        }
        buffer.rewind()
        return buffer
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fd = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }
}
