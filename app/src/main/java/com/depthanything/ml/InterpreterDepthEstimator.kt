package com.depthanything.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Depth estimator using the V1 Interpreter API with GpuDelegate + direct ByteBuffer I/O.
 *
 * Interpreter.run(ByteBuffer, ByteBuffer) copies GPU output directly into a pre-allocated
 * direct ByteBuffer via JNI (libtensorflowlite_jni.so). Readback takes ~1-5ms vs
 * 230-340ms with CompiledModel's TensorBuffer.readFloat().
 *
 * Requires litert:1.4.0 + litert-gpu:1.4.1 + litert-gpu-api:1.4.1
 * (NOT compatible with litert:2.x which stripped addDelegate from Options)
 */
class InterpreterDepthEstimator(
    private val context: Context,
    override val mode: InferenceMode,
    private val modelFileName: String
) : DepthEstimator {

    companion object {
        private const val TAG = "DepthAnything"
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private var inputWidth = 0
    private var inputHeight = 0
    private var outputWidth = 0
    private var outputHeight = 0

    // Pre-allocated direct ByteBuffers (off-heap native memory, reused per frame)
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null

    init {
        initialize()
    }

    private fun initialize() {
        Log.i(TAG, "[Interpreter] Loading: $modelFileName (${mode.label})")

        // Detect shape
        val fd = context.assets.openFd(modelFileName)
        val channel = FileInputStream(fd.fileDescriptor).channel
        val modelBuf = channel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            fd.startOffset, fd.declaredLength
        )
        val shapeInterp = Interpreter(modelBuf)
        val inShape = shapeInterp.getInputTensor(0).shape()
        val outShape = shapeInterp.getOutputTensor(0).shape()
        shapeInterp.close()

        inputHeight = inShape[1]; inputWidth = inShape[2]
        when {
            outShape.size == 3 -> { outputHeight = outShape[1]; outputWidth = outShape[2] }
            outShape.size == 4 && outShape[3] == 1 -> { outputHeight = outShape[1]; outputWidth = outShape[2] }
            outShape.size == 4 && outShape[1] == 1 -> { outputHeight = outShape[2]; outputWidth = outShape[3] }
            else -> { outputHeight = outShape[1]; outputWidth = outShape[2] }
        }
        Log.i(TAG, "[Interpreter] Shape: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")

        // Set up GPU delegate with FP32 precision
        val options = Interpreter.Options()
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            val delegateOptions = GpuDelegate.Options()
            delegateOptions.setPrecisionLossAllowed(false)  // FP32 compute
            delegateOptions.setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL)
            delegateOptions.setInferencePreference(
                GpuDelegateFactory.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
            )
            gpuDelegate = GpuDelegate(delegateOptions)
            options.addDelegate(gpuDelegate!!)
            Log.i(TAG, "[Interpreter] GPU delegate added (OpenCL, FP32)")
        } else {
            options.setNumThreads(4)
            Log.w(TAG, "[Interpreter] GPU not supported, falling back to CPU")
        }

        // Create interpreter
        val fd2 = context.assets.openFd(modelFileName)
        val channel2 = FileInputStream(fd2.fileDescriptor).channel
        val modelBuf2 = channel2.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            fd2.startOffset, fd2.declaredLength
        )
        interpreter = Interpreter(modelBuf2, options)

        // Allocate direct ByteBuffers
        inputBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * 3 * 4)
            .order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(outputHeight * outputWidth * 4)
            .order(ByteOrder.nativeOrder())

        // Warmup: trigger GPU shader compilation
        inputBuffer!!.rewind(); outputBuffer!!.rewind()
        try { interpreter!!.run(inputBuffer, outputBuffer) } catch (_: Exception) {}

        Log.i(TAG, "[Interpreter] Ready")
    }

    override fun predict(bitmap: Bitmap): DepthResult {
        val interp = interpreter ?: throw IllegalStateException("Not initialized")
        val inBuf = inputBuffer ?: throw IllegalStateException("Not initialized")
        val outBuf = outputBuffer ?: throw IllegalStateException("Not initialized")

        // Preprocess: bitmap -> direct ByteBuffer (NHWC float32)
        val floatBuf = ImageUtils.bitmapToFloatBuffer(
            bitmap, inputWidth, inputHeight, nchw = false, normalize = true
        )
        inBuf.rewind()
        inBuf.asFloatBuffer().put(floatBuf)

        // Inference: GPU runs, output written directly into outBuf
        val startTime = System.nanoTime()
        outBuf.rewind()
        interp.run(inBuf, outBuf)
        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000

        // Read output from direct ByteBuffer (already in CPU memory, no readFloat!)
        outBuf.rewind()
        val fb = outBuf.asFloatBuffer()
        val outputArray = FloatArray(outputWidth * outputHeight)
        fb.get(outputArray)

        val grayscale = ImageUtils.depthFloatsToGrayscale(outputArray, outputWidth, outputHeight)
        val scaled = Bitmap.createScaledBitmap(grayscale, bitmap.width, bitmap.height, true)
        if (scaled !== grayscale) grayscale.recycle()

        return DepthResult(scaled, elapsedMs, mode)
    }

    override fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
        inputBuffer = null
        outputBuffer = null
    }
}
