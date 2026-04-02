package com.depthanything.sample

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Depth estimator using the OLD Interpreter API with GpuDelegate.
 *
 * Why this is fast:
 * - Interpreter.run(ByteBuffer, ByteBuffer) copies GPU output directly into a
 *   pre-allocated direct ByteBuffer via JNI (libtensorflowlite_jni.so).
 *   The readback takes ~1-5ms vs 230-340ms with CompiledModel's TensorBuffer.readFloat().
 * - The GpuDelegate uses OpenCL (same backend as CompiledModel's LITERT_CL).
 * - Direct ByteBuffers are off-heap native memory, avoiding Java array allocation.
 *
 * Requires litert:1.4.x + litert-gpu:1.4.x + litert-gpu-api:1.4.x
 */
class InterpreterDepthEstimator(context: Context, modelFileName: String) : AutoCloseable {

    companion object {
        private const val TAG = "DepthAnything"
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null

    val inputWidth: Int
    val inputHeight: Int
    private val outputWidth: Int
    private val outputHeight: Int

    // Pre-allocated direct ByteBuffers for zero-allocation inference
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer

    // Pre-allocated pixel/float arrays
    private val inputPixels: IntArray
    private val outputPixels: IntArray
    private val resizedBitmap: Bitmap
    private val depthBitmap: Bitmap
    private val scaleMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    // Pre-baked Inferno LUT
    private val infernoLut = IntArray(256) { Colormap.inferno(255 - it) }

    init {
        Log.i(TAG, "[Interpreter] Loading model: $modelFileName")

        // Detect shape using a temporary interpreter
        val fd = context.assets.openFd(modelFileName)
        val channel = FileInputStream(fd.fileDescriptor).channel
        val modelBuffer = channel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            fd.startOffset, fd.declaredLength
        )
        val shapeInterp = Interpreter(modelBuffer)
        val inShape = shapeInterp.getInputTensor(0).shape()
        val outShape = shapeInterp.getOutputTensor(0).shape()
        shapeInterp.close()

        inputHeight = inShape[1]; inputWidth = inShape[2]
        outputHeight = outShape[1]; outputWidth = outShape[2]
        Log.i(TAG, "[Interpreter] Shape: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")

        // Set up GPU delegate with FP32 precision
        val options = Interpreter.Options()
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            val delegateOptions = GpuDelegate.Options()
            // setPrecisionLossAllowed(false) => FP32 compute, same as CompiledModel FP32
            delegateOptions.setPrecisionLossAllowed(false)
            // Force OpenCL backend for maximum compatibility with Tensor G3
            delegateOptions.setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENCL)
            // SUSTAINED_SPEED for real-time pipeline
            delegateOptions.setInferencePreference(
                GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
            )
            gpuDelegate = GpuDelegate(delegateOptions)
            options.addDelegate(gpuDelegate!!)
            Log.i(TAG, "[Interpreter] GPU delegate added (OpenCL, FP32)")
        } else {
            options.setNumThreads(4)
            Log.w(TAG, "[Interpreter] GPU not supported, falling back to CPU")
        }

        // Re-map the model buffer (the previous one may have been consumed)
        val fd2 = context.assets.openFd(modelFileName)
        val channel2 = FileInputStream(fd2.fileDescriptor).channel
        val modelBuffer2 = channel2.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            fd2.startOffset, fd2.declaredLength
        )

        interpreter = Interpreter(modelBuffer2, options)
        Log.i(TAG, "[Interpreter] Interpreter created successfully")

        // Allocate direct ByteBuffers (off-heap, native memory)
        // Input: [1, H, W, 3] floats = H * W * 3 * 4 bytes
        val inputBytes = inputHeight * inputWidth * 3 * 4
        inputBuffer = ByteBuffer.allocateDirect(inputBytes).order(ByteOrder.nativeOrder())

        // Output: [1, H, W] or [1, H, W, 1] floats
        val outputElements = outputHeight * outputWidth
        val outputBytes = outputElements * 4
        outputBuffer = ByteBuffer.allocateDirect(outputBytes).order(ByteOrder.nativeOrder())

        // Pre-allocate pixel arrays
        inputPixels = IntArray(inputWidth * inputHeight)
        outputPixels = IntArray(outputWidth * outputHeight)
        resizedBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        depthBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)

        // Warm up: run one inference to trigger GPU shader compilation
        inputBuffer.rewind()
        outputBuffer.rewind()
        try {
            interpreter.run(inputBuffer, outputBuffer)
            Log.i(TAG, "[Interpreter] Warmup inference completed")
        } catch (e: Exception) {
            Log.w(TAG, "[Interpreter] Warmup failed (non-fatal): ${e.message}")
        }

        Log.i(TAG, "[Interpreter] Ready: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")
    }

    /**
     * Run depth estimation. Writes colored depth into [outputBitmap].
     * Returns total time (preprocess + inference + postprocess) in ms.
     */
    fun predict(inputBitmap: Bitmap, outputBitmap: Bitmap): Long {
        var t = System.nanoTime()

        // Preprocess: resize + normalize into direct ByteBuffer
        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(
            inputWidth.toFloat() / inputBitmap.width,
            inputHeight.toFloat() / inputBitmap.height
        )
        canvas.drawBitmap(inputBitmap, scaleMatrix, paint)
        resizedBitmap.getPixels(inputPixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        inputBuffer.rewind()
        val fb = inputBuffer.asFloatBuffer()
        for (pixel in inputPixels) {
            fb.put((Color.red(pixel) / 255f - MEAN[0]) / STD[0])
            fb.put((Color.green(pixel) / 255f - MEAN[1]) / STD[1])
            fb.put((Color.blue(pixel) / 255f - MEAN[2]) / STD[2])
        }
        val preMs = (System.nanoTime() - t) / 1_000_000

        // Inference: GPU runs the model, output is written directly into outputBuffer
        t = System.nanoTime()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        val infMs = (System.nanoTime() - t) / 1_000_000

        // Postprocess: read floats from direct ByteBuffer (already in CPU memory)
        t = System.nanoTime()
        outputBuffer.rewind()
        val outFb = outputBuffer.asFloatBuffer()

        // Min-max normalization + colormap
        var min = Float.MAX_VALUE; var max = Float.MIN_VALUE
        val count = outputWidth * outputHeight
        // First pass: find min/max
        for (i in 0 until count) {
            val v = outFb.get(i)
            if (v < min) min = v
            if (v > max) max = v
        }
        val scale = 255f / (max - min).coerceAtLeast(1e-6f)

        // Second pass: colormap
        for (i in 0 until count) {
            outputPixels[i] = infernoLut[((outFb.get(i) - min) * scale).toInt().coerceIn(0, 255)]
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

        val cmapMs = (t2 - t) / 1_000_000
        val drawMs = (t3 - t2) / 1_000_000
        val postMs = (t3 - t) / 1_000_000

        Log.i(TAG, "[Interp] pre=${preMs}ms inf=${infMs}ms post=${postMs}ms (cmap=${cmapMs} draw=${drawMs})")

        return preMs + infMs + postMs
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
        resizedBitmap.recycle()
        depthBitmap.recycle()
    }
}
