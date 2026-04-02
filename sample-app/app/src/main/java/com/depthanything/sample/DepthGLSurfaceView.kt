package com.depthanything.sample

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLSurfaceView
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.concurrent.withLock

private const val TAG = "DepthGLSurface"
private const val MODEL_FILE = "depth_anything_v2_keras.tflite"

class DepthGLSurfaceView(context: Context) : GLSurfaceView(context) {

    private val pipeline = NativeDepthPipeline()
    private val renderer = DepthRenderer()
    private val processing = AtomicBoolean(false)

    private val frameLock = ReentrantLock()
    private var pendingPixels: IntArray? = null
    private var pendingWidth = 0
    private var pendingHeight = 0
    private var pendingRotation = 0
    private var hasNewFrame = false

    var inputWidth = 518; private set
    var inputHeight = 518; private set
    var outputWidth = 518; private set
    var outputHeight = 518; private set

    var fps = 0; private set
    var isZeroCopy = false; private set
    var initError: String? = null; private set
    var onFpsUpdate: ((Int, Boolean) -> Unit)? = null

    // Kotlin CompiledModel (FP32 GPU) — keeps model alive
    private var compiledModel: CompiledModel? = null
    private var environment: com.google.ai.edge.litert.Environment? = null

    init {
        setEGLContextClientVersion(3)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun detectModelShape(context: Context) {
        try {
            val fd = context.assets.openFd(MODEL_FILE)
            val channel = java.io.FileInputStream(fd.fileDescriptor).channel
            val buf = channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                fd.startOffset, fd.declaredLength
            )
            val interp = org.tensorflow.lite.Interpreter(buf)
            val inShape = interp.getInputTensor(0).shape()
            val outShape = interp.getOutputTensor(0).shape()
            interp.close()
            inputHeight = inShape[1]; inputWidth = inShape[2]
            outputHeight = outShape[1]; outputWidth = outShape[2]
        } catch (e: Exception) {
            Log.e(TAG, "Shape detection failed", e)
        }
    }

    fun submitFrame(imageProxy: ImageProxy) {
        if (processing.get()) {
            imageProxy.close()
            return
        }
        val w = imageProxy.width
        val h = imageProxy.height
        val rotation = imageProxy.imageInfo.rotationDegrees
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * w
        val srcW = w + rowPadding / pixelStride

        val src = Bitmap.createBitmap(srcW, h, Bitmap.Config.ARGB_8888)
        buffer.rewind()
        src.copyPixelsFromBuffer(buffer)
        imageProxy.close()

        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        src.recycle()

        frameLock.withLock {
            pendingPixels = pixels
            pendingWidth = w
            pendingHeight = h
            pendingRotation = rotation
            hasNewFrame = true
        }
        requestRender()
    }

    fun destroy() {
        pipeline.close()
        compiledModel?.close()
        environment?.close()
    }

    private inner class DepthRenderer : Renderer {
        private var frameCount = 0
        private var lastFpsTime = System.currentTimeMillis()

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            Log.i(TAG, "Surface created — initializing pipeline")

            // 1. Init GL resources (on GL thread)
            val glOk = pipeline.nativeInitGl(inputWidth, inputHeight, outputWidth, outputHeight)
            if (!glOk) {
                initError = "GL init failed"
                Log.e(TAG, initError!!)
                return
            }

            // 2. Create Kotlin CompiledModel with FP32 GPU (proven to work)
            try {
                environment = com.google.ai.edge.litert.Environment.create()
                val options = CompiledModel.Options(Accelerator.GPU)
                try {
                    options.gpuOptions = CompiledModel.GpuOptions(
                        null, null, null,
                        CompiledModel.GpuOptions.Precision.FP32,
                        null, null, null, null, null, null, null, null, null, null, null
                    )
                } catch (_: Exception) {}

                compiledModel = CompiledModel.create(
                    context.assets, MODEL_FILE, options, environment
                )
                Log.i(TAG, "Kotlin CompiledModel created with FP32 GPU")
            } catch (e: Exception) {
                initError = "CompiledModel failed: ${e.message}"
                Log.e(TAG, initError!!, e)
                return
            }

            // 3. Extract native handles and pass to C++
            val envHandle = NativeDepthPipeline.getNativeHandle(environment!!)
            val modelHandle = NativeDepthPipeline.getNativeHandle(compiledModel!!)
            Log.i(TAG, "Native handles: env=$envHandle model=$modelHandle")

            val handleOk = pipeline.nativeSetHandles(envHandle, modelHandle)
            if (!handleOk) {
                initError = "nativeSetHandles failed"
                Log.e(TAG, initError!!)
                return
            }

            isZeroCopy = pipeline.nativeIsZeroCopy()
            Log.i(TAG, "Pipeline ready (zero-copy: $isZeroCopy)")
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {}

        override fun onDrawFrame(gl: GL10?) {
            if (!pipeline.nativeIsInitialized()) return

            var pixels: IntArray? = null
            var w = 0; var h = 0; var rot = 0

            frameLock.withLock {
                if (hasNewFrame) {
                    pixels = pendingPixels
                    w = pendingWidth
                    h = pendingHeight
                    rot = pendingRotation
                    hasNewFrame = false
                }
            }

            if (pixels != null) {
                processing.set(true)
                pipeline.nativeProcessFrame(pixels!!, w, h, rot)
                processing.set(false)
            }

            pipeline.nativeRender(width, height)

            frameCount++
            val now = System.currentTimeMillis()
            if (now - lastFpsTime >= 1000) {
                fps = frameCount
                isZeroCopy = pipeline.nativeIsZeroCopy()
                frameCount = 0
                lastFpsTime = now
                onFpsUpdate?.invoke(fps, isZeroCopy)
            }
        }
    }
}
