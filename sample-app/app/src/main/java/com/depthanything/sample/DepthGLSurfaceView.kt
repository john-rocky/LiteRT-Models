package com.depthanything.sample

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.opengl.EGL14
import android.opengl.GLSurfaceView
import android.util.Log
import androidx.camera.core.ImageProxy
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.concurrent.withLock

private const val TAG = "DepthGLSurface"
private const val MODEL_FILE = "depth_anything_v2_keras.tflite"

/**
 * GLSurfaceView that runs the C++ NDK zero-copy depth pipeline.
 * Camera frames are preprocessed on CPU, inference + colormap + render on GPU.
 */
class DepthGLSurfaceView(context: Context) : GLSurfaceView(context) {

    private val pipeline = NativeDepthPipeline()
    private val renderer = DepthRenderer()
    private val processing = AtomicBoolean(false)

    // Frame data (protected by lock)
    private val frameLock = ReentrantLock()
    private var pendingPixels: IntArray? = null
    private var pendingWidth = 0
    private var pendingHeight = 0
    private var pendingRotation = 0
    private var hasNewFrame = false

    // Model dimensions (from legacy Interpreter shape detection)
    var inputWidth = 518; private set
    var inputHeight = 518; private set
    var outputWidth = 518; private set
    var outputHeight = 518; private set

    var fps = 0; private set
    var isZeroCopy = false; private set
    var initError: String? = null; private set

    var onFpsUpdate: ((Int, Boolean) -> Unit)? = null

    init {
        setEGLContextClientVersion(3)
        setZOrderOnTop(true)
        holder.setFormat(android.graphics.PixelFormat.OPAQUE)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    /**
     * Detect model input/output shape using legacy TFLite Interpreter.
     * Must be called before the GL surface is created.
     */
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
            Log.i(TAG, "Model shape: ${inputWidth}x${inputHeight} -> ${outputWidth}x${outputHeight}")
        } catch (e: Exception) {
            Log.e(TAG, "Shape detection failed", e)
        }
    }

    /**
     * Submit a camera frame for processing. Called from CameraX analyzer thread.
     * Drops frames if the pipeline is still busy.
     */
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

        // Decode RGBA_8888 into IntArray
        val src = Bitmap.createBitmap(srcW, h, Bitmap.Config.ARGB_8888)
        buffer.rewind()
        src.copyPixelsFromBuffer(buffer)
        imageProxy.close()

        // Extract pixels from the valid region
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
    }

    private inner class DepthRenderer : Renderer {
        private var frameCount = 0
        private var lastFpsTime = System.currentTimeMillis()

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            Log.i(TAG, "Surface created, initializing native pipeline")

            val ctx = context
            val ok = pipeline.nativeInit(
                ctx.assets, MODEL_FILE,
                inputWidth, inputHeight, outputWidth, outputHeight
            )

            if (!ok) {
                initError = "Native pipeline init failed"
                Log.e(TAG, initError!!)
                return
            }

            isZeroCopy = pipeline.nativeIsZeroCopy()
            Log.i(TAG, "Pipeline ready (zero-copy: $isZeroCopy)")
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            // Viewport is set in native render
        }

        override fun onDrawFrame(gl: GL10?) {
            if (!pipeline.nativeIsInitialized()) return

            // Check for new frame
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

            // FPS counter
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
