package com.depthanything.sample

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "DepthAnything"
private const val MODEL_FILE = "depth_anything_v2_keras.tflite"
private const val REQUEST_CAMERA = 100

class MainActivity : ComponentActivity() {

    private lateinit var depthImageView: ImageView
    private lateinit var fpsText: TextView
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var estimator: DepthEstimator? = null
    private var depthDisplayBitmap: Bitmap? = null
    private var isProcessing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val frame = FrameLayout(this)

        depthImageView = ImageView(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            scaleType = ImageView.ScaleType.CENTER_CROP
        }
        frame.addView(depthImageView)

        fpsText = TextView(this).apply {
            textSize = 24f
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 2f, 2f, 0xFF000000.toInt())
            text = "Loading..."
        }
        frame.addView(fpsText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START
        ).apply { setMargins(32, 48, 0, 0) })

        setContentView(frame)

        try {
            estimator = DepthEstimator(this, MODEL_FILE)
            depthDisplayBitmap = Bitmap.createBitmap(
                estimator!!.inputWidth, estimator!!.inputHeight,
                Bitmap.Config.ARGB_8888
            )
        } catch (e: Exception) {
            Log.e(TAG, "Model init failed", e)
            fpsText.text = "Model load failed: ${e.message}"
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    private fun startCamera() {
        val est = estimator ?: return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            @Suppress("DEPRECATION")
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(est.inputWidth, est.inputHeight))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frameCount = 0
            var lastFpsTime = System.currentTimeMillis()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                if (isProcessing || depthDisplayBitmap == null) {
                    imageProxy.close()
                    return@setAnalyzer
                }
                isProcessing = true

                val cameraBitmap = imageProxyToBitmap(imageProxy, est.inputWidth, est.inputHeight)
                imageProxy.close()

                est.predict(cameraBitmap, depthDisplayBitmap!!)
                cameraBitmap.recycle()

                depthImageView.post {
                    depthImageView.setImageBitmap(depthDisplayBitmap)
                }

                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000) {
                    val fps = frameCount
                    frameCount = 0
                    lastFpsTime = now
                    fpsText.post { fpsText.text = "$fps FPS" }
                }
                isProcessing = false
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, imageAnalysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private val tempMatrix = Matrix()
    private val tempPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    private fun imageProxyToBitmap(imageProxy: ImageProxy, targetW: Int, targetH: Int): Bitmap {
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * imageProxy.width
        val srcW = imageProxy.width + rowPadding / pixelStride

        val src = Bitmap.createBitmap(srcW, imageProxy.height, Bitmap.Config.ARGB_8888)
        buffer.rewind()
        src.copyPixelsFromBuffer(buffer)

        val rotation = imageProxy.imageInfo.rotationDegrees.toFloat()
        val dst = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(dst)
        tempMatrix.reset()

        if (rotation != 0f) {
            val rotW = if (rotation == 90f || rotation == 270f) imageProxy.height else imageProxy.width
            val rotH = if (rotation == 90f || rotation == 270f) imageProxy.width else imageProxy.height
            tempMatrix.postRotate(rotation, imageProxy.width / 2f, imageProxy.height / 2f)
            tempMatrix.postTranslate((rotW - imageProxy.width) / 2f, (rotH - imageProxy.height) / 2f)
            tempMatrix.postScale(targetW.toFloat() / rotW, targetH.toFloat() / rotH)
        } else {
            tempMatrix.setScale(targetW.toFloat() / imageProxy.width, targetH.toFloat() / imageProxy.height)
        }

        canvas.drawBitmap(src, tempMatrix, tempPaint)
        src.recycle()
        return dst
    }

    override fun onDestroy() {
        super.onDestroy()
        estimator?.close()
        cameraExecutor.shutdown()
    }
}
