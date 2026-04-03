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
private const val TARGET_SIZE = 518
private const val REQUEST_CAMERA = 100

class MainActivity : ComponentActivity() {

    private lateinit var depthImageView: ImageView
    private lateinit var fpsText: TextView
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var pipeline: NcnnDepthPipeline? = null
    private var depthBitmap: Bitmap? = null
    private var isProcessing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val frame = FrameLayout(this)
        depthImageView = ImageView(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            scaleType = ImageView.ScaleType.CENTER_CROP
        }
        frame.addView(depthImageView)

        fpsText = TextView(this).apply {
            textSize = 24f
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 2f, 2f, 0xFF000000.toInt())
            text = "Loading NCNN..."
        }
        frame.addView(fpsText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START
        ).apply { setMargins(32, 48, 0, 0) })

        setContentView(frame)

        try {
            pipeline = NcnnDepthPipeline()
            val ok = pipeline!!.nativeInit(
                assets, "dptv2_s.param", "dptv2_s.bin",
                TARGET_SIZE, true  // useGpu=true for Vulkan
            )
            if (!ok) throw RuntimeException("NCNN init failed")
            depthBitmap = Bitmap.createBitmap(TARGET_SIZE, TARGET_SIZE, Bitmap.Config.ARGB_8888)
            val mode = if (pipeline!!.nativeIsVulkan()) "Vulkan" else "CPU"
            fpsText.text = "Ready ($mode)"
        } catch (e: Exception) {
            Log.e(TAG, "Init failed", e)
            fpsText.text = "Failed: ${e.message}"
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else ActivityCompat.requestPermissions(this,
            arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA && grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) startCamera()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(TARGET_SIZE, TARGET_SIZE))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frameCount = 0; var lastFpsTime = System.currentTimeMillis()

            analysis.setAnalyzer(cameraExecutor) { proxy ->
                if (isProcessing || depthBitmap == null) { proxy.close(); return@setAnalyzer }
                isProcessing = true

                val w = proxy.width; val h = proxy.height
                val rot = proxy.imageInfo.rotationDegrees
                val plane = proxy.planes[0]; val buffer = plane.buffer
                val pixelStride = plane.pixelStride; val rowStride = plane.rowStride
                val srcW = w + (rowStride - pixelStride * w) / pixelStride
                val src = Bitmap.createBitmap(srcW, h, Bitmap.Config.ARGB_8888)
                buffer.rewind(); src.copyPixelsFromBuffer(buffer); proxy.close()
                val pixels = IntArray(w * h)
                src.getPixels(pixels, 0, w, 0, 0, w, h); src.recycle()

                val result = pipeline?.nativeInfer(pixels, w, h, rot)
                if (result != null && result.size == TARGET_SIZE * TARGET_SIZE) {
                    depthBitmap!!.setPixels(result, 0, TARGET_SIZE, 0, 0, TARGET_SIZE, TARGET_SIZE)
                    depthImageView.post { depthImageView.setImageBitmap(depthBitmap) }
                }

                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000) {
                    val fps = frameCount; frameCount = 0; lastFpsTime = now
                    val mode = if (pipeline?.nativeIsVulkan() == true) "Vulkan" else "CPU"
                    fpsText.post { fpsText.text = "$fps FPS ($mode)" }
                }
                isProcessing = false
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() { super.onDestroy(); pipeline?.close(); cameraExecutor.shutdown() }
}
