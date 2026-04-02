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
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
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
            FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START
        ).apply { setMargins(32, 48, 0, 0) })

        setContentView(frame)

        try {
            estimator = DepthEstimator(this, MODEL_FILE)
            depthDisplayBitmap = Bitmap.createBitmap(
                estimator!!.inputWidth, estimator!!.inputHeight, Bitmap.Config.ARGB_8888)
        } catch (e: Exception) {
            Log.e(TAG, "Model init failed", e)
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
        val est = estimator ?: return
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(est.inputWidth, est.inputHeight))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frameCount = 0; var lastFpsTime = System.currentTimeMillis()

            analysis.setAnalyzer(cameraExecutor) { proxy ->
                if (isProcessing || depthDisplayBitmap == null) { proxy.close(); return@setAnalyzer }
                isProcessing = true
                val bmp = proxyToBitmap(proxy, est.inputWidth, est.inputHeight)
                proxy.close()
                est.predict(bmp, depthDisplayBitmap!!)
                bmp.recycle()
                depthImageView.post { depthImageView.setImageBitmap(depthDisplayBitmap) }
                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000) {
                    val fps = frameCount; frameCount = 0; lastFpsTime = now
                    fpsText.post { fpsText.text = "$fps FPS" }
                }
                isProcessing = false
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private val mat = Matrix(); private val pnt = Paint(Paint.FILTER_BITMAP_FLAG)
    private fun proxyToBitmap(proxy: ImageProxy, tw: Int, th: Int): Bitmap {
        val p = proxy.planes[0]; val buf = p.buffer
        val sw = proxy.width + (p.rowStride - p.pixelStride * proxy.width) / p.pixelStride
        val src = Bitmap.createBitmap(sw, proxy.height, Bitmap.Config.ARGB_8888)
        buf.rewind(); src.copyPixelsFromBuffer(buf)
        val rot = proxy.imageInfo.rotationDegrees.toFloat()
        val dst = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888)
        val c = Canvas(dst); mat.reset()
        if (rot != 0f) {
            val rw = if (rot == 90f || rot == 270f) proxy.height else proxy.width
            val rh = if (rot == 90f || rot == 270f) proxy.width else proxy.height
            mat.postRotate(rot, proxy.width / 2f, proxy.height / 2f)
            mat.postTranslate((rw - proxy.width) / 2f, (rh - proxy.height) / 2f)
            mat.postScale(tw.toFloat() / rw, th.toFloat() / rh)
        } else mat.setScale(tw.toFloat() / proxy.width, th.toFloat() / proxy.height)
        c.drawBitmap(src, mat, pnt); src.recycle(); return dst
    }

    override fun onDestroy() { super.onDestroy(); estimator?.close(); cameraExecutor.shutdown() }
}
