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
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "DepthAnything"
private const val REQUEST_CAMERA = 100

class MainActivity : ComponentActivity() {

    private lateinit var depthImageView: ImageView
    private lateinit var fpsText: TextView
    private lateinit var modelSpinner: Spinner
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var estimator: InterpreterDepthEstimator? = null
    private var depthDisplayBitmap: Bitmap? = null
    private var isProcessing = false
    private var currentModel = ""

    // Available models (must exist in assets/)
    private val models = arrayOf(
        "depth_anything_v2_keras.tflite",
        "depth_anything_v2_keras_fp16w.tflite",
        "depth_anything_v2_keras_392x518.tflite",
        "depth_anything_v2_keras_392x518_fp16w.tflite",
        "depth_anything_v2_nhwc_clamped.tflite",
        "depth_anything_v2_nhwc_clamped_fp16w.tflite",
    )

    // Short display names
    private val modelNames = arrayOf(
        "Keras 518 FP32 (99MB)",
        "Keras 518 FP16w (50MB)",
        "Keras 392 FP32 (98MB)",
        "Keras 392 FP16w (49MB)",
        "Clamped 392 FP32 (99MB)",
        "Clamped 392 FP16w (50MB)",
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val frame = FrameLayout(this)

        depthImageView = ImageView(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            scaleType = ImageView.ScaleType.CENTER_CROP
        }
        frame.addView(depthImageView)

        // Top bar: FPS + model selector
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 0)
        }

        fpsText = TextView(this).apply {
            textSize = 20f
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 2f, 2f, 0xFF000000.toInt())
            text = "Loading..."
        }
        topBar.addView(fpsText)

        modelSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item, modelNames)
            setBackgroundColor(0x88000000.toInt())
        }
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                val model = models[pos]
                if (model != currentModel) {
                    loadModel(model)
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        topBar.addView(modelSpinner)

        frame.addView(topBar, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP))

        setContentView(frame)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else ActivityCompat.requestPermissions(this,
            arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
    }

    private fun loadModel(modelFile: String) {
        isProcessing = true  // pause inference
        runOnUiThread { fpsText.text = "Loading..." }

        cameraExecutor.execute {
            try {
                // Close old estimator (recycles internal bitmaps)
                val oldEstimator = estimator
                estimator = null
                oldEstimator?.close()

                // Create new estimator and bitmap
                val newEstimator = InterpreterDepthEstimator(this, modelFile)
                val newBitmap = Bitmap.createBitmap(
                    newEstimator.inputWidth, newEstimator.inputHeight, Bitmap.Config.ARGB_8888)

                // Swap atomically
                estimator = newEstimator
                depthDisplayBitmap = newBitmap
                currentModel = modelFile

                runOnUiThread {
                    fpsText.text = "Ready: ${newEstimator.inputWidth}x${newEstimator.inputHeight}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed: $modelFile", e)
                runOnUiThread { fpsText.text = "Failed: ${e.message}" }
            }
            isProcessing = false
        }
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
                .setTargetResolution(android.util.Size(518, 518))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frameCount = 0; var lastFpsTime = System.currentTimeMillis()

            analysis.setAnalyzer(cameraExecutor) { proxy ->
                if (isProcessing || estimator == null || depthDisplayBitmap == null) {
                    proxy.close(); return@setAnalyzer
                }
                isProcessing = true
                val bmp = proxyToBitmap(proxy, estimator!!.inputWidth, estimator!!.inputHeight)
                proxy.close()
                estimator!!.predict(bmp, depthDisplayBitmap!!)
                bmp.recycle()
                depthImageView.post { depthImageView.setImageBitmap(depthDisplayBitmap) }
                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000) {
                    val fps = frameCount; frameCount = 0; lastFpsTime = now
                    fpsText.post { fpsText.text = "$fps FPS — $currentModel" }
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
