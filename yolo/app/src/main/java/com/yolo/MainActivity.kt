package com.yolo

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
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "YOLO"

class MainActivity : ComponentActivity() {

    private var detector: ObjectDetector? = null
    private val executor = Executors.newSingleThreadExecutor()

    @Volatile
    private var isProcessing = false
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var currentModel = ""

    private val mat = Matrix()
    private val pnt = Paint(Paint.FILTER_BITMAP_FLAG)

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionCanvasView
    private lateinit var fpsText: TextView
    private lateinit var modelSpinner: Spinner

    private val models = arrayOf("yolo11n.tflite", "yolo26n.tflite")
    private val modelNames = arrayOf("YOLO11n", "YOLO26n")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) initCamera()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initCamera() {
        val root = FrameLayout(this)
        previewView = PreviewView(this)
        overlayView = DetectionCanvasView(this)

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 0)
        }

        fpsText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 18f
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
                if (model != currentModel) loadModel(model)
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        topBar.addView(modelSpinner)

        root.addView(previewView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(topBar, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP))
        setContentView(root)

        startCamera()
    }

    private fun loadModel(modelFile: String) {
        isProcessing = true
        fpsText.post { fpsText.text = "Loading $modelFile..." }
        executor.execute {
            try {
                detector?.close()
                detector = ObjectDetector(this, modelFile)
                currentModel = modelFile
                fpsText.post { fpsText.text = "$modelFile GPU ready" }
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}")
                fpsText.post { fpsText.text = "Failed: ${e.message}" }
            }
            isProcessing = false
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            @Suppress("DEPRECATION")
            val preview = Preview.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(executor) { proxy ->
                if (isProcessing || detector == null) {
                    proxy.close()
                    return@setAnalyzer
                }
                isProcessing = true

                val bmp = proxyToBitmap(proxy)
                proxy.close()

                val (dets, _) = detector!!.detect(bmp)
                val bmpW = bmp.width
                val bmpH = bmp.height
                bmp.recycle()

                overlayView.post { overlayView.setDetections(dets, bmpW, bmpH) }

                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000) {
                    val fps = frameCount
                    frameCount = 0
                    lastFpsTime = now
                    fpsText.post { fpsText.text = "$fps FPS | ${dets.size} detections" }
                }

                isProcessing = false
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun proxyToBitmap(proxy: ImageProxy): Bitmap {
        val plane = proxy.planes[0]
        val buf = plane.buffer
        val sw = proxy.width + (plane.rowStride - plane.pixelStride * proxy.width) / plane.pixelStride

        val src = Bitmap.createBitmap(sw, proxy.height, Bitmap.Config.ARGB_8888)
        buf.rewind()
        src.copyPixelsFromBuffer(buf)

        val rot = proxy.imageInfo.rotationDegrees.toFloat()
        if (rot == 0f && sw == proxy.width) return src

        val rw = if (rot == 90f || rot == 270f) proxy.height else proxy.width
        val rh = if (rot == 90f || rot == 270f) proxy.width else proxy.height
        val dst = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
        val c = Canvas(dst)
        mat.reset()
        mat.postRotate(rot, proxy.width / 2f, proxy.height / 2f)
        mat.postTranslate((rw - proxy.width) / 2f, (rh - proxy.height) / 2f)
        c.drawBitmap(src, mat, pnt)
        src.recycle()
        return dst
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        detector?.close()
    }
}
