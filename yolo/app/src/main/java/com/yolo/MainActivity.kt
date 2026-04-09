package com.yolo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "YOLO"

class MainActivity : ComponentActivity() {

    private var detector: ObjectDetector? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private var currentModel = ""

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
        pipeline?.enabled = false
        fpsText.post { fpsText.text = "Loading $modelFile..." }
        backgroundExecutor.execute {
            try {
                detector?.close()
                detector = ObjectDetector(this, modelFile)
                currentModel = modelFile
                fpsText.post { fpsText.text = "$modelFile GPU ready" }
                pipeline?.enabled = true
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}")
                fpsText.post { fpsText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun startCamera() {
        pipeline = RealtimeCameraPipeline(
            activity = this,
            previewView = previewView,
        ) { bmp -> runInference(bmp) }.also {
            it.enabled = false  // wait until model is loaded
            it.start(this)
        }
    }

    private fun runInference(bmp: Bitmap) {
        val det = detector ?: return
        val (dets, _) = det.detect(bmp)
        val bmpW = bmp.width
        val bmpH = bmp.height

        overlayView.post { overlayView.setDetections(dets, bmpW, bmpH) }

        val fps = pipeline?.fps ?: 0
        fpsText.post { fpsText.text = "$fps FPS | ${dets.size} detections" }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        detector?.close()
    }
}
