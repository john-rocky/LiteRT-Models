package com.yolox

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "YOLOX"
private const val MODEL = "yolox_nano.tflite"

class MainActivity : ComponentActivity() {

    private var detector: YoloxDetector? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DetectionCanvasView
    private lateinit var fpsText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission(),
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
        fpsText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 18f
            setPadding(24, 48, 24, 0)
            text = "Compiling GPU model…"
        }

        root.addView(
            previewView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        root.addView(
            overlayView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        root.addView(
            fpsText,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP,
            ),
        )
        setContentView(root)

        startCamera()
        loadModel()
    }

    private fun loadModel() {
        // GPU compilation takes ~15s on first launch — keep it off the main thread.
        backgroundExecutor.execute {
            try {
                detector = YoloxDetector(this)
                fpsText.post { fpsText.text = "GPU ready" }
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
            it.enabled = false  // wait until the model is loaded
            it.start(this)
        }
    }

    private fun runInference(bmp: Bitmap) {
        val det = detector ?: return
        val dets = det.detect(bmp)
        overlayView.post { overlayView.setDetections(dets, bmp.width, bmp.height) }
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
