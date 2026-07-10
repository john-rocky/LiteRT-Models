package com.portrait

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

private const val TAG = "Portrait"

class MainActivity : ComponentActivity() {

    private var sketcher: PortraitSketcher? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SketchOverlayView
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val launcher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted -> if (granted) initUi() }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) initUi() else launcher.launch(Manifest.permission.CAMERA)
    }

    private fun initUi() {
        val root = FrameLayout(this)
        previewView = PreviewView(this)
        overlayView = SketchOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFF000000.toInt()); textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading Portrait (GPU)..."
        }
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(previewView, FrameLayout.LayoutParams(280, 280, Gravity.BOTTOM or Gravity.END))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        setContentView(root)
        loadModel(); startCamera()
    }

    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                sketcher = PortraitSketcher(this)
                statusText.post { statusText.text = "Portrait GPU ready — center your face" }
                pipeline?.enabled = true
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}", e)
                statusText.post { statusText.text = "Load failed: ${e.message}" }
            }
        }
    }

    private fun startCamera() {
        pipeline = RealtimeCameraPipeline(activity = this, previewView = previewView) { bmp ->
            runInference(bmp)
        }.also { it.enabled = false; it.start(this) }
    }

    private fun runInference(bmp: Bitmap) {
        val s = sketcher ?: return
        val sq = minOf(bmp.width, bmp.height)
        val crop = Bitmap.createBitmap(bmp, (bmp.width - sq) / 2, (bmp.height - sq) / 2, sq, sq)
        val (sketch, ms) = s.sketch(crop)
        crop.recycle()
        overlayView.post { overlayView.setSketch(sketch) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "Portrait Sketch GPU  |  $fps FPS  |  ${ms}ms" }
    }

    override fun onDestroy() {
        super.onDestroy(); pipeline?.close(); backgroundExecutor.shutdown(); sketcher?.close()
    }
}
