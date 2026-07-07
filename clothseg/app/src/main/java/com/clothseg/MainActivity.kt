package com.clothseg

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

private const val TAG = "ClothSeg"

class MainActivity : ComponentActivity() {

    private var segmenter: ClothSegmenter? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SegOverlayView
    private lateinit var statusText: TextView

    private val O = ClothSegmenter.OUT
    private val ovPixels = IntArray(O * O)
    private val ovBitmap = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    // 0 bg (transparent), 1 upper (cyan), 2 lower (orange), 3 full (magenta)
    private val COLORS = intArrayOf(0, (0xA0 shl 24) or 0x00C8FF, (0xA0 shl 24) or 0xFF9600, (0xA0 shl 24) or 0xE600C8)

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
        overlayView = SegOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading Cloth Seg (GPU)..."
        }
        root.addView(previewView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        setContentView(root)
        loadModel()
        startCamera()
    }

    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                segmenter = ClothSegmenter(this)
                statusText.post { statusText.text = "Cloth Seg GPU ready — upper/lower/full clothing" }
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
        val s = segmenter ?: return
        val (cls, ms) = s.segment(bmp)
        for (i in 0 until O * O) ovPixels[i] = COLORS[cls[i].toInt()]
        ovBitmap.setPixels(ovPixels, 0, O, 0, 0, O, O)
        overlayView.post { overlayView.setOverlay(ovBitmap) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "Cloth Seg GPU  |  $fps FPS  |  ${ms}ms  |  upper=cyan lower=orange" }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        segmenter?.close()
    }
}
