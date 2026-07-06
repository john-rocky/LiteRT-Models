package com.yolact

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
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

private const val TAG = "YOLACT"

class MainActivity : ComponentActivity() {

    private var segmenter: YolactSegmenter? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: InstanceOverlayView
    private lateinit var statusText: TextView

    private val S = YolactSegmenter.SIZE
    private val overlayPixels = IntArray(S * S)
    private val overlayBitmap = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)

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
        overlayView = InstanceOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading YOLACT (GPU)..."
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
                segmenter = YolactSegmenter(this)
                statusText.post { statusText.text = "YOLACT GPU ready — COCO instance segmentation" }
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
        val seg = segmenter ?: return
        val (instances, ms) = seg.segment(bmp)
        // composite colored instance masks into one SIZE×SIZE overlay bitmap
        java.util.Arrays.fill(overlayPixels, 0)
        for (ins in instances) {
            val c = Palette.color(ins.cls)
            val col = (0xA0 shl 24) or (c and 0x00FFFFFF)   // ~63% alpha
            val m = ins.mask
            for (i in m.indices) if (m[i]) overlayPixels[i] = col
        }
        overlayBitmap.setPixels(overlayPixels, 0, S, 0, 0, S, S)
        overlayView.post { overlayView.setResult(overlayBitmap, instances) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "YOLACT GPU  |  $fps FPS  |  ${ms}ms  |  ${instances.size} inst" }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        segmenter?.close()
    }
}
