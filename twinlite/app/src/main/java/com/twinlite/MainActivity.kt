package com.twinlite

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

private const val TAG = "TwinLiteNet"

class MainActivity : ComponentActivity() {

    private var segmenter: TwinLiteSegmenter? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SegOverlayView
    private lateinit var statusText: TextView

    private val W = TwinLiteSegmenter.W; private val H = TwinLiteSegmenter.H
    private val ovPixels = IntArray(W * H)
    private val ovBitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
    private val GREEN = (0x88 shl 24) or 0x28E05A   // drivable area
    private val RED = (0xFF shl 24) or 0xFF3030     // lane line

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
            textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading TwinLiteNet (GPU)..."
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
                segmenter = TwinLiteSegmenter(this)
                statusText.post { statusText.text = "TwinLiteNet GPU ready — drivable area + lanes" }
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
        val (da, ll, ms) = s.segment(bmp)
        for (i in 0 until W * H) {
            ovPixels[i] = when {
                ll[i].toInt() == 1 -> RED
                da[i].toInt() == 1 -> GREEN
                else -> 0
            }
        }
        ovBitmap.setPixels(ovPixels, 0, W, 0, 0, W, H)
        val bw = bmp.width; val bh = bmp.height
        overlayView.post { overlayView.setOverlay(ovBitmap, bw, bh) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "TwinLiteNet GPU  |  $fps FPS  |  ${ms}ms  |  drivable + lanes" }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        segmenter?.close()
    }
}
