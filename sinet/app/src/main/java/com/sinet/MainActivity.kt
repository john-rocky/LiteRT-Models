package com.sinet

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

private const val TAG = "SINet"

class MainActivity : ComponentActivity() {

    private var detector: CamouflageDetector? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: CamoOverlayView
    private lateinit var statusText: TextView

    private val O = CamouflageDetector.OUT
    private val ovPixels = IntArray(O * O)
    private val ovBitmap = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)

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
        overlayView = CamoOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading SINet (GPU)..."
        }
        root.addView(previewView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        setContentView(root)
        loadModel(); startCamera()
    }

    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                detector = CamouflageDetector(this)
                statusText.post { statusText.text = "SINet GPU ready — camouflaged object detection" }
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
        val d = detector ?: return
        val (heat, ms) = d.detect(bmp)
        for (i in 0 until O * O) {
            val a = (heat[i] * 170).toInt().coerceIn(0, 255)   // semi-transparent red highlight
            ovPixels[i] = (a shl 24) or 0xFF3020
        }
        ovBitmap.setPixels(ovPixels, 0, O, 0, 0, O, O)
        val bw = bmp.width; val bh = bmp.height
        overlayView.post { overlayView.setHeat(ovBitmap, bw, bh) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "SINet GPU  |  $fps FPS  |  ${ms}ms  |  concealed objects" }
    }

    override fun onDestroy() {
        super.onDestroy(); pipeline?.close(); backgroundExecutor.shutdown(); detector?.close()
    }
}
