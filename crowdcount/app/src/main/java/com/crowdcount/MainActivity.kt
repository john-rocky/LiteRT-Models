package com.crowdcount

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
import kotlin.math.roundToInt

private const val TAG = "DMCount"

class MainActivity : ComponentActivity() {

    private var counter: CrowdCounter? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: DensityOverlayView
    private lateinit var statusText: TextView
    private lateinit var countText: TextView

    private val O = CrowdCounter.OUT
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
        overlayView = DensityOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading DM-Count (GPU)..."
        }
        countText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(8f, 0f, 0f, 0xFF000000.toInt())
            textSize = 56f; gravity = Gravity.CENTER_HORIZONTAL; setPadding(0, 0, 0, 64)
        }
        root.addView(previewView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        root.addView(countText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM))
        setContentView(root)
        loadModel(); startCamera()
    }

    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                counter = CrowdCounter(this)
                statusText.post { statusText.text = "DM-Count GPU ready — crowd counting" }
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
        val c = counter ?: return
        val result = c.count(bmp)
        // Normalize the density map per frame for display (the count uses the raw sum).
        var maxV = 1e-5f
        for (v in result.density) if (v > maxV) maxV = v
        for (i in 0 until O * O) {
            val v = (result.density[i] / maxV).coerceIn(0f, 1f)
            val a = (v * 220).toInt()
            val g = ((1f - v) * 160).toInt()          // faint orange -> strong red
            ovPixels[i] = (a shl 24) or (0xFF shl 16) or (g shl 8)
        }
        ovBitmap.setPixels(ovPixels, 0, O, 0, 0, O, O)
        val bw = bmp.width; val bh = bmp.height
        overlayView.post { overlayView.setHeat(ovBitmap, bw, bh) }
        val people = result.count.roundToInt()
        countText.post { countText.text = "$people" }
        val fps = pipeline?.fps ?: 0
        statusText.post {
            statusText.text = "DM-Count GPU  |  $fps FPS  |  ${result.inferenceMs}ms  |  ~$people people"
        }
    }

    override fun onDestroy() {
        super.onDestroy(); pipeline?.close(); backgroundExecutor.shutdown(); counter?.close()
    }
}
