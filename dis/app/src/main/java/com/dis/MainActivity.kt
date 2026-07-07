package com.dis

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
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

private const val TAG = "DIS"

class MainActivity : ComponentActivity() {

    private var segmenter: CutoutSegmenter? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: CutoutOverlayView
    private lateinit var statusText: TextView

    private val O = CutoutSegmenter.OUT
    private val compPixels = IntArray(O * O)
    private val fgPixels = IntArray(O * O)
    private val compBitmap = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    private val fgScaled = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    private val BG_R = 240; private val BG_G = 240; private val BG_B = 245   // studio white bg

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
        overlayView = CutoutOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading DIS (GPU)..."
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
                segmenter = CutoutSegmenter(this)
                statusText.post { statusText.text = "DIS GPU ready — high-precision cutout" }
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
        val (alpha, ms) = s.matte(bmp)
        Canvas0(bmp)
        fgScaled.getPixels(fgPixels, 0, O, 0, 0, O, O)
        for (i in 0 until O * O) {
            val a = alpha[i]; val p = fgPixels[i]
            val r = (((p shr 16) and 0xFF) * a + BG_R * (1 - a)).toInt()
            val g = (((p shr 8) and 0xFF) * a + BG_G * (1 - a)).toInt()
            val b = ((p and 0xFF) * a + BG_B * (1 - a)).toInt()
            compPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        compBitmap.setPixels(compPixels, 0, O, 0, 0, O, O)
        overlayView.post { overlayView.setComposite(compBitmap) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "DIS GPU  |  $fps FPS  |  ${ms}ms  |  cutout" }
    }

    private fun Canvas0(bmp: Bitmap) {
        android.graphics.Canvas(fgScaled).drawBitmap(
            bmp, Matrix().apply { setScale(O.toFloat() / bmp.width, O.toFloat() / bmp.height) }, null)
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        segmenter?.close()
    }
}
