package com.ormbg

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

private const val TAG = "ormbg"

class MainActivity : ComponentActivity() {

    private var remover: BgRemover? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: MatteOverlayView
    private lateinit var statusText: TextView

    private val O = BgRemover.OUT
    private val compPixels = IntArray(O * O)
    private val fgPixels = IntArray(O * O)
    private val compBitmap = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    private val fgScaled = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    // replacement background (studio green)
    private val BG_R = 30; private val BG_G = 190; private val BG_B = 120

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
        overlayView = MatteOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f; setPadding(24, 48, 24, 0); text = "Loading ormbg (GPU)..."
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
                remover = BgRemover(this)
                statusText.post { statusText.text = "ormbg GPU ready — background removal" }
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
        val r = remover ?: return
        val (alpha, ms) = r.matte(bmp)
        // downscale the frame to O×O and composite foreground over the replacement background
        android.graphics.Canvas(fgScaled).drawBitmap(
            bmp, android.graphics.Matrix().apply {
                setScale(O.toFloat() / bmp.width, O.toFloat() / bmp.height)
            }, null)
        fgScaled.getPixels(fgPixels, 0, O, 0, 0, O, O)
        for (i in 0 until O * O) {
            val a = alpha[i]
            val p = fgPixels[i]
            val fr = (p shr 16) and 0xFF; val fg = (p shr 8) and 0xFF; val fb = p and 0xFF
            val rr = (fr * a + BG_R * (1 - a)).toInt()
            val gg = (fg * a + BG_G * (1 - a)).toInt()
            val bb = (fb * a + BG_B * (1 - a)).toInt()
            compPixels[i] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
        }
        compBitmap.setPixels(compPixels, 0, O, 0, 0, O, O)
        overlayView.post { overlayView.setComposite(compBitmap) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "ormbg GPU  |  $fps FPS  |  ${ms}ms  |  bg removed" }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        remover?.close()
    }
}
