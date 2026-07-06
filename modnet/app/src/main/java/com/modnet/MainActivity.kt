package com.modnet

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "MODNet"

class MainActivity : ComponentActivity() {

    private var matter: Matter? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView

    // Tap to cycle the replacement background.
    private val bgColors = intArrayOf(
        Color.rgb(0, 177, 64), Color.rgb(20, 20, 24), Color.rgb(240, 240, 245),
        Color.rgb(0, 120, 215), Color.rgb(210, 70, 90))
    @Volatile private var bgIdx = 0

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
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }
        imageView = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER }
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f
            setPadding(24, 48, 24, 0)
            text = "Loading MODNet (GPU)..."
        }
        root.addView(imageView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        root.setOnClickListener { bgIdx = (bgIdx + 1) % bgColors.size }
        setContentView(root)

        loadModel()
        startCamera()
    }

    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                matter = Matter(this)
                statusText.post { statusText.text = "MODNet GPU ready — tap to change background" }
                pipeline?.enabled = true
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}", e)
                statusText.post { statusText.text = "Load failed: ${e.message}" }
            }
        }
    }

    private fun startCamera() {
        // previewView = null: we display the composited matte result instead of the raw preview
        pipeline = RealtimeCameraPipeline(activity = this, previewView = null) { bmp ->
            runInference(bmp)
        }.also { it.enabled = false; it.start(this) }
    }

    private fun runInference(bmp: Bitmap) {
        val m = matter ?: return
        val (composite, ms) = m.matte(bmp, bgColors[bgIdx])
        val shown = composite.copy(Bitmap.Config.ARGB_8888, false)
        imageView.post { imageView.setImageBitmap(shown) }
        val fps = pipeline?.fps ?: 0
        statusText.post { statusText.text = "MODNet GPU  |  $fps FPS  |  ${ms}ms/frame  |  tap: bg" }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        matter?.close()
    }
}
