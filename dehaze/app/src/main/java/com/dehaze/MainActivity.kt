package com.dehaze

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

private const val TAG = "Dehaze"

class MainActivity : ComponentActivity() {

    private var dehazer: Dehazer? = null
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView

    // Tap to compare: true shows the dehazed frame, false the raw camera frame.
    @Volatile private var showDehazed = true

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
            text = "Loading DehazeFormer (GPU)..."
        }
        root.addView(imageView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        root.setOnClickListener { showDehazed = !showDehazed }
        setContentView(root)

        loadModel()
        startCamera()
    }

    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                dehazer = Dehazer(this)
                statusText.post { statusText.text = "DehazeFormer GPU ready — tap to compare" }
                pipeline?.enabled = true
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}", e)
                statusText.post { statusText.text = "Load failed: ${e.message}" }
            }
        }
    }

    private fun startCamera() {
        // previewView = null: we display the dehazed frame instead of the raw preview
        pipeline = RealtimeCameraPipeline(activity = this, previewView = null) { bmp ->
            runInference(bmp)
        }.also { it.enabled = false; it.start(this) }
    }

    private fun runInference(bmp: Bitmap) {
        val d = dehazer ?: return
        if (showDehazed) {
            val (dehazed, ms) = d.dehaze(bmp)
            imageView.post { imageView.setImageBitmap(dehazed) }
            val fps = pipeline?.fps ?: 0
            statusText.post {
                statusText.text = "DehazeFormer GPU  |  $fps FPS  |  ${ms}ms/frame  |  tap: original"
            }
        } else {
            val shown = bmp.copy(Bitmap.Config.ARGB_8888, false)
            imageView.post { imageView.setImageBitmap(shown) }
            statusText.post { statusText.text = "Original camera frame  |  tap: dehaze" }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        dehazer?.close()
    }
}
