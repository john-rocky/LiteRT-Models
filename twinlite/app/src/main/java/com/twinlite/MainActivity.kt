package com.twinlite

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.FrameLayout
import android.view.TextureView
import android.widget.TextView
import com.google.ai.edge.litert.Accelerator
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "TwinLiteNet"

class MainActivity : ComponentActivity() {

    private var segmenter: TwinLiteSegmenter? = null
    private var loadedMs: Long = 0
    private var fps = 0
    private var frameCount = 0
    private var fpsWindowStart = System.currentTimeMillis()
    private var pipeline: RealtimeCameraPipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SegOverlayView
    private lateinit var videoView: TextureView
    private var videoPipeline: VideoFramePipeline? = null
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
        videoView = TextureView(this)
        overlayView = SegOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 16f; setPadding(24, 48, 24, 0); text = if (BuildConfig.USE_NPU) "Loading TwinLiteNet (NPU)..." else "Loading TwinLiteNet (GPU)..."
        }
        // A bundled clip drives both builds, so the two accelerators see identical
        // frames and the comparison does not depend on where a camera was pointed.
        root.addView(videoView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        setContentView(root)
        // The clip starts only once the model is ready: decoding in parallel competes
        // with shader construction and inflated the GPU load figure by 1.9x.
        loadModel()
    }

    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                val acc = if (BuildConfig.USE_NPU) Accelerator.NPU else Accelerator.GPU
                val asset = if (BuildConfig.USE_NPU) "twinlite_npu_aot.tflite" else "twinlite.tflite"
                val seg = TwinLiteSegmenter(this, asset, acc)
                segmenter = seg
                loadedMs = seg.loadMs
                val label = if (BuildConfig.USE_NPU) "NPU (Hexagon, AOT)" else "GPU (Adreno)"
                statusText.post {
                    statusText.text = "TwinLiteNet — $label — ready in ${seg.loadMs} ms"
                }
                // startVideo() hops to the UI thread, so enabling must happen inside it
                // — setting it here would run before the pipeline exists.
                runOnUiThread { startVideo() }
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}", e)
                statusText.post { statusText.text = "Load failed: ${e.message}" }
            }
        }
    }

    private fun startVideo() {
        videoPipeline = VideoFramePipeline(this, videoView, "demo_road.mp4") { bmp ->
            runInference(bmp)
        }.also { it.enabled = true; it.start() }
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
        frameCount++
        val now = System.currentTimeMillis()
        if (now - fpsWindowStart >= 1000) {
            fps = (frameCount * 1000L / (now - fpsWindowStart)).toInt()
            frameCount = 0; fpsWindowStart = now
        }
        // The running line must name the accelerator it is actually on, and keep the
        // load figure visible — it is overwritten within a frame otherwise.
        val accLabel = if (BuildConfig.USE_NPU) "NPU" else "GPU"
        statusText.post {
            statusText.text =
                // No frame rate here: it would report the demo pipeline's bitmap
                // grab, not the model. Load and inference are the model's own numbers.
                "TwinLiteNet $accLabel   |   loaded ${loadedMs} ms   |   inference ${ms} ms"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        videoPipeline?.stop()
        backgroundExecutor.shutdown()
        segmenter?.close()
    }
}
