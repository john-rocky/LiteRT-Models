package com.ormbg

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.TextureView
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.google.ai.edge.litert.Accelerator
import java.util.concurrent.Executors

private const val TAG = "ormbg"

class MainActivity : ComponentActivity() {

    private var remover: BgRemover? = null
    private var pipeline: VideoFramePipeline? = null
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private lateinit var textureView: TextureView
    private lateinit var overlayView: MatteOverlayView
    private lateinit var statusText: TextView

    // One flavor per accelerator, so a run is never a mix of the two.
    private val accelerator = if (BuildConfig.USE_NPU) Accelerator.NPU else Accelerator.GPU
    private val accLabel = if (BuildConfig.USE_NPU) "NPU" else "GPU"
    // Both flavors read the same published file; only the accelerator differs.
    private val modelFile = "ormbg.tflite"

    // The demo composites at a reduced resolution so the loop keeps up with the clip.
    private val O = 256
    private val compPixels = IntArray(O * O)
    private val fgPixels = IntArray(O * O)
    private val compBitmap = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    private val fgScaled = Bitmap.createBitmap(O, O, Bitmap.Config.ARGB_8888)
    // replacement background (studio green)
    private val BG_R = 30; private val BG_G = 190; private val BG_B = 120

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = FrameLayout(this)
        textureView = TextureView(this)
        overlayView = MatteOverlayView(this)
        statusText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt()); setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 13f; setPadding(24, 120, 24, 0); text = "Loading ormbg ($accLabel)..."
        }
        root.addView(textureView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(statusText, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        setContentView(root)
        loadModel()
    }

    /**
     * The load is timed with nothing else running, because video decoding on the same
     * device inflated it by ~80% when the two overlapped. Playback starts only once the
     * model is ready.
     */
    private fun loadModel() {
        backgroundExecutor.execute {
            try {
                val r = BgRemover.fromAssets(this, modelFile, accelerator)
                remover = r
                statusText.post {
                    statusText.text = "ormbg $accLabel   |   loaded ${r.loadMs} ms"
                    startVideo()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}", e)
                statusText.post { statusText.text = "Load failed: ${e.message}" }
            }
        }
    }

    private fun startVideo() {
        pipeline = VideoFramePipeline(this, textureView, "demo_person.mp4") { bmp ->
            runInference(bmp)
        }.also { it.enabled = true; it.start() }
    }

    private fun runInference(bmp: Bitmap) {
        val r = remover ?: return
        val t0 = System.nanoTime()
        val alpha = (r.process(bmp) ?: return).downsample(O)
        val ms = (System.nanoTime() - t0) / 1_000_000
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
        val bw = bmp.width; val bh = bmp.height
        overlayView.post { overlayView.setComposite(compBitmap, bw, bh) }
        // No FPS here: it would measure the decode-and-composite loop, not the model.
        val loaded = r.loadMs
        val model = r.modelMs
        statusText.post {
            statusText.text =
                "ormbg $accLabel   |   loaded $loaded ms   |   inference $model ms   |   frame $ms ms"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.stop()
        backgroundExecutor.shutdown()
        remover?.close()
    }
}
