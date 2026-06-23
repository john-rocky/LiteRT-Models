package com.edgetamvideo

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/**
 * EdgeTAM video tap-to-track: pick a video, tap an object on the first frame, and the object is
 * segmented + tracked across the following frames on-device (CompiledModel GPU). Encoder/memory/
 * decoder all run on the GPU; the rolling memory bank is managed in Kotlin.
 */
class MainActivity : AppCompatActivity() {

    private val MAX_FRAMES = 30
    @Volatile private var tracker: EdgeTamVideoTracker? = null
    private var frames: List<Bitmap> = emptyList()
    private var overlays: MutableList<Bitmap> = mutableListOf()
    @Volatile private var busy = false
    private lateinit var status: TextView
    private lateinit var imageView: ImageView
    private val ui = Handler(Looper.getMainLooper())
    private var playRunnable: Runnable? = null

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) onPick(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 48, 24, 24) }
        val button = Button(this).apply {
            text = "Select video"
            setOnClickListener { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) }
        }
        status = TextView(this).apply {
            textSize = 16f; setTextColor(Color.WHITE); gravity = Gravity.CENTER
            text = "EdgeTAM Video — pick a video, then tap an object on the first frame"; setPadding(0, 24, 0, 16)
        }
        imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            adjustViewBounds = true; scaleType = ImageView.ScaleType.FIT_CENTER
            setOnTouchListener { _, e -> if (e.action == MotionEvent.ACTION_DOWN) onTap(e.x, e.y); true }
        }
        root.addView(button); root.addView(status); root.addView(imageView)
        setContentView(root.also { it.setBackgroundColor(Color.BLACK) })

        thread { selfTest() }
    }

    /** On-device GPU tracking self-test: a white circle moving left->right. The mask should follow it. */
    private fun selfTest() {
        try {
            val n = 6; val side = 512; val r = 70f
            val tf = (0 until n).map { i ->
                val cx = 150f + (380f - 150f) * i / (n - 1); val cy = 256f
                Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888).also { b ->
                    android.graphics.Canvas(b).apply {
                        drawColor(Color.BLACK)
                        drawCircle(cx, cy, r, android.graphics.Paint().apply { color = Color.WHITE })
                    }
                }
            }
            val seg = ensureTracker()
            val t0 = System.nanoTime()
            val m0 = seg.startTracking(tf[0], 150f / side * 1024f, 256f / side * 1024f)
            logMask("SELFTEST f0", m0, 150f, side)
            for (i in 1 until n) {
                val mi = seg.trackFrame(i, tf[i])
                val expCx = 150f + (380f - 150f) * i / (n - 1)
                logMask("SELFTEST f$i", mi, expCx, side)
            }
            val per = (System.nanoTime() - t0) / 1_000_000 / n
            Log.i("EdgeTAMVideo", "SELFTEST done ${per}ms/frame on ${seg.accelerator}")
            tf.forEach { it.recycle() }
            seg.reset()
            runOnUiThread { status.text = "EdgeTAM Video ready (${seg.accelerator}) — pick a video, tap an object" }
        } catch (e: Exception) {
            Log.e("EdgeTAMVideo", "SELFTEST FAIL", e)
        }
    }

    private fun logMask(tag: String, mask: FloatArray, expCxFrame: Float, side: Int) {
        var fg = 0; var sx = 0.0; var sy = 0.0
        for (my in 0 until 256) for (mx in 0 until 256) if (mask[my * 256 + mx] > 0f) { fg++; sx += mx; sy += my }
        val cxFrame = if (fg > 0) (sx / fg) * side / 256.0 else -1.0
        Log.i("EdgeTAMVideo", "$tag mask_fg=$fg centroidX(frame)=${"%.0f".format(cxFrame)} (circle≈${"%.0f".format(expCxFrame)})")
    }

    private fun onPick(uri: Uri) {
        stopPlayback()
        runOnUiThread { status.text = "Extracting frames…" }
        thread {
            try {
                val fl = extractFrames(uri)
                if (fl.isEmpty()) { runOnUiThread { status.text = "No frames extracted" }; return@thread }
                frames = fl
                runOnUiThread {
                    imageView.setImageBitmap(frames[0])
                    status.text = "Tap an object on the first frame (${frames.size} frames)"
                }
            } catch (e: Exception) {
                Log.e("EdgeTAMVideo", "extract fail", e)
                runOnUiThread { status.text = "ERROR: ${e.message}" }
            }
        }
    }

    private fun onTap(vx: Float, vy: Float) {
        val fl = frames
        if (fl.isEmpty() || busy) return
        val first = fl[0]
        val pts = floatArrayOf(vx, vy)
        Matrix().also { imageView.imageMatrix.invert(it); it.mapPoints(pts) }
        val bx = pts[0]; val by = pts[1]
        if (bx < 0 || by < 0 || bx >= first.width || by >= first.height) return
        busy = true
        stopPlayback()
        thread {
            try {
                val seg = ensureTracker()
                val t0 = System.nanoTime()
                overlays = ArrayList(fl.size)
                // conditioning frame 0
                val m0 = seg.startTracking(first, bx / first.width * 1024f, by / first.height * 1024f)
                overlays.add(overlay(first, m0))
                runOnUiThread { imageView.setImageBitmap(overlays[0]); status.text = "Tracking… 1/${fl.size}" }
                for (i in 1 until fl.size) {
                    val mi = seg.trackFrame(i, fl[i])
                    overlays.add(overlay(fl[i], mi))
                    val done = i + 1
                    runOnUiThread { status.text = "Tracking… $done/${fl.size}" }
                }
                val ms = (System.nanoTime() - t0) / 1_000_000
                val per = ms / fl.size
                Log.i("EdgeTAMVideo", "tracked ${fl.size} frames in ${ms}ms (${per}ms/frame) on ${seg.accelerator}")
                runOnUiThread { status.text = "Done (${per} ms/frame, ${seg.accelerator}) — playing. Tap to re-track." }
                startPlayback()
            } catch (e: Exception) {
                Log.e("EdgeTAMVideo", "track fail", e)
                runOnUiThread { status.text = "ERROR: ${e.message}" }
            } finally { busy = false }
        }
    }

    /** Tint masked pixels (mask is 256x256 logits over the square-resized frame). */
    private fun overlay(src: Bitmap, mask: FloatArray): Bitmap {
        val w = src.width; val h = src.height
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val px = IntArray(w * h); out.getPixels(px, 0, w, 0, 0, w, h)
        for (y in 0 until h) {
            val my = (y * 256 / h).coerceIn(0, 255)
            for (x in 0 until w) {
                val mx = (x * 256 / w).coerceIn(0, 255)
                if (mask[my * 256 + mx] > 0f) {
                    val p = px[y * w + x]
                    val r = (p shr 16 and 0xFF) * 6 / 10
                    val g = (p shr 8 and 0xFF) * 6 / 10 + 102
                    val b = (p and 0xFF) * 6 / 10 + 102
                    px[y * w + x] = (0xFF shl 24) or (r shl 16) or (g.coerceAtMost(255) shl 8) or b.coerceAtMost(255)
                }
            }
        }
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }

    private fun startPlayback() {
        val ov = overlays
        if (ov.isEmpty()) return
        var idx = 0
        playRunnable = object : Runnable {
            override fun run() {
                if (ov.isEmpty()) return
                imageView.setImageBitmap(ov[idx % ov.size]); idx++
                ui.postDelayed(this, 100)
            }
        }
        ui.post(playRunnable!!)
    }

    private fun stopPlayback() { playRunnable?.let { ui.removeCallbacks(it) }; playRunnable = null }

    /** Extract up to MAX_FRAMES evenly-spaced frames. */
    private fun extractFrames(uri: Uri): List<Bitmap> {
        val mmr = MediaMetadataRetriever()
        mmr.setDataSource(this, uri)
        val durMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
        val out = ArrayList<Bitmap>()
        val n = if (durMs > 0) minOf(MAX_FRAMES, maxOf(2, (durMs / 150).toInt())) else 1
        for (i in 0 until n) {
            val tUs = if (n > 1) (i.toLong() * (durMs - 1) / (n - 1)) * 1000 else 0L
            val bmp = (if (android.os.Build.VERSION.SDK_INT >= 27)
                mmr.getScaledFrameAtTime(tUs, MediaMetadataRetriever.OPTION_CLOSEST, 640, 640)
            else mmr.getFrameAtTime(tUs, MediaMetadataRetriever.OPTION_CLOSEST))
            if (bmp != null) out.add(bmp)
        }
        mmr.release()
        return out
    }

    @Synchronized
    private fun ensureTracker(): EdgeTamVideoTracker =
        tracker ?: EdgeTamVideoTracker(this).also { tracker = it }

    override fun onDestroy() { super.onDestroy(); stopPlayback(); tracker?.close() }
}
