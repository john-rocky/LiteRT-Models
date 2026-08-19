package com.sam2

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/**
 * SAM 2.1 video tracking demo: pick a video, tap the object in the first frame, and the tracker
 * propagates the mask across frames fully on the LiteRT GPU (Sam2VideoTracker host loop). Masked
 * frames are cached and scrubbable with the seek bar. Defaults to the 2-slot memory bank for
 * interactivity; the 7-slot bank is more faithful but ~2x slower per frame on Mali.
 */
class Sam2VideoActivity : AppCompatActivity() {

    companion object {
        private const val MAX_FRAMES = 16     // frames sampled across the clip
        private const val NUM_MASK_MEM = 2    // memory-bank slots (2 = fast, 7 = faithful)
    }

    @Volatile private var tracker: Sam2VideoTracker? = null
    private val frames = ArrayList<Bitmap>()
    private val overlays = ArrayList<Bitmap?>()
    @Volatile private var firstFrame: Bitmap? = null

    private lateinit var status: TextView
    private lateinit var imageView: ImageView
    private lateinit var seek: SeekBar

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) onPickVideo(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(24, 48, 24, 24); setBackgroundColor(Color.BLACK)
        }
        val button = Button(this).apply {
            text = "Select video"
            setOnClickListener {
                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
            }
        }
        status = TextView(this).apply {
            textSize = 15f; setTextColor(Color.WHITE); gravity = Gravity.CENTER
            text = "Pick a video, then tap the object in the first frame."; setPadding(0, 24, 0, 12)
        }
        imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
            adjustViewBounds = true; scaleType = ImageView.ScaleType.FIT_CENTER
            setOnTouchListener { _, e -> if (e.action == MotionEvent.ACTION_DOWN) onTapFirstFrame(e.x, e.y); true }
        }
        seek = SeekBar(this).apply {
            isEnabled = false
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) = showFrame(p)
                override fun onStartTrackingTouch(sb: SeekBar) {}
                override fun onStopTrackingTouch(sb: SeekBar) {}
            })
        }
        root.addView(button); root.addView(status); root.addView(imageView); root.addView(seek)
        setContentView(root)

        // Headless self-test: `am start -n com.sam2/.Sam2VideoActivity --ez selftest true`
        // tracks a synthetic moving disk and logs per-frame fg/obj/ms to logcat (tag SAM2V),
        // so the on-device host loop can be verified without UI taps.
        if (intent.getBooleanExtra("selftest", false)) thread { selfTest() }

        // Demo mode: `--ez demo true [--es demofile <name>] [--ef clickx 0.52 --ef clicky 0.42]`
        // loads a video staged in filesDir, tracks a preset click, then loops the masked
        // frames on screen — for recording an on-device demo without UI taps.
        if (intent.getBooleanExtra("demo", false)) thread { runDemo() }
    }

    private fun runDemo() {
        try {
            val name = intent.getStringExtra("demofile") ?: "demo_cat.mp4"
            val file = java.io.File(filesDir, name)
            if (!file.exists()) {
                runOnUiThread { status.text = "Demo video not staged: $name" }
                return
            }
            loadFrames { it.setDataSource(file.absolutePath) }
            if (frames.isEmpty()) {
                runOnUiThread { status.text = "No frames read from $name" }
                return
            }
            val cx = intent.getFloatExtra("clickx", 0.52f) * Sam2VideoTracker.SIZE
            val cy = intent.getFloatExtra("clicky", 0.42f) * Sam2VideoTracker.SIZE
            runTracking(cx, cy)
            loopPlayback()
        } catch (e: Exception) {
            android.util.Log.e("SAM2V", "DEMO FAIL ${e.message}", e)
            runOnUiThread { status.text = "DEMO FAIL: ${e.message}" }
        }
    }

    /** Cycle through the masked frames on the main thread so a screen recording sees motion. */
    private fun loopPlayback() {
        val handler = android.os.Handler(android.os.Looper.getMainLooper())
        val n = overlays.size
        val step = object : Runnable {
            var i = 0
            override fun run() {
                if (overlays.isEmpty()) return
                showFrame(i % n)
                seek.progress = i % n
                i++
                handler.postDelayed(this, 220)
            }
        }
        handler.post(step)
    }

    private fun selfTest() {
        try {
            val trk = ensureTracker()
            trk.reset()
            val n = MAX_FRAMES
            val size = Sam2VideoTracker.SIZE
            val paint = android.graphics.Paint().apply { color = Color.WHITE }
            var firstFg = 0
            val t0 = System.nanoTime()
            for (i in 0 until n) {
                val bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
                android.graphics.Canvas(bmp).apply {
                    drawColor(Color.BLACK)
                    val cx = 400f + 28f * i          // disk drifts diagonally
                    val cy = 512f + 12f * i
                    drawCircle(cx, cy, 200f, paint)
                }
                val res = if (i == 0) trk.startFrame(0, bmp, 400f, 512f) else trk.trackFrame(i, bmp)
                val fg = res.mask.count { it > 0f }
                if (i == 0) firstFg = fg
                android.util.Log.i(
                    "SAM2V", "SELFTEST frame $i fg=$fg obj=%.2f appearing=%b".format(res.objScore, res.appearing)
                )
                bmp.recycle()
            }
            val msPerFrame = (System.nanoTime() - t0) / 1_000_000 / n
            android.util.Log.i(
                "SAM2V", "SELFTEST DONE frames=$n ${trk.accelerator} ${NUM_MASK_MEM}-slot " +
                    "~${msPerFrame}ms/frame first_fg=$firstFg"
            )
            runOnUiThread { status.text = "Self-test done — see logcat (tag SAM2V)." }
        } catch (e: Exception) {
            android.util.Log.e("SAM2V", "SELFTEST FAIL ${e.message}", e)
            runOnUiThread { status.text = "SELFTEST FAIL: ${e.message}"; status.setTextColor(Color.RED) }
        }
    }

    /** Extract MAX_FRAMES evenly across the clip; `configure` sets the retriever's source. */
    private fun loadFrames(configure: (MediaMetadataRetriever) -> Unit) {
        frames.forEach { it.recycle() }
        frames.clear(); overlays.clear(); firstFrame = null
        val retriever = MediaMetadataRetriever()
        try {
            configure(retriever)
            val durUs = (retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L) * 1000L
            val n = MAX_FRAMES
            for (i in 0 until n) {
                val tUs = if (n == 1) 0L else durUs * i / (n - 1)
                val f = retriever.getFrameAtTime(tUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    ?: continue
                frames.add(f.copy(Bitmap.Config.ARGB_8888, false))
                overlays.add(null)
                f.recycle()
            }
        } finally {
            retriever.release()
        }
    }

    private fun onPickVideo(uri: Uri) {
        status.text = "Extracting frames…"
        thread {
            loadFrames { it.setDataSource(this, uri) }
            if (frames.isEmpty()) {
                runOnUiThread { status.text = "No frames could be read from this video." }
                return@thread
            }
            firstFrame = frames[0]
            runOnUiThread {
                imageView.setImageBitmap(frames[0])
                seek.max = frames.size - 1; seek.progress = 0; seek.isEnabled = false
                status.text = "Tap the object in this first frame to start tracking."
            }
        }
    }

    private fun onTapFirstFrame(vx: Float, vy: Float) {
        val bmp = firstFrame ?: return
        val pts = floatArrayOf(vx, vy)
        Matrix().also { imageView.imageMatrix.invert(it); it.mapPoints(pts) }
        val bx = pts[0]; val by = pts[1]
        if (bx < 0 || by < 0 || bx >= bmp.width || by >= bmp.height) return
        firstFrame = null  // consume the tap so a second tap doesn't restart
        val clickX = bx / bmp.width * Sam2VideoTracker.SIZE
        val clickY = by / bmp.height * Sam2VideoTracker.SIZE
        runOnUiThread { status.text = "Tracking on GPU…" }
        thread { runTracking(clickX, clickY) }
    }

    private fun runTracking(clickX: Float, clickY: Float) {
        try {
            val trk = ensureTracker()
            trk.reset()
            val t0 = System.nanoTime()
            var appearing = 0
            for (i in frames.indices) {
                val res = if (i == 0) trk.startFrame(0, frames[0], clickX, clickY)
                          else trk.trackFrame(i, frames[i])
                overlays[i] = overlay(frames[i], res.mask)
                if (res.appearing) appearing++
                val done = i + 1
                runOnUiThread {
                    status.text = "Tracking… frame $done/${frames.size}"
                    if (i == 0) { imageView.setImageBitmap(overlays[0]) }
                }
            }
            val msPerFrame = (System.nanoTime() - t0) / 1_000_000 / frames.size
            runOnUiThread {
                seek.isEnabled = true; seek.progress = 0; showFrame(0)
                status.text = "Done — ${frames.size} frames, ~${msPerFrame} ms/frame " +
                    "(${trk.accelerator}, ${NUM_MASK_MEM}-slot), object in $appearing. Scrub to review."
            }
        } catch (e: Exception) {
            runOnUiThread { status.text = "ERROR: ${e.message}"; status.setTextColor(Color.RED) }
        }
    }

    private fun showFrame(i: Int) {
        if (i in overlays.indices) imageView.setImageBitmap(overlays[i] ?: frames[i])
    }

    /** Tint the masked region (mask is 256x256 logits over the square-resized frame). */
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
                    px[y * w + x] =
                        (0xFF shl 24) or (r shl 16) or (g.coerceAtMost(255) shl 8) or b.coerceAtMost(255)
                }
            }
        }
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }

    @Synchronized
    private fun ensureTracker(): Sam2VideoTracker =
        tracker ?: Sam2VideoTracker(this, NUM_MASK_MEM).also { tracker = it }

    override fun onDestroy() {
        super.onDestroy()
        tracker?.close()
        frames.forEach { it.recycle() }
    }
}
