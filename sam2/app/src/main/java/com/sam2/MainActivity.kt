package com.sam2

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import android.net.Uri
import android.os.Bundle
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
import androidx.exifinterface.media.ExifInterface
import kotlin.concurrent.thread

/**
 * SAM2 (Hiera-Tiny) tap-to-segment: pick any image, then tap an object to segment it on-device (GPU).
 * The encoder runs once per image; each tap runs the mask decoder. On launch a headless benchmark
 * reports the median warm encoder / decoder latency to logcat (tag "SAM2", key "BENCH") — the
 * LiteRT-side numbers for the LiteRT-vs-MLX comparison.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "SAM2"
        private const val BENCH_ITERS = 20
    }

    @Volatile private var segmenter: Sam2Segmenter? = null
    @Volatile private var bitmap: Bitmap? = null
    @Volatile private var encoded = false
    private lateinit var status: TextView
    private lateinit var imageView: ImageView

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) onPick(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 48, 24, 24) }
        val button = Button(this).apply {
            text = "Select image"
            setOnClickListener { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
        }
        status = TextView(this).apply {
            textSize = 16f; setTextColor(Color.WHITE); gravity = Gravity.CENTER
            text = "SAM2 — compiling GPU + benchmarking…"; setPadding(0, 24, 0, 16)
        }
        imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            adjustViewBounds = true; scaleType = ImageView.ScaleType.FIT_CENTER
            setOnTouchListener { _, e -> if (e.action == MotionEvent.ACTION_DOWN) onTap(e.x, e.y); true }
        }
        root.addView(button); root.addView(status); root.addView(imageView)
        setContentView(root.also { it.setBackgroundColor(Color.BLACK) })

        thread { benchmark() }
    }

    /** Headless GPU-compile check + warm-latency benchmark on a synthetic circle. */
    private fun benchmark() {
        try {
            val test = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
            Canvas(test).apply {
                drawColor(Color.BLACK)
                drawCircle(256f, 256f, 120f, Paint().apply { color = Color.WHITE })
            }
            val seg = ensureSegmenter()

            // Warm up: the first encode triggers GPU kernel compilation (excluded from timing).
            val compileMs = seg.encode(test)
            seg.segment(512f, 512f)

            val encTimes = LongArray(BENCH_ITERS) { seg.encode(test) }
            val decTimes = LongArray(BENCH_ITERS) {
                val t = System.nanoTime(); seg.segment(512f, 512f); (System.nanoTime() - t) / 1_000_000
            }
            val encMed = median(encTimes)
            val decMed = median(decTimes)
            val fg = seg.segment(512f, 512f).count { it > 0f }
            Log.i(TAG, "BENCH ${seg.accelerator} compile=${compileMs}ms " +
                "enc_median=${encMed}ms dec_median=${decMed}ms iters=$BENCH_ITERS mask_fg=$fg")
            runOnUiThread {
                status.text = "SAM2 ${seg.accelerator} — enc ${encMed}ms / dec ${decMed}ms " +
                    "(median, compile ${compileMs}ms) — pick an image, tap an object"
            }
        } catch (e: Exception) {
            Log.e(TAG, "BENCH FAIL ${e.message}", e)
            runOnUiThread { status.text = "ERROR: ${e.message}"; status.setTextColor(Color.RED) }
        }
    }

    private fun median(values: LongArray): Long {
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }

    private fun onPick(uri: Uri) {
        val bmp = loadBitmap(uri)
        bitmap = bmp; encoded = false
        runOnUiThread { imageView.setImageBitmap(bmp); status.text = "Encoding on GPU…" }
        thread {
            try {
                val seg = ensureSegmenter()
                val ms = seg.encode(bmp)
                encoded = true
                runOnUiThread { status.text = "Ready (${seg.accelerator} ${ms} ms) — tap an object" }
            } catch (e: Exception) {
                runOnUiThread { status.text = "ERROR: ${e.message}"; status.setTextColor(Color.RED) }
            }
        }
    }

    private fun onTap(vx: Float, vy: Float) {
        val bmp = bitmap ?: return
        if (!encoded) return
        val pts = floatArrayOf(vx, vy)
        Matrix().also { imageView.imageMatrix.invert(it); it.mapPoints(pts) }
        val bx = pts[0]; val by = pts[1]
        if (bx < 0 || by < 0 || bx >= bmp.width || by >= bmp.height) return
        thread {
            try {
                val t = System.nanoTime()
                val mask = segmenter!!.segment(bx / bmp.width * 1024f, by / bmp.height * 1024f)
                val ov = overlay(bmp, mask)
                val ms = (System.nanoTime() - t) / 1_000_000
                runOnUiThread { imageView.setImageBitmap(ov); status.text = "segmented (${ms} ms) — tap again" }
            } catch (e: Exception) {
                runOnUiThread { status.text = "ERROR: ${e.message}" }
            }
        }
    }

    /** Tint the masked region (mask is 256x256 logits over the square-resized image). */
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

    @Synchronized
    private fun ensureSegmenter(): Sam2Segmenter =
        segmenter ?: Sam2Segmenter(this).also { segmenter = it }

    private fun loadBitmap(uri: Uri): Bitmap {
        val bytes = contentResolver.openInputStream(uri)!!.use { it.readBytes() }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1600) sample *= 2
        var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample })
        val deg = when (ExifInterface(bytes.inputStream()).getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (deg != 0f) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(deg) }, true)
        return bmp
    }

    override fun onDestroy() {
        super.onDestroy(); segmenter?.close()
    }
}
