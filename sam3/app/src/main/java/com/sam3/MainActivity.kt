package com.sam3

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * SAM 3.1 text-prompted detection + segmentation on-device: pick an image, type what to
 * find ("wheel", "paper bag"), get boxes + instance masks. Vision + head run on the
 * LiteRT CompiledModel GPU; the text encoder runs on CPU (see Sam3Detector).
 */
class MainActivity : Activity() {

    private val tag = "SAM3"
    private val bg = Executors.newSingleThreadExecutor()
    private var det: Sam3Detector? = null

    private lateinit var status: TextView
    private lateinit var prompt: EditText
    private lateinit var imageView: ImageView
    private var bitmap: Bitmap? = null

    private val colors = intArrayOf(
        Color.rgb(255, 64, 64), Color.rgb(64, 200, 96), Color.rgb(80, 128, 255),
        Color.rgb(255, 200, 32), Color.rgb(200, 80, 255), Color.rgb(32, 220, 220))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 90, 36, 36) }
        status = TextView(this).apply { textSize = 15f; text = "Loading (first compile takes a while)…" }
        val pick = Button(this).apply { text = "🖼  Pick image"; setOnClickListener { pickImage() } }
        prompt = EditText(this).apply {
            hint = "What to find, e.g. \"wheel\""
            inputType = InputType.TYPE_CLASS_TEXT
            setText("wheel")
        }
        val go = Button(this).apply { text = "🔍  Detect"; setOnClickListener { runDetect() } }
        imageView = ImageView(this).apply { adjustViewBounds = true }
        root.addView(status); root.addView(pick); root.addView(prompt); root.addView(go); root.addView(imageView)
        setContentView(root)

        bg.execute {
            // Tracker autotest mode: when its fixtures are installed, run the video
            // tracker instead of the image pipeline (loading both graph sets at once
            // would double the GPU memory). Remove files/tracker to go back.
            if (TrackerAutotest.shouldRun(this)) {
                runOnUiThread { status.text = "Tracker autotest (compiling 7 graphs)…" }
                try {
                    val verdict = TrackerAutotest.run(this, java.io.File(filesDir, "tracker")) {
                        line -> runOnUiThread { status.text = line }
                    }
                    runOnUiThread {
                        status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                        status.text = verdict
                    }
                } catch (e: Throwable) {
                    Log.e(tag, "tracker", e)
                    java.io.File(filesDir, "tracker_result.txt")
                        .writeText("TRACKER FAIL: ${Log.getStackTraceString(e)}")
                    runOnUiThread {
                        status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                        status.text = "TRACKER FAIL: ${e.message}"
                    }
                }
                return@execute
            }
            try {
                det = Sam3Detector(this)
                runOnUiThread { status.text = "Ready — pick an image and type a prompt." }
                autotest()
            } catch (e: Throwable) {
                Log.e(tag, "load", e)
                java.io.File(filesDir, "probe_sam3.txt").writeText("LOAD FAIL: ${Log.getStackTraceString(e)}")
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    /**
     * Headless verification hook: when files/autotest.jpg exists, run the prompts listed in
     * files/autotest_prompts.txt (one per line, default "wheel") against it, append results
     * (timings, per-query score/box) to files/probe_sam3.txt, save the last overlay to
     * files/overlay_out.png. Lets adb drive a full on-device check without the picker.
     */
    private fun autotest() {
        val d = det ?: return
        val img = java.io.File(filesDir, "autotest.jpg")
        if (!img.exists()) return
        val probe = java.io.File(filesDir, "probe_sam3.txt")
        try {
            val bm = BitmapFactory.decodeFile(img.absolutePath)
            val promptsFile = java.io.File(filesDir, "autotest_prompts.txt")
            val prompts = if (promptsFile.exists())
                promptsFile.readLines().filter { it.isNotBlank() } else listOf("wheel")
            val sb = StringBuilder()
            for (pr in prompts) {
                val t0 = System.nanoTime()
                val dets = d.detect(bm, pr)
                val ms = (System.nanoTime() - t0) / 1_000_000
                sb.append("PROMPT '").append(pr).append("' total=").append(ms)
                    .append("ms vis=").append(d.lastVisionMs).append(" txt=").append(d.lastTextMs)
                    .append(" head=").append(d.lastHeadMs).append(" kept=").append(dets.size).append("\n")
                for (dt in dets) {
                    sb.append("  score=%.4f box=%.4f,%.4f,%.4f,%.4f maskpx=%d\n".format(
                        dt.score, dt.box[0], dt.box[1], dt.box[2], dt.box[3],
                        dt.mask.count { it > 0.5f }))
                }
                val ov = overlay(bm, dets)
                java.io.FileOutputStream(java.io.File(filesDir, "overlay_out.png")).use {
                    ov.compress(Bitmap.CompressFormat.PNG, 90, it)
                }
                runOnUiThread { imageView.setImageBitmap(ov); status.text = "autotest: $pr done" }
            }
            // raw float32 taps for the Mac-side parity check
            fun dump(name: String, a: FloatArray?) {
                if (a == null) return
                val bb = java.nio.ByteBuffer.allocate(a.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                bb.asFloatBuffer().put(a)
                java.io.File(filesDir, name).writeBytes(bb.array())
            }
            dump("tap_input.bin", d.lastInput); dump("tap_vis.bin", d.visFeat)
            dump("tap_text.bin", d.lastTextMem); dump("tap_pad.bin", d.lastPad); dump("tap_head.bin", d.lastHeadOut)
            sb.append("DONE\n")
            probe.writeText(sb.toString())
        } catch (e: Throwable) {
            Log.e(tag, "autotest", e)
            probe.writeText("AUTOTEST FAIL: ${Log.getStackTraceString(e)}")
        }
    }

    private fun pickImage() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE); type = "image/*"
        }, 1)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val uri = data?.data ?: return
        if (resultCode != RESULT_OK) return
        try {
            bitmap = load(uri)
            imageView.setImageBitmap(bitmap)
            status.text = "Image set (${bitmap!!.width}x${bitmap!!.height})."
        } catch (e: Throwable) { status.text = "Failed: ${e.message}" }
    }

    private fun load(uri: Uri): Bitmap {
        contentResolver.openInputStream(uri).use { return BitmapFactory.decodeStream(it) }
    }

    private fun runDetect() {
        val d = det ?: return
        val bm = bitmap ?: run { status.text = "Pick an image first."; return }
        val text = prompt.text.toString().ifBlank { "object" }
        runOnUiThread { status.text = "Detecting \"$text\"…" }
        bg.execute {
            try {
                val t0 = System.nanoTime()
                val dets = d.detect(bm, text)
                val ms = (System.nanoTime() - t0) / 1_000_000
                val overlay = overlay(bm, dets)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "✓ ${dets.size} × \"$text\" in ${ms}ms " +
                        "(vis ${d.lastVisionMs} + txt ${d.lastTextMs} + head ${d.lastHeadMs})"
                    imageView.setImageBitmap(overlay)
                }
            } catch (e: Throwable) {
                Log.e(tag, "detect", e); runOnUiThread { status.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun overlay(bm: Bitmap, dets: List<Sam3Detector.Detection>): Bitmap {
        val w = bm.width; val h = bm.height
        val out = bm.copy(Bitmap.Config.ARGB_8888, true)
        val px = IntArray(w * h); out.getPixels(px, 0, w, 0, 0, w, h)
        val m = Sam3Detector.MASK
        dets.forEachIndexed { j, dt ->
            val c = colors[j % colors.size]
            val cr = Color.red(c); val cg = Color.green(c); val cb = Color.blue(c)
            for (y in 0 until h) {
                val my = (y * m / h).coerceIn(0, m - 1)
                for (x in 0 until w) {
                    val mx = (x * m / w).coerceIn(0, m - 1)
                    if (dt.mask[my * m + mx] > 0.5f) {
                        val i = y * w + x; val p = px[i]
                        px[i] = Color.rgb(
                            (((p shr 16) and 0xFF) + cr) / 2,
                            (((p shr 8) and 0xFF) + cg) / 2,
                            ((p and 0xFF) + cb) / 2)
                    }
                }
            }
        }
        out.setPixels(px, 0, w, 0, 0, w, h)
        val cv = Canvas(out)
        val paint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = w / 200f + 2f }
        val tp = Paint().apply { textSize = w / 40f + 12f }
        dets.forEachIndexed { j, dt ->
            paint.color = colors[j % colors.size]; tp.color = colors[j % colors.size]
            val (cx, cy, bw, bh) = dt.box
            cv.drawRect((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, paint)
            cv.drawText("%.2f".format(dt.score), (cx - bw / 2) * w + 6, (cy - bh / 2) * h + tp.textSize, tp)
        }
        return out
    }

    override fun onDestroy() { super.onDestroy(); bg.shutdown(); det?.close() }
}
