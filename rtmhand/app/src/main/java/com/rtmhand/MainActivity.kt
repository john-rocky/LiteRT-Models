package com.rtmhand

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * RTMPose hand pose demo (21 hand keypoints), fully on the CompiledModel GPU. Top-down: center-crops to a
 * 256x256 hand box, estimates the 21 keypoints, and draws the hand skeleton (one color per finger). Works on
 * a bundled image and any image picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "RTMHAND"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: RtmHandEstimator? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var poseView: HandView

    // 21-keypoint hand skeleton (wrist + 5 fingers x 4 joints).
    private val edges = arrayOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 0 to 9, 9 to 10, 10 to 11, 11 to 12,
        0 to 13, 13 to 14, 14 to 15, 15 to 16, 0 to 17, 17 to 18, 18 to 19, 19 to 20,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
        status = TextView(this).apply { textSize = 15f; text = "Loading RTMPose-Hand on GPU…" }
        val pick = Button(this).apply {
            text = "🖼  Pick image"; isEnabled = false
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        poseView = HandView(this)
        root.addView(status); root.addView(pick)
        root.addView(poseView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 980))
        setContentView(root)

        bg.execute {
            try {
                net = RtmHandEstimator(this)
                // Optional bundled demo image; if absent just wait for a picked image.
                try {
                    run(cropSquare(BitmapFactory.decodeStream(assets.open("test_image.jpg"))), warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an image to estimate hand pose." }
                }
                runOnUiThread { pick.isEnabled = true }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != pickReq || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        runOnUiThread { status.text = "Estimating…" }
        bg.execute {
            try { run(cropSquare(loadOriented(uri)), warm = false) }
            catch (e: Throwable) { Log.e(tag, "estimate failed", e); runOnUiThread { status.text = "Failed: ${e.message}" } }
        }
    }

    private fun run(crop: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(crop)
        if (warm) n.estimate(rgb)
        val t0 = System.nanoTime()
        val kpts = n.estimate(rgb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        val visible = kpts.count { it.score > 0.2f }
        Log.i(tag, "estimate ${ms}ms visible=$visible/21")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU hand pose ✓  ${ms} ms  ·  $visible/21 keypoints  ·  RTMPose-Hand, CompiledModel GPU"
            poseView.set(crop, kpts, edges); poseView.invalidate()
        }
    }

    private fun loadOriented(uri: Uri): Bitmap {
        val bm = contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) } ?: error("cannot decode image")
        val rot = contentResolver.openInputStream(uri).use {
            when (ExifInterface(it!!).getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        }
        if (rot == 0f) return bm
        return Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, Matrix().apply { postRotate(rot) }, true)
    }

    private fun cropSquare(src: Bitmap): Bitmap {
        val s = minOf(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, RtmHandEstimator.W, RtmHandEstimator.H, true)
    }

    private fun bitmapToRgb(bm: Bitmap): FloatArray {
        val n = bm.width * bm.height; val px = IntArray(n)
        bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            val p = px[i]
            out[i * 3] = ((p shr 16) and 0xFF).toFloat(); out[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
            out[i * 3 + 2] = (p and 0xFF).toFloat()
        }
        return out
    }

    override fun onDestroy() { super.onDestroy(); bg.shutdown(); net?.close() }

    class HandView(ctx: Context) : View(ctx) {
        private var bm: Bitmap? = null
        private var kpts: List<RtmHandEstimator.Keypoint> = emptyList()
        private var edges: Array<Pair<Int, Int>> = emptyArray()
        private val bone = Paint().apply { color = Color.rgb(0, 220, 0); strokeWidth = 5f; isAntiAlias = true }
        private val dot = Paint().apply { isAntiAlias = true }
        private val imgPaint = Paint().apply { isFilterBitmap = true }
        private val fingerColor = intArrayOf(
            Color.rgb(255, 40, 40), Color.rgb(255, 140, 0), Color.rgb(230, 200, 0), Color.rgb(0, 200, 0), Color.rgb(0, 120, 255),
        )

        fun set(b: Bitmap, k: List<RtmHandEstimator.Keypoint>, e: Array<Pair<Int, Int>>) { bm = b; kpts = k; edges = e }

        override fun onDraw(canvas: Canvas) {
            val b = bm ?: return
            val s = minOf(width.toFloat() / b.width, height.toFloat() / b.height)
            val w = b.width * s; val h = b.height * s
            val ox = (width - w) / 2; val oy = (height - h) / 2
            canvas.drawBitmap(b, null, android.graphics.RectF(ox, oy, ox + w, oy + h), imgPaint)
            fun px(i: Int) = ox + kpts[i].x * s
            fun py(i: Int) = oy + kpts[i].y * s
            for ((a, c) in edges) {
                if (kpts[a].score > 0.2f && kpts[c].score > 0.2f) canvas.drawLine(px(a), py(a), px(c), py(c), bone)
            }
            for (i in kpts.indices) if (kpts[i].score > 0.2f) {
                dot.color = if (i == 0) Color.WHITE else fingerColor[(i - 1) / 4]
                canvas.drawCircle(px(i), py(i), 6f, dot)
            }
        }
    }
}
