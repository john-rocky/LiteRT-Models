package com.rtmanimal

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
import kotlin.math.min

/**
 * RTMPose-m animal pose demo (AP-10K), fully on the CompiledModel GPU. Top-down: center-crops the image to a
 * square animal box, estimates 17 AP-10K keypoints, and draws the skeleton. Works on a bundled image at launch
 * and any image picked from the gallery.
 */
class MainActivity : Activity() {

    private val tag = "RTMANIMAL"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: RtmAnimalPoseEstimator? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var poseView: PoseView

    // AP-10K 17-keypoint skeleton: eyes/nose/neck/tail + four limbs (shoulder/hip → elbow/knee → paw).
    private val skeleton = arrayOf(
        0 to 1, 0 to 2, 1 to 2, 2 to 3, 3 to 4,
        3 to 5, 5 to 6, 6 to 7, 3 to 8, 8 to 9, 9 to 10,
        4 to 11, 11 to 12, 12 to 13, 4 to 14, 14 to 15, 15 to 16,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
        status = TextView(this).apply { textSize = 15f; text = "Loading RTMPose-Animal on GPU…" }
        val pick = Button(this).apply {
            text = "🖼  Pick image"; isEnabled = false
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        poseView = PoseView(this)
        root.addView(status); root.addView(pick)
        root.addView(poseView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 960))
        setContentView(root)

        bg.execute {
            try {
                net = RtmAnimalPoseEstimator(this)
                try {
                    run(cropSquare(BitmapFactory.decodeStream(assets.open("test_image.jpg"))), warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an animal image to estimate pose." }
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
        runOnUiThread { status.text = "Estimating pose…" }
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
        val visible = kpts.count { it.score > 0.3f }
        Log.i(tag, "estimate ${ms}ms visible=$visible/17")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU animal pose ✓  ${ms} ms  ·  $visible/17 keypoints  ·  RTMPose-m AP-10K, CompiledModel GPU"
            poseView.set(crop, kpts, skeleton); poseView.invalidate()
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

    /** Center-crop to a square, then resize to the model's 256x256. */
    private fun cropSquare(src: Bitmap): Bitmap {
        val s = min(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, RtmAnimalPoseEstimator.W, RtmAnimalPoseEstimator.H, true)
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

    class PoseView(ctx: Context) : View(ctx) {
        private var bm: Bitmap? = null
        private var kpts: List<RtmAnimalPoseEstimator.Keypoint> = emptyList()
        private var edges: Array<Pair<Int, Int>> = emptyArray()
        private val bone = Paint().apply { color = Color.rgb(0, 230, 0); strokeWidth = 6f; isAntiAlias = true }
        private val joint = Paint().apply { color = Color.rgb(255, 40, 40); isAntiAlias = true }
        private val imgPaint = Paint().apply { isFilterBitmap = true }

        fun set(b: Bitmap, k: List<RtmAnimalPoseEstimator.Keypoint>, e: Array<Pair<Int, Int>>) { bm = b; kpts = k; edges = e }

        override fun onDraw(canvas: Canvas) {
            val b = bm ?: return
            val s = min(width.toFloat() / b.width, height.toFloat() / b.height)
            val w = b.width * s; val h = b.height * s
            val ox = (width - w) / 2; val oy = (height - h) / 2
            canvas.drawBitmap(b, null, android.graphics.RectF(ox, oy, ox + w, oy + h), imgPaint)
            fun px(i: Int) = ox + kpts[i].x * s
            fun py(i: Int) = oy + kpts[i].y * s
            for ((a, c) in edges) {
                if (kpts[a].score > 0.3f && kpts[c].score > 0.3f) canvas.drawLine(px(a), py(a), px(c), py(c), bone)
            }
            for (i in kpts.indices) if (kpts[i].score > 0.3f) canvas.drawCircle(px(i), py(i), 7f, joint)
        }
    }
}
