package com.gfpgan

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.exifinterface.media.ExifInterface
import java.util.concurrent.Executors

private const val TAG = "GFPGAN"
private const val MAX_INPUT_SIZE = 1024

class MainActivity : ComponentActivity() {

    private lateinit var compareView: CompareView
    private lateinit var statusText: TextView
    private lateinit var pickButton: Button

    private var restorer: FaceRestorer? = null
    private var detector: FaceDetector? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var isProcessing = false
    private var lastBefore: Bitmap? = null
    private var lastAfter: Bitmap? = null

    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { processImage(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 48, 0, 0)
        }

        statusText = TextView(this).apply {
            textSize = 16f
            setPadding(24, 8, 24, 8)
            text = "Loading model..."
        }
        root.addView(statusText)

        pickButton = Button(this).apply {
            text = "Select Face Photo"
            isEnabled = false
            setOnClickListener { imagePicker.launch("image/*") }
        }
        root.addView(pickButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.CENTER_HORIZONTAL })

        compareView = CompareView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        }
        root.addView(compareView)

        setContentView(root)

        executor.execute {
            try {
                restorer = FaceRestorer(this)
                detector = try { FaceDetector(this) } catch (e: Exception) {
                    Log.w(TAG, "Face detector unavailable, using center crop", e); null
                }
                runOnUiThread {
                    statusText.text = "Ready — select a face photo to restore"
                    pickButton.isEnabled = true
                }
                runAutoTestIfPresent()
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun processImage(uri: Uri) {
        val r = restorer ?: return
        if (isProcessing) return
        isProcessing = true
        pickButton.isEnabled = false
        statusText.text = "Restoring..."

        executor.execute {
            try {
                val bitmap = loadBitmap(uri) ?: throw Exception("Failed to load image")
                val before = prepareAligned(bitmap)
                bitmap.recycle()

                val t = System.nanoTime()
                val after = r.restore(before)
                val ms = (System.nanoTime() - t) / 1_000_000

                lastBefore?.recycle(); lastAfter?.recycle()
                lastBefore = before; lastAfter = after

                runOnUiThread {
                    compareView.setImages(before, after)
                    statusText.text = "Restored 512x512 in ${ms}ms — drag to compare"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Restore failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    /** Headless test (no picker): if <externalFiles>/gfp_in.png exists, restore it and save
     *  gfp_before.png / gfp_after.png alongside it. Used to drive a restoration over adb. */
    private fun runAutoTestIfPresent() {
        val r = restorer ?: return
        val dir = getExternalFilesDir(null) ?: return
        val inFile = java.io.File(dir, "gfp_in.png")
        if (!inFile.exists()) return
        try {
            val src = android.graphics.BitmapFactory.decodeFile(inFile.absolutePath) ?: return
            val before = prepareAligned(src)
            val t = System.nanoTime()
            val after = r.restore(before)
            val ms = (System.nanoTime() - t) / 1_000_000
            java.io.FileOutputStream(java.io.File(dir, "gfp_before.png")).use { before.compress(Bitmap.CompressFormat.PNG, 100, it) }
            java.io.FileOutputStream(java.io.File(dir, "gfp_after.png")).use { after.compress(Bitmap.CompressFormat.PNG, 100, it) }
            Log.i(TAG, "AUTOTEST ok: restored in ${ms}ms, saved gfp_before/after.png")
            runOnUiThread { compareView.setImages(before, after); statusText.text = "Auto-test: restored in ${ms}ms — drag to compare" }
        } catch (e: Exception) {
            Log.e(TAG, "AUTOTEST failed", e)
        }
    }

    /** Detect the largest face and FFHQ-align it to 512x512. Falls back to a center-square crop. */
    private fun prepareAligned(src: Bitmap): Bitmap {
        val d = detector
        if (d != null) {
            try {
                val sz = FaceDetector.SIZE
                val det = Bitmap.createScaledBitmap(src, sz, sz, true)
                val px = IntArray(sz * sz)
                det.getPixels(px, 0, sz, 0, 0, sz, sz)
                val rgb = FloatArray(sz * sz * 3)
                var i = 0
                for (p in px) {
                    rgb[i++] = ((p shr 16) and 0xFF).toFloat()
                    rgb[i++] = ((p shr 8) and 0xFF).toFloat()
                    rgb[i++] = (p and 0xFF).toFloat()
                }
                det.recycle()
                val face = d.detect(rgb).maxByOrNull { it.score }
                if (face != null) {
                    val sx = src.width.toFloat() / sz; val sy = src.height.toFloat() / sz
                    val lm = FloatArray(10)
                    for (j in 0 until 5) { lm[2 * j] = face.landmarks[2 * j] * sx; lm[2 * j + 1] = face.landmarks[2 * j + 1] * sy }
                    Log.i(TAG, "Face detected (score ${"%.2f".format(face.score)}), FFHQ-aligned")
                    return FaceAligner.align(src, lm)
                }
                Log.w(TAG, "No face detected — center crop")
            } catch (e: Exception) {
                Log.w(TAG, "Alignment failed — center crop", e)
            }
        }
        return restorer!!.toFaceInput(centerSquareCrop(src))
    }

    private fun centerSquareCrop(src: Bitmap): Bitmap {
        val s = minOf(src.width, src.height)
        val left = (src.width - s) / 2
        val top = (src.height - s) / 2
        return Bitmap.createBitmap(src, left, top, s, s)
    }

    private fun loadBitmap(uri: Uri): Bitmap? {
        val input = contentResolver.openInputStream(uri) ?: return null
        val opts = BitmapFactory.Options()
        opts.inJustDecodeBounds = true
        BitmapFactory.decodeStream(input, null, opts)
        input.close()

        var sampleSize = 1
        while (opts.outWidth / sampleSize > MAX_INPUT_SIZE || opts.outHeight / sampleSize > MAX_INPUT_SIZE) {
            sampleSize *= 2
        }

        val input2 = contentResolver.openInputStream(uri) ?: return null
        opts.inJustDecodeBounds = false
        opts.inSampleSize = sampleSize
        val bitmap = BitmapFactory.decodeStream(input2, null, opts)
        input2.close()

        val input3 = contentResolver.openInputStream(uri) ?: return bitmap
        val exif = ExifInterface(input3)
        input3.close()
        val rotation = when (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (rotation == 0f || bitmap == null) return bitmap

        val matrix = Matrix().apply { postRotate(rotation) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotated
    }

    override fun onDestroy() {
        super.onDestroy()
        restorer?.close()
        detector?.close()
        executor.shutdown()
    }
}
