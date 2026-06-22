package com.da3

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import kotlin.concurrent.thread

/**
 * DA3-SMALL monocular depth: pick any image from the photo library, run on-device GPU inference,
 * and show input | depth side by side. The model (~1.8 s/image on Pixel 8a GPU) is built once and reused.
 */
class MainActivity : AppCompatActivity() {

    @Volatile private var predictor: DA3Predictor? = null
    private lateinit var status: TextView
    private lateinit var inputView: ImageView
    private lateinit var depthView: ImageView

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) process(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 24)
        }
        val button = Button(this).apply {
            text = "Select image"
            setOnClickListener {
                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
        }
        status = TextView(this).apply {
            textSize = 16f; setTextColor(Color.WHITE); gravity = Gravity.CENTER
            text = "DA3-SMALL depth — pick an image"
            setPadding(0, 24, 0, 0)
        }
        fun imageView() = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                .also { it.marginStart = 8; it.marginEnd = 8 }
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        inputView = imageView(); depthView = imageView()
        val pair = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.topMargin = 24 }
            addView(inputView); addView(depthView)
        }
        root.addView(button); root.addView(status); root.addView(pair)
        setContentView(ScrollView(this).apply { setBackgroundColor(Color.BLACK); addView(root) })
    }

    private fun process(uri: Uri) {
        status.text = "Running…"
        thread {
            try {
                val bmp = loadBitmap(uri)
                if (predictor == null) {
                    runOnUiThread { status.text = "Loading model (GPU compile)…" }
                    predictor = DA3Predictor(this)
                }
                val res = predictor!!.predict(bmp)
                val depth = res.depthBitmap()
                runOnUiThread {
                    status.text = "DA3-SMALL  |  ${res.accelerator}  |  ${res.inferenceMs} ms"
                    status.setTextColor(if (res.accelerator == "GPU") Color.GREEN else Color.YELLOW)
                    inputView.setImageBitmap(bmp)
                    depthView.setImageBitmap(depth)
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "ERROR: ${e.message}"; status.setTextColor(Color.RED) }
            }
        }
    }

    /** Decode (downsampled) + apply EXIF rotation. */
    private fun loadBitmap(uri: Uri): Bitmap {
        val bytes = contentResolver.openInputStream(uri)!!.use { it.readBytes() }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        var sample = 1
        val longSide = maxOf(bounds.outWidth, bounds.outHeight)
        while (longSide / sample > 1600) sample *= 2          // cap long side ~1600px
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
        val orient = ExifInterface(bytes.inputStream())
            .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        val deg = when (orient) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (deg != 0f) {
            val m = Matrix().apply { postRotate(deg) }
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        }
        return bmp
    }

    override fun onDestroy() {
        super.onDestroy()
        predictor?.close()
    }
}
