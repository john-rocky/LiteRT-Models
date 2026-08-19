package com.tipsv2

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.GridLayout
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
 * TIPSv2-B/14 + DPT heads: pick an image from the photo library, run one GPU inference and show
 * input | depth | normals | ADE20K segmentation (with a legend of the classes found).
 * The model (~0.9 s/image for all three heads on Pixel 8a GPU) is compiled once and reused.
 */
class MainActivity : AppCompatActivity() {

    @Volatile private var predictor: TipsPredictor? = null
    private lateinit var status: TextView
    private lateinit var inputView: ImageView
    private lateinit var depthView: ImageView
    private lateinit var normalsView: ImageView
    private lateinit var segView: ImageView
    private lateinit var depthLabel: TextView
    private lateinit var legend: GridLayout

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
            text = "TIPSv2-B/14 depth · normals · segmentation — pick an image"
            setPadding(0, 24, 0, 0)
        }
        fun imageView() = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                .also { it.marginStart = 8; it.marginEnd = 8 }
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        fun caption(text: String) = TextView(this).apply {
            this.text = text; textSize = 13f; setTextColor(Color.LTGRAY); gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        fun row(vararg views: ImageView) = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.topMargin = 16 }
            views.forEach { addView(it) }
        }
        fun captions(vararg texts: String) = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            texts.forEach { addView(caption(it)) }
        }
        inputView = imageView(); depthView = imageView()
        normalsView = imageView(); segView = imageView()
        depthLabel = caption("depth (metric, Spectral: near = red)")
        legend = GridLayout(this).apply {
            columnCount = 3
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.topMargin = 16 }
        }
        root.addView(button); root.addView(status)
        root.addView(row(inputView, depthView))
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; addView(caption("input")); addView(depthLabel)
        })
        root.addView(row(normalsView, segView))
        root.addView(captions("surface normals (xyz → RGB)", "ADE20K segmentation (150 classes)"))
        root.addView(legend)
        setContentView(ScrollView(this).apply {
            setBackgroundColor(Color.BLACK)
            fitsSystemWindows = true      // targetSdk 35 is edge-to-edge: keep the button below the status bar
            addView(root)
        })
    }

    private fun process(uri: Uri) {
        status.text = "Running…"
        thread {
            try {
                val bmp = loadBitmap(uri)
                if (predictor == null) {
                    runOnUiThread { status.text = "Loading model (GPU compile)…" }
                    predictor = TipsPredictor(this)
                }
                val res = predictor!!.predict(bmp)
                val depth = res.depthBitmap()
                val normals = res.normalsBitmap()
                val (seg, classes) = res.segmentation()
                runOnUiThread {
                    status.text = "TIPSv2-B/14 DPT  |  ${res.accelerator}  |  ${res.inferenceMs} ms"
                    status.setTextColor(if (res.accelerator == "GPU") Color.GREEN else Color.YELLOW)
                    inputView.setImageBitmap(bmp)
                    depthView.setImageBitmap(depth)
                    normalsView.setImageBitmap(normals)
                    segView.setImageBitmap(seg)
                    depthLabel.text = "depth %.2f–%.2f m (near = red)".format(res.depthMinMetres, res.depthMaxMetres)
                    showLegend(classes)
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "ERROR: ${e.message}"; status.setTextColor(Color.RED) }
            }
        }
    }

    /** Legend of the classes present (largest area first, up to 12). */
    private fun showLegend(classes: List<Pair<Int, Int>>) {
        legend.removeAllViews()
        val total = classes.sumOf { it.second }.coerceAtLeast(1)
        for ((cls, count) in classes.take(12)) {
            val pct = 100f * count / total
            if (pct < 0.5f) break
            legend.addView(LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(8, 6, 8, 6)
                addView(TextView(this@MainActivity).apply {
                    setBackgroundColor(TipsResult.ADE_PALETTE[cls])
                    layoutParams = LinearLayout.LayoutParams(32, 32).also { it.marginEnd = 10 }
                })
                addView(TextView(this@MainActivity).apply {
                    text = "%s %.0f%%".format(TipsResult.ADE_CLASSES[cls], pct)
                    textSize = 12f; setTextColor(Color.WHITE)
                })
            })
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
