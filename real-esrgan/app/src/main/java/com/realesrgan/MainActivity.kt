package com.realesrgan

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

private const val TAG = "RealESRGAN"
private const val MAX_INPUT_SIZE = 512

class MainActivity : ComponentActivity() {

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var progressBar: ProgressBar

    private var upscaler: Upscaler? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var isProcessing = false

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

        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            setPadding(24, 0, 24, 0)
            max = 100
            visibility = View.GONE
        }
        root.addView(progressBar)

        pickButton = Button(this).apply {
            text = "Select Image"
            isEnabled = false
            setOnClickListener { imagePicker.launch("image/*") }
        }
        root.addView(pickButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.CENTER_HORIZONTAL })

        imageView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
        }
        root.addView(imageView)

        setContentView(root)

        executor.execute {
            try {
                upscaler = Upscaler(this)
                runOnUiThread {
                    statusText.text = "Ready — select an image to upscale 4x"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun processImage(uri: Uri) {
        if (isProcessing || upscaler == null) return
        isProcessing = true
        pickButton.isEnabled = false
        progressBar.progress = 0
        progressBar.visibility = View.VISIBLE
        statusText.text = "Loading image..."

        executor.execute {
            try {
                val bitmap = loadBitmap(uri) ?: throw Exception("Failed to load image")
                val w = bitmap.width
                val h = bitmap.height

                runOnUiThread {
                    statusText.text = "Upscaling ${w}x${h} -> ${w * 4}x${h * 4}..."
                }

                val t = System.nanoTime()
                val result = upscaler!!.upscale(bitmap) { progress ->
                    runOnUiThread {
                        progressBar.progress = (progress * 100).toInt()
                    }
                }
                val ms = (System.nanoTime() - t) / 1_000_000
                bitmap.recycle()

                runOnUiThread {
                    imageView.setImageBitmap(result)
                    progressBar.visibility = View.GONE
                    statusText.text = "${w}x${h} -> ${result.width}x${result.height} in ${ms}ms"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Upscale failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    progressBar.visibility = View.GONE
                    pickButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    private fun loadBitmap(uri: Uri): Bitmap? {
        val input = contentResolver.openInputStream(uri) ?: return null
        val opts = BitmapFactory.Options()
        opts.inJustDecodeBounds = true
        BitmapFactory.decodeStream(input, null, opts)
        input.close()

        // Downsample if too large
        var sampleSize = 1
        while (opts.outWidth / sampleSize > MAX_INPUT_SIZE || opts.outHeight / sampleSize > MAX_INPUT_SIZE) {
            sampleSize *= 2
        }

        val input2 = contentResolver.openInputStream(uri) ?: return null
        opts.inJustDecodeBounds = false
        opts.inSampleSize = sampleSize
        val bitmap = BitmapFactory.decodeStream(input2, null, opts)
        input2.close()

        // Handle EXIF rotation
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
        upscaler?.close()
        executor.shutdown()
    }
}
