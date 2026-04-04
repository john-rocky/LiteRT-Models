package com.rmbg

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
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

private const val TAG = "RMBG"
private const val MAX_IMAGE_SIZE = 1024

class MainActivity : ComponentActivity() {

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var toggleButton: Button

    private var remover: BackgroundRemover? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var isProcessing = false

    private var originalBitmap: Bitmap? = null
    private var resultBitmap: Bitmap? = null
    private var showingResult = false

    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { processImage(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF1A1A1A.toInt())
            setPadding(0, 48, 0, 0)
        }

        statusText = TextView(this).apply {
            textSize = 15f
            setTextColor(0xFFCCCCCC.toInt())
            setPadding(24, 8, 24, 8)
            text = "Loading model..."
        }
        root.addView(statusText)

        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, 4, 0, 4)
        }

        pickButton = Button(this).apply {
            text = "Select Image"
            isEnabled = false
            setOnClickListener { imagePicker.launch("image/*") }
        }
        buttonRow.addView(pickButton)

        toggleButton = Button(this).apply {
            text = "Toggle"
            isEnabled = false
            setOnClickListener { toggleView() }
        }
        buttonRow.addView(toggleButton)

        root.addView(buttonRow)

        imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF2A2A2A.toInt())
        }
        root.addView(imageView)

        setContentView(root)

        executor.execute {
            try {
                remover = BackgroundRemover(this)
                runOnUiThread {
                    statusText.text = "Ready (${remover!!.acceleratorName}) — select an image"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun processImage(uri: Uri) {
        if (isProcessing || remover == null) return
        isProcessing = true
        pickButton.isEnabled = false
        toggleButton.isEnabled = false
        statusText.text = "Processing..."

        executor.execute {
            try {
                val bitmap = decodeBitmap(uri) ?: throw Exception("Failed to load image")
                originalBitmap?.recycle()
                originalBitmap = bitmap

                val result = remover!!.removeBackground(bitmap)
                resultBitmap?.recycle()
                resultBitmap = result
                showingResult = true

                runOnUiThread {
                    imageView.setBackgroundColor(Color.WHITE)
                    imageView.setImageBitmap(result)
                    statusText.text = "${bitmap.width}x${bitmap.height} in ${remover!!.lastInferenceMs}ms — toggle to compare"
                    pickButton.isEnabled = true
                    toggleButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Process failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    private fun toggleView() {
        if (showingResult) {
            imageView.setBackgroundColor(0xFF2A2A2A.toInt())
            imageView.setImageBitmap(originalBitmap)
            showingResult = false
            statusText.text = "Original"
        } else {
            imageView.setBackgroundColor(Color.WHITE)
            imageView.setImageBitmap(resultBitmap)
            showingResult = true
            statusText.text = "Background removed — ${remover!!.lastInferenceMs}ms"
        }
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        val input = contentResolver.openInputStream(uri) ?: return null
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeStream(input, null, opts)
        input.close()

        var sampleSize = 1
        while (opts.outWidth / sampleSize > MAX_IMAGE_SIZE || opts.outHeight / sampleSize > MAX_IMAGE_SIZE) {
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
        remover?.close()
        originalBitmap?.recycle()
        resultBitmap?.recycle()
        executor.shutdown()
    }
}
