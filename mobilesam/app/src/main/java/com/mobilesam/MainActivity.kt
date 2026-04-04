package com.mobilesam

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

private const val TAG = "MobileSAM"
private const val MAX_IMAGE_SIZE = 1024

class MainActivity : ComponentActivity() {

    private lateinit var segmentationView: SegmentationView
    private lateinit var statusText: TextView
    private lateinit var pickButton: Button

    private var segmenter: MobileSAMSegmenter? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var isProcessing = false
    private var currentBitmap: Bitmap? = null

    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { loadImage(it) } }

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
            text = "Loading models..."
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

        val clearButton = Button(this).apply {
            text = "Clear"
            setOnClickListener {
                segmentationView.clearMask()
                statusText.text = "Tap on the image to segment"
            }
        }
        buttonRow.addView(clearButton)

        root.addView(buttonRow)

        segmentationView = SegmentationView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
            onTap = { x, y -> onImageTap(x, y) }
        }
        root.addView(segmentationView)

        setContentView(root)

        // Load models
        executor.execute {
            try {
                segmenter = MobileSAMSegmenter(this)
                runOnUiThread {
                    statusText.text = "Ready (${segmenter!!.acceleratorName}) — select an image"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun loadImage(uri: Uri) {
        if (isProcessing || segmenter == null) return
        isProcessing = true
        pickButton.isEnabled = false
        statusText.text = "Encoding image..."

        executor.execute {
            try {
                val bitmap = decodeBitmap(uri) ?: throw Exception("Failed to load image")
                currentBitmap?.recycle()
                currentBitmap = bitmap

                runOnUiThread { segmentationView.setImage(bitmap) }

                // Run encoder
                segmenter!!.encodeImage(bitmap)

                runOnUiThread {
                    statusText.text = "Encoded ${bitmap.width}x${bitmap.height} in ${segmenter!!.lastEncodeTimeMs}ms — tap to segment"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Encode failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    private fun onImageTap(x: Float, y: Float) {
        if (isProcessing || segmenter == null) return
        isProcessing = true
        statusText.text = "Segmenting..."

        executor.execute {
            try {
                val mask = segmenter!!.segment(x, y)
                runOnUiThread {
                    segmentationView.setMask(mask)
                    statusText.text = "Decode: ${segmenter!!.lastDecodeTimeMs}ms | Tap again or select new image"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Segment failed", e)
                runOnUiThread { statusText.text = "Error: ${e.message}" }
            }
            isProcessing = false
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
        segmenter?.close()
        currentBitmap?.recycle()
        executor.shutdown()
    }
}
