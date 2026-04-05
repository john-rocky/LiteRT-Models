package com.dsine

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

private const val TAG = "DSINE"
private const val MAX_IMAGE_SIZE = 1024

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var originalView: ImageView
    private lateinit var normalView: ImageView
    private lateinit var toggleButton: Button

    private var estimator: NormalEstimator? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var isProcessing = false
    private var currentBitmap: Bitmap? = null
    private var normalBitmap: Bitmap? = null
    private var showingNormal = true

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
            text = "Show Original"
            isEnabled = false
            setOnClickListener { toggleView() }
        }
        buttonRow.addView(toggleButton)

        root.addView(buttonRow)

        // Image views (stacked, toggle visibility)
        val imageContainer = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
            setPadding(16, 8, 16, 16)
        }

        originalView = ImageView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF2A2A2A.toInt())
            visibility = View.GONE
        }
        imageContainer.addView(originalView)

        normalView = ImageView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF2A2A2A.toInt())
        }
        imageContainer.addView(normalView)

        root.addView(imageContainer)

        setContentView(root)

        // Load model
        executor.execute {
            try {
                estimator = NormalEstimator(this)
                runOnUiThread {
                    statusText.text = "Ready (${estimator!!.acceleratorName}) — select an image"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun loadImage(uri: Uri) {
        if (isProcessing || estimator == null) return
        isProcessing = true
        pickButton.isEnabled = false
        toggleButton.isEnabled = false
        statusText.text = "Estimating normals..."

        executor.execute {
            try {
                val bitmap = decodeBitmap(uri) ?: throw Exception("Failed to load image")
                currentBitmap?.recycle()
                currentBitmap = bitmap

                runOnUiThread {
                    originalView.setImageBitmap(bitmap)
                }

                normalBitmap?.recycle()
                val normals = estimator!!.estimate(bitmap)
                normalBitmap = normals

                runOnUiThread {
                    normalView.setImageBitmap(normals)
                    showingNormal = true
                    normalView.visibility = View.VISIBLE
                    originalView.visibility = View.GONE
                    toggleButton.text = "Show Original"
                    toggleButton.isEnabled = true
                    statusText.text = "Estimated ${bitmap.width}x${bitmap.height} " +
                        "in ${estimator!!.lastInferenceMs}ms"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Estimation failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    private fun toggleView() {
        showingNormal = !showingNormal
        if (showingNormal) {
            normalView.visibility = View.VISIBLE
            originalView.visibility = View.GONE
            toggleButton.text = "Show Original"
        } else {
            normalView.visibility = View.GONE
            originalView.visibility = View.VISIBLE
            toggleButton.text = "Show Normals"
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
        estimator?.close()
        currentBitmap?.recycle()
        normalBitmap?.recycle()
        executor.shutdown()
    }
}
