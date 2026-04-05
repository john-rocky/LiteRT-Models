package com.clip

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Typeface
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

private const val TAG = "CLIP"
private const val MAX_IMAGE_SIZE = 1024
private const val TOP_K = 10

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var imageView: ImageView
    private lateinit var resultsContainer: LinearLayout

    private var classifier: CLIPClassifier? = null
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
        root.addView(buttonRow)

        // Scrollable content area
        val scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }

        val contentLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 8, 16, 16)
        }

        imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF2A2A2A.toInt())
        }
        contentLayout.addView(imageView)

        // Results header
        val resultsHeader = TextView(this).apply {
            text = "Classification Results"
            textSize = 16f
            setTextColor(0xFFFFFFFF.toInt())
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, 16, 0, 8)
            visibility = View.GONE
        }
        contentLayout.addView(resultsHeader)

        resultsContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        contentLayout.addView(resultsContainer)

        scrollView.addView(contentLayout)
        root.addView(scrollView)

        setContentView(root)

        // Load model
        executor.execute {
            try {
                classifier = CLIPClassifier(this)
                val labels = classifier!!.getLabels()
                val hasText = classifier!!.hasTextEncoder()
                runOnUiThread {
                    statusText.text = "Ready (${classifier!!.acceleratorName}) — " +
                        "${labels.size} labels" +
                        if (hasText) " + custom" else ""
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun loadImage(uri: Uri) {
        if (isProcessing || classifier == null) return
        isProcessing = true
        pickButton.isEnabled = false
        statusText.text = "Classifying..."
        resultsContainer.removeAllViews()

        executor.execute {
            try {
                val bitmap = decodeBitmap(uri) ?: throw Exception("Failed to load image")
                currentBitmap?.recycle()
                currentBitmap = bitmap

                runOnUiThread { imageView.setImageBitmap(bitmap) }

                val results = classifier!!.classify(bitmap)
                val topResults = results.take(TOP_K)

                runOnUiThread {
                    displayResults(topResults)
                    statusText.text = "Classified ${bitmap.width}x${bitmap.height} " +
                        "in ${classifier!!.lastInferenceMs}ms"
                    pickButton.isEnabled = true

                    // Show results header
                    val header = (imageView.parent as LinearLayout).getChildAt(1) as TextView
                    header.visibility = View.VISIBLE
                }
            } catch (e: Exception) {
                Log.e(TAG, "Classification failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    private fun displayResults(results: List<CLIPClassifier.ClassificationResult>) {
        resultsContainer.removeAllViews()

        val maxScore = results.firstOrNull()?.score ?: 0f

        for (result in results) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, 6, 0, 6)
            }

            // Label
            val labelText = TextView(this).apply {
                text = result.label
                textSize = 14f
                setTextColor(0xFFFFFFFF.toInt())
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.25f)
            }
            row.addView(labelText)

            // Bar container
            val barContainer = FrameLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(0, 28, 0.55f).apply {
                    marginStart = 8
                    marginEnd = 8
                }
                setBackgroundColor(0xFF333333.toInt())
            }

            // Bar fill
            val barWidth = if (maxScore > 0f) result.score / maxScore else 0f
            val bar = View(this).apply {
                layoutParams = FrameLayout.LayoutParams(
                    0, ViewGroup.LayoutParams.MATCH_PARENT
                )
                setBackgroundColor(getBarColor(result.score))
            }
            barContainer.addView(bar)

            // Set bar width after layout
            barContainer.post {
                bar.layoutParams = bar.layoutParams.apply {
                    width = (barContainer.width * barWidth).toInt()
                }
            }

            row.addView(barContainer)

            // Percentage
            val pctText = TextView(this).apply {
                text = String.format("%.1f%%", result.score * 100f)
                textSize = 13f
                setTextColor(0xFFAAAAAA.toInt())
                gravity = Gravity.END
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.2f)
            }
            row.addView(pctText)

            resultsContainer.addView(row)
        }
    }

    private fun getBarColor(score: Float): Int {
        // Green for high confidence, orange for medium, red for low
        return when {
            score > 0.5f -> 0xFF4CAF50.toInt()   // Green
            score > 0.2f -> 0xFFFF9800.toInt()    // Orange
            score > 0.1f -> 0xFFFFC107.toInt()    // Amber
            else -> 0xFF9E9E9E.toInt()            // Grey
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
        classifier?.close()
        currentBitmap?.recycle()
        executor.shutdown()
    }
}
