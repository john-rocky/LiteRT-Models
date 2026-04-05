package com.smolvlm

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.exifinterface.media.ExifInterface
import java.util.concurrent.Executors

private const val TAG = "SmolVLM"
private const val MAX_IMAGE_SIZE = 1024

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var promptInput: EditText
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    private var vlm: VLMInference? = null
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
            setPadding(24, 48, 24, 24)
        }

        statusText = TextView(this).apply {
            textSize = 15f
            setTextColor(0xFFCCCCCC.toInt())
            setPadding(0, 8, 0, 8)
            text = "Loading model..."
        }
        root.addView(statusText)

        // Image + Pick button
        val imageRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 4, 0, 4)
        }

        pickButton = Button(this).apply {
            text = "Select Image"
            isEnabled = false
            setOnClickListener { imagePicker.launch("image/*") }
        }
        imageRow.addView(pickButton)

        imageView = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(200, 200).apply { marginStart = 16 }
            scaleType = ImageView.ScaleType.CENTER_CROP
            setBackgroundColor(0xFF2A2A2A.toInt())
        }
        imageRow.addView(imageView)
        root.addView(imageRow)

        // Prompt input
        val promptRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 8, 0, 8)
        }

        promptInput = EditText(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            hint = "Ask about the image..."
            setText("Describe this image.")
            textSize = 15f
            setTextColor(0xFFFFFFFF.toInt())
            setHintTextColor(0xFF888888.toInt())
            setBackgroundColor(0xFF2A2A2A.toInt())
            setPadding(16, 12, 16, 12)
            isSingleLine = true
        }
        promptRow.addView(promptInput)

        val askButton = Button(this).apply {
            text = "Ask"
            setOnClickListener { generateResponse() }
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { marginStart = 8 }
        }
        promptRow.addView(askButton)
        root.addView(promptRow)

        // Response
        val responseLabel = TextView(this).apply {
            text = "Response"
            textSize = 16f
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 16, 0, 8)
        }
        root.addView(responseLabel)

        val scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }

        resultText = TextView(this).apply {
            textSize = 18f
            setTextColor(0xFFEEEEEE.toInt())
            setBackgroundColor(0xFF2A2A2A.toInt())
            setPadding(24, 24, 24, 24)
            text = "Select an image and ask a question..."
            setTextIsSelectable(true)
        }
        scrollView.addView(resultText)
        root.addView(scrollView)

        setContentView(root)

        // Load model
        executor.execute {
            try {
                vlm = VLMInference(this)
                runOnUiThread {
                    statusText.text = "Ready (vision: ${vlm!!.acceleratorName})"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun loadImage(uri: Uri) {
        val bitmap = decodeBitmap(uri) ?: return
        currentBitmap?.recycle()
        currentBitmap = bitmap
        imageView.setImageBitmap(bitmap)
        statusText.text = "Image loaded (${bitmap.width}x${bitmap.height}) — ask a question"
    }

    private fun generateResponse() {
        val bitmap = currentBitmap
        if (bitmap == null || isProcessing || vlm == null) return

        val prompt = promptInput.text.toString().trim()
        if (prompt.isEmpty()) return

        isProcessing = true
        pickButton.isEnabled = false
        statusText.text = "Generating..."
        resultText.text = ""

        executor.execute {
            try {
                val text = vlm!!.generate(bitmap, prompt) { partial ->
                    runOnUiThread { resultText.text = partial }
                }
                runOnUiThread {
                    resultText.text = text.ifEmpty { "(no response)" }
                    statusText.text = "Vision: ${vlm!!.lastEncodeMs}ms | " +
                        "Generate: ${vlm!!.lastDecodeMs}ms"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                }
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
        vlm?.close()
        currentBitmap?.recycle()
        executor.shutdown()
    }
}
