package com.lama

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

private const val TAG = "LaMa"
private const val MAX_IMAGE_SIZE = 1024

class MainActivity : ComponentActivity() {

    private lateinit var maskDrawView: MaskDrawView
    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var inpaintButton: Button
    private lateinit var toggleButton: Button
    private lateinit var clearButton: Button

    private var inpainter: Inpainter? = null
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

        val row1 = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, 4, 0, 0)
        }
        pickButton = Button(this).apply {
            text = "Image"
            isEnabled = false
            setOnClickListener { imagePicker.launch("image/*") }
        }
        row1.addView(pickButton)

        inpaintButton = Button(this).apply {
            text = "Inpaint"
            isEnabled = false
            setOnClickListener { runInpaint() }
        }
        row1.addView(inpaintButton)

        toggleButton = Button(this).apply {
            text = "Toggle"
            isEnabled = false
            setOnClickListener { maskDrawView.toggleResult() }
        }
        row1.addView(toggleButton)

        clearButton = Button(this).apply {
            text = "Clear"
            setOnClickListener {
                maskDrawView.clearMask()
                toggleButton.isEnabled = false
                statusText.text = "Draw on the area to erase"
            }
        }
        row1.addView(clearButton)
        root.addView(row1)

        maskDrawView = MaskDrawView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }
        root.addView(maskDrawView)

        setContentView(root)

        executor.execute {
            try {
                inpainter = Inpainter(this)
                runOnUiThread {
                    statusText.text = "Ready (${inpainter!!.acceleratorName}) — select an image"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun loadImage(uri: Uri) {
        val bitmap = decodeBitmap(uri) ?: return
        currentBitmap?.recycle()
        currentBitmap = bitmap
        maskDrawView.setImage(bitmap)
        inpaintButton.isEnabled = true
        toggleButton.isEnabled = false
        statusText.text = "${bitmap.width}x${bitmap.height} — draw on the area to erase"
    }

    private fun runInpaint() {
        val image = currentBitmap ?: return
        val mask = maskDrawView.getMask() ?: return
        if (isProcessing || inpainter == null) return
        isProcessing = true
        inpaintButton.isEnabled = false
        statusText.text = "Inpainting..."

        executor.execute {
            try {
                val result = inpainter!!.inpaint(image, mask)
                runOnUiThread {
                    maskDrawView.setResult(result)
                    statusText.text = "Done in ${inpainter!!.lastInferenceMs}ms — toggle to compare"
                    inpaintButton.isEnabled = true
                    toggleButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Inpaint failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    inpaintButton.isEnabled = true
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
        inpainter?.close()
        currentBitmap?.recycle()
        executor.shutdown()
    }
}
