package com.moge

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

private const val TAG = "MoGe"
private const val MAX_IMAGE_SIZE = 1024

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var imageView: ImageView
    private lateinit var pointCloudView: PointCloudView
    private lateinit var infoText: TextView
    private lateinit var modeGroup: RadioGroup

    private var predictor: MoGePredictor? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var isProcessing = false

    private var currentBitmap: Bitmap? = null
    private var normalBitmap: Bitmap? = null
    private var depthBitmap: Bitmap? = null
    private var currentResult: MoGeResult? = null
    private var currentMode = Mode.NORMAL

    private enum class Mode { NORMAL, DEPTH, CLOUD, INFO }

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
            textSize = 14f
            setTextColor(0xFFCCCCCC.toInt())
            setPadding(24, 8, 24, 4)
            text = "Loading model..."
        }
        root.addView(statusText)

        // Pick button
        pickButton = Button(this).apply {
            text = "Select Image"
            isEnabled = false
            setOnClickListener { imagePicker.launch("image/*") }
        }
        val pickRow = LinearLayout(this).apply {
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, 0, 0, 4)
        }
        pickRow.addView(pickButton)
        root.addView(pickRow)

        // Mode selector
        modeGroup = RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(8, 0, 8, 4)
        }
        val modes = listOf("Normal" to Mode.NORMAL, "Depth" to Mode.DEPTH, "3D" to Mode.CLOUD, "Info" to Mode.INFO)
        for ((label, mode) in modes) {
            val rb = RadioButton(this).apply {
                text = label
                setTextColor(0xFFCCCCCC.toInt())
                id = mode.ordinal
            }
            modeGroup.addView(rb)
        }
        modeGroup.check(Mode.NORMAL.ordinal)
        modeGroup.setOnCheckedChangeListener { _, id ->
            currentMode = Mode.entries[id]
            updateDisplay()
        }
        root.addView(modeGroup)

        // Content area
        val container = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
            setPadding(8, 4, 8, 8)
        }

        imageView = ImageView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF2A2A2A.toInt())
        }
        container.addView(imageView)

        pointCloudView = PointCloudView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            visibility = View.GONE
        }
        container.addView(pointCloudView)

        infoText = TextView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setPadding(24, 16, 24, 16)
            textSize = 13f
            setTextColor(0xFFCCCCCC.toInt())
            setBackgroundColor(0xFF2A2A2A.toInt())
            visibility = View.GONE
        }
        container.addView(infoText)

        root.addView(container)
        setContentView(root)

        // Load model
        executor.execute {
            try {
                predictor = MoGePredictor(this)
                runOnUiThread {
                    statusText.text = "Ready (${predictor!!.acceleratorName}) — select an image"
                    pickButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    private fun loadImage(uri: Uri) {
        if (isProcessing || predictor == null) return
        isProcessing = true
        pickButton.isEnabled = false
        statusText.text = "Predicting geometry..."

        executor.execute {
            try {
                val bitmap = decodeBitmap(uri) ?: throw Exception("Failed to load image")
                currentBitmap?.recycle()
                currentBitmap = bitmap

                val result = predictor!!.predict(bitmap)
                currentResult = result

                normalBitmap?.recycle()
                normalBitmap = result.normalBitmap()

                depthBitmap?.recycle()
                depthBitmap = result.depthBitmap()

                runOnUiThread {
                    statusText.text = "${bitmap.width}x${bitmap.height} → ${result.inferenceMs}ms (${predictor!!.acceleratorName})"
                    pickButton.isEnabled = true
                    updateDisplay()
                }

                // Update point cloud on GL thread
                pointCloudView.setPointCloud(result)

            } catch (e: Exception) {
                Log.e(TAG, "Prediction failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    private fun updateDisplay() {
        imageView.visibility = View.GONE
        pointCloudView.visibility = View.GONE
        infoText.visibility = View.GONE

        when (currentMode) {
            Mode.NORMAL -> {
                imageView.visibility = View.VISIBLE
                normalBitmap?.let { imageView.setImageBitmap(it) }
                    ?: currentBitmap?.let { imageView.setImageBitmap(it) }
            }
            Mode.DEPTH -> {
                imageView.visibility = View.VISIBLE
                depthBitmap?.let { imageView.setImageBitmap(it) }
                    ?: currentBitmap?.let { imageView.setImageBitmap(it) }
            }
            Mode.CLOUD -> {
                pointCloudView.visibility = View.VISIBLE
                pointCloudView.requestRender()
            }
            Mode.INFO -> {
                infoText.visibility = View.VISIBLE
                val r = currentResult
                if (r != null) {
                    // Compute depth stats
                    var minZ = Float.MAX_VALUE
                    var maxZ = Float.MIN_VALUE
                    var validCount = 0
                    for (i in 0 until r.width * r.height) {
                        if (r.mask[i] > 0.5f) {
                            val z = r.points[i * 3 + 2]
                            if (z < minZ) minZ = z
                            if (z > maxZ) maxZ = z
                            validCount++
                        }
                    }
                    infoText.text = buildString {
                        appendLine("MoGe-2 ViT-S Geometry Estimation")
                        appendLine()
                        appendLine("Model input:  ${r.width} x ${r.height}")
                        appendLine("Inference:    ${r.inferenceMs} ms")
                        appendLine("Accelerator:  ${predictor?.acceleratorName ?: "?"}")
                        appendLine()
                        appendLine("Metric scale: %.4f".format(r.metricScale))
                        appendLine("Valid pixels: $validCount / ${r.width * r.height} (%.1f%%)".format(
                            validCount * 100f / (r.width * r.height)
                        ))
                        appendLine()
                        appendLine("Depth range (z):")
                        appendLine("  min: %.4f".format(minZ))
                        appendLine("  max: %.4f".format(maxZ))
                        appendLine("  range: %.4f".format(maxZ - minZ))
                        appendLine()
                        appendLine("Point map: affine 3D (exp remap)")
                        appendLine("Normal: L2-normalized surface normals")
                        appendLine("Mask: sigmoid confidence (>0.5 valid)")
                    }
                } else {
                    infoText.text = "Select an image to see geometry info."
                }
            }
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

    override fun onPause() {
        super.onPause()
        pointCloudView.onPause()
    }

    override fun onResume() {
        super.onResume()
        pointCloudView.onResume()
    }

    override fun onDestroy() {
        super.onDestroy()
        predictor?.close()
        currentBitmap?.recycle()
        normalBitmap?.recycle()
        depthBitmap?.recycle()
        executor.shutdown()
    }
}
