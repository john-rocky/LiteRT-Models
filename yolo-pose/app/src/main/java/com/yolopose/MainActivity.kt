package com.yolopose

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private const val TAG = "YOLOPose"
private const val MAX_IMAGE_SIZE = 1600

class MainActivity : ComponentActivity() {

    private enum class Mode { CAMERA, IMAGE, VIDEO }

    private var estimator: PoseEstimator? = null
    private val executor = Executors.newSingleThreadExecutor()

    @Volatile
    private var isProcessing = false
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()

    @Volatile
    private var mode = Mode.CAMERA

    @Volatile
    private var cancelVideo = false

    private var cameraProvider: ProcessCameraProvider? = null
    private var staticBitmap: Bitmap? = null
    private var displayedVideoFrame: Bitmap? = null

    private val mat = Matrix()
    private val pnt = Paint(Paint.FILTER_BITMAP_FLAG)

    private lateinit var previewView: PreviewView
    private lateinit var staticImageView: ImageView
    private lateinit var overlayView: PoseCanvasView
    private lateinit var fpsText: TextView
    private lateinit var cameraButton: Button
    private lateinit var imageButton: Button
    private lateinit var videoButton: Button

    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { startImageMode(it) } }

    private val videoPicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { startVideoMode(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) initUi()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initUi()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initUi() {
        val root = FrameLayout(this)
        previewView = PreviewView(this)
        staticImageView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFF000000.toInt())
            visibility = View.GONE
        }
        overlayView = PoseCanvasView(this).apply { fitCenter = false }

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 0)
        }

        fpsText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 18f
            text = "Loading model..."
        }
        topBar.addView(fpsText)

        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 8, 0, 0)
        }
        cameraButton = Button(this).apply {
            text = "Camera"
            setOnClickListener { switchToCamera() }
        }
        imageButton = Button(this).apply {
            text = "Image"
            setOnClickListener { imagePicker.launch("image/*") }
        }
        videoButton = Button(this).apply {
            text = "Video"
            setOnClickListener { videoPicker.launch("video/*") }
        }
        buttonRow.addView(cameraButton)
        buttonRow.addView(imageButton)
        buttonRow.addView(videoButton)
        topBar.addView(buttonRow)

        root.addView(
            previewView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        root.addView(
            staticImageView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        root.addView(
            overlayView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        root.addView(
            topBar,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP,
            ),
        )
        setContentView(root)
        updateModeButtons()

        loadModel()
        startCamera()
    }

    private fun updateModeButtons() {
        // Only disable Camera while it's already active. Image / Video always
        // remain tappable so the user can re-pick another file at any time.
        cameraButton.isEnabled = mode != Mode.CAMERA
    }

    private fun switchToCamera() {
        if (mode == Mode.CAMERA) return
        cancelVideo = true  // signal video loop to exit (no-op for image mode)
        if (mode == Mode.IMAGE) exitToCamera()
    }

    private fun loadModel() {
        isProcessing = true
        executor.execute {
            try {
                estimator = PoseEstimator(this)
                fpsText.post { fpsText.text = "yolo26n-pose GPU ready" }
            } catch (e: Exception) {
                Log.e(TAG, "Load failed", e)
                fpsText.post { fpsText.text = "Failed: ${e.message}" }
            }
            isProcessing = false
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider

            @Suppress("DEPRECATION")
            val preview = Preview.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(executor) { proxy ->
                if (mode != Mode.CAMERA || isProcessing || estimator == null) {
                    proxy.close()
                    return@setAnalyzer
                }
                isProcessing = true

                val bmp = proxyToBitmap(proxy)
                proxy.close()

                val (poses, _) = estimator!!.detect(bmp)
                val bmpW = bmp.width
                val bmpH = bmp.height
                bmp.recycle()

                overlayView.post { overlayView.setPoses(poses, bmpW, bmpH) }

                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000) {
                    val fps = frameCount
                    frameCount = 0
                    lastFpsTime = now
                    fpsText.post { fpsText.text = "$fps FPS | ${poses.size} persons" }
                }

                isProcessing = false
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis,
            )
        }, ContextCompat.getMainExecutor(this))
    }

    // ── Image mode ─────────────────────────────────────────────────────────

    private fun startImageMode(uri: Uri) {
        if (estimator == null) {
            fpsText.text = "Model not ready"
            return
        }
        cancelVideo = true  // stop any running video
        mode = Mode.IMAGE
        cameraProvider?.unbindAll()
        previewView.visibility = View.GONE
        staticImageView.visibility = View.VISIBLE
        overlayView.fitCenter = true
        overlayView.setPoses(emptyList(), 1, 1)
        fpsText.text = "Loading image..."
        updateModeButtons()

        executor.execute {
            try {
                val bitmap = decodeBitmap(uri) ?: throw Exception("Failed to decode image")
                val (poses, infMs) = estimator!!.detect(bitmap)
                val w = bitmap.width
                val h = bitmap.height
                runOnUiThread {
                    staticBitmap?.recycle()
                    staticBitmap = bitmap
                    staticImageView.setImageBitmap(bitmap)
                    overlayView.setPoses(poses, w, h)
                    fpsText.text = "${w}x${h} | ${poses.size} persons | ${infMs}ms"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Image processing failed", e)
                runOnUiThread {
                    fpsText.text = "Image error: ${e.message}"
                    exitToCamera()
                }
            }
        }
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        // Probe size for downsampling
        contentResolver.openInputStream(uri).use { input ->
            input ?: return null
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeStream(input, null, opts)
            var sampleSize = 1
            while (opts.outWidth / sampleSize > MAX_IMAGE_SIZE ||
                opts.outHeight / sampleSize > MAX_IMAGE_SIZE
            ) {
                sampleSize *= 2
            }

            val decoded = contentResolver.openInputStream(uri).use { input2 ->
                input2 ?: return null
                opts.inJustDecodeBounds = false
                opts.inSampleSize = sampleSize
                BitmapFactory.decodeStream(input2, null, opts)
            } ?: return null

            // Honor EXIF orientation (front-camera selfies, sideways phone shots, etc.)
            val rotation = contentResolver.openInputStream(uri).use { input3 ->
                input3 ?: return@use 0f
                val exif = ExifInterface(input3)
                when (
                    exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL,
                    )
                ) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                    ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                    ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                    else -> 0f
                }
            }
            if (rotation == 0f) return decoded

            val m = Matrix().apply { postRotate(rotation) }
            val rotated = Bitmap.createBitmap(
                decoded, 0, 0, decoded.width, decoded.height, m, true,
            )
            decoded.recycle()
            return rotated
        }
    }

    // ── Video mode ─────────────────────────────────────────────────────────

    private fun startVideoMode(uri: Uri) {
        if (estimator == null) {
            fpsText.text = "Model not ready"
            return
        }
        // Drop any leftover image-mode bitmap
        staticBitmap?.recycle()
        staticBitmap = null
        staticImageView.setImageDrawable(null)

        mode = Mode.VIDEO
        cancelVideo = false
        cameraProvider?.unbindAll()
        previewView.visibility = View.GONE
        staticImageView.visibility = View.VISIBLE
        overlayView.fitCenter = true
        overlayView.setPoses(emptyList(), 1, 1)
        fpsText.text = "Loading video..."
        updateModeButtons()

        executor.execute { processVideo(uri) }
    }

    private fun processVideo(uri: Uri) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(this, uri)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L
            if (durationMs <= 0L) {
                fpsText.post { fpsText.text = "Cannot read video duration" }
                exitToCamera()
                return
            }

            // Step at ~30 fps. MediaMetadataRetriever can be slow on some devices,
            // so the actual processed FPS will usually be lower than the source.
            val stepMs = 33L
            var t = 0L
            var frameIdx = 0
            val started = System.currentTimeMillis()

            while (t < durationMs && !cancelVideo && mode == Mode.VIDEO) {
                val frame = retriever.getFrameAtTime(
                    t * 1000L,
                    MediaMetadataRetriever.OPTION_CLOSEST,
                ) ?: break

                val (poses, infMs) = estimator!!.detect(frame)
                val w = frame.width
                val h = frame.height
                val idx = frameIdx
                val cur = t

                // Hand the bitmap to the UI thread; recycle the previous one there
                // so we never recycle a bitmap the ImageView is still drawing.
                runOnUiThread {
                    val old = displayedVideoFrame
                    displayedVideoFrame = frame
                    staticImageView.setImageBitmap(frame)
                    overlayView.setPoses(poses, w, h)
                    fpsText.text = "Frame $idx | ${poses.size} persons | ${infMs}ms | " +
                        "${cur / 1000}.${(cur % 1000) / 100}s / ${durationMs / 1000}s"
                    old?.recycle()
                }

                frameIdx++
                t += stepMs
            }

            val elapsed = System.currentTimeMillis() - started
            val avgFps = if (elapsed > 0) frameIdx * 1000.0 / elapsed else 0.0
            fpsText.post {
                fpsText.text = if (cancelVideo) {
                    "Stopped at frame $frameIdx (${"%.1f".format(avgFps)} fps)"
                } else {
                    "Done $frameIdx frames (${"%.1f".format(avgFps)} fps)"
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Video processing failed", e)
            fpsText.post { fpsText.text = "Video error: ${e.message}" }
        } finally {
            try {
                retriever.release()
            } catch (_: Exception) {
            }
            exitToCamera()
        }
    }

    private fun exitToCamera() {
        runOnUiThread {
            mode = Mode.CAMERA
            previewView.visibility = View.VISIBLE
            staticImageView.visibility = View.GONE
            staticImageView.setImageDrawable(null)
            displayedVideoFrame?.recycle()
            displayedVideoFrame = null
            staticBitmap?.recycle()
            staticBitmap = null
            overlayView.fitCenter = false
            overlayView.setPoses(emptyList(), 1, 1)
            updateModeButtons()
            // Re-bind the camera so live preview resumes immediately.
            startCamera()
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private fun proxyToBitmap(proxy: ImageProxy): Bitmap {
        val plane = proxy.planes[0]
        val buf = plane.buffer
        val sw = proxy.width + (plane.rowStride - plane.pixelStride * proxy.width) / plane.pixelStride

        val src = Bitmap.createBitmap(sw, proxy.height, Bitmap.Config.ARGB_8888)
        buf.rewind()
        src.copyPixelsFromBuffer(buf)

        val rot = proxy.imageInfo.rotationDegrees.toFloat()
        if (rot == 0f && sw == proxy.width) return src

        val rw = if (rot == 90f || rot == 270f) proxy.height else proxy.width
        val rh = if (rot == 90f || rot == 270f) proxy.width else proxy.height
        val dst = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
        val c = Canvas(dst)
        mat.reset()
        mat.postRotate(rot, proxy.width / 2f, proxy.height / 2f)
        mat.postTranslate((rw - proxy.width) / 2f, (rh - proxy.height) / 2f)
        c.drawBitmap(src, mat, pnt)
        src.recycle()
        return dst
    }

    override fun onDestroy() {
        super.onDestroy()
        cancelVideo = true
        executor.shutdown()
        estimator?.close()
        displayedVideoFrame?.recycle()
        displayedVideoFrame = null
        staticBitmap?.recycle()
        staticBitmap = null
    }
}
