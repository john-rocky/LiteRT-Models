package com.yolotracking

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors

private const val TAG = "YOLOTracking"

class MainActivity : ComponentActivity() {

    private var detector: ObjectDetector? = null
    private var reIdExtractor: ReIDExtractor? = null
    private val tracker = DeepSORTTracker()
    private val executor = Executors.newSingleThreadExecutor()

    @Volatile
    private var isProcessing = false
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var totalFrameCount = 0

    private val reIdInterval = 3

    private val mat = Matrix()
    private val pnt = Paint(Paint.FILTER_BITMAP_FLAG)

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: TrackingCanvasView
    private lateinit var fpsText: TextView
    private lateinit var videoButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var progressText: TextView

    private val videoPicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { startVideoProcessing(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) initCamera()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initCamera() {
        val root = FrameLayout(this)
        previewView = PreviewView(this)
        overlayView = TrackingCanvasView(this)

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 0)
        }

        fpsText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 18f
            text = "Loading..."
        }
        topBar.addView(fpsText)

        // Video button
        videoButton = Button(this).apply {
            text = "Process Video"
            setOnClickListener { videoPicker.launch("video/*") }
        }
        topBar.addView(videoButton)

        // Progress bar (hidden by default)
        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            visibility = View.GONE
            max = 100
        }
        topBar.addView(progressBar)

        progressText = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setShadowLayer(4f, 0f, 0f, 0xFF000000.toInt())
            textSize = 14f
            visibility = View.GONE
        }
        topBar.addView(progressText)

        root.addView(previewView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(overlayView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(topBar, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP))
        setContentView(root)

        loadModels()
        startCamera()
    }

    private fun loadModels() {
        fpsText.post { fpsText.text = "Loading YOLO + OSNet..." }
        executor.execute {
            try {
                detector = ObjectDetector(this, "yolo11n.tflite")
                Log.i(TAG, "YOLO loaded")
                fpsText.post { fpsText.text = "YOLO loaded, loading Re-ID..." }

                reIdExtractor = ReIDExtractor(this, "osnet_x0_25.tflite")
                Log.i(TAG, "Re-ID loaded")
                fpsText.post { fpsText.text = "Ready" }
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}", e)
                fpsText.post { fpsText.text = "Failed: ${e.message}" }
            }
            isProcessing = false
        }
    }

    // ---- Video processing ----

    private fun startVideoProcessing(uri: Uri) {
        val det = detector ?: return
        val reid = reIdExtractor ?: return

        videoButton.isEnabled = false
        progressBar.visibility = View.VISIBLE
        progressBar.progress = 0
        progressText.visibility = View.VISIBLE
        progressText.text = "Preparing..."

        val processor = VideoProcessor(this, det, reid, reIdInterval = reIdInterval)

        executor.execute {
            processor.process(uri, object : VideoProcessor.ProgressListener {
                override fun onProgress(currentFrame: Int, totalFrames: Int) {
                    val pct = (currentFrame * 100) / totalFrames
                    runOnUiThread {
                        progressBar.progress = pct
                        progressText.text = "Processing: $currentFrame / $totalFrames frames ($pct%)"
                    }
                }

                override fun onComplete(outputFile: File) {
                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        progressText.text = "Saved: ${outputFile.name}"
                        videoButton.isEnabled = true
                        Toast.makeText(
                            this@MainActivity,
                            "Saved to ${outputFile.absolutePath}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

                override fun onError(message: String) {
                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        progressText.text = "Error: $message"
                        videoButton.isEnabled = true
                    }
                }
            })
        }
    }

    // ---- Camera live tracking ----

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

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
                if (isProcessing || detector == null || reIdExtractor == null) {
                    proxy.close()
                    return@setAnalyzer
                }
                isProcessing = true

                val bmp = proxyToBitmap(proxy)
                proxy.close()
                val bmpW = bmp.width
                val bmpH = bmp.height

                // 1. Detect
                val (rawDets, _) = detector!!.detect(bmp)

                // 2. Extract Re-ID features
                val detsWithFeatures = reIdExtractor!!.extractFeatures(bmp, rawDets)
                bmp.recycle()
                totalFrameCount++

                // 3. Track
                val tracked = tracker.update(detsWithFeatures)
                val activeTracks = tracker.activeTracks

                overlayView.post {
                    overlayView.setResults(tracked, activeTracks, bmpW, bmpH)
                }

                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTime >= 1000) {
                    val fps = frameCount
                    frameCount = 0
                    lastFpsTime = now
                    val trackCount = tracked.size
                    fpsText.post { fpsText.text = "$fps FPS | $trackCount tracks" }
                }

                isProcessing = false
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

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
        executor.shutdown()
        detector?.close()
        reIdExtractor?.close()
    }
}
