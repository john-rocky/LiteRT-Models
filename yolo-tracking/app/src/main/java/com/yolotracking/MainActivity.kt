package com.yolotracking

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.util.concurrent.Executors

private const val TAG = "YOLOTracking"

class MainActivity : ComponentActivity() {

    private var detector: ObjectDetector? = null
    private var reIdExtractor: ReIDExtractor? = null
    private val tracker = DeepSORTTracker()

    // Reusable two-thread pipeline (manages camera + bitmap pool)
    private var pipeline: RealtimeCameraPipeline? = null

    // Background executor for model loading and video processing
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    private var totalFrameCount = 0

    // 0 = Accurate (every frame), 1 = Balanced (1/2), 2 = Fast (1/3)
    private var mode = 0
    private val balancedInterval = 2
    private val fastInterval = 3

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: TrackingCanvasView
    private lateinit var fpsText: TextView
    private lateinit var modeSpinner: Spinner
    private lateinit var videoButton: Button
    private lateinit var playButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var progressText: TextView
    private var lastOutputFile: File? = null

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

        // Mode spinner
        val modeNames = arrayOf("Accurate", "Balanced (Re-ID 1/${balancedInterval})", "Fast (Re-ID 1/${fastInterval})")
        modeSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item, modeNames)
            setBackgroundColor(0x88000000.toInt())
        }
        modeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                mode = pos
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        topBar.addView(modeSpinner)

        // Video button
        videoButton = Button(this).apply {
            text = "Process Video"
            setOnClickListener { videoPicker.launch("video/*") }
        }
        topBar.addView(videoButton)

        // Play button (hidden until video is processed)
        playButton = Button(this).apply {
            text = "Play Result"
            visibility = View.GONE
            setOnClickListener { lastOutputFile?.let { playVideo(it) } }
        }
        topBar.addView(playButton)

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
        backgroundExecutor.execute {
            try {
                detector = ObjectDetector(this, "yolo11n.tflite")
                Log.i(TAG, "YOLO loaded")
                fpsText.post { fpsText.text = "YOLO loaded, loading Re-ID..." }

                reIdExtractor = ReIDExtractor(this, "osnet_x0_25.tflite")
                Log.i(TAG, "Re-ID loaded")
                fpsText.post { fpsText.text = "Ready" }

                pipeline?.enabled = true
            } catch (e: Exception) {
                Log.e(TAG, "Load failed: ${e.message}", e)
                fpsText.post { fpsText.text = "Failed: ${e.message}" }
            }
        }
    }

    /** Per-frame ML pipeline: YOLO → (optional Re-ID) → DeepSORT → overlay. */
    private fun runInference(bmp: Bitmap) {
        val det = detector ?: return
        val reid = reIdExtractor ?: return

        val bmpW = bmp.width
        val bmpH = bmp.height

        // 1. Detect
        val (rawDets, _) = det.detect(bmp)

        // 2. Re-ID (skip on intermediate frames in Balanced/Fast mode)
        val interval = when (mode) {
            1 -> balancedInterval
            2 -> fastInterval
            else -> 1
        }
        val runReId = (totalFrameCount % interval == 0)
        val detsWithFeatures = if (runReId) {
            reid.extractFeatures(bmp, rawDets)
        } else {
            rawDets
        }
        totalFrameCount++

        // 3. Track
        val tracked = tracker.update(detsWithFeatures)
        val activeTracks = tracker.activeTracks

        overlayView.post {
            overlayView.setResults(tracked, activeTracks, bmpW, bmpH)
        }

        // Update FPS display from pipeline counter
        val fps = pipeline?.fps ?: 0
        val modeName = when (mode) { 0 -> "Accurate"; 1 -> "Balanced"; else -> "Fast" }
        fpsText.post { fpsText.text = "$fps FPS | ${tracked.size} tracks | $modeName" }
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

        // Video processing always uses high accuracy (Re-ID every frame)
        val processor = VideoProcessor(this, det, reid, reIdInterval = 1)

        backgroundExecutor.execute {
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
                        lastOutputFile = outputFile
                        playButton.visibility = View.VISIBLE
                        // Auto-play the result
                        playVideo(outputFile)
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

    private fun playVideo(file: File) {
        try {
            val uri = FileProvider.getUriForFile(
                this, "com.yolotracking.fileprovider", file
            )
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "video/mp4")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to play video", e)
            Toast.makeText(this, "Cannot open video: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    // ---- Camera live tracking ----

    private fun startCamera() {
        pipeline = RealtimeCameraPipeline(
            activity = this,
            previewView = previewView,
        ) { bmp -> runInference(bmp) }.also {
            it.enabled = false  // wait until models are loaded
            it.start(this)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
        backgroundExecutor.shutdown()
        detector?.close()
        reIdExtractor?.close()
    }
}
