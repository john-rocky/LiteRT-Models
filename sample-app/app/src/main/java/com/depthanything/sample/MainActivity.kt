package com.depthanything.sample

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors

private const val TAG = "DepthAnything"
private const val MODEL_FILE = "depth_anything_v2_keras.tflite"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { DepthCameraApp() }
    }
}

@Composable
fun DepthCameraApp() {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) launcher.launch(Manifest.permission.CAMERA)
    }

    if (hasCameraPermission) {
        CameraDepthScreen()
    } else {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera permission required")
        }
    }
}

@Composable
fun CameraDepthScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var fps by remember { mutableIntStateOf(0) }
    var initError by remember { mutableStateOf<String?>(null) }

    val estimator = remember {
        try {
            DepthEstimator(context, MODEL_FILE)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init model", e)
            initError = e.message
            null
        }
    }

    DisposableEffect(Unit) {
        onDispose { estimator?.close() }
    }

    Box(Modifier.fillMaxSize()) {
        if (initError != null) {
            Text(
                "Model load failed: $initError",
                modifier = Modifier.align(Alignment.Center).padding(16.dp),
                color = MaterialTheme.colorScheme.error
            )
            return
        }

        val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

        // Use ImageView for zero-copy depth rendering (no Compose recomposition)
        var depthImageView: ImageView? = null

        // Pre-allocate reusable bitmaps
        val depthDisplayBitmap = remember(estimator) {
            estimator?.let {
                Bitmap.createBitmap(it.inputWidth, it.inputHeight, Bitmap.Config.ARGB_8888)
            }
        }

        AndroidView(
            factory = { ctx ->
                // FrameLayout with camera preview + depth overlay
                val frame = android.widget.FrameLayout(ctx)

                val previewView = PreviewView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }
                frame.addView(previewView)

                val imageView = ImageView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    imageAlpha = 180  // ~70% opacity
                }
                frame.addView(imageView)
                depthImageView = imageView

                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    @Suppress("DEPRECATION")
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(android.util.Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()

                    var frameCount = 0
                    var lastFpsTime = System.currentTimeMillis()

                    // Pre-allocate camera frame bitmap
                    var cameraBitmap: Bitmap? = null

                    imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (estimator != null && depthDisplayBitmap != null) {
                            // Reuse or create camera bitmap
                            val w = imageProxy.width
                            val h = imageProxy.height
                            val rotation = imageProxy.imageInfo.rotationDegrees
                            val rotW = if (rotation == 90 || rotation == 270) h else w
                            val rotH = if (rotation == 90 || rotation == 270) w else h

                            if (cameraBitmap == null || cameraBitmap!!.width != rotW || cameraBitmap!!.height != rotH) {
                                cameraBitmap?.recycle()
                                cameraBitmap = Bitmap.createBitmap(rotW, rotH, Bitmap.Config.ARGB_8888)
                            }

                            fillBitmapFromImageProxy(imageProxy, cameraBitmap!!)

                            val ms = estimator.predict(cameraBitmap!!, depthDisplayBitmap)

                            imageView.post {
                                imageView.setImageBitmap(depthDisplayBitmap)
                            }

                            frameCount++
                            val now = System.currentTimeMillis()
                            if (now - lastFpsTime >= 1000) {
                                val currentFps = frameCount
                                frameCount = 0
                                lastFpsTime = now
                                imageView.post { fps = currentFps }
                            }
                        }
                        imageProxy.close()
                    }

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA,
                        preview, imageAnalysis
                    )
                }, ContextCompat.getMainExecutor(ctx))

                frame
            },
            modifier = Modifier.fillMaxSize()
        )

        // FPS counter
        Text(
            text = "${fps} FPS",
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp),
            fontSize = 24.sp,
            color = androidx.compose.ui.graphics.Color.White
        )
    }
}

/** Fill pre-allocated bitmap from ImageProxy (RGBA_8888) with rotation. */
private val tempMatrix = Matrix()
private val tempPaint = Paint(Paint.FILTER_BITMAP_FLAG)

private fun fillBitmapFromImageProxy(imageProxy: ImageProxy, dst: Bitmap) {
    val plane = imageProxy.planes[0]
    val buffer = plane.buffer
    val pixelStride = plane.pixelStride
    val rowStride = plane.rowStride
    val rowPadding = rowStride - pixelStride * imageProxy.width
    val srcW = imageProxy.width + rowPadding / pixelStride

    // Decode into a temp bitmap (unavoidable for copyPixelsFromBuffer)
    val src = Bitmap.createBitmap(srcW, imageProxy.height, Bitmap.Config.ARGB_8888)
    buffer.rewind()
    src.copyPixelsFromBuffer(buffer)

    val rotation = imageProxy.imageInfo.rotationDegrees.toFloat()
    val canvas = Canvas(dst)
    tempMatrix.reset()

    if (rotation != 0f) {
        // Rotate around center, then scale to fill dst
        val rotW = if (rotation == 90f || rotation == 270f) imageProxy.height else imageProxy.width
        val rotH = if (rotation == 90f || rotation == 270f) imageProxy.width else imageProxy.height
        tempMatrix.postRotate(rotation, imageProxy.width / 2f, imageProxy.height / 2f)
        tempMatrix.postTranslate(
            (rotW - imageProxy.width) / 2f,
            (rotH - imageProxy.height) / 2f
        )
        tempMatrix.postScale(dst.width.toFloat() / rotW, dst.height.toFloat() / rotH)
    } else {
        tempMatrix.setScale(
            dst.width.toFloat() / imageProxy.width,
            dst.height.toFloat() / imageProxy.height
        )
    }

    canvas.drawBitmap(src, tempMatrix, tempPaint)
    src.recycle()
}
