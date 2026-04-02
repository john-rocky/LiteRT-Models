package com.depthanything.sample

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
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
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
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

    var depthBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var fps by remember { mutableIntStateOf(0) }
    var initError by remember { mutableStateOf<String?>(null) }

    // Initialize depth estimator
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

    // Output bitmap (reused)
    val outputBitmap = remember(estimator) {
        estimator?.let {
            Bitmap.createBitmap(it.inputWidth, it.inputHeight, Bitmap.Config.ARGB_8888)
        }
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

        // Camera preview (background)
        val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.BACKPRESSURE_STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()

                    var frameCount = 0
                    var lastFpsTime = System.currentTimeMillis()

                    imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        val bitmap = imageProxyToBitmap(imageProxy)
                        if (bitmap != null && estimator != null && outputBitmap != null) {
                            val ms = estimator.predict(bitmap, outputBitmap)
                            bitmap.recycle()
                            depthBitmap = outputBitmap.copy(Bitmap.Config.ARGB_8888, false)

                            frameCount++
                            val now = System.currentTimeMillis()
                            if (now - lastFpsTime >= 1000) {
                                fps = frameCount
                                frameCount = 0
                                lastFpsTime = now
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
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // Depth overlay
        depthBitmap?.let { bmp ->
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "Depth map",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
                alpha = 0.7f
            )
        }

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

private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
    val plane = imageProxy.planes[0]
    val buffer = plane.buffer
    val pixelStride = plane.pixelStride
    val rowStride = plane.rowStride
    val rowPadding = rowStride - pixelStride * imageProxy.width

    val bitmap = Bitmap.createBitmap(
        imageProxy.width + rowPadding / pixelStride,
        imageProxy.height,
        Bitmap.Config.ARGB_8888
    )
    bitmap.copyPixelsFromBuffer(buffer)

    // Crop padding and apply rotation
    val cropped = Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height)
    if (cropped !== bitmap) bitmap.recycle()

    val rotation = imageProxy.imageInfo.rotationDegrees
    if (rotation == 0) return cropped

    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    val rotated = Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
    if (rotated !== cropped) cropped.recycle()
    return rotated
}
