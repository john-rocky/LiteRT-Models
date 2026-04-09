package com.yolotracking

import android.content.Context
import android.graphics.*
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * Re-ID feature extractor using OSNet on LiteRT CompiledModel GPU.
 * Input:  256x128 RGB normalized with ImageNet mean/std
 * Output: 512-dim L2-normalized embedding
 *
 * Optimized: no Bitmap allocation per crop (uses src/dst Rect drawing),
 * pre-allocated feature buffer, in-place L2 normalization.
 */
class ReIDExtractor(context: Context, modelFileName: String = "osnet_x0_25.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "ReID"
        private const val INPUT_H = 256
        private const val INPUT_W = 128
        private const val EMBED_DIM = 512
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        // Precomputed: 1/255 / std, and -mean/std for fused normalize
        private val SCALE_R = 1f / (255f * STD[0])
        private val SCALE_G = 1f / (255f * STD[1])
        private val SCALE_B = 1f / (255f * STD[2])
        private val BIAS_R = -MEAN[0] / STD[0]
        private val BIAS_G = -MEAN[1] / STD[1]
        private val BIAS_B = -MEAN[2] / STD[2]
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    // Pre-allocated buffers — zero allocation per crop
    private val resizedBitmap = Bitmap.createBitmap(INPUT_W, INPUT_H, Bitmap.Config.ARGB_8888)
    private val resizedCanvas = Canvas(resizedBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val srcRect = Rect()
    private val dstRect = Rect(0, 0, INPUT_W, INPUT_H)
    private val inputPixels = IntArray(INPUT_H * INPUT_W)
    private val inputFloats = FloatArray(INPUT_H * INPUT_W * 3)
    private val featureBuf = FloatArray(EMBED_DIM)

    init {
        Log.i(TAG, "Loading Re-ID model: $modelFileName")

        val options = CompiledModel.Options(Accelerator.GPU)
        try {
            options.gpuOptions = CompiledModel.GpuOptions(
                null, null, null,
                CompiledModel.GpuOptions.Precision.FP32,
                null, null, null, null, null, null, null, null, null, null, null
            )
        } catch (_: Exception) {}

        compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        inputBuffers = compiledModel.createInputBuffers()
        Log.i(TAG, "Re-ID model ready (${INPUT_H}x${INPUT_W} -> ${EMBED_DIM}D)")
    }

    fun extractFeatures(bitmap: Bitmap, detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return detections

        val t = System.nanoTime()
        val result = detections.map { det ->
            val feature = extractSingle(bitmap, det)
            det.copy(feature = feature)
        }
        val ms = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Re-ID: ${detections.size} crops in ${ms}ms (${if (detections.isNotEmpty()) ms / detections.size else 0}ms/crop)")
        return result
    }

    private fun extractSingle(bitmap: Bitmap, det: Detection): FloatArray {
        // Draw crop directly to resizedBitmap using src/dst Rects (no intermediate Bitmap)
        val x = det.xMin.toInt().coerceIn(0, bitmap.width - 1)
        val y = det.yMin.toInt().coerceIn(0, bitmap.height - 1)
        val r = det.xMax.toInt().coerceIn(x + 1, bitmap.width)
        val b = det.yMax.toInt().coerceIn(y + 1, bitmap.height)
        srcRect.set(x, y, r, b)

        resizedCanvas.drawBitmap(bitmap, srcRect, dstRect, paint)

        // Normalize with fused scale+bias (single pass, no division per pixel)
        resizedBitmap.getPixels(inputPixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H)
        var idx = 0
        for (pixel in inputPixels) {
            inputFloats[idx++] = ((pixel shr 16) and 0xFF) * SCALE_R + BIAS_R
            inputFloats[idx++] = ((pixel shr 8) and 0xFF) * SCALE_G + BIAS_G
            inputFloats[idx++] = (pixel and 0xFF) * SCALE_B + BIAS_B
        }
        inputBuffers[0].writeFloat(inputFloats)

        // Inference
        val results = compiledModel.run(inputBuffers)
        val raw = results[0].readFloat()

        // L2 normalize into pre-allocated buffer
        var norm = 0f
        for (v in raw) norm += v * v
        norm = kotlin.math.sqrt(norm).coerceAtLeast(1e-8f)
        val invNorm = 1f / norm
        for (i in 0 until raw.size.coerceAtMost(EMBED_DIM)) {
            featureBuf[i] = raw[i] * invNorm
        }
        return featureBuf.copyOf()  // copy needed since Track stores reference
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
    }
}
