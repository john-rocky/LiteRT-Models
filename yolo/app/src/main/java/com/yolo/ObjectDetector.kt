package com.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * YOLO11n object detector using LiteRT CompiledModel (GPU).
 * Output shape: [1, 84, N] where 84 = 4 bbox (cx,cy,w,h) + 80 class scores.
 * All buffers pre-allocated — zero allocation per frame.
 */
class ObjectDetector(context: Context, modelFileName: String = "yolo11n.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "YOLO11"
        private const val NUM_CLASSES = 80
        private const val BBOX_CHANNELS = 4
        private const val CONF_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
        private const val MAX_DETECTIONS = 20
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    val inputSize: Int
    private val numDetections: Int

    // Pre-allocated buffers
    private val inputPixels: IntArray
    private val inputFloats: FloatArray
    private val resizedBitmap: Bitmap
    private val scaleMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        Log.i(TAG, "Loading model: $modelFileName")

        // Read shape via Interpreter
        val fd = context.assets.openFd(modelFileName)
        val channel = java.io.FileInputStream(fd.fileDescriptor).channel
        val buf = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val interp = org.tensorflow.lite.Interpreter(buf)
        val inShape = interp.getInputTensor(0).shape()   // [1, H, W, 3]
        val outShape = interp.getOutputTensor(0).shape()  // [1, 84, N]
        interp.close()

        inputSize = inShape[1]  // 384
        numDetections = outShape[2]  // 3024
        Log.i(TAG, "Shape: ${inputSize}x${inputSize} -> [${outShape.joinToString(",")}]")

        val options = CompiledModel.Options(Accelerator.GPU)
        try {
            options.gpuOptions = CompiledModel.GpuOptions(
                null, null, null,
                CompiledModel.GpuOptions.Precision.FP32,
                null, null, null, null, null, null, null, null, null, null, null
            )
        } catch (_: Exception) {}
        compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        Log.i(TAG, "GPU FP32 compiled OK")
        inputBuffers = compiledModel.createInputBuffers()

        inputPixels = IntArray(inputSize * inputSize)
        inputFloats = FloatArray(inputSize * inputSize * 3)
        resizedBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)

        Log.i(TAG, "Model ready: ${inputSize}x${inputSize}, $numDetections candidates")
    }

    /**
     * Run object detection. Returns detections in original image coords + time in ms.
     */
    fun detect(bitmap: Bitmap): Pair<List<Detection>, Long> {
        var t = System.nanoTime()

        // Preprocess: resize + normalize to 0-1 (YOLO standard)
        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(
            inputSize.toFloat() / bitmap.width,
            inputSize.toFloat() / bitmap.height
        )
        canvas.drawBitmap(bitmap, scaleMatrix, paint)
        resizedBitmap.getPixels(inputPixels, 0, inputSize, 0, 0, inputSize, inputSize)

        var idx = 0
        for (pixel in inputPixels) {
            inputFloats[idx++] = ((pixel shr 16) and 0xFF) / 255f
            inputFloats[idx++] = ((pixel shr 8) and 0xFF) / 255f
            inputFloats[idx++] = (pixel and 0xFF) / 255f
        }
        inputBuffers[0].writeFloat(inputFloats)
        val preMs = (System.nanoTime() - t) / 1_000_000

        // Inference
        t = System.nanoTime()
        val resultBuffers = compiledModel.run(inputBuffers)
        val infMs = (System.nanoTime() - t) / 1_000_000

        // Postprocess
        t = System.nanoTime()
        val raw = resultBuffers[0].readFloat()
        val dets = postProcess(raw, bitmap.width, bitmap.height)
        val postMs = (System.nanoTime() - t) / 1_000_000

        Log.i(TAG, "pre=${preMs}ms inf=${infMs}ms post=${postMs}ms dets=${dets.size}")

        return dets to (preMs + infMs + postMs)
    }

    private fun postProcess(raw: FloatArray, origW: Int, origH: Int): List<Detection> {
        val candidates = mutableListOf<Detection>()

        for (i in 0 until numDetections) {
            // Find best class
            var maxScore = 0f
            var maxClass = 0
            for (c in 0 until NUM_CLASSES) {
                val score = raw[(BBOX_CHANNELS + c) * numDetections + i]
                if (score > maxScore) {
                    maxScore = score
                    maxClass = c
                }
            }
            if (maxScore < CONF_THRESHOLD) continue

            // Bbox: cx, cy, w, h — pixel coords relative to input size (0..inputSize)
            val rawCx = raw[0 * numDetections + i]
            val rawCy = raw[1 * numDetections + i]
            val rawW = raw[2 * numDetections + i]
            val rawH = raw[3 * numDetections + i]

            if (candidates.size < 3) {
                Log.d(TAG, "det[$i] score=${"%.2f".format(maxScore)} class=${CocoLabels.name(maxClass)} " +
                    "raw bbox=[${"%.1f".format(rawCx)}, ${"%.1f".format(rawCy)}, ${"%.1f".format(rawW)}, ${"%.1f".format(rawH)}]")
            }

            // Normalized coords (0-1) → original image space
            candidates.add(Detection(
                classId = maxClass,
                className = CocoLabels.name(maxClass),
                score = maxScore,
                xMin = ((rawCx - rawW / 2f) * origW).coerceIn(0f, origW.toFloat()),
                yMin = ((rawCy - rawH / 2f) * origH).coerceIn(0f, origH.toFloat()),
                xMax = ((rawCx + rawW / 2f) * origW).coerceIn(0f, origW.toFloat()),
                yMax = ((rawCy + rawH / 2f) * origH).coerceIn(0f, origH.toFloat()),
            ))
        }

        return nms(candidates.sortedByDescending { it.score }, IOU_THRESHOLD)
            .take(MAX_DETECTIONS)
    }

    private fun nms(sorted: List<Detection>, iouThresh: Float): List<Detection> {
        val result = mutableListOf<Detection>()
        val active = BooleanArray(sorted.size) { true }

        for (i in sorted.indices) {
            if (!active[i]) continue
            result.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (active[j] && iou(sorted[i], sorted[j]) > iouThresh) {
                    active[j] = false
                }
            }
        }
        return result
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1 = maxOf(a.xMin, b.xMin)
        val y1 = maxOf(a.yMin, b.yMin)
        val x2 = minOf(a.xMax, b.xMax)
        val y2 = minOf(a.yMax, b.yMax)
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaA = (a.xMax - a.xMin) * (a.yMax - a.yMin)
        val areaB = (b.xMax - b.xMin) * (b.yMax - b.yMin)
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
    }
}
