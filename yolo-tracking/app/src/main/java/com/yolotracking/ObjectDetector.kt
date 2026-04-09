package com.yolotracking

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
 * Output shape: [1, 84, N] where 84 = 4 bbox + 80 class scores.
 *
 * Post-processing optimized: primitive arrays only, zero allocation per frame,
 * cache-friendly class scan, in-place NMS.
 */
class ObjectDetector(context: Context, modelFileName: String = "yolo11n.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "YOLO"
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
    private val isXyxy: Boolean

    // Pre-allocated input buffers
    private val inputPixels: IntArray
    private val inputFloats: FloatArray
    private val resizedBitmap: Bitmap
    private val scaleMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    // Pre-allocated post-processing buffers (zero allocation per frame)
    private val candScores: FloatArray
    private val candClasses: IntArray
    private val candX1: FloatArray
    private val candY1: FloatArray
    private val candX2: FloatArray
    private val candY2: FloatArray
    private val candAreas: FloatArray
    private val sortIndices: IntArray
    private val nmsActive: BooleanArray

    init {
        Log.i(TAG, "Loading model: $modelFileName")

        val fd = context.assets.openFd(modelFileName)
        val channel = java.io.FileInputStream(fd.fileDescriptor).channel
        val buf = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val interp = org.tensorflow.lite.Interpreter(buf)
        val inShape = interp.getInputTensor(0).shape()
        val outShape = interp.getOutputTensor(0).shape()
        interp.close()

        inputSize = inShape[1]
        numDetections = outShape[2]
        isXyxy = modelFileName.contains("yolo26")
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

        // Pre-allocate post-processing buffers (worst case = numDetections)
        candScores = FloatArray(numDetections)
        candClasses = IntArray(numDetections)
        candX1 = FloatArray(numDetections)
        candY1 = FloatArray(numDetections)
        candX2 = FloatArray(numDetections)
        candY2 = FloatArray(numDetections)
        candAreas = FloatArray(numDetections)
        sortIndices = IntArray(numDetections)
        nmsActive = BooleanArray(numDetections)

        Log.i(TAG, "Model ready: ${inputSize}x${inputSize}, $numDetections candidates")
    }

    fun detect(bitmap: Bitmap): Pair<List<Detection>, Long> {
        var t = System.nanoTime()

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

        t = System.nanoTime()
        val resultBuffers = compiledModel.run(inputBuffers)
        val runMs = (System.nanoTime() - t) / 1_000_000

        t = System.nanoTime()
        val raw = resultBuffers[0].readFloat()
        val readMs = (System.nanoTime() - t) / 1_000_000
        val infMs = runMs + readMs  // true GPU time = run + sync/read

        t = System.nanoTime()
        val dets = postProcess(raw, bitmap.width, bitmap.height)
        val postMs = (System.nanoTime() - t) / 1_000_000

        Log.i(TAG, "pre=${preMs}ms run=${runMs}ms read=${readMs}ms post=${postMs}ms total_inf=${infMs}ms dets=${dets.size}")
        return dets to (preMs + infMs + postMs)
    }

    private fun postProcess(raw: FloatArray, origW: Int, origH: Int): List<Detection> {
        val n = numDetections
        var candCount = 0

        // Single-pass: for each candidate, find best class and filter
        for (i in 0 until n) {
            var maxScore = 0f
            var maxClass = 0
            var classOff = BBOX_CHANNELS * n
            for (c in 0 until NUM_CLASSES) {
                val score = raw[classOff + i]
                if (score > maxScore) {
                    maxScore = score
                    maxClass = c
                }
                classOff += n
            }
            if (maxScore < CONF_THRESHOLD) continue

            val r0 = raw[i]
            val r1 = raw[n + i]
            val r2 = raw[n * 2 + i]
            val r3 = raw[n * 3 + i]

            val x1: Float; val y1: Float; val x2: Float; val y2: Float
            if (isXyxy) {
                x1 = r0 * origW; y1 = r1 * origH
                x2 = r2 * origW; y2 = r3 * origH
            } else {
                val hw = r2 / 2f; val hh = r3 / 2f
                x1 = (r0 - hw) * origW; y1 = (r1 - hh) * origH
                x2 = (r0 + hw) * origW; y2 = (r1 + hh) * origH
            }

            val cx1 = x1.coerceIn(0f, origW.toFloat())
            val cy1 = y1.coerceIn(0f, origH.toFloat())
            val cx2 = x2.coerceIn(0f, origW.toFloat())
            val cy2 = y2.coerceIn(0f, origH.toFloat())

            candScores[candCount] = maxScore
            candClasses[candCount] = maxClass
            candX1[candCount] = cx1
            candY1[candCount] = cy1
            candX2[candCount] = cx2
            candY2[candCount] = cy2
            candAreas[candCount] = (cx2 - cx1) * (cy2 - cy1)
            candCount++
        }

        if (candCount == 0) return emptyList()

        // --- Pass 2: sort indices by score descending (insertion sort, fast for small N) ---
        for (i in 0 until candCount) sortIndices[i] = i
        for (i in 1 until candCount) {
            val key = sortIndices[i]
            val keyScore = candScores[key]
            var j = i - 1
            while (j >= 0 && candScores[sortIndices[j]] < keyScore) {
                sortIndices[j + 1] = sortIndices[j]
                j--
            }
            sortIndices[j + 1] = key
        }

        // --- Pass 3: NMS on primitive arrays ---
        for (i in 0 until candCount) nmsActive[i] = true
        val result = mutableListOf<Detection>()

        for (si in 0 until candCount) {
            val i = sortIndices[si]
            if (!nmsActive[i]) continue

            result.add(Detection(
                classId = candClasses[i],
                className = CocoLabels.name(candClasses[i]),
                score = candScores[i],
                xMin = candX1[i], yMin = candY1[i],
                xMax = candX2[i], yMax = candY2[i],
            ))
            if (result.size >= MAX_DETECTIONS) break

            // Suppress overlapping lower-score candidates
            val ax1 = candX1[i]; val ay1 = candY1[i]
            val ax2 = candX2[i]; val ay2 = candY2[i]
            val aArea = candAreas[i]

            for (sj in si + 1 until candCount) {
                val j = sortIndices[sj]
                if (!nmsActive[j]) continue

                val ix1 = maxOf(ax1, candX1[j])
                val iy1 = maxOf(ay1, candY1[j])
                val ix2 = minOf(ax2, candX2[j])
                val iy2 = minOf(ay2, candY2[j])
                val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
                val iou = inter / (aArea + candAreas[j] - inter + 1e-6f)
                if (iou > IOU_THRESHOLD) nmsActive[j] = false
            }
        }

        return result
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        resizedBitmap.recycle()
    }
}
