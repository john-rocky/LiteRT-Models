package com.yolox

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOX-Nano object detector on LiteRT CompiledModel (GPU).
 *
 * Model (Apache-2.0, Megvii YOLOX) converted to a GPU-clean TFLite via onnx2tf:
 *   input : images [1, 416, 416, 3]  NHWC, BGR, 0-255, NO normalization (letterboxed)
 *   output: Identity [1, 3549, 85]   raw head output (anchor-major)
 *           85 = 4 box (x, y, w, h) + 1 obj + 80 class scores
 *
 * The graph applies sigmoid to obj/class but does NOT decode boxes, so the grid +
 * stride decode (and NMS) is done here. Verified GPU-clean: zero GATHER/TOPK/CAST,
 * zero >4D tensors, zero dynamic dims; runs ~21 FPS on Pixel 8a.
 */
class YoloxDetector(
    context: Context,
    modelFileName: String = "yolox_nano.tflite",
    vararg accelerators: Accelerator = arrayOf(Accelerator.GPU),
) : Closeable {

    companion object {
        private const val TAG = "YOLOX"
        const val INPUT_SIZE = 416
        const val NUM_CLASSES = 80
        const val NUM_FIELDS = NUM_CLASSES + 5  // 85
        val STRIDES = intArrayOf(8, 16, 32)
        const val DEFAULT_SCORE_THRESHOLD = 0.30f
        const val IOU_THRESHOLD = 0.45f
        const val MAX_DETECTIONS = 50
        private const val PAD_VALUE = 114  // YOLOX letterbox gray
    }

    private val model: CompiledModel = CompiledModel.create(
        context.assets, modelFileName, CompiledModel.Options(*accelerators), null,
    )
    private val inputBuffers: List<TensorBuffer> = model.createInputBuffers()
    private val outputBuffers: List<TensorBuffer> = model.createOutputBuffers()

    // Pre-allocated buffers (zero allocation per frame)
    private val inputPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val inputFloats = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)
    private val resizedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val scaleMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    // Per-anchor grid origin + stride, precomputed once (3549 anchors for 416 input)
    private val gridX: IntArray
    private val gridY: IntArray
    private val gridStride: IntArray

    init {
        val gx = ArrayList<Int>()
        val gy = ArrayList<Int>()
        val gs = ArrayList<Int>()
        for (stride in STRIDES) {
            val n = INPUT_SIZE / stride
            for (yy in 0 until n) for (xx in 0 until n) {
                gx.add(xx); gy.add(yy); gs.add(stride)
            }
        }
        gridX = gx.toIntArray()
        gridY = gy.toIntArray()
        gridStride = gs.toIntArray()
        Log.i(TAG, "Ready: ${INPUT_SIZE}x$INPUT_SIZE, ${gridX.size} anchors")
    }

    fun detect(bitmap: Bitmap, scoreThreshold: Float = DEFAULT_SCORE_THRESHOLD): List<Detection> {
        // Letterbox preprocess (YOLOX-canonical): uniform scale preserving aspect, pad to
        // 416 with gray 114 at top-left, write BGR 0-255 (no normalization). Stretching
        // would distort objects and loosen boxes.
        val ratio = min(
            INPUT_SIZE.toFloat() / bitmap.width,
            INPUT_SIZE.toFloat() / bitmap.height,
        )
        val canvas = Canvas(resizedBitmap)
        canvas.drawColor(Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE))
        scaleMatrix.setScale(ratio, ratio)
        canvas.drawBitmap(bitmap, scaleMatrix, paint)
        resizedBitmap.getPixels(inputPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        var idx = 0
        for (pixel in inputPixels) {
            inputFloats[idx++] = (pixel and 0xFF).toFloat()          // B
            inputFloats[idx++] = ((pixel shr 8) and 0xFF).toFloat()  // G
            inputFloats[idx++] = ((pixel shr 16) and 0xFF).toFloat() // R
        }

        inputBuffers[0].writeFloat(inputFloats)
        model.run(inputBuffers, outputBuffers)
        val raw = outputBuffers[0].readFloat()  // [3549 * 85], anchor-major

        return postProcess(raw, ratio, bitmap.width, bitmap.height, scoreThreshold)
    }

    private fun postProcess(
        raw: FloatArray,
        ratio: Float,
        origW: Int,
        origH: Int,
        scoreThreshold: Float,
    ): List<Detection> {
        val candidates = ArrayList<Detection>()

        for (i in gridX.indices) {
            val base = i * NUM_FIELDS
            val obj = raw[base + 4]
            if (obj < scoreThreshold) continue  // early reject (cls score only lowers it)

            // Best class
            var maxCls = 0
            var maxClsScore = 0f
            for (c in 0 until NUM_CLASSES) {
                val s = raw[base + 5 + c]
                if (s > maxClsScore) { maxClsScore = s; maxCls = c }
            }
            val score = obj * maxClsScore
            if (score < scoreThreshold) continue

            // Decode box (grid + stride) in 416-space, then un-letterbox (divide by ratio).
            val stride = gridStride[i]
            val cx = (raw[base + 0] + gridX[i]) * stride
            val cy = (raw[base + 1] + gridY[i]) * stride
            val w = exp(raw[base + 2]) * stride
            val h = exp(raw[base + 3]) * stride

            candidates.add(
                Detection(
                    classId = maxCls,
                    className = CocoLabels.name(maxCls),
                    score = score,
                    xMin = ((cx - w / 2f) / ratio).coerceIn(0f, origW.toFloat()),
                    yMin = ((cy - h / 2f) / ratio).coerceIn(0f, origH.toFloat()),
                    xMax = ((cx + w / 2f) / ratio).coerceIn(0f, origW.toFloat()),
                    yMax = ((cy + h / 2f) / ratio).coerceIn(0f, origH.toFloat()),
                ),
            )
        }

        return nms(candidates.sortedByDescending { it.score }, IOU_THRESHOLD).take(MAX_DETECTIONS)
    }

    /** Per-class non-maximum suppression. */
    private fun nms(sorted: List<Detection>, iouThresh: Float): List<Detection> {
        val result = ArrayList<Detection>()
        val active = BooleanArray(sorted.size) { true }
        for (i in sorted.indices) {
            if (!active[i]) continue
            result.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (active[j] && sorted[j].classId == sorted[i].classId &&
                    iou(sorted[i], sorted[j]) > iouThresh
                ) {
                    active[j] = false
                }
            }
        }
        return result
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1 = max(a.xMin, b.xMin)
        val y1 = max(a.yMin, b.yMin)
        val x2 = min(a.xMax, b.xMax)
        val y2 = min(a.yMax, b.yMax)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a.xMax - a.xMin) * (a.yMax - a.yMin)
        val areaB = (b.xMax - b.xMin) * (b.yMax - b.yMin)
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
        resizedBitmap.recycle()
    }
}
