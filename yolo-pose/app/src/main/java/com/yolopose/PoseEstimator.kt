package com.yolopose

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
 * YOLO26n-pose estimator using LiteRT CompiledModel (GPU).
 *
 * Input shape:  [1, 3, 384, 384] NCHW (litert-torch native, planar RGB float32 0..1).
 * Output shape: [1, 56, N] where 56 = 4 bbox + 1 person conf + 17 keypoints * 3 (x, y, vis).
 * Bbox channels are (cx, cy, w, h) — the legacy one-to-many head emits xywh,
 * NOT xyxy. (xyxy is only produced by the end2end head which we bypass.)
 *
 * Coordinate convention: the wrapped one-to-many head (end2end=False, export=True)
 * keeps bbox and keypoint xy values in input image pixel space (0..inputSize).
 * The decoder rescales by origDim / inputSize to map back to the original bitmap.
 */
class PoseEstimator(
    context: Context,
    modelFileName: String = "yolo26n_pose.tflite",
) : AutoCloseable {

    companion object {
        private const val TAG = "YOLOPose"
        private const val NUM_KEYPOINTS = CocoKeypoints.NUM_KEYPOINTS  // 17
        private const val BBOX_CHANNELS = 4
        private const val CONF_CHANNELS = 1
        private const val CHANNELS_PER_KP = 3                          // x, y, vis
        private const val EXPECTED_CHANNELS =
            BBOX_CHANNELS + CONF_CHANNELS + NUM_KEYPOINTS * CHANNELS_PER_KP  // 56

        private const val CONF_THRESHOLD = 0.30f
        private const val IOU_THRESHOLD = 0.45f
        private const val MAX_PERSONS = 10
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    val inputSize: Int
    private val numChannels: Int
    private val numDetections: Int

    // Pre-allocated I/O
    private val inputPixels: IntArray
    private val inputFloats: FloatArray
    private val resizedBitmap: Bitmap
    private val scaleMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        Log.i(TAG, "Loading model: $modelFileName")

        // Probe shape via Interpreter
        val fd = context.assets.openFd(modelFileName)
        val channel = java.io.FileInputStream(fd.fileDescriptor).channel
        val buf = channel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength,
        )
        val interp = org.tensorflow.lite.Interpreter(buf)
        val inShape = interp.getInputTensor(0).shape()    // [1, 3, H, W] NCHW
        val outShape = interp.getOutputTensor(0).shape()  // [1, 56, N]
        interp.close()

        inputSize = inShape[2]  // NCHW: index 2 = H, 3 = W (square)
        numChannels = outShape[1]
        numDetections = outShape[2]
        Log.i(TAG, "Shape: NCHW ${inputSize}x${inputSize} -> [${outShape.joinToString(",")}]")
        require(numChannels == EXPECTED_CHANNELS) {
            "Unexpected pose channel count: $numChannels (expected $EXPECTED_CHANNELS)"
        }

        val options = CompiledModel.Options(Accelerator.GPU)
        try {
            options.gpuOptions = CompiledModel.GpuOptions(
                null, null, null,
                CompiledModel.GpuOptions.Precision.FP32,
                null, null, null, null, null, null, null, null, null, null, null,
            )
        } catch (_: Exception) {}
        compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        Log.i(TAG, "GPU FP32 compiled OK")
        inputBuffers = compiledModel.createInputBuffers()

        inputPixels = IntArray(inputSize * inputSize)
        inputFloats = FloatArray(inputSize * inputSize * 3)
        resizedBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)

        Log.i(TAG, "PoseEstimator ready: ${inputSize}x${inputSize}, $numDetections candidates")
    }

    /**
     * Run pose estimation. Returns persons in original image coords + total time in ms.
     */
    fun detect(bitmap: Bitmap): Pair<List<Pose>, Long> {
        var t = System.nanoTime()

        // Preprocess: resize + 0..1 normalize, planar NCHW (R plane | G plane | B plane)
        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(
            inputSize.toFloat() / bitmap.width,
            inputSize.toFloat() / bitmap.height,
        )
        canvas.drawBitmap(bitmap, scaleMatrix, paint)
        resizedBitmap.getPixels(inputPixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val planeSize = inputSize * inputSize
        for (i in inputPixels.indices) {
            val pixel = inputPixels[i]
            inputFloats[i] = ((pixel shr 16) and 0xFF) / 255f
            inputFloats[planeSize + i] = ((pixel shr 8) and 0xFF) / 255f
            inputFloats[2 * planeSize + i] = (pixel and 0xFF) / 255f
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
        val persons = postProcess(raw, bitmap.width, bitmap.height)
        val postMs = (System.nanoTime() - t) / 1_000_000

        Log.i(TAG, "pre=${preMs}ms inf=${infMs}ms post=${postMs}ms persons=${persons.size}")

        return persons to (preMs + infMs + postMs)
    }

    private fun postProcess(raw: FloatArray, origW: Int, origH: Int): List<Pose> {
        // Coordinates in raw output are in input image pixel space (0..inputSize).
        // Rescale to original bitmap dimensions.
        val sx = origW.toFloat() / inputSize
        val sy = origH.toFloat() / inputSize

        val candidates = ArrayList<Pose>(32)

        for (i in 0 until numDetections) {
            // Raw layout: [channel * N + i] (flattened from [1, C, N])
            val conf = raw[4 * numDetections + i]
            if (conf < CONF_THRESHOLD) continue

            // Legacy one-to-many head emits bbox as (cx, cy, w, h) in input pixels.
            val cx = raw[0 * numDetections + i] * sx
            val cy = raw[1 * numDetections + i] * sy
            val bw = raw[2 * numDetections + i] * sx
            val bh = raw[3 * numDetections + i] * sy

            val xMin = (cx - bw * 0.5f).coerceIn(0f, origW.toFloat())
            val yMin = (cy - bh * 0.5f).coerceIn(0f, origH.toFloat())
            val xMax = (cx + bw * 0.5f).coerceIn(0f, origW.toFloat())
            val yMax = (cy + bh * 0.5f).coerceIn(0f, origH.toFloat())
            if (xMax <= xMin || yMax <= yMin) continue

            val kps = FloatArray(NUM_KEYPOINTS * 3)
            val kpBase = (BBOX_CHANNELS + CONF_CHANNELS) * numDetections
            for (k in 0 until NUM_KEYPOINTS) {
                val kx = raw[kpBase + (k * 3) * numDetections + i]
                val ky = raw[kpBase + (k * 3 + 1) * numDetections + i]
                val kv = raw[kpBase + (k * 3 + 2) * numDetections + i]
                kps[k * 3] = (kx * sx).coerceIn(0f, origW.toFloat())
                kps[k * 3 + 1] = (ky * sy).coerceIn(0f, origH.toFloat())
                kps[k * 3 + 2] = kv
            }

            candidates.add(
                Pose(
                    score = conf,
                    xMin = xMin,
                    yMin = yMin,
                    xMax = xMax,
                    yMax = yMax,
                    keypoints = kps,
                ),
            )
        }

        return nms(candidates.sortedByDescending { it.score }, IOU_THRESHOLD)
            .take(MAX_PERSONS)
    }

    private fun nms(sorted: List<Pose>, iouThresh: Float): List<Pose> {
        val result = ArrayList<Pose>(sorted.size)
        val active = BooleanArray(sorted.size) { true }
        for (i in sorted.indices) {
            if (!active[i]) continue
            result.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (active[j] && iou(sorted[i], sorted[j]) > iouThresh) active[j] = false
            }
        }
        return result
    }

    private fun iou(a: Pose, b: Pose): Float {
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
