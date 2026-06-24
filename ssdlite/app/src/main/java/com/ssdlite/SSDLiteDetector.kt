package com.ssdlite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * SSDLite320-MobileNetV3-Large object detector on LiteRT CompiledModel (GPU).
 *
 * Model (torchvision, BSD-3) converted PATCH-FREE via litert-torch with the
 * 4D-head-tap technique: the graph returns each FPN level's RAW head conv outputs
 * (NCHW), so anchor-decode + multiclass-NMS run here. Tapping the raw heads keeps
 * the graph fully GPU-compatible (no GATHER_ND / TOPK / >4D) — exporting the
 * model's built-in DefaultBoxGenerator + NMS postprocess would not be.
 *
 *   input : [1, 3, 320, 320]  NCHW, RGB, normalized (pixel/127.5 - 1) → [-1, 1]
 *   output: 12 tensors, (cls, box) per 6 feature levels (H = W = 20,10,5,3,2,1):
 *             cls[i] = [1, 6*91, H, W]   box[i] = [1, 6*4, H, W]
 *           6 anchors/loc, 91 classes (COCO 90 + background at index 0).
 *
 * Decode mirrors torchvision SSD.postprocess_detections + BoxCoder(10,10,5,5):
 * softmax over 91 → best non-background class → score threshold → box decode
 * against the precomputed default boxes → per-class NMS. Verified bit-faithful
 * against stock torchvision on the FP16 tflite (box-match 298/300 @ IoU 0.99).
 */
class SSDLiteDetector(
    context: Context,
    modelFileName: String = "ssdlite_mobilenetv3_320_fp16.tflite",
    vararg accelerators: Accelerator = arrayOf(Accelerator.GPU),
) : Closeable {

    companion object {
        private const val TAG = "SSDLite"
        const val INPUT_SIZE = 320
        const val NUM_CLASSES = 91          // includes background at index 0
        const val ANCHORS_PER_LOC = 6
        val LEVELS = intArrayOf(20, 10, 5, 3, 2, 1)

        // DefaultBoxGenerator(min_ratio=0.2, max_ratio=0.95) → 7 scales (last is the cap).
        private val SCALES = floatArrayOf(0.2f, 0.35f, 0.5f, 0.65f, 0.8f, 0.95f, 1.0f)
        private val ASPECT_RATIOS = intArrayOf(2, 3)

        // BoxCoder(weights = (dx, dy, dw, dh)) and the torch.exp() input clamp.
        private const val WX = 10f
        private const val WY = 10f
        private const val WW = 5f
        private const val WH = 5f
        private val BBOX_CLIP = ln(1000f / 16f)

        const val DEFAULT_SCORE_THRESHOLD = 0.4f
        const val IOU_THRESHOLD = 0.55f     // torchvision SSD nms_thresh
        const val MAX_DETECTIONS = 100
    }

    private val model: CompiledModel = CompiledModel.create(
        context.assets, modelFileName, CompiledModel.Options(*accelerators), null,
    )
    private val inputBuffers: List<TensorBuffer> = model.createInputBuffers()
    private val outputBuffers: List<TensorBuffer> = model.createOutputBuffers()

    // NCHW input, pre-allocated (zero per-frame allocation)
    private val inputPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val inputFloats = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
    private val resizedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val scaleMatrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    // Precomputed default boxes (xyxy, 320-space), order: level → row(y) → col(x) → anchor.
    private val anchors: FloatArray
    private val levelAnchorOffset = IntArray(LEVELS.size)

    init {
        var total = 0
        for (li in LEVELS.indices) {
            levelAnchorOffset[li] = total
            total += LEVELS[li] * LEVELS[li] * ANCHORS_PER_LOC
        }
        anchors = FloatArray(total * 4)
        buildAnchors()
        Log.i(TAG, "Ready: ${INPUT_SIZE}x$INPUT_SIZE, $total anchors")
    }

    /**
     * Reproduce torchvision's DefaultBoxGenerator for a 320×320 input (matches the
     * model's generator to ~3e-5). Anchor order — level → row → col → anchor — is
     * exactly the order the head's NCHW channels reshape to, so anchors and head
     * outputs stay aligned without any GATHER on device.
     */
    private fun buildAnchors() {
        var idx = 0
        for (li in LEVELS.indices) {
            val f = LEVELS[li]
            val wh = whPairs(li)                  // 6 (w,h) pairs, normalized [0,1]
            for (i in 0 until f) {                // row (y)
                for (j in 0 until f) {            // col (x)
                    val cx = (j + 0.5f) / f
                    val cy = (i + 0.5f) / f
                    for (a in 0 until ANCHORS_PER_LOC) {
                        val w = wh[a * 2]
                        val h = wh[a * 2 + 1]
                        anchors[idx++] = (cx - 0.5f * w) * INPUT_SIZE
                        anchors[idx++] = (cy - 0.5f * h) * INPUT_SIZE
                        anchors[idx++] = (cx + 0.5f * w) * INPUT_SIZE
                        anchors[idx++] = (cy + 0.5f * h) * INPUT_SIZE
                    }
                }
            }
        }
    }

    /** The 6 (w,h) pairs for a level, clipped to [0,1]; mirrors _generate_wh_pairs. */
    private fun whPairs(level: Int): FloatArray {
        val sk = SCALES[level]
        val skp = sqrt(SCALES[level] * SCALES[level + 1])
        val out = ArrayList<Float>(12)
        out.add(sk); out.add(sk)                  // aspect ratio 1, scale s_k
        out.add(skp); out.add(skp)                // aspect ratio 1, scale s'_k
        for (ar in ASPECT_RATIOS) {               // ar and 1/ar
            val sq = sqrt(ar.toFloat())
            out.add(sk * sq); out.add(sk / sq)
            out.add(sk / sq); out.add(sk * sq)
        }
        return FloatArray(out.size) { out[it].coerceIn(0f, 1f) }
    }

    fun detect(bitmap: Bitmap, scoreThreshold: Float = DEFAULT_SCORE_THRESHOLD): List<Detection> {
        // Preprocess: stretch-resize to 320×320 (SSD uses fixed-size resize, not
        // letterbox), RGB, NCHW, normalized to [-1, 1]. Matches the torchvision
        // GeneralizedRCNNTransform (mean = std = 0.5).
        scaleMatrix.setScale(
            INPUT_SIZE.toFloat() / bitmap.width,
            INPUT_SIZE.toFloat() / bitmap.height,
        )
        Canvas(resizedBitmap).drawBitmap(bitmap, scaleMatrix, paint)
        resizedBitmap.getPixels(inputPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val plane = INPUT_SIZE * INPUT_SIZE
        for (i in inputPixels.indices) {
            val p = inputPixels[i]
            inputFloats[i] = ((p shr 16) and 0xFF) / 127.5f - 1f             // R
            inputFloats[plane + i] = ((p shr 8) and 0xFF) / 127.5f - 1f      // G
            inputFloats[2 * plane + i] = (p and 0xFF) / 127.5f - 1f          // B
        }
        inputBuffers[0].writeFloat(inputFloats)
        model.run(inputBuffers, outputBuffers)

        return postProcess(scoreThreshold, bitmap.width, bitmap.height)
    }

    private fun postProcess(scoreThreshold: Float, origW: Int, origH: Int): List<Detection> {
        val candidates = ArrayList<Detection>()
        val scaleX = origW.toFloat() / INPUT_SIZE
        val scaleY = origH.toFloat() / INPUT_SIZE

        // Output order is (cls, box) per level — outputBuffers[2*li], [2*li+1].
        for (li in LEVELS.indices) {
            val h = LEVELS[li]
            val w = h
            val hw = h * w
            val cls = outputBuffers[2 * li].readFloat()      // [6*91 * hw] NCHW
            val box = outputBuffers[2 * li + 1].readFloat()  // [6*4  * hw] NCHW
            val offset = levelAnchorOffset[li]

            for (p in 0 until hw) {                          // p = row*w + col
                for (a in 0 until ANCHORS_PER_LOC) {
                    // cls channel for (anchor a, class k) = a*91 + k; spatial stride = hw.
                    val clsBase = a * NUM_CLASSES * hw + p
                    var maxAll = cls[clsBase]                // running max over all 91 (stability)
                    var bestLogit = Float.NEGATIVE_INFINITY  // best foreground (k = 1..90)
                    var bestK = 1
                    var k = 0
                    while (k < NUM_CLASSES) {
                        val l = cls[clsBase + k * hw]
                        if (l > maxAll) maxAll = l
                        if (k >= 1 && l > bestLogit) { bestLogit = l; bestK = k }
                        k++
                    }
                    // softmax prob ≤ exp(bestLogit - maxAll) since Σexp ≥ 1 — cheap reject.
                    val probUpperBound = exp(bestLogit - maxAll)
                    if (probUpperBound < scoreThreshold) continue
                    var sum = 0f
                    k = 0
                    while (k < NUM_CLASSES) { sum += exp(cls[clsBase + k * hw] - maxAll); k++ }
                    val score = probUpperBound / sum
                    if (score < scoreThreshold) continue

                    // Decode box: BoxCoder deltas against the default box (xyxy, 320-space).
                    val ai = (offset + p * ANCHORS_PER_LOC + a) * 4
                    val ax1 = anchors[ai]
                    val ay1 = anchors[ai + 1]
                    val aw = anchors[ai + 2] - ax1
                    val ah = anchors[ai + 3] - ay1
                    val acx = ax1 + 0.5f * aw
                    val acy = ay1 + 0.5f * ah
                    val boxBase = a * 4 * hw + p
                    val dx = box[boxBase] / WX
                    val dy = box[boxBase + hw] / WY
                    val dw = min(box[boxBase + 2 * hw] / WW, BBOX_CLIP)
                    val dh = min(box[boxBase + 3 * hw] / WH, BBOX_CLIP)
                    val pcx = dx * aw + acx
                    val pcy = dy * ah + acy
                    val pw = exp(dw) * aw
                    val ph = exp(dh) * ah

                    // 320-space xyxy → clip to image → map back to original bitmap pixels.
                    val x1 = (pcx - 0.5f * pw).coerceIn(0f, INPUT_SIZE.toFloat())
                    val y1 = (pcy - 0.5f * ph).coerceIn(0f, INPUT_SIZE.toFloat())
                    val x2 = (pcx + 0.5f * pw).coerceIn(0f, INPUT_SIZE.toFloat())
                    val y2 = (pcy + 0.5f * ph).coerceIn(0f, INPUT_SIZE.toFloat())

                    candidates.add(
                        Detection(
                            classId = bestK,
                            className = CocoLabels.name(bestK),
                            score = score,
                            xMin = x1 * scaleX, yMin = y1 * scaleY,
                            xMax = x2 * scaleX, yMax = y2 * scaleY,
                        ),
                    )
                }
            }
        }
        return nms(candidates.sortedByDescending { it.score }, IOU_THRESHOLD).take(MAX_DETECTIONS)
    }

    /** Per-class greedy non-maximum suppression (torchvision batched_nms equivalent). */
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
