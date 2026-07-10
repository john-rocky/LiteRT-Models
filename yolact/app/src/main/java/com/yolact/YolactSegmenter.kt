package com.yolact

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/** One detected instance: class, score, box (in 0..1), and a SIZE×SIZE binary mask. */
data class Instance(val cls: Int, val score: Float, val x1: Float, val y1: Float,
                    val x2: Float, val y2: Float, val mask: BooleanArray)

/**
 * YOLACT-ResNet50 real-time instance segmentation on LiteRT CompiledModel (GPU).
 *
 * The GPU graph (backbone + FPN + protonet + heads) emits raw outputs:
 *   loc  [1, 19248, 4]   box regressions vs the priors
 *   conf [1, 19248, 81]  class scores (softmax, incl. background)
 *   mask [1, 19248, 32]  mask coefficients (tanh)
 *   proto[1, 138, 138, 32] prototype masks
 * The decode is host-side (the RF-DETR raw-head pattern): SSD box-decode against the
 * baked priors, per-class NMS, then lincomb masks (sigmoid(proto @ coeff), cropped to
 * the box). Everything network-side runs fully on the GPU delegate.
 */
class YolactSegmenter(context: Context, modelFileName: String = "yolact.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "YOLACT"
        const val SIZE = 550
        const val N = 19248
        const val NC = 81            // 80 classes + background
        const val K = 32             // mask dim
        const val PS = 138           // proto spatial
        private val MEAN = floatArrayOf(103.94f, 116.78f, 123.68f)   // BGR
        private val STD = floatArrayOf(57.38f, 57.12f, 58.40f)
        private const val SCORE_THRESH = 0.3f
        private const val IOU_THRESH = 0.5f
        private const val MAX_DET = 40
    }

    private val model: CompiledModel
    private val inBufs: List<TensorBuffer>
    private val outBufs: List<TensorBuffer>
    private val priors: FloatArray            // [N*4] cx,cy,w,h
    private var iLoc = 0; private var iConf = 1; private var iMask = 2; private var iProto = 3

    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val options = CompiledModel.Options(Accelerator.GPU)
        model = CompiledModel.create(context.assets, modelFileName, options, null)
        inBufs = model.createInputBuffers()
        outBufs = model.createOutputBuffers()
        // map output slots by element count (loc=N*4, conf=N*81, mask=N*32, proto=PS*PS*32)
        for (i in outBufs.indices) when (outBufs[i].readFloat().size) {
            N * 4 -> iLoc = i; N * NC -> iConf = i; N * K -> iMask = i; PS * PS * K -> iProto = i
        }
        priors = readBin(context, "priors.bin")
        Log.i(TAG, "GPU compiled OK — loc=$iLoc conf=$iConf mask=$iMask proto=$iProto, priors=${priors.size / 4}")
    }

    private fun readBin(ctx: Context, name: String): FloatArray {
        val b = ctx.assets.open(name).use { it.readBytes() }
        val fb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.limit()).also { fb.get(it) }
    }

    fun segment(bitmap: Bitmap): Pair<List<Instance>, Long> {
        val t = System.nanoTime()
        val canvas = Canvas(resized)
        matrix.setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height)
        canvas.drawBitmap(bitmap, matrix, paint)
        resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            inputFloats[i] = ((p and 0xFF) - MEAN[0]) / STD[0]                  // B
            inputFloats[plane + i] = (((p shr 8) and 0xFF) - MEAN[1]) / STD[1]  // G
            inputFloats[2 * plane + i] = (((p shr 16) and 0xFF) - MEAN[2]) / STD[2]  // R
        }
        inBufs[0].writeFloat(inputFloats)
        model.run(inBufs, outBufs)
        val loc = outBufs[iLoc].readFloat()
        val conf = outBufs[iConf].readFloat()
        val mask = outBufs[iMask].readFloat()
        val proto = outBufs[iProto].readFloat()   // [PS*PS*K], row-major (h,w,k)

        // per-anchor top class + SSD box decode for candidates over threshold
        data class Cand(val a: Int, val cls: Int, val sc: Float, val box: FloatArray)
        val cands = ArrayList<Cand>()
        for (a in 0 until N) {
            var best = 1; var bv = conf[a * NC + 1]
            for (c in 2 until NC) { val v = conf[a * NC + c]; if (v > bv) { bv = v; best = c } }
            if (bv < SCORE_THRESH) continue
            val pcx = priors[a * 4]; val pcy = priors[a * 4 + 1]; val pw = priors[a * 4 + 2]; val ph = priors[a * 4 + 3]
            val cx = pcx + loc[a * 4] * 0.1f * pw
            val cy = pcy + loc[a * 4 + 1] * 0.1f * ph
            val w = pw * Math.exp((loc[a * 4 + 2] * 0.2f).toDouble()).toFloat()
            val h = ph * Math.exp((loc[a * 4 + 3] * 0.2f).toDouble()).toFloat()
            cands.add(Cand(a, best - 1, bv, floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        }
        cands.sortByDescending { it.sc }
        // per-class greedy NMS
        val kept = ArrayList<Cand>()
        val removed = BooleanArray(cands.size)
        for (i in cands.indices) {
            if (removed[i]) continue
            kept.add(cands[i])
            if (kept.size >= MAX_DET) break
            for (j in i + 1 until cands.size) {
                if (!removed[j] && cands[j].cls == cands[i].cls && iou(cands[i].box, cands[j].box) > IOU_THRESH)
                    removed[j] = true
            }
        }

        // lincomb masks: mask = sigmoid(proto @ coeff), cropped to box, thresholded at 0.5
        val out = ArrayList<Instance>(kept.size)
        for (d in kept) {
            val coeff = FloatArray(K) { mask[d.a * K + it] }
            val m = BooleanArray(SIZE * SIZE)
            val px1 = (d.box[0] * SIZE).toInt().coerceIn(0, SIZE); val py1 = (d.box[1] * SIZE).toInt().coerceIn(0, SIZE)
            val px2 = (d.box[2] * SIZE).toInt().coerceIn(0, SIZE); val py2 = (d.box[3] * SIZE).toInt().coerceIn(0, SIZE)
            for (yy in py1 until py2) {
                val protoY = yy * PS / SIZE
                for (xx in px1 until px2) {
                    val protoX = xx * PS / SIZE
                    val base = (protoY * PS + protoX) * K
                    var s = 0f
                    for (k in 0 until K) s += proto[base + k] * coeff[k]
                    if (s > 0f) m[yy * SIZE + xx] = true   // sigmoid(s)>0.5  <=>  s>0
                }
            }
            out.add(Instance(d.cls, d.sc, d.box[0], d.box[1], d.box[2], d.box[3], m))
        }
        return out to ((System.nanoTime() - t) / 1_000_000)
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = maxOf(a[0], b[0]); val y1 = maxOf(a[1], b[1]); val x2 = minOf(a[2], b[2]); val y2 = minOf(a[3], b[3])
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (ua <= 0f) 0f else inter / ua
    }

    override fun close() {
        model.close()
        if (!resized.isRecycled) resized.recycle()
    }
}
