package com.edgetam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.cos
import kotlin.math.sin

/**
 * EdgeTAM (on-device SAM2) promptable segmentation on LiteRT CompiledModel GPU — tap-to-segment.
 * MobileSAM split: image ENCODER (GPU, once per image) -> embeddings + FPN; prompt encoder (Kotlin, per tap)
 * -> sparse embedding; mask DECODER (GPU, per tap) -> mask. Verified corr 1.0 vs the full PyTorch model.
 */
class EdgeTamSegmenter(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "EdgeTAM"
        private const val SIZE = 1024
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val IE = 256 * 64 * 64      // image_embeddings
        private const val F0 = 32 * 256 * 256     // fpn0
        private const val F1 = 64 * 128 * 128     // fpn1
        private const val TWO_PI = (2.0 * Math.PI).toFloat()
    }

    private val encoder: CompiledModel
    private val decoder: CompiledModel
    private val gaussian: FloatArray   // (2,128) row-major: [0..127]=x-proj, [128..255]=y-proj
    private val pointEmbed1: FloatArray
    private val notAPoint: FloatArray

    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val canvasBmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    private var cachedIe = FloatArray(0)
    private var cachedF0 = FloatArray(0)
    private var cachedF1 = FloatArray(0)
    var accelerator = ""; private set

    init {
        // Both graphs run on the ML Drift GPU. The encoder's RepViT SqueezeExcite global-average-pool
        // (x.mean((2,3))) lowered to a single multi-axis SUM that ML Drift mis-computes -> NaN -> empty masks;
        // it is reformulated as two single-axis means at conversion time (see scripts/etam_sefinal.py),
        // which is numerically identical and GPU-correct (verified mask_fg≈11264 vs the PyTorch model).
        encoder = load(context, "edgetam_encoder.tflite", Accelerator.GPU)
        decoder = load(context, "edgetam_decoder.tflite", Accelerator.GPU)
        val buf = context.assets.open("edgetam_prompt.bin").use { it.readBytes() }
        val fb = java.nio.ByteBuffer.wrap(buf).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val all = FloatArray(768).also { fb.get(it) }
        gaussian = all.copyOfRange(0, 256)
        pointEmbed1 = all.copyOfRange(256, 512)
        notAPoint = all.copyOfRange(512, 768)
    }

    private fun load(ctx: Context, file: String, accel: Accelerator): CompiledModel {
        val m = CompiledModel.create(ctx.assets, file, CompiledModel.Options(accel), null)
        Log.i(TAG, "$file loaded on $accel"); accelerator = accel.toString(); return m
    }

    /** Run the image encoder once and cache the embeddings (call when a new image is picked). */
    fun encode(src: Bitmap): Long {
        val t = System.nanoTime()
        Canvas(canvasBmp).drawBitmap(src, null, RectF(0f, 0f, SIZE.toFloat(), SIZE.toFloat()), paint)
        canvasBmp.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val p = pixels[i]
            inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        val encIn = encoder.createInputBuffers()
        encIn[0].writeFloat(inputFloats)
        // single concatenated output [ie | fpn0 | fpn1] split by offset (order-independent)
        val flat = encoder.run(encIn)[0].readFloat()
        cachedIe = flat.copyOfRange(0, IE)
        cachedF0 = flat.copyOfRange(IE, IE + F0)
        cachedF1 = flat.copyOfRange(IE + F0, IE + F0 + F1)
        encIn.forEach { it.close() }
        return (System.nanoTime() - t) / 1_000_000
    }

    /** Segment at a positive point in model (0..1024) coordinates. Returns a 256x256 mask (logits). */
    fun segment(modelX: Float, modelY: Float): FloatArray {
        val sparse = FloatArray(512)
        val ccx = 2f * ((modelX + 0.5f) / SIZE) - 1f
        val ccy = 2f * ((modelY + 0.5f) / SIZE) - 1f
        for (k in 0 until 128) {
            val proj = TWO_PI * (ccx * gaussian[k] + ccy * gaussian[128 + k])
            sparse[k] = sin(proj) + pointEmbed1[k]
            sparse[128 + k] = cos(proj) + pointEmbed1[128 + k]
        }
        for (k in 0 until 256) sparse[256 + k] = notAPoint[k]

        // single concatenated decoder input [ie | sparse | fpn0 | fpn1] (order-independent)
        val flatIn = FloatArray(IE + 512 + F0 + F1)
        System.arraycopy(cachedIe, 0, flatIn, 0, IE)
        System.arraycopy(sparse, 0, flatIn, IE, 512)
        System.arraycopy(cachedF0, 0, flatIn, IE + 512, F0)
        System.arraycopy(cachedF1, 0, flatIn, IE + 512 + F0, F1)
        val di = decoder.createInputBuffers()
        di[0].writeFloat(flatIn)
        val masks = decoder.run(di)[0].readFloat()   // (3,256,256)
        di.forEach { it.close() }
        return masks.copyOfRange(0, 256 * 256)        // mask 0 (the whole-object mask)
    }

    override fun close() {
        encoder.close(); decoder.close(); canvasBmp.recycle()
    }
}
