package com.sam3

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SAM 3.1 text-prompted detection + instance segmentation (image side of the
 * Object-Multiplex checkpoint) on the LiteRT CompiledModel API, three graphs:
 *   vision : image [1,3,1008,1008] -> [fpn288 | fpn144 | fpn72]           (GPU)
 *   text   : token embeddings [1,32,1024] -> text memory [1, 32*256]      (CPU — the
 *            CLIP-L residual stream reaches |x|~1.2e3; fp16 GPU execution corrupts the
 *            prompt embedding for some prompts, and this graph is fast anyway)
 *   head   : [fpn x3 | text_mem | pad(32)] -> [logits(200) | boxes(200x4 cxcywh,
 *            normalized) | presence(1) | mask logits (200x288x288)]       (GPU)
 * Host: BPE tokenize, fp16 token-embedding lookup, score threshold, mask sigmoid.
 * score = sigmoid(logit) * sigmoid(presence); keep > threshold.
 */
class Sam3Detector(private val ctx: Context) : Closeable {

    companion object {
        const val SIZE = 1008
        const val TOK = 32
        const val TDIM = 1024
        const val QUERIES = 200
        const val MASK = 288
        const val N_VIS = 256 * (288 * 288 + 144 * 144 + 72 * 72)
    }

    data class Detection(
        val score: Float,
        /** cx, cy, w, h normalized to [0,1] */
        val box: FloatArray,
        /** sigmoid mask probabilities, MASK x MASK, row-major */
        val mask: FloatArray,
    )

    private fun f(name: String) = File(ctx.filesDir, name).also {
        check(it.exists()) { "Missing ${it.name}. Run scripts/install_to_device.sh first." }
    }

    private val tokenizer = BpeTokenizer(ctx)
    private val visModel = CompiledModel.create(f("sam3_vision.tflite").absolutePath,
        CompiledModel.Options(Accelerator.GPU), null)
    private val visIn = visModel.createInputBuffers()
    private val visOut = visModel.createOutputBuffers()
    private val textModel = CompiledModel.create(f("sam3_text.tflite").absolutePath,
        CompiledModel.Options(Accelerator.CPU), null)
    private val textIn = textModel.createInputBuffers()
    private val textOut = textModel.createOutputBuffers()
    private val headModel = CompiledModel.create(f("sam3_head.tflite").absolutePath,
        CompiledModel.Options(Accelerator.GPU), null)
    private val headIn = headModel.createInputBuffers()
    private val headOut = headModel.createOutputBuffers()

    /** fp16 [49408 x 1024] row-major token-embedding table, kept as raw bytes (101 MB). */
    private val tokEmb: ByteBuffer =
        ByteBuffer.wrap(f("sam3_token_embed.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN)

    var lastVisionMs = 0L; var lastTextMs = 0L; var lastHeadMs = 0L

    private fun halfToFloat(h: Int): Float {
        val s = (h ushr 15) and 0x1; val e = (h ushr 10) and 0x1F; val m = h and 0x3FF
        val bits = when {
            e == 0 -> if (m == 0) s shl 31 else {
                var mant = m; var exp = -1
                do { mant = mant shl 1; exp++ } while (mant and 0x400 == 0)
                (s shl 31) or ((127 - 15 - exp) shl 23) or ((mant and 0x3FF) shl 13)
            }
            e == 0x1F -> (s shl 31) or (0xFF shl 23) or (m shl 13)
            else -> (s shl 31) or ((e - 15 + 127) shl 23) or (m shl 13)
        }
        return Float.fromBits(bits)
    }

    private fun preprocess(bm: Bitmap): FloatArray {
        val s = Bitmap.createScaledBitmap(bm, SIZE, SIZE, true)
        val px = IntArray(SIZE * SIZE); s.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
        val out = FloatArray(3 * SIZE * SIZE)
        val plane = SIZE * SIZE
        for (i in px.indices) {
            val p = px[i]
            out[i] = (((p shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
            out[plane + i] = (((p shr 8) and 0xFF) / 255f - 0.5f) / 0.5f
            out[2 * plane + i] = ((p and 0xFF) / 255f - 0.5f) / 0.5f
        }
        return out
    }

    // vision features are per-image; cache them so re-prompting skips the big graph
    private var cachedKey: Bitmap? = null
    lateinit var visFeat: FloatArray
    /** debug taps for the on-device parity check (autotest) */
    var lastTextMem: FloatArray? = null
    var lastPad: FloatArray? = null
    var lastHeadOut: FloatArray? = null
    var lastInput: FloatArray? = null

    private fun runVision(bm: Bitmap) {
        if (cachedKey === bm) { lastVisionMs = 0; return }
        val t0 = System.nanoTime()
        val inp = preprocess(bm)
        lastInput = inp
        visIn[0].writeFloat(inp)
        visModel.run(visIn, visOut)
        visFeat = visOut[0].readFloat()
        lastVisionMs = (System.nanoTime() - t0) / 1_000_000
        cachedKey = bm
    }

    private fun runText(prompt: String): Pair<FloatArray, FloatArray> {
        val t0 = System.nanoTime()
        val ids = tokenizer.encode(prompt)
        val emb = FloatArray(TOK * TDIM)
        for (t in 0 until TOK) {
            val base = ids[t] * TDIM * 2
            for (d in 0 until TDIM) {
                emb[t * TDIM + d] = halfToFloat(tokEmb.getShort(base + d * 2).toInt() and 0xFFFF)
            }
        }
        textIn[0].writeFloat(emb)
        textModel.run(textIn, textOut)
        val mem = textOut[0].readFloat()                        // [32*256]
        val pad = FloatArray(TOK) { if (ids[it] == 0) 1f else 0f }
        lastTextMem = mem; lastPad = pad
        lastTextMs = (System.nanoTime() - t0) / 1_000_000
        return Pair(mem, pad)
    }

    /** @return detections above [threshold], unsorted (query order). */
    fun detect(bm: Bitmap, prompt: String, threshold: Float = 0.5f): List<Detection> {
        runVision(bm)
        val (mem, pad) = runText(prompt)
        val t0 = System.nanoTime()
        val headInput = FloatArray(N_VIS + TOK * 256 + TOK)
        System.arraycopy(visFeat, 0, headInput, 0, N_VIS)
        System.arraycopy(mem, 0, headInput, N_VIS, TOK * 256)
        System.arraycopy(pad, 0, headInput, N_VIS + TOK * 256, TOK)
        headIn[0].writeFloat(headInput)
        headModel.run(headIn, headOut)
        val y = headOut[0].readFloat()
        lastHeadOut = y
        lastHeadMs = (System.nanoTime() - t0) / 1_000_000
        val presence = 1f / (1f + kotlin.math.exp(-y[1000]))
        val out = ArrayList<Detection>()
        for (q in 0 until QUERIES) {
            val score = 1f / (1f + kotlin.math.exp(-y[q])) * presence
            if (score <= threshold) continue
            val box = FloatArray(4) { y[200 + q * 4 + it] }
            val mask = FloatArray(MASK * MASK)
            val base = 1001 + q * MASK * MASK
            for (i in mask.indices) mask[i] = 1f / (1f + kotlin.math.exp(-y[base + i]))
            out.add(Detection(score, box, mask))
        }
        return out
    }

    override fun close() {
        visIn.forEach { it.close() }; visOut.forEach { it.close() }; visModel.close()
        textIn.forEach { it.close() }; textOut.forEach { it.close() }; textModel.close()
        headIn.forEach { it.close() }; headOut.forEach { it.close() }; headModel.close()
    }
}
