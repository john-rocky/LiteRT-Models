package com.sam2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin

/**
 * SAM 2.1 Hiera-Tiny VIDEO tracking on LiteRT CompiledModel GPU.
 *
 * On-device host loop over the four fixed-shape per-frame graphs produced by
 * scripts/convert_sam2_video.py (encode / memcond / decode / memorize). This is the exact
 * orchestration verified in scripts/verify_sam2_video.py (min mask-IoU 0.9999 vs the HF
 * PyTorch reference over a 10-frame clip): the rolling memory bank and the best-mask /
 * no-object / mask-for-mem bookkeeping live here in Kotlin; every heavy tensor op is a GPU
 * graph.
 *
 * The memory attention keeps the batch dim (rank-4 [1, heads, N, d]) so it dodges the ML
 * Drift rank-3 batched-attention miscompute; the residual fp16 error over the N x 4096
 * memory keys is accumulation only and does not reach the mask.
 *
 * Models are large (encoder alone is ~80 MB fp16), so they load from filesDir via the
 * file-path CompiledModel overload — push them first with scripts/install_video_to_device.sh.
 */
class Sam2VideoTracker(context: Context, private val numMaskMem: Int = 2) : AutoCloseable {

    companion object {
        private const val TAG = "SAM2V"
        const val SIZE = 1024
        private const val LOW = 256                       // mask decoder output side
        private const val IE = 256 * 64 * 64              // pix features (top level)
        private const val F0 = 32 * 256 * 256             // high-res s0
        private const val F1 = 64 * 128 * 128             // high-res s1
        private const val HW = 4096                       // 64x64 memory tokens
        private const val MEMCH = 64                      // memory channel dim
        private const val NPTR_FRAMES = 16                // object pointers kept
        private const val PTR_SPLIT = 4                   // tokens per pointer (256 / 64)
        private const val NPTR = NPTR_FRAMES * PTR_SPLIT  // 64 pointer tokens
        private const val DEC_MASKS = 4 * LOW * LOW       // 4 mask logits
        private const val MEM_SCALE = 20f
        private const val MEM_BIAS = -10f
        private const val NO_OBJ = -1024f
        private const val MASK_NEG = -1e9f
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val TWO_PI = (2.0 * Math.PI).toFloat()
    }

    private val encoder: CompiledModel
    private val memcond: CompiledModel
    private val decoder: CompiledModel
    private val memorize: CompiledModel

    // Prompt-encoder constants.
    private val gaussian: FloatArray   // (2,128) row-major
    private val pointEmbed1: FloatArray
    private val pointEmbed0: FloatArray
    private val notAPoint: FloatArray
    private val trackSparse: FloatArray            // no-point sparse prompt for tracked frames (512)
    private val mtpe: FloatArray                   // memory temporal PE (7,64)
    private val noObjPtr: FloatArray               // (256)
    private val tposW: FloatArray                  // (64,256) object-pointer temporal projection
    private val tposB: FloatArray                  // (64)

    // Rolling memory bank, keyed by frame index.
    private val spatialBank = HashMap<Int, FloatArray>()   // frame -> (4096*64) token-major
    private val ptrBank = HashMap<Int, FloatArray>()       // frame -> (256)
    private var condFrame = -1

    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val squareBmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    var accelerator = ""; private set

    init {
        val dir = context.filesDir
        encoder = load(dir, "sam2v_encode.tflite")
        memcond = load(dir, "sam2v_memcond$numMaskMem.tflite")
        decoder = load(dir, "sam2v_decode.tflite")
        memorize = load(dir, "sam2v_memorize.tflite")

        val p = readFloats(File(dir, "sam2v_prompt.bin"))          // 1024
        gaussian = p.copyOfRange(0, 256)
        pointEmbed1 = p.copyOfRange(256, 512)
        pointEmbed0 = p.copyOfRange(512, 768)
        notAPoint = p.copyOfRange(768, 1024)
        trackSparse = readFloats(File(dir, "sam2v_track_sparse.bin"))   // 512
        mtpe = readFloats(File(dir, "sam2v_mtpe.bin"))                  // 448
        noObjPtr = readFloats(File(dir, "sam2v_no_obj_ptr.bin"))        // 256
        val tp = readFloats(File(dir, "sam2v_tpos_proj.bin"))          // 16448
        tposW = tp.copyOfRange(0, 64 * 256)
        tposB = tp.copyOfRange(64 * 256, 64 * 256 + 64)
    }

    private fun load(dir: File, file: String): CompiledModel {
        val path = File(dir, file)
        require(path.exists()) { "$file not in filesDir — run scripts/install_video_to_device.sh first" }
        val m = CompiledModel.create(path.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
        Log.i(TAG, "$file loaded on GPU"); accelerator = Accelerator.GPU.toString(); return m
    }

    private fun readFloats(f: File): FloatArray {
        val b = f.readBytes()
        val fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.size / 4).also { fb.get(it) }
    }

    /** Per-frame result: the low-res (256x256) mask logits, the object-score logit, and appearing flag. */
    class FrameResult(val mask: FloatArray, val objScore: Float, val appearing: Boolean)

    /** Reset the bank; call before tracking a new clip. */
    fun reset() {
        spatialBank.clear(); ptrBank.clear(); condFrame = -1
    }

    /** The single running graph output holders (reused across frames to avoid per-frame allocation). */
    private val encFlat = FloatArray(IE + F0 + F1)

    private fun encode(frame: Bitmap): Triple<FloatArray, FloatArray, FloatArray> {
        Canvas(squareBmp).drawBitmap(frame, null, RectF(0f, 0f, SIZE.toFloat(), SIZE.toFloat()), paint)
        squareBmp.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val px = pixels[i]
            inputFloats[i] = (((px shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            inputFloats[plane + i] = (((px shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            inputFloats[2 * plane + i] = ((px and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        val inb = encoder.createInputBuffers()
        inb[0].writeFloat(inputFloats)
        val flat = encoder.run(inb)[0].readFloat()
        inb.forEach { it.close() }
        val pixRaw = flat.copyOfRange(0, IE)
        val hi0 = flat.copyOfRange(IE, IE + F0)
        val hi1 = flat.copyOfRange(IE + F0, IE + F0 + F1)
        return Triple(pixRaw, hi0, hi1)
    }

    /** Sparse prompt (512) for a single positive/negative click in model (0..1024) coords. */
    private fun clickSparse(x: Float, y: Float, positive: Boolean): FloatArray {
        val sparse = FloatArray(512)
        val ccx = 2f * ((x + 0.5f) / SIZE) - 1f
        val ccy = 2f * ((y + 0.5f) / SIZE) - 1f
        val pe = if (positive) pointEmbed1 else pointEmbed0
        for (k in 0 until 128) {
            val proj = TWO_PI * (ccx * gaussian[k] + ccy * gaussian[128 + k])
            sparse[k] = sin(proj) + pe[k]
            sparse[128 + k] = cos(proj) + pe[128 + k]
        }
        for (k in 0 until 256) sparse[256 + k] = notAPoint[k]
        return sparse
    }

    /** get_1d_sine_pe(pos, 256): [sin(pos/dim_t) | cos(pos/dim_t)], dim_t = 10000^(2*(i/2)/128). */
    private fun sinePe(pos: Float): FloatArray {
        val out = FloatArray(256)
        for (i in 0 until 128) {
            val dimT = Math.pow(10000.0, (2 * (i / 2)).toDouble() / 128.0).toFloat()
            val v = pos / dimT
            out[i] = sin(v)
            out[128 + i] = cos(v)
        }
        return out
    }

    /** Object-pointer temporal position embedding: tposW @ sinePe(t_diff / 15) + tposB. */
    private fun ptrPos(tDiff: Int): FloatArray {
        val pe = sinePe(tDiff / (NPTR_FRAMES - 1f))
        val out = FloatArray(MEMCH)
        for (r in 0 until MEMCH) {
            var s = tposB[r]
            val base = r * 256
            for (c in 0 until 256) s += tposW[base + c] * pe[c]
            out[r] = s
        }
        return out
    }

    /** Assemble the fixed memcond input for a tracked frame t (mirrors verify_sam2_video.track). */
    private fun buildMemcondInput(t: Int, pixRaw: FloatArray): FloatArray {
        val n = numMaskMem
        val memLen = n * HW * MEMCH
        val flat = FloatArray(IE + memLen + n * MEMCH + 2 * NPTR * MEMCH + n * HW + NPTR)
        var off = 0
        System.arraycopy(pixRaw, 0, flat, off, IE); off += IE
        val memBase = off; off += memLen
        val tpeBase = off; off += n * MEMCH
        val ptrTokBase = off; off += NPTR * MEMCH
        val ptrPosBase = off; off += NPTR * MEMCH
        val kmBase = off
        java.util.Arrays.fill(flat, kmBase, kmBase + n * HW + NPTR, MASK_NEG)

        // ---- spatial slots: conditioning frame first, then the most-distant recent frames ----
        var slot = 0
        fun fillSpatial(mem: FloatArray, tpeRow: Int) {
            System.arraycopy(mem, 0, flat, memBase + slot * HW * MEMCH, HW * MEMCH)
            System.arraycopy(mtpe, tpeRow * MEMCH, flat, tpeBase + slot * MEMCH, MEMCH)
            java.util.Arrays.fill(flat, kmBase + slot * HW, kmBase + (slot + 1) * HW, 0f)
            slot++
        }
        fillSpatial(spatialBank[condFrame]!!, 6)                       // cond frame uses mtpe row 6
        for (offset in (n - 1) downTo 1) {
            val pf = t - offset
            val mem = spatialBank[pf]
            if (mem != null && pf != condFrame) fillSpatial(mem, offset - 1)
        }

        // ---- object pointers: conditioning pointer, then recent tracked pointers ----
        val ptrs = ArrayList<Pair<Int, FloatArray>>()
        ptrs.add(Pair(t - condFrame, ptrBank[condFrame]!!))
        var td = 1
        while (td < NPTR_FRAMES) {
            val pf = t - td
            if (pf < 0) break
            val p = ptrBank[pf]
            if (p != null && pf != condFrame) ptrs.add(Pair(td, p))
            td++
        }
        for (i in ptrs.indices) {
            val (diff, p) = ptrs[i]
            val pos = ptrPos(diff)
            for (j in 0 until PTR_SPLIT) {
                val tok = i * PTR_SPLIT + j
                System.arraycopy(p, j * MEMCH, flat, ptrTokBase + tok * MEMCH, MEMCH)
                System.arraycopy(pos, 0, flat, ptrPosBase + tok * MEMCH, MEMCH)
                flat[kmBase + n * HW + tok] = 0f
            }
        }
        return flat
    }

    /** Decoder: [pix_feat | hi0 | hi1 | sparse | nomem] -> masks/iou/ptr/obj. Returns FrameResult + best ptr. */
    private fun decode(pixFeat: FloatArray, hi0: FloatArray, hi1: FloatArray, sparse: FloatArray, nomem: Float):
        Pair<DecodeOut, Int> {
        val flat = FloatArray(IE + F0 + F1 + 512 + 1)
        var o = 0
        System.arraycopy(pixFeat, 0, flat, o, IE); o += IE
        System.arraycopy(hi0, 0, flat, o, F0); o += F0
        System.arraycopy(hi1, 0, flat, o, F1); o += F1
        System.arraycopy(sparse, 0, flat, o, 512); o += 512
        flat[o] = nomem
        val inb = decoder.createInputBuffers()
        inb[0].writeFloat(flat)
        val out = decoder.run(inb)[0].readFloat()
        inb.forEach { it.close() }
        val iouBase = DEC_MASKS
        var best = 1
        for (k in 2 until 4) if (out[iouBase + k] > out[iouBase + best]) best = k
        return Pair(DecodeOut(out), best)
    }

    private class DecodeOut(val raw: FloatArray) {
        fun mask(i: Int): FloatArray = raw.copyOfRange(i * LOW * LOW, (i + 1) * LOW * LOW)
        fun ptr(i: Int): FloatArray = raw.copyOfRange(DEC_MASKS + 4 + i * 256, DEC_MASKS + 4 + (i + 1) * 256)
        val obj: Float get() = raw[raw.size - 1]
    }

    /** memory encoder: [pix_raw | mask_for_mem(1024x1024) | occ] -> spatial memory (4096x64). */
    private fun memorizeFrame(pixRaw: FloatArray, maskForMem: FloatArray, occ: Float): FloatArray {
        val flat = FloatArray(2 * IE + 1)
        System.arraycopy(pixRaw, 0, flat, 0, IE)
        System.arraycopy(maskForMem, 0, flat, IE, SIZE * SIZE)
        flat[2 * IE] = occ
        val inb = memorize.createInputBuffers()
        inb[0].writeFloat(flat)
        val mem = memorize.run(inb)[0].readFloat()
        inb.forEach { it.close() }
        return mem
    }

    /** Bilinear 256->1024, align_corners=false (matches torch F.interpolate used at conversion). */
    private fun upsample(low: FloatArray): FloatArray {
        val out = FloatArray(SIZE * SIZE)
        val scale = LOW.toFloat() / SIZE
        for (y in 0 until SIZE) {
            var sy = (y + 0.5f) * scale - 0.5f
            if (sy < 0f) sy = 0f
            val y0 = sy.toInt().coerceAtMost(LOW - 1)
            val y1 = (y0 + 1).coerceAtMost(LOW - 1)
            val wy = sy - y0
            for (x in 0 until SIZE) {
                var sx = (x + 0.5f) * scale - 0.5f
                if (sx < 0f) sx = 0f
                val x0 = sx.toInt().coerceAtMost(LOW - 1)
                val x1 = (x0 + 1).coerceAtMost(LOW - 1)
                val wx = sx - x0
                val a = low[y0 * LOW + x0]; val b = low[y0 * LOW + x1]
                val c = low[y1 * LOW + x0]; val d = low[y1 * LOW + x1]
                out[y * SIZE + x] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) +
                    c * (1 - wx) * wy + d * wx * wy
            }
        }
        return out
    }

    /**
     * Conditioning frame: encode + prompt-decode + memorize; seeds the bank. `positive` marks a
     * foreground click at model (0..1024) coords. Returns the low-res mask.
     */
    fun startFrame(frameIdx: Int, frame: Bitmap, clickX: Float, clickY: Float): FrameResult {
        condFrame = frameIdx
        val (pixRaw, hi0, hi1) = encode(frame)
        val sparse = clickSparse(clickX, clickY, positive = true)
        val (dec, best) = decode(pixRaw, hi0, hi1, sparse, nomem = 1f)
        return finishFrame(frameIdx, pixRaw, dec, best, isPrompt = true)
    }

    /** Tracked frame: encode + memory-conditioned decode + memorize; extends the bank. */
    fun trackFrame(frameIdx: Int, frame: Bitmap): FrameResult {
        check(condFrame >= 0) { "call startFrame first" }
        val (pixRaw, hi0, hi1) = encode(frame)
        val memcondFlat = buildMemcondInput(frameIdx, pixRaw)
        val mcIn = memcond.createInputBuffers()
        mcIn[0].writeFloat(memcondFlat)
        val pixFeat = memcond.run(mcIn)[0].readFloat()
        mcIn.forEach { it.close() }
        val (dec, best) = decode(pixFeat, hi0, hi1, trackSparse, nomem = 0f)
        return finishFrame(frameIdx, pixRaw, dec, best, isPrompt = false)
    }

    private fun finishFrame(frameIdx: Int, pixRaw: FloatArray, dec: DecodeOut, best: Int, isPrompt: Boolean):
        FrameResult {
        val appearing = dec.obj > 0f
        val low = if (appearing) dec.mask(best) else FloatArray(LOW * LOW) { NO_OBJ }
        val objPtr = if (appearing) dec.ptr(best) else noObjPtr
        val high = upsample(low)
        val mfm = FloatArray(SIZE * SIZE)
        if (isPrompt) {
            for (i in mfm.indices) mfm[i] = (if (high[i] > 0f) 1f else 0f) * MEM_SCALE + MEM_BIAS
        } else {
            for (i in mfm.indices) mfm[i] = (1f / (1f + exp(-high[i]))) * MEM_SCALE + MEM_BIAS
        }
        val mem = memorizeFrame(pixRaw, mfm, if (appearing) 0f else 1f)
        spatialBank[frameIdx] = mem
        ptrBank[frameIdx] = objPtr
        return FrameResult(low, dec.obj, appearing)
    }

    override fun close() {
        encoder.close(); memcond.close(); decoder.close(); memorize.close(); squareBmp.recycle()
    }
}
