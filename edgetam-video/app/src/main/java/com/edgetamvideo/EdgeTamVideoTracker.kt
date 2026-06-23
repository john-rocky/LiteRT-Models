package com.edgetamvideo

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
 * EdgeTAM (on-device SAM2) VIDEO object tracking on LiteRT CompiledModel GPU.
 *
 * Four stateless per-frame graphs run on the GPU; the rolling memory bank lives here in Kotlin
 * (the standard on-device SAM2 split). Verified numerically identical (IoU ~1.0) to the HF PyTorch
 * model frame-by-frame — see litert-upstream/edgetam/video_spike/deploy_ref_flat.py.
 *
 *   encode    frame[3x1024x1024] -> [pix_raw | hi0 | hi1]
 *   memcond   [pix_raw | memory | mem_pos | key_mask] -> pix_feat   (tracking frames)
 *   decode    [pix_feat | hi0 | hi1 | sparse] -> [masks(3) | iou(3) | objptr(3) | objscore]
 *   memorize  [pix_raw | mask_for_mem] -> [spatial_mem | spatial_pos]
 *
 * Conditioning frame (first tap): pix_feat = pix_raw + no_memory (no memcond).
 */
class EdgeTamVideoTracker(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "EdgeTAMVideo"
        private const val SIZE = 1024
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        // flat tensor sizes
        private const val IE = 256 * 64 * 64        // pix_raw / pix_feat / hi1     = 1048576
        private const val H0 = 32 * 256 * 256       // hi0                          = 2097152
        private const val H1 = 64 * 128 * 128       // hi1                          = 1048576
        private const val NMM = 7                   // spatial memory frames
        private const val MAXP = 16                 // object-pointer frames
        private const val MEMCH = 64                // memory channel dim
        private const val SPT = 512                 // perceiver tokens per frame
        private const val MEM = NMM * SPT + MAXP * 4 // total memory tokens          = 3648
        private const val MC = MEM * MEMCH          // memory flat                   = 233472
        private const val NO_OBJ = -1024f
        private const val SCALE = 20f
        private const val BIAS = -10f
        private const val TWO_PI = (2.0 * Math.PI).toFloat()
        private const val DEC_OUT = 196608 + 3 + 768 + 1
    }

    private val encode = load(context, "encode.tflite")
    private val memcond = load(context, "memcond.tflite")
    private val decode = load(context, "decode.tflite")
    private val memorize = load(context, "memorize.tflite")

    private val noMemory = readBin(context, "no_memory.bin", 256)          // (256)
    private val mtpe = readBin(context, "mtpe.bin", 7 * 64)                 // (7,64)
    private val noObjptr = readBin(context, "no_objptr.bin", 256)          // (256)
    private val trackSparse = readBin(context, "track_sparse.bin", 512)    // (2,256)
    private val prompt = readBin(context, "video_prompt.bin", 768)         // gaussian(256)|pe1(256)|nap(256)
    private val gaussian = prompt.copyOfRange(0, 256)
    private val pointEmbed1 = prompt.copyOfRange(256, 512)
    private val notAPoint = prompt.copyOfRange(512, 768)

    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val canvasBmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    var accelerator = ""; private set

    // rolling memory state (per single object)
    private class Spatial(val frame: Int, val mem: FloatArray, val pos: FloatArray)
    private class Pointer(val frame: Int, val ptr: FloatArray)
    private val spatialBank = ArrayList<Spatial>()
    private val pointerBank = ArrayList<Pointer>()

    private fun load(ctx: Context, file: String): CompiledModel {
        val m = CompiledModel.create(ctx.assets, file, CompiledModel.Options(Accelerator.GPU), null)
        accelerator = "GPU"
        Log.i(TAG, "$file loaded on GPU")
        return m
    }

    private fun readBin(ctx: Context, file: String, n: Int): FloatArray {
        val bytes = ctx.assets.open(file).use { it.readBytes() }
        val fb = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(n).also { fb.get(it) }
    }

    private fun run1(m: CompiledModel, input: FloatArray): FloatArray {
        val inb = m.createInputBuffers()
        inb[0].writeFloat(input)
        val out = m.run(inb)[0].readFloat()
        inb.forEach { it.close() }
        return out
    }

    fun reset() { spatialBank.clear(); pointerBank.clear() }

    /** Encode a frame -> [pix_raw | hi0 | hi1]. */
    private fun encodeFrame(src: Bitmap): FloatArray {
        Canvas(canvasBmp).drawBitmap(src, null, RectF(0f, 0f, SIZE.toFloat(), SIZE.toFloat()), paint)
        canvasBmp.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val p = pixels[i]
            inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        return run1(encode, inputFloats)
    }

    /** Sparse prompt for a positive point in model (0..1024) coords: [point | not_a_point]. */
    private fun pointSparse(modelX: Float, modelY: Float): FloatArray {
        val sp = FloatArray(512)
        val ccx = 2f * ((modelX + 0.5f) / SIZE) - 1f
        val ccy = 2f * ((modelY + 0.5f) / SIZE) - 1f
        for (k in 0 until 128) {
            val proj = TWO_PI * (ccx * gaussian[k] + ccy * gaussian[128 + k])
            sp[k] = sin(proj) + pointEmbed1[k]
            sp[128 + k] = cos(proj) + pointEmbed1[128 + k]
        }
        for (k in 0 until 256) sp[256 + k] = notAPoint[k]
        return sp
    }

    /** get_1d_sine_pe(off/15, 64): [sin(32) | cos(32)]. */
    private fun sinePe(off: Int): FloatArray {
        val out = FloatArray(64)
        val pos = off / 15f
        for (i in 0 until 32) {
            val dimT = Math.pow(10000.0, (2.0 * (i / 2)) / 32.0).toFloat()
            val v = pos / dimT
            out[i] = sin(v); out[32 + i] = cos(v)
        }
        return out
    }

    /** Assemble fixed memory [7x512 spatial | 16x4 ptr] + key_mask from the rolling bank. */
    private fun assemble(fi: Int): Triple<FloatArray, FloatArray, FloatArray> {
        val memory = FloatArray(MC)
        val mpos = FloatArray(MC)
        val mask = FloatArray(MEM) { -1e9f }
        // spatial: cond (offset 0 -> mtpe[6]) then offsets 6..1
        val condFrame = spatialBank[0].frame
        val realSpatial = ArrayList<Pair<Spatial, Int>>()
        realSpatial.add(spatialBank[0] to 6)
        for (off in NMM - 1 downTo 1) {
            val pf = fi - off
            if (pf == condFrame) continue
            val hit = spatialBank.firstOrNull { it.frame == pf } ?: continue
            realSpatial.add(hit to off - 1)
        }
        for ((slot, sp) in realSpatial.withIndex()) {
            val (s, mtpeIdx) = sp
            val base = slot * SPT * MEMCH
            System.arraycopy(s.mem, 0, memory, base, SPT * MEMCH)
            for (t in 0 until SPT) for (c in 0 until MEMCH) {
                mpos[base + t * MEMCH + c] = s.pos[t * MEMCH + c] + mtpe[mtpeIdx * 64 + c]
            }
        }
        for (i in 0 until realSpatial.size * SPT) mask[i] = 0f
        // pointers: most recent <=16 frames, each 256 -> 4x64, pos = sine(fi-frame)
        val recent = pointerBank.sortedByDescending { it.frame }.take(MAXP)
        var pi = 0
        val ptrBase = NMM * SPT * MEMCH
        for (p in recent) {
            val pp = sinePe(fi - p.frame)
            for (t in 0 until 4) {
                val dst = ptrBase + pi * MEMCH
                System.arraycopy(p.ptr, t * MEMCH, memory, dst, MEMCH)
                System.arraycopy(pp, 0, mpos, dst, MEMCH)
                mask[NMM * SPT + pi] = 0f
                pi++
            }
        }
        return Triple(memory, mpos, mask)
    }

    /** Decode + select best mask + gate + push memory. Returns 256x256 logit mask. */
    private fun decodeStore(fi: Int, pixFeat: FloatArray, hi0: FloatArray, hi1: FloatArray,
                            pixRaw: FloatArray, sparse: FloatArray): FloatArray {
        val decIn = FloatArray(IE + H0 + H1 + 512)
        System.arraycopy(pixFeat, 0, decIn, 0, IE)
        System.arraycopy(hi0, 0, decIn, IE, H0)
        System.arraycopy(hi1, 0, decIn, IE + H0, H1)
        System.arraycopy(sparse, 0, decIn, IE + H0 + H1, 512)
        val out = run1(decode, decIn)                            // [masks(196608)|iou(3)|objptr(768)|objscore(1)]
        var best = 0
        for (k in 1 until 3) if (out[196608 + k] > out[196608 + best]) best = k
        val appearing = out[DEC_OUT - 1] > 0f
        val mask = FloatArray(256 * 256)
        if (appearing) System.arraycopy(out, best * 65536, mask, 0, 65536)
        else java.util.Arrays.fill(mask, NO_OBJ)
        val ptr = FloatArray(256)
        if (appearing) System.arraycopy(out, 196611 + best * 256, ptr, 0, 256)
        else System.arraycopy(noObjptr, 0, ptr, 0, 256)
        // mask_for_mem = (bilinear256->1024 > 0) * scale + bias
        val mfm = maskForMem(mask)
        val memIn = FloatArray(2 * IE)
        System.arraycopy(pixRaw, 0, memIn, 0, IE)
        System.arraycopy(mfm, 0, memIn, IE, IE)
        val mo = run1(memorize, memIn)                           // [spatial_mem(32768)|spatial_pos(32768)]
        val sm = mo.copyOfRange(0, SPT * MEMCH)
        val sp = mo.copyOfRange(SPT * MEMCH, 2 * SPT * MEMCH)
        spatialBank.add(Spatial(fi, sm, sp))
        pointerBank.add(Pointer(fi, ptr))
        return mask
    }

    /** Bilinear upsample 256->1024 (align_corners=False) of logits, then (>0)*20-10. */
    private fun maskForMem(mask: FloatArray): FloatArray {
        val out = FloatArray(SIZE * SIZE)
        val scale = 256f / SIZE
        for (oy in 0 until SIZE) {
            var sy = (oy + 0.5f) * scale - 0.5f
            if (sy < 0f) sy = 0f; if (sy > 255f) sy = 255f
            val y0 = sy.toInt(); val y1 = if (y0 < 255) y0 + 1 else y0; val fy = sy - y0
            for (ox in 0 until SIZE) {
                var sx = (ox + 0.5f) * scale - 0.5f
                if (sx < 0f) sx = 0f; if (sx > 255f) sx = 255f
                val x0 = sx.toInt(); val x1 = if (x0 < 255) x0 + 1 else x0; val fx = sx - x0
                val v00 = mask[y0 * 256 + x0]; val v01 = mask[y0 * 256 + x1]
                val v10 = mask[y1 * 256 + x0]; val v11 = mask[y1 * 256 + x1]
                val v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy
                out[oy * SIZE + ox] = if (v > 0f) SCALE + BIAS else BIAS
            }
        }
        return out
    }

    /** Conditioning frame: tap a point -> mask, init the bank. */
    fun startTracking(frame: Bitmap, modelX: Float, modelY: Float): FloatArray {
        reset()
        val eo = encodeFrame(frame)
        val pixRaw = eo.copyOfRange(0, IE); val hi0 = eo.copyOfRange(IE, IE + H0); val hi1 = eo.copyOfRange(IE + H0, IE + H0 + H1)
        val pixFeat = FloatArray(IE)            // pix_raw + no_memory (per channel)
        for (c in 0 until 256) { val nm = noMemory[c]; val b = c * 4096; for (s in 0 until 4096) pixFeat[b + s] = pixRaw[b + s] + nm }
        return decodeStore(0, pixFeat, hi0, hi1, pixRaw, pointSparse(modelX, modelY))
    }

    /** Subsequent frame: track using the memory bank. */
    fun trackFrame(fi: Int, frame: Bitmap): FloatArray {
        val eo = encodeFrame(frame)
        val pixRaw = eo.copyOfRange(0, IE); val hi0 = eo.copyOfRange(IE, IE + H0); val hi1 = eo.copyOfRange(IE + H0, IE + H0 + H1)
        val (memory, mpos, mask) = assemble(fi)
        val mcIn = FloatArray(IE + 2 * MC + MEM)
        System.arraycopy(pixRaw, 0, mcIn, 0, IE)
        System.arraycopy(memory, 0, mcIn, IE, MC)
        System.arraycopy(mpos, 0, mcIn, IE + MC, MC)
        System.arraycopy(mask, 0, mcIn, IE + 2 * MC, MEM)
        val pixFeat = run1(memcond, mcIn)
        return decodeStore(fi, pixFeat, hi0, hi1, pixRaw, trackSparse)
    }

    override fun close() { encode.close(); memcond.close(); decode.close(); memorize.close(); canvasBmp.recycle() }
}
