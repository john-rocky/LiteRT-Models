package com.clothseg

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
 * Cloth segmentation (U²-Net) on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 768, 768]  NCHW, RGB, (x/255 - 0.5)/0.5.
 * Output: [1, 4, 768, 768]  logits — argmax over 4 classes: 0 bg, 1 upper, 2 lower, 3 full body.
 *
 * U²-Net (RSU blocks) — pure CNN, fully GPU (defensive align_corners=False). ~88 ms/frame.
 */
class ClothSegmenter(context: Context, modelFileName: String = "clothseg.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "ClothSeg"
        const val S = 768
        const val OUT = 256   // downscaled class map for fast UI compositing
        const val NC = 4
    }

    private val model: CompiledModel
    private val inBufs: List<TensorBuffer>
    private val outBufs: List<TensorBuffer>

    private val inputFloats = FloatArray(3 * S * S)
    private val pixels = IntArray(S * S)
    private val resized = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val classMap = ByteArray(OUT * OUT)

    init {
        val options = CompiledModel.Options(Accelerator.GPU)
        model = CompiledModel.create(context.assets, modelFileName, options, null)
        inBufs = model.createInputBuffers()
        outBufs = model.createOutputBuffers()
        Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
    }

    /** Returns an OUT×OUT class map (0 bg / 1 upper / 2 lower / 3 full) + time (ms). */
    fun segment(bitmap: Bitmap): Pair<ByteArray, Long> {
        val t = System.nanoTime()
        Canvas(resized).drawBitmap(
            bitmap, matrix.apply { setScale(S.toFloat() / bitmap.width, S.toFloat() / bitmap.height) }, paint)
        resized.getPixels(pixels, 0, S, 0, 0, S, S)
        val plane = S * S
        for (i in 0 until plane) {
            val p = pixels[i]
            inputFloats[i] = (((p shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
            inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - 0.5f) / 0.5f
            inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - 0.5f) / 0.5f
        }
        inBufs[0].writeFloat(inputFloats)
        model.run(inBufs, outBufs)
        val o = outBufs[0].readFloat()   // [4*plane] planar: class c at c*plane + idx

        val step = S / OUT
        for (y in 0 until OUT) {
            for (x in 0 until OUT) {
                val idx = (y * step) * S + (x * step)
                var best = 0; var bv = o[idx]
                for (c in 1 until NC) { val v = o[c * plane + idx]; if (v > bv) { bv = v; best = c } }
                classMap[y * OUT + x] = best.toByte()
            }
        }
        return classMap to ((System.nanoTime() - t) / 1_000_000)
    }

    override fun close() {
        model.close()
        if (!resized.isRecycled) resized.recycle()
    }
}
