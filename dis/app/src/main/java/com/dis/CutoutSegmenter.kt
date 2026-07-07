package com.dis

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
 * DIS (IS-Net, general-use) high-precision object cutout on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 1024, 1024]  NCHW, RGB, x/255 - 0.5.
 * Output: [1, 1, 1024, 1024]  sigmoid alpha (0..1).
 *
 * IS-Net (RSU blocks) — pure CNN, fully GPU (defensive align_corners=False). ~11 ms/frame.
 */
class CutoutSegmenter(context: Context, modelFileName: String = "dis.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "DIS"
        const val SIZE = 1024
        const val OUT = 256   // downscaled alpha for fast UI compositing
    }

    private val model: CompiledModel
    private val inBufs: List<TensorBuffer>
    private val outBufs: List<TensorBuffer>

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
        Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
    }

    /** Returns an [OUT]×[OUT] alpha matte (0..1) + time (ms). */
    fun matte(bitmap: Bitmap): Pair<FloatArray, Long> {
        val t = System.nanoTime()
        Canvas(resized).drawBitmap(
            bitmap, matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) }, paint)
        resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            inputFloats[i] = ((p shr 16) and 0xFF) / 255f - 0.5f
            inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f - 0.5f
            inputFloats[2 * plane + i] = (p and 0xFF) / 255f - 0.5f
        }
        inBufs[0].writeFloat(inputFloats)
        model.run(inBufs, outBufs)
        val full = outBufs[0].readFloat()   // [1024*1024] alpha 0..1

        val out = FloatArray(OUT * OUT)
        val step = SIZE / OUT
        for (y in 0 until OUT) {
            val sy = y * step
            for (x in 0 until OUT) out[y * OUT + x] = full[sy * SIZE + x * step]
        }
        return out to ((System.nanoTime() - t) / 1_000_000)
    }

    override fun close() {
        model.close()
        if (!resized.isRecycled) resized.recycle()
    }
}
