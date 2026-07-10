package com.twinlite

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
 * TwinLiteNet drivable-area + lane-line segmentation on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 360, 640]  NCHW, RGB, x/255.
 * Output: two [1, 2, 360, 640] logit maps — drivable_area and lane_line (argmax over 2 classes).
 *
 * ESPNet-based, pure CNN, fully GPU (ConvTranspose2d → ZeroStuffConvT2d). 3.1 MB.
 * Returns two byte masks (1 = foreground) at the model resolution.
 */
class TwinLiteSegmenter(context: Context, modelFileName: String = "twinlite.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "TwinLiteNet"
        const val W = 640; const val H = 360
    }

    private val model: CompiledModel
    private val inBufs: List<TensorBuffer>
    private val outBufs: List<TensorBuffer>
    private var iDa = 0; private var iLl = 1

    private val inputFloats = FloatArray(3 * H * W)
    private val pixels = IntArray(W * H)
    private val resized = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val daMask = ByteArray(W * H)
    private val llMask = ByteArray(W * H)

    init {
        val options = CompiledModel.Options(Accelerator.GPU)
        model = CompiledModel.create(context.assets, modelFileName, options, null)
        inBufs = model.createInputBuffers()
        outBufs = model.createOutputBuffers()
        Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
    }

    /** Returns (drivableMask, laneMask) each W*H bytes (1 = foreground) + time (ms). */
    fun segment(bitmap: Bitmap): Triple<ByteArray, ByteArray, Long> {
        val t = System.nanoTime()
        Canvas(resized).drawBitmap(
            bitmap, matrix.apply { setScale(W.toFloat() / bitmap.width, H.toFloat() / bitmap.height) }, paint)
        resized.getPixels(pixels, 0, W, 0, 0, W, H)
        val plane = W * H
        for (i in 0 until plane) {
            val p = pixels[i]
            inputFloats[i] = ((p shr 16) and 0xFF) / 255f
            inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f
            inputFloats[2 * plane + i] = (p and 0xFF) / 255f
        }
        inBufs[0].writeFloat(inputFloats)
        model.run(inBufs, outBufs)
        val da = outBufs[iDa].readFloat()   // [2*plane]: class0 at [i], class1 at [plane+i]
        val ll = outBufs[iLl].readFloat()
        for (i in 0 until plane) {
            daMask[i] = if (da[plane + i] > da[i]) 1 else 0
            llMask[i] = if (ll[plane + i] > ll[i]) 1 else 0
        }
        return Triple(daMask, llMask, (System.nanoTime() - t) / 1_000_000)
    }

    override fun close() {
        model.close()
        if (!resized.isRecycled) resized.recycle()
    }
}
