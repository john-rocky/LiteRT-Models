package com.sinet

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
 * SINet-V2 camouflaged / concealed object detection on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 352, 352]  NCHW, RGB, ImageNet-normalized.
 * Output: [1, 1, 352, 352]  sigmoid map — high = camouflaged object.
 *
 * Res2Net-50 backbone + conv decoder — pure CNN, fully GPU (ZeroPadMaxPool + align_corners=False).
 */
class CamouflageDetector(context: Context, modelFileName: String = "sinet.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "SINet"
        const val SIZE = 352
        const val OUT = 256
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val model: CompiledModel
    private val inBufs: List<TensorBuffer>
    private val outBufs: List<TensorBuffer>
    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val heat = FloatArray(OUT * OUT)

    init {
        model = CompiledModel.create(context.assets, modelFileName, CompiledModel.Options(Accelerator.GPU), null)
        inBufs = model.createInputBuffers(); outBufs = model.createOutputBuffers()
        Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
    }

    /** Returns an OUT×OUT camouflage map (0..1) + time (ms). */
    fun detect(bitmap: Bitmap): Pair<FloatArray, Long> {
        val t = System.nanoTime()
        Canvas(resized).drawBitmap(bitmap, matrix.apply {
            setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) }, paint)
        resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        inBufs[0].writeFloat(inputFloats)
        model.run(inBufs, outBufs)
        val m = outBufs[0].readFloat()   // [352*352] sigmoid
        val step = SIZE / OUT
        for (y in 0 until OUT) for (x in 0 until OUT) heat[y * OUT + x] = m[(y * step) * SIZE + x * step]
        return heat to ((System.nanoTime() - t) / 1_000_000)
    }

    override fun close() { model.close(); if (!resized.isRecycled) resized.recycle() }
}
