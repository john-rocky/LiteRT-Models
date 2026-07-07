package com.crowdcount

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
 * DM-Count crowd counting on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 512, 512]  NCHW, RGB, ImageNet-normalized.
 * Output: [1, 1, 64, 64]    density map — sum of the map = estimated person count.
 *
 * VGG19 backbone + conv regression head (NeurIPS 2020) — pure CNN, fully GPU.
 * The mid-graph 2x bilinear upsample (align_corners=True, banned on the GPU delegate)
 * is re-authored exactly as two constant-matrix multiplies at conversion time.
 */
class CrowdCounter(context: Context, modelFileName: String = "dmcount.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "DMCount"
        const val SIZE = 512
        const val OUT = 64
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    /** One inference result: [density] is an OUT×OUT map whose sum is [count]. */
    data class Result(val density: FloatArray, val count: Float, val inferenceMs: Long)

    private val model: CompiledModel
    private val inBufs: List<TensorBuffer>
    private val outBufs: List<TensorBuffer>
    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        model = CompiledModel.create(context.assets, modelFileName, CompiledModel.Options(Accelerator.GPU), null)
        inBufs = model.createInputBuffers(); outBufs = model.createOutputBuffers()
        Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
    }

    /** Runs DM-Count on [bitmap] and returns the density map + person count + time. */
    fun count(bitmap: Bitmap): Result {
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
        val density = outBufs[0].readFloat()   // [64*64] non-negative density
        var sum = 0f
        for (v in density) sum += v
        return Result(density, sum, (System.nanoTime() - t) / 1_000_000)
    }

    override fun close() { model.close(); if (!resized.isRecycled) resized.recycle() }
}
