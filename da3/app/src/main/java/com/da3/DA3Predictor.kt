package com.da3

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.abs

/**
 * Depth Anything 3 (DA3-SMALL mono) depth predictor: single CompiledModel GPU inference.
 * The model is fixed to the native portrait aspect (896x504). Arbitrary images are letterboxed
 * (aspect preserved, padded) into 896x504, and the depth is cropped back to the content region.
 * Input [1,3,896,504] NCHW, ImageNet-normalized; output [1,1,896,504] depth.
 */
class DA3Predictor(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "DA3"
        private const val MODEL_FILE = "da3_small_gpu_fp16.tflite"
        const val MODEL_H = 896
        const val MODEL_W = 504
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>
    private val inputFloats = FloatArray(3 * MODEL_H * MODEL_W)
    private val pixels = IntArray(MODEL_H * MODEL_W)
    private val canvasBmp = Bitmap.createBitmap(MODEL_W, MODEL_H, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    var acceleratorName = ""; private set

    init {
        compiledModel = try {
            val m = CompiledModel.create(context.assets, MODEL_FILE, CompiledModel.Options(Accelerator.GPU), null)
            acceleratorName = "GPU"; Log.i(TAG, "Model GPU ready"); m
        } catch (e: Exception) {
            Log.w(TAG, "GPU compile failed: ${e.message}, falling back to CPU")
            val m = CompiledModel.create(context.assets, MODEL_FILE, CompiledModel.Options(Accelerator.CPU), null)
            acceleratorName = "CPU"; Log.i(TAG, "Model CPU ready"); m
        }
        inputBuffers = compiledModel.createInputBuffers()
    }

    fun predict(src: Bitmap): DA3Result {
        val t = System.nanoTime()

        // letterbox into MODEL_W x MODEL_H, preserving aspect (no distortion)
        val canvas = Canvas(canvasBmp)
        canvas.drawColor(Color.BLACK)
        val srcAspect = src.width.toFloat() / src.height
        val dstAspect = MODEL_W.toFloat() / MODEL_H
        val dst = if (srcAspect > dstAspect) {            // wider than model → fit width
            val fitH = MODEL_W / srcAspect; val y = (MODEL_H - fitH) * 0.5f
            RectF(0f, y, MODEL_W.toFloat(), y + fitH)
        } else {                                          // taller → fit height
            val fitW = MODEL_H * srcAspect; val x = (MODEL_W - fitW) * 0.5f
            RectF(x, 0f, x + fitW, MODEL_H.toFloat())
        }
        canvas.drawBitmap(src, null, dst, paint)
        canvasBmp.getPixels(pixels, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)

        val plane = MODEL_H * MODEL_W
        for (i in pixels.indices) {
            val p = pixels[i]
            inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
            inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
            inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        inputBuffers[0].writeFloat(inputFloats)

        val out = compiledModel.run(inputBuffers)
        val depth = out[0].readFloat()

        val ms = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Inference: ${ms}ms ($acceleratorName)")
        val cl = dst.left.toInt().coerceIn(0, MODEL_W);  val ct = dst.top.toInt().coerceIn(0, MODEL_H)
        val cr = dst.right.toInt().coerceIn(0, MODEL_W); val cb = dst.bottom.toInt().coerceIn(0, MODEL_H)
        return DA3Result(depth, MODEL_W, MODEL_H, cl, ct, cr, cb, ms, acceleratorName)
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        canvasBmp.recycle()
    }
}

data class DA3Result(
    val depth: FloatArray, val width: Int, val height: Int,
    val cropL: Int, val cropT: Int, val cropR: Int, val cropB: Int,
    val inferenceMs: Long, val accelerator: String
) {
    /** Turbo-colormap depth, cropped to the content region (original aspect), normalized over content only. */
    fun depthBitmap(): Bitmap {
        val cw = (cropR - cropL).coerceAtLeast(1)
        val ch = (cropB - cropT).coerceAtLeast(1)
        var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE
        for (y in cropT until cropB) for (x in cropL until cropR) {
            val d = depth[y * width + x]; if (d < mn) mn = d; if (d > mx) mx = d
        }
        val range = if (mx > mn) mx - mn else 1f
        val px = IntArray(cw * ch)
        for (y in 0 until ch) for (x in 0 until cw) {
            val tt = ((depth[(y + cropT) * width + (x + cropL)] - mn) / range).coerceIn(0f, 1f)
            val r = (255 * (1.5f - abs(4 * tt - 3)).coerceIn(0f, 1f)).toInt()
            val g = (255 * (1.5f - abs(4 * tt - 2)).coerceIn(0f, 1f)).toInt()
            val b = (255 * (1.5f - abs(4 * tt - 1)).coerceIn(0f, 1f)).toInt()
            px[y * cw + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888).also {
            it.setPixels(px, 0, cw, 0, 0, cw, ch)
        }
    }
}
