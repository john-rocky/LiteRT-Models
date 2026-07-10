package com.gfpgan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * GFPGAN v1.4 blind face restoration on LiteRT CompiledModel (GPU).
 *
 * The StyleGAN2 ModulatedConv2d was re-authored to a 4D form at conversion time (no >4D tensors,
 * constant conv filters), so the whole graph runs fully on the GPU delegate.
 *
 * Tensor layout is NCHW (litert-torch preserves channels-first): input/output [1, 3, 512, 512],
 * normalized to [-1, 1]. The 431 MB fp16 model is staged in filesDir (too large to bundle) —
 * run scripts/install_to_device.sh once before first launch.
 */
class FaceRestorer(context: Context, modelFileName: String = "gfpgan_fp16.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "GFPGAN"
        const val SIZE = 512
        private const val HW = SIZE * SIZE
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>

    private val inputFloats = FloatArray(3 * HW)
    private val pixels = IntArray(HW)
    private val outPixels = IntArray(HW)
    private val square = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val f = File(context.filesDir, modelFileName)
        if (!f.exists()) {
            throw IllegalStateException(
                "Model not found: ${f.absolutePath}. Push it first with scripts/install_to_device.sh."
            )
        }
        Log.i(TAG, "Loading model: ${f.absolutePath}")
        compiledModel = CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
        inputBuffers = compiledModel.createInputBuffers()
        Log.i(TAG, "GPU compiled OK — ${SIZE}x${SIZE} face restoration")
    }

    /** Resize any bitmap to 512x512 (the aligned-face input the model expects). */
    fun toFaceInput(input: Bitmap): Bitmap {
        val c = Canvas(square)
        c.drawBitmap(
            input, null,
            android.graphics.Rect(0, 0, SIZE, SIZE), paint
        )
        return square.copy(Bitmap.Config.ARGB_8888, false)
    }

    /** Restore a 512x512 face bitmap. Returns a fresh 512x512 restored bitmap. */
    fun restore(face512: Bitmap): Bitmap {
        face512.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        // NCHW planes (R plane, G plane, B plane), normalized to [-1, 1]
        for (i in 0 until HW) {
            val p = pixels[i]
            inputFloats[i] = Color.red(p) / 127.5f - 1f
            inputFloats[HW + i] = Color.green(p) / 127.5f - 1f
            inputFloats[2 * HW + i] = Color.blue(p) / 127.5f - 1f
        }
        inputBuffers[0].writeFloat(inputFloats)

        val out = compiledModel.run(inputBuffers)[0].readFloat()   // NCHW, [-1, 1]

        for (i in 0 until HW) {
            val r = ((out[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((out[HW + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((out[2 * HW + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val result = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        result.setPixels(outPixels, 0, SIZE, 0, 0, SIZE, SIZE)
        return result
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        compiledModel.close()
        square.recycle()
    }
}
