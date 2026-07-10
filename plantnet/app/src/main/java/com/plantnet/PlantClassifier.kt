package com.plantnet

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
 * PlantNet-300K fine-grained plant species classifier on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 224, 224]  NCHW, RGB, ImageNet-normalized.
 * Output: [1, 1081]         species logits (PlantNet-300K, Latin names).
 *
 * A torchvision ResNet18 — pure CNN. One re-authoring patch (baked into the graph,
 * see scripts/): the ResNet stem MaxPool's -inf-pad PADV2 is replaced with a 0-pad +
 * unpadded maxpool (exact post-ReLU), which the Mali delegate accepts.
 */
class PlantClassifier(context: Context, modelFileName: String = "plantnet.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "PlantNet"
        const val SIZE = 224
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

    init {
        val options = CompiledModel.Options(Accelerator.GPU)
        model = CompiledModel.create(context.assets, modelFileName, options, null)
        inBufs = model.createInputBuffers()
        outBufs = model.createOutputBuffers()
        Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
    }

    /** Classify. Returns top-[topK] (species name, probability) + time (ms). */
    fun classify(bitmap: Bitmap, topK: Int = 5): Pair<List<Pair<String, Float>>, Long> {
        val t = System.nanoTime()
        // center-crop to square, resize to 224
        val side = minOf(bitmap.width, bitmap.height)
        val sx = (bitmap.width - side) / 2f
        val sy = (bitmap.height - side) / 2f
        val canvas = Canvas(resized)
        matrix.reset()
        matrix.postTranslate(-sx, -sy)
        matrix.postScale(SIZE.toFloat() / side, SIZE.toFloat() / side)
        canvas.drawBitmap(bitmap, matrix, paint)
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
        val logits = outBufs[0].readFloat()   // [1081]

        val idx = logits.indices.sortedByDescending { logits[it] }.take(topK)
        val mx = logits[idx.first()]
        var sum = 0.0
        for (v in logits) sum += Math.exp((v - mx).toDouble())
        val preds = idx.map { i ->
            PlantNetLabels.NAMES[i] to (Math.exp((logits[i] - mx).toDouble()) / sum).toFloat()
        }
        return preds to ((System.nanoTime() - t) / 1_000_000)
    }

    override fun close() {
        model.close()
        if (!resized.isRecycled) resized.recycle()
    }
}
