package com.sinet

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator

/**
 * SINet-V2 camouflaged / concealed object detection on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 352, 352]  NCHW, RGB, ImageNet-normalized.
 * Output: [1, 1, 352, 352]  sigmoid map — high = camouflaged object.
 *
 * Res2Net-50 backbone + conv decoder — pure CNN, fully GPU (ZeroPadMaxPool + align_corners=False).
 *
 * Reference adopter of the shared components: [CompiledModelRunner] owns the GPU
 * model lifecycle and [ImageTensor] does the Bitmap → NCHW ImageNet-normalized tensor.
 */
class CamouflageDetector(context: Context, modelFileName: String = "sinet.tflite") : AutoCloseable {

    companion object {
        private const val TAG = "SINet"
        const val SIZE = 352
        const val OUT = 256
    }

    private val runner = CompiledModelRunner.fromAssets(context, modelFileName, Accelerator.GPU)
    private val imageTensor = ImageTensor(
        width = SIZE,
        height = SIZE,
        mean = ImageTensor.IMAGENET_MEAN,
        std = ImageTensor.IMAGENET_STD,
        layout = ImageTensor.Layout.NCHW,
        channelOrder = ImageTensor.ChannelOrder.RGB,
        scaleTo01 = true,
    )
    private val heat = FloatArray(OUT * OUT)

    init {
        Log.i(TAG, "GPU compiled OK — ${runner.inputBuffers.size} in / ${runner.outputBuffers.size} out")
    }

    /** Returns an OUT×OUT camouflage map (0..1) + time (ms). */
    fun detect(bitmap: Bitmap): Pair<FloatArray, Long> {
        val t = System.nanoTime()
        runner.writeInput(0, imageTensor.load(bitmap))
        runner.run()
        val map = runner.readOutput(0)   // [352*352] sigmoid
        // Downsample the full SIZE×SIZE map to OUT×OUT. Compute the source index per
        // element (multiply then divide) so it spans the whole map — a precomputed
        // step = SIZE / OUT would truncate to 1 and crop the top-left OUT×OUT corner.
        for (y in 0 until OUT) {
            val srcRow = (y * SIZE / OUT) * SIZE
            for (x in 0 until OUT) {
                heat[y * OUT + x] = map[srcRow + x * SIZE / OUT]
            }
        }
        return heat to ((System.nanoTime() - t) / 1_000_000)
    }

    override fun close() {
        runner.close()
        imageTensor.release()
    }
}
