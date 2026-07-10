package com.rtmanimal

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * RTMPose-m animal pose (mmpose, AP-10K) on the LiteRT CompiledModel GPU.
 *   image[1,3,256,256] (mmpose mean/std) -> simcc_x[1,17,512], simcc_y[1,17,512]
 *
 * Top-down: a square animal crop in, 17 AP-10K keypoints out, decoded by argmax over each 1D x/y SimCC
 * (bins = pixels x split=2). ~5 ms on a Pixel 8a, fully GPU. Output[0] is simcc_x, output[1] is simcc_y.
 */
class RtmAnimalPoseEstimator(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val W = 256
        const val H = 256
        const val K = 17                 // AP-10K animal keypoints
        const val SPLIT = 2              // SimCC bin split factor (bins = pixels * split)
        const val MODEL = "rtm_animal_fp16.tflite"
        // mmpose normalization (RGB, 0-255 scale)
        val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }

    data class Keypoint(val x: Float, val y: Float, val score: Float)

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, MODEL)
        check(f.exists()) { "Model not found: $MODEL. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    /** rgb: W*H*3 row-major [0,255] (a centered animal crop). Returns 17 keypoints in crop pixels. */
    fun estimate(rgb: FloatArray): List<Keypoint> {
        val hw = W * H
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            chw[i] = (rgb[i * 3] - MEAN[0]) / STD[0]
            chw[hw + i] = (rgb[i * 3 + 1] - MEAN[1]) / STD[1]
            chw[2 * hw + i] = (rgb[i * 3 + 2] - MEAN[2]) / STD[2]
        }
        inBuf[0].writeFloat(chw)
        model.run(inBuf, outBuf)
        val sx = outBuf[0].readFloat()   // [K * W * SPLIT]
        val sy = outBuf[1].readFloat()   // [K * H * SPLIT]
        val xBins = W * SPLIT; val yBins = H * SPLIT
        val out = ArrayList<Keypoint>(K)
        for (k in 0 until K) {
            val (xi, xv) = argmax(sx, k * xBins, xBins)
            val (yi, yv) = argmax(sy, k * yBins, yBins)
            out.add(Keypoint(xi / SPLIT.toFloat(), yi / SPLIT.toFloat(), (xv + yv) / 2f))
        }
        return out
    }

    private fun argmax(a: FloatArray, off: Int, n: Int): Pair<Int, Float> {
        var bi = 0; var bv = a[off]
        for (i in 1 until n) { val v = a[off + i]; if (v > bv) { bv = v; bi = i } }
        return bi to bv
    }

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
