package com.rtmhand

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * RTMPose hand pose (21 hand keypoints) on the LiteRT CompiledModel GPU.
 *   image[1,3,256,256] (ImageNet 0-255 normalized) -> simcc_x[1,21,256*2], simcc_y[1,21,256*2]
 *
 * RTMPose-family (CSPNeXt + RTMCC/SimCC head). Same GPU re-authoring as the body model: SafeRMSNorm for the
 * RTMCC ScaleNorm fp16 overflow + GAU act@act BMM -> broadcast-reduce (both baked into the .tflite). SimCC
 * decodes each keypoint by argmax over its 1D x/y bins (bins = pixels * split=2). Top-down: one centered subject.
 */
class RtmHandEstimator(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val W = 256
        const val H = 256
        const val K = 21
        const val SPLIT = 2
        const val MODEL = "rtmhand_fp16.tflite"
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

    /** rgb: W*H*3 row-major [0,255] (the centered crop). Returns K keypoints. */
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
        val a = outBuf[0].readFloat(); val b = outBuf[1].readFloat()
        val xBins = W * SPLIT; val yBins = H * SPLIT
        // outputs are simcc_x and simcc_y; tell them apart by bin count, or by order when equal (square input)
        val sx: FloatArray; val sy: FloatArray
        if (xBins != yBins) { sx = if (a.size == K * xBins) a else b; sy = if (a.size == K * yBins) a else b }
        else { sx = a; sy = b }
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
