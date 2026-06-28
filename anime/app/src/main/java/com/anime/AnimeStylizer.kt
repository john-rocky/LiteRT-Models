package com.anime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * AnimeGANv2 (bryandlee/animegan2-pytorch) photo→anime stylization on the LiteRT CompiledModel GPU.
 *   image[1,3,256,256] (RGB scaled to [-1,1]) -> stylized[1,3,256,256] ([-1,1])
 *
 * Two styles (paprika = general anime, face_paint_512_v2 = anime face portrait), each a ~4 MB fp16 graph,
 * lazily loaded & cached. ~10 ms / 256x256 on a Pixel 8a, fully GPU. Numerically-exact GPU re-authorings
 * (baked in): ReflectionPad→zero-pad, GroupNorm(1)→SafeGroupNorm, bilinear align_corners=False, and conv-weight
 * scaling via GroupNorm scale-invariance (keeps the large conv activations fp16-precise on Mali).
 */
class AnimeStylizer(private val ctx: Context) : Closeable {

    companion object {
        const val SIZE = 256
        val STYLES = listOf("paprika", "face_paint_512_v2")
    }

    private class Net(val model: CompiledModel) {
        val inBuf = model.createInputBuffers()
        val outBuf = model.createOutputBuffers()
    }

    private val nets = HashMap<String, Net>()

    private fun net(style: String): Net = nets.getOrPut(style) {
        val f = File(ctx.filesDir, "anime_${style}_fp16.tflite")
        check(f.exists()) { "Model not found: ${f.name}. Push first: scripts/install_to_device.sh" }
        Net(CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null))
    }

    /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns a stylized SIZE x SIZE bitmap. */
    fun stylize(rgb: FloatArray, style: String): Bitmap {
        val n = net(style)
        val hw = SIZE * SIZE
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {                       // RGB [0,255] -> [-1,1], NCHW planar
            chw[i] = rgb[i * 3] / 127.5f - 1f
            chw[hw + i] = rgb[i * 3 + 1] / 127.5f - 1f
            chw[2 * hw + i] = rgb[i * 3 + 2] / 127.5f - 1f
        }
        n.inBuf[0].writeFloat(chw)
        n.model.run(n.inBuf, n.outBuf)
        val out = n.outBuf[0].readFloat()             // [3*SIZE*SIZE] NCHW, [-1,1]
        val px = IntArray(hw)
        for (i in 0 until hw) {
            val r = (((out[i] + 1f) * 127.5f)).coerceIn(0f, 255f).toInt()
            val g = (((out[hw + i] + 1f) * 127.5f)).coerceIn(0f, 255f).toInt()
            val b = (((out[2 * hw + i] + 1f) * 127.5f)).coerceIn(0f, 255f).toInt()
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        nets.values.forEach { it.inBuf.forEach { b -> b.close() }; it.outBuf.forEach { b -> b.close() }; it.model.close() }
        nets.clear()
    }
}
