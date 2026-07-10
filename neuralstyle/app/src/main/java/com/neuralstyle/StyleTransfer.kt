package com.neuralstyle

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * Fast Neural Style Transfer (PyTorch examples TransformerNet) on the LiteRT CompiledModel GPU.
 *   image[1,3,256,256] (RGB, 0-255) -> stylized[1,3,256,256] (RGB, 0-255)
 *
 * Four styles (candy / mosaic / rain_princess / udnie), each a 3.5 MB fp16 graph, lazily loaded & cached.
 * ~9 ms / 256x256 on a Pixel 8a, fully GPU. Two numerically-exact GPU re-authorings (baked into the .tflite):
 *  - ReflectionPad2d -> zero-pad (GATHER_ND -> PAD; border-only difference).
 *  - the large conv activations (max ~5000) lose fp16 precision on Mali, so conv weights are scaled down
 *    (InstanceNorm is scale-invariant -> exact) + SafeInstanceNorm reduces in a down-scaled domain.
 */
class StyleTransfer(private val ctx: Context) : Closeable {

    companion object {
        const val SIZE = 256
        val STYLES = listOf("candy", "mosaic", "rain_princess", "udnie")
    }

    private class Net(val model: CompiledModel) {
        val inBuf = model.createInputBuffers()
        val outBuf = model.createOutputBuffers()
    }

    private val nets = HashMap<String, Net>()

    private fun net(style: String): Net = nets.getOrPut(style) {
        val f = File(ctx.filesDir, "style_${style}_fp16.tflite")
        check(f.exists()) { "Model not found: ${f.name}. Push first: scripts/install_to_device.sh" }
        Net(CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null))
    }

    /** rgb: SIZE*SIZE*3 row-major [0,255]. Returns a stylized SIZE x SIZE bitmap. */
    fun stylize(rgb: FloatArray, style: String): Bitmap {
        val n = net(style)
        val hw = SIZE * SIZE
        val chw = FloatArray(3 * hw)
        for (i in 0 until hw) {
            chw[i] = rgb[i * 3]; chw[hw + i] = rgb[i * 3 + 1]; chw[2 * hw + i] = rgb[i * 3 + 2]
        }
        n.inBuf[0].writeFloat(chw)
        n.model.run(n.inBuf, n.outBuf)
        val out = n.outBuf[0].readFloat()            // [3*SIZE*SIZE] NCHW, 0-255
        val px = IntArray(hw)
        for (i in 0 until hw) {
            val r = out[i].coerceIn(0f, 255f).toInt()
            val g = out[hw + i].coerceIn(0f, 255f).toInt()
            val b = out[2 * hw + i].coerceIn(0f, 255f).toInt()
            px[i] = Color.rgb(r, g, b)
        }
        return Bitmap.createBitmap(px, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        nets.values.forEach { it.inBuf.forEach { b -> b.close() }; it.outBuf.forEach { b -> b.close() }; it.model.close() }
        nets.clear()
    }
}
