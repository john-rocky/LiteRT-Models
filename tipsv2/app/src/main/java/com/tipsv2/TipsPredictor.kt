package com.tipsv2

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
import java.io.File

/**
 * TIPSv2-B/14 + DPT heads: one CompiledModel GPU graph that returns metric depth, surface normals
 * and ADE20K semantic logits for a 448x448 image.
 *
 * Input  [1,3,448,448] NCHW, RGB in [0,1] (no ImageNet normalization — TIPSv2 convention).
 * Outputs [1,1,448,448] depth (metres, NYU scale) · [1,3,448,448] unit normals ·
 *         [1,150,256,256] segmentation logits at the DPT head's native resolution (argmax here).
 * Arbitrary images are letterboxed into the square (aspect preserved) and the results cropped back.
 */
class TipsPredictor(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "TIPSv2"
        private const val MODEL_FILE = "tipsv2_b14_dpt_fp16.tflite"
        const val SIZE = 448
        const val SEG_RES = 256
        const val NUM_CLASSES = 150
    }

    private val compiledModel: CompiledModel
    private val inputBuffers: List<TensorBuffer>
    private val outputBuffers: List<TensorBuffer>
    private val inputFloats = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)
    private val canvasBmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    var acceleratorName = ""; private set

    init {
        // 318 MB fp16 — too large for APK assets; staged into filesDir by scripts/install_to_device.sh
        val modelFile = File(context.filesDir, MODEL_FILE)
        if (!modelFile.exists()) {
            throw IllegalStateException(
                "Model not found. Run scripts/install_to_device.sh (adb push + run-as cp into files/)"
            )
        }
        val path = modelFile.absolutePath
        compiledModel = try {
            val m = CompiledModel.create(path, CompiledModel.Options(Accelerator.GPU), null)
            acceleratorName = "GPU"; Log.i(TAG, "Model GPU ready"); m
        } catch (e: Exception) {
            Log.w(TAG, "GPU compile failed: ${e.message}, falling back to CPU")
            val m = CompiledModel.create(path, CompiledModel.Options(Accelerator.CPU), null)
            acceleratorName = "CPU"; Log.i(TAG, "Model CPU ready"); m
        }
        inputBuffers = compiledModel.createInputBuffers()
        outputBuffers = compiledModel.createOutputBuffers()
    }

    fun predict(src: Bitmap): TipsResult {
        val t = System.nanoTime()

        // letterbox into the square, preserving aspect
        val canvas = Canvas(canvasBmp)
        canvas.drawColor(Color.BLACK)
        val dst = if (src.width >= src.height) {
            val h = SIZE * src.height.toFloat() / src.width; val y = (SIZE - h) * 0.5f
            RectF(0f, y, SIZE.toFloat(), y + h)
        } else {
            val w = SIZE * src.width.toFloat() / src.height; val x = (SIZE - w) * 0.5f
            RectF(x, 0f, x + w, SIZE.toFloat())
        }
        canvas.drawBitmap(src, null, dst, paint)
        canvasBmp.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)

        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val p = pixels[i]
            inputFloats[i] = ((p shr 16) and 0xFF) / 255f
            inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f
            inputFloats[2 * plane + i] = (p and 0xFF) / 255f
        }
        inputBuffers[0].writeFloat(inputFloats)

        compiledModel.run(inputBuffers, outputBuffers)
        val depth = outputBuffers[0].readFloat()      // [448*448]
        val normals = outputBuffers[1].readFloat()    // [3*448*448]
        val seg = outputBuffers[2].readFloat()        // [150*256*256]

        val ms = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Inference: ${ms}ms ($acceleratorName)")
        val crop = intArrayOf(
            dst.left.toInt().coerceIn(0, SIZE), dst.top.toInt().coerceIn(0, SIZE),
            dst.right.toInt().coerceIn(0, SIZE), dst.bottom.toInt().coerceIn(0, SIZE)
        )
        return TipsResult(depth, normals, seg, crop, ms, acceleratorName)
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        compiledModel.close()
        canvasBmp.recycle()
    }
}

class TipsResult(
    val depth: FloatArray, val normals: FloatArray, val segLogits: FloatArray,
    /** content rect inside the 448 square: left, top, right, bottom */
    private val crop: IntArray,
    val inferenceMs: Long, val accelerator: String
) {
    private val s = TipsPredictor.SIZE
    private val cw get() = (crop[2] - crop[0]).coerceAtLeast(1)
    private val ch get() = (crop[3] - crop[1]).coerceAtLeast(1)

    val depthMinMetres: Float get() = cropped(depth).minOrNull() ?: 0f
    val depthMaxMetres: Float get() = cropped(depth).maxOrNull() ?: 0f

    private fun cropped(plane: FloatArray): FloatArray {
        val out = FloatArray(cw * ch); var k = 0
        for (y in crop[1] until crop[3]) for (x in crop[0] until crop[2]) out[k++] = plane[y * s + x]
        return out
    }

    /** Metric depth -> inverse depth, 2nd/98th-percentile normalized, Spectral colormap (near = red). */
    fun depthBitmap(): Bitmap {
        val d = cropped(depth)
        val disp = FloatArray(d.size) { if (d[it] > 0f) 1f / d[it] else 0f }
        val sorted = disp.copyOf(); sorted.sort()
        val lo = sorted[(0.02f * (sorted.size - 1)).toInt()]
        val hi = sorted[(0.98f * (sorted.size - 1)).toInt()]
        val range = if (hi > lo) hi - lo else 1e-6f
        val px = IntArray(disp.size) { i ->
            val n = 1f - ((disp[i] - lo) / range).coerceIn(0f, 1f)
            SPECTRAL[(n * 255f).toInt().coerceIn(0, 255)]
        }
        return Bitmap.createBitmap(px, cw, ch, Bitmap.Config.ARGB_8888)
    }

    /** Unit normals -> RGB = (n + 1) / 2 per channel (x→R, y→G, z→B). */
    fun normalsBitmap(): Bitmap {
        val plane = s * s
        val px = IntArray(cw * ch); var k = 0
        for (y in crop[1] until crop[3]) for (x in crop[0] until crop[2]) {
            val i = y * s + x
            val r = ((normals[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((normals[plane + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((normals[2 * plane + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            px[k++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(px, cw, ch, Bitmap.Config.ARGB_8888)
    }

    /** Per-pixel argmax over the 150 ADE20K logits (256x256 head grid) -> palette, nearest-upscaled
     *  to the 448 square and cropped. Also returns the class histogram for the legend. */
    fun segmentation(): Pair<Bitmap, List<Pair<Int, Int>>> {
        val r = TipsPredictor.SEG_RES
        val plane = r * r
        val label = IntArray(plane)
        for (p in 0 until plane) {
            var best = 0; var bv = segLogits[p]
            for (c in 1 until TipsPredictor.NUM_CLASSES) {
                val v = segLogits[c * plane + p]
                if (v > bv) { bv = v; best = c }
            }
            label[p] = best
        }
        val hist = IntArray(TipsPredictor.NUM_CLASSES)
        val px = IntArray(cw * ch); var k = 0
        for (y in crop[1] until crop[3]) for (x in crop[0] until crop[2]) {
            val cls = label[(y * r / s) * r + (x * r / s)]
            hist[cls]++
            px[k++] = ADE_PALETTE[cls]
        }
        val top = hist.withIndex().filter { it.value > 0 }.sortedByDescending { it.value }
            .map { it.index to it.value }
        return Bitmap.createBitmap(px, cw, ch, Bitmap.Config.ARGB_8888) to top
    }

    companion object {
        // matplotlib "Spectral" colormap, 256 RGB entries
        private const val LUT =
            "9e0142a00343a20643a40844a70b44a90d45ab0f45ad1246af1446b11747b41947b61b48b81e48ba2049bc2249be254a" +
            "c1274ac32a4bc52c4bc72e4cc9314ccb334dcd364dd0384ed23a4ed43d4fd63f4fd7414ed8434ed9444dda464ddc484c" +
            "dd4a4cde4c4bdf4e4be1504be2514ae3534ae45549e55749e75948e85b48e95c47ea5e47eb6046ed6246ee6445ef6645" +
            "f06744f26944f36b43f46d43f47044f57245f57547f57748f67a49f67c4af67f4bf7814cf7844ef8864ff88950f88c51" +
            "f98e52f99153f99355fa9656fa9857fa9b58fb9d59fba05bfba35cfca55dfca85efcaa5ffdad60fdaf62fdb163fdb365" +
            "fdb567fdb768fdb96afdbb6cfdbd6dfdbf6ffdc171fdc372fdc574fdc776fec877feca79fecc7bfece7cfed07efed27f" +
            "fed481fed683fed884feda86fedc88fede89fee08bfee18dfee28ffee491fee593fee695fee797fee999feea9bfeeb9d" +
            "feec9ffeeda1feefa3fff0a6fff1a8fff2aafff3acfff5aefff6b0fff7b2fff8b4fffab6fffbb8fffcbafffdbcfffebe" +
            "ffffbefefebdfdfebbfcfebafbfdb8fafdb7f9fcb5f8fcb4f7fcb2f6fbb0f5fbaff4faadf3faacf2faaaf1f9a9f0f9a7" +
            "eff9a6eef8a4edf8a3ecf7a1ebf7a0eaf79ee9f69de8f69be7f59ae6f598e4f498e1f399dff299ddf19adaf09ad8ef9b" +
            "d6ee9bd3ed9cd1ed9ccfec9dcdeb9dcaea9ec8e99ec6e89fc3e79fc1e6a0bfe5a0bce4a0bae3a1b8e2a1b5e1a2b3e0a2" +
            "b1dfa3aedea3acdda4aadca4a7dba4a4daa4a2d9a49fd8a49cd7a499d6a497d5a494d4a491d3a48fd2a48cd1a489d0a4" +
            "86cfa584cea581cda57ecca57ccaa579c9a576c8a574c7a571c6a56ec5a56bc4a569c3a566c2a564c0a662bda760bba8" +
            "5eb9a95cb7aa5ab4ab58b2ac56b0ad54aead52abae50a9af4ea7b04ba4b149a2b247a0b3459eb4439bb54199b63f97b7" +
            "3d95b83b92b93990ba378ebb358bbc3389bd3387bc3585bb3682ba3880b93a7eb83b7cb73d79b63f77b54175b44273b3" +
            "4471b2466eb1486cb0496aaf4b68ae4d65ad4e63ac5061aa525fa9545ca8555aa75758a65956a55b53a45c51a35e4fa2"
        private val SPECTRAL = IntArray(256) { i ->
            val r = LUT.substring(i * 6, i * 6 + 2).toInt(16)
            val g = LUT.substring(i * 6 + 2, i * 6 + 4).toInt(16)
            val b = LUT.substring(i * 6 + 4, i * 6 + 6).toInt(16)
            (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        /** ADE20K 150-class palette (mmsegmentation order = the model's id2label order). */
        val ADE_PALETTE = intArrayOf(
            0xFF787878.toInt(), 0xFFB47878.toInt(), 0xFF06E6E6.toInt(), 0xFF503232.toInt(), 0xFF04C803.toInt(), 0xFF787850.toInt(), 0xFF8C8C8C.toInt(), 0xFFCC05FF.toInt(), 0xFFE6E6E6.toInt(), 0xFF04FA07.toInt(),
            0xFFE005FF.toInt(), 0xFFEBFF07.toInt(), 0xFF96053D.toInt(), 0xFF787846.toInt(), 0xFF08FF33.toInt(), 0xFFFF0652.toInt(), 0xFF8FFF8C.toInt(), 0xFFCCFF04.toInt(), 0xFFFF3307.toInt(), 0xFFCC4603.toInt(),
            0xFF0066C8.toInt(), 0xFF3DE6FA.toInt(), 0xFFFF0633.toInt(), 0xFF0B66FF.toInt(), 0xFFFF0747.toInt(), 0xFFFF09E0.toInt(), 0xFF0907E6.toInt(), 0xFFDCDCDC.toInt(), 0xFFFF095C.toInt(), 0xFF7009FF.toInt(),
            0xFF08FFD6.toInt(), 0xFF07FFE0.toInt(), 0xFFFFB806.toInt(), 0xFF0AFF47.toInt(), 0xFFFF290A.toInt(), 0xFF07FFFF.toInt(), 0xFFE0FF08.toInt(), 0xFF6608FF.toInt(), 0xFFFF3D06.toInt(), 0xFFFFC207.toInt(),
            0xFFFF7A08.toInt(), 0xFF00FF14.toInt(), 0xFFFF0829.toInt(), 0xFFFF0599.toInt(), 0xFF0633FF.toInt(), 0xFFEB0CFF.toInt(), 0xFFA09614.toInt(), 0xFF00A3FF.toInt(), 0xFF8C8C8C.toInt(), 0xFFFA0A0F.toInt(),
            0xFF14FF00.toInt(), 0xFF1FFF00.toInt(), 0xFFFF1F00.toInt(), 0xFFFFE000.toInt(), 0xFF99FF00.toInt(), 0xFF0000FF.toInt(), 0xFFFF4700.toInt(), 0xFF00EBFF.toInt(), 0xFF00ADFF.toInt(), 0xFF1F00FF.toInt(),
            0xFF0BC8C8.toInt(), 0xFFFF5200.toInt(), 0xFF00FFF5.toInt(), 0xFF003DFF.toInt(), 0xFF00FF70.toInt(), 0xFF00FF85.toInt(), 0xFFFF0000.toInt(), 0xFFFFA300.toInt(), 0xFFFF6600.toInt(), 0xFFC2FF00.toInt(),
            0xFF008FFF.toInt(), 0xFF33FF00.toInt(), 0xFF0052FF.toInt(), 0xFF00FF29.toInt(), 0xFF00FFAD.toInt(), 0xFF0A00FF.toInt(), 0xFFADFF00.toInt(), 0xFF00FF99.toInt(), 0xFFFF5C00.toInt(), 0xFFFF00FF.toInt(),
            0xFFFF00F5.toInt(), 0xFFFF0066.toInt(), 0xFFFFAD00.toInt(), 0xFFFF0014.toInt(), 0xFFFFB8B8.toInt(), 0xFF001FFF.toInt(), 0xFF00FF3D.toInt(), 0xFF0047FF.toInt(), 0xFFFF00CC.toInt(), 0xFF00FFC2.toInt(),
            0xFF00FF52.toInt(), 0xFF000AFF.toInt(), 0xFF0070FF.toInt(), 0xFF3300FF.toInt(), 0xFF00C2FF.toInt(), 0xFF007AFF.toInt(), 0xFF00FFA3.toInt(), 0xFFFF9900.toInt(), 0xFF00FF0A.toInt(), 0xFFFF7000.toInt(),
            0xFF8FFF00.toInt(), 0xFF5200FF.toInt(), 0xFFA3FF00.toInt(), 0xFFFFEB00.toInt(), 0xFF08B8AA.toInt(), 0xFF8500FF.toInt(), 0xFF00FF5C.toInt(), 0xFFB800FF.toInt(), 0xFFFF001F.toInt(), 0xFF00B8FF.toInt(),
            0xFF00D6FF.toInt(), 0xFFFF0070.toInt(), 0xFF5CFF00.toInt(), 0xFF00E0FF.toInt(), 0xFF70E0FF.toInt(), 0xFF46B8A0.toInt(), 0xFFA300FF.toInt(), 0xFF9900FF.toInt(), 0xFF47FF00.toInt(), 0xFFFF00A3.toInt(),
            0xFFFFCC00.toInt(), 0xFFFF008F.toInt(), 0xFF00FFEB.toInt(), 0xFF85FF00.toInt(), 0xFFFF00EB.toInt(), 0xFFF500FF.toInt(), 0xFFFF007A.toInt(), 0xFFFFF500.toInt(), 0xFF0ABED4.toInt(), 0xFFD6FF00.toInt(),
            0xFF00CCFF.toInt(), 0xFF1400FF.toInt(), 0xFFFFFF00.toInt(), 0xFF0099FF.toInt(), 0xFF0029FF.toInt(), 0xFF00FFCC.toInt(), 0xFF2900FF.toInt(), 0xFF29FF00.toInt(), 0xFFAD00FF.toInt(), 0xFF00F5FF.toInt(),
            0xFF4700FF.toInt(), 0xFF7A00FF.toInt(), 0xFF00FFB8.toInt(), 0xFF005CFF.toInt(), 0xFFB8FF00.toInt(), 0xFF0085FF.toInt(), 0xFFFFD600.toInt(), 0xFF19C2C2.toInt(), 0xFF66FF00.toInt(), 0xFF5C00FF.toInt()
        )

        val ADE_CLASSES = listOf(
            "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
            "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table",
            "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa",
            "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat",
            "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
            "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink",
            "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case",
            "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind",
            "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove",
            "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
            "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
            "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister",
            "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
            "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel",
            "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
            "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle",
            "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
            "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
            "bulletin board", "shower", "radiator", "glass", "clock", "flag"
        )
    }
}
