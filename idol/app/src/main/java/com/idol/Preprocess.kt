package com.idol

import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/** The picture has nothing to cut out (the app's "No object found in this picture."). */
class NoObjectException(val reason: String) : Exception("no object found: $reason")

/** What the encoders see, and how it was cut out (scripts/idol_glue.py preprocess). */
class Preprocessed(
    /** 1024 x 1024 RGB: the object on black. */
    val composite: Image8,
    /** The uint8 alpha at the resized size, before the minimum filter (the matting graph's, or the picture's own). */
    val alphaRaw: ByteArray,
    val alphaWidth: Int,
    val alphaHeight: Int,
    /** The crop square (left, top, right, bottom) in the resized picture. */
    val box: IntArray,
    /** True when the picture brought its own alpha (RGBA with a pixel below 255) and no matting ran. */
    val ownAlpha: Boolean,
)

/**
 * The host steps around the two matting graphs, ported line by line from scripts/idol_glue.py: birefnet_alpha (the
 * author's BiRefNet.remove_background) and dis_alpha (IS-Net DIS as its authors run it, plus the alpha floor).
 */
object Matting {
    const val SIZE = 1024
    private val BIREFNET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val BIREFNET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** uint8 DIS alpha values below this are set to 0 (idol_glue.DIS_ALPHA_FLOOR, DIS only). */
    const val DIS_ALPHA_FLOOR = 8

    /** The RGB picture as float32 planes [3, h, w], each value c (or c / 255 with [divide]). */
    private fun planes(rgb: Image8, divide: Boolean): FloatArray {
        val n = rgb.width * rgb.height
        val out = FloatArray(3 * n)
        val s = rgb.data
        for (i in 0 until n) {
            for (c in 0 until 3) {
                val v = (s[i * 3 + c].toInt() and 0xFF).toFloat()
                out[c * n + i] = if (divide) v / 255.0f else v
            }
        }
        return out
    }

    /** BiRefNet's input [1, 3, 1024, 1024]: RGB / 255, bilinear (align_corners) to 1024 x 1024, ImageNet mean / std. */
    fun birefnetInput(rgb: Image8): FloatArray {
        val t = TorchOps.bilinear(planes(rgb, true), 3, rgb.height, rgb.width, SIZE, SIZE, alignCorners = true)
        val n = SIZE * SIZE
        for (c in 0 until 3) {
            val m = BIREFNET_MEAN[c]
            val s = BIREFNET_STD[c]
            for (i in c * n until (c + 1) * n) t[i] = (t[i] - m) / s
        }
        return t
    }

    /** BiRefNet's sigmoid map [1, 1, 1024, 1024] -> uint8 alpha at w x h: bilinear (align_corners), x 255, truncated. */
    fun birefnetAlpha(map: FloatArray, w: Int, h: Int): ByteArray {
        val a = TorchOps.bilinear(map, 1, SIZE, SIZE, h, w, alignCorners = true)
        return ByteArray(a.size) { i -> toUint8(a[i]) }
    }

    /** DIS's input [1, 3, 1024, 1024]: bilinear to 1024 x 1024 on 0..255, then x / 255 - 0.5. */
    fun disInput(rgb: Image8): FloatArray {
        val t = TorchOps.bilinear(planes(rgb, false), 3, rgb.height, rgb.width, SIZE, SIZE, alignCorners = false)
        for (i in t.indices) t[i] = t[i] / 255.0f - 0.5f
        return t
    }

    /**
     * DIS's sigmoid map -> uint8 alpha at w x h: bilinear back, min-max normalised (the division guarded by 1e-12 as in
     * the authors' code), x 255 truncated, values below [floor] set to 0.
     */
    fun disAlpha(map: FloatArray, w: Int, h: Int, floor: Int = DIS_ALPHA_FLOOR): ByteArray {
        val a = TorchOps.bilinear(map, 1, SIZE, SIZE, h, w, alignCorners = false)
        var mi = Float.POSITIVE_INFINITY
        var ma = Float.NEGATIVE_INFINITY
        for (v in a) {
            if (v < mi) mi = v
            if (v > ma) ma = v
        }
        val d = max(ma - mi, 1e-12f)
        val out = ByteArray(a.size)
        for (i in a.indices) {
            val u = toUint8((a[i] - mi) / d).toInt() and 0xFF
            out[i] = (if (u >= floor) u else 0).toByte()
        }
        return out
    }

    /** torch: (a.clamp(0, 1) * 255).to(torch.uint8), the cast truncating. */
    private fun toUint8(v: Float): Byte {
        val c = if (v < 0f) 0f else if (v > 1f) 1f else v
        return (c * 255f).toInt().toByte()
    }
}

object Preprocess {
    const val IMAGE_SIZE = 1024
    val CROP_MARGIN = mapOf("synthetic" to 1.2, "real" to 1.7)

    /** The resize size of the author's first step: short side 1024, Python round() (half to even), at least 1. */
    fun firstSize(width: Int, height: Int): Pair<Int, Int> {
        val scale = IMAGE_SIZE.toDouble() / min(width, height)
        // kotlin.math.round is Math.rint: ties to the even integer, as Python's round()
        return max(1, round(width * scale).toInt()) to max(1, round(height * scale).toInt())
    }

    /** First step alone (the matting input of a picture without its own alpha): the resized picture. */
    fun resized(image: Image8): Image8 {
        val (w, h) = firstSize(image.width, image.height)
        return Pil.resize(image, w, h)
    }

    /**
     * scripts/idol_glue.py preprocess: resize the short side to 1024 (Lanczos), alpha from [alphaFn] unless the picture
     * has its own, 3 x 3 minimum filter on the alpha, square crop of side margin x the box of the nonzero alpha,
     * resize to 1024 (Lanczos), composite on black.
     *
     * The app adds guards the author's code does not have (it raises or crops nonsense there): a picture of one flat
     * colour, an alpha with no pixel at 128 or above, and a crop square under one pixel are "no object".
     */
    fun run(image: Image8, mode: String, alphaFn: (Image8) -> ByteArray): Preprocessed {
        val margin = requireNotNull(CROP_MARGIN[mode]) { "mode $mode" }
        var img = resized(image)
        val w = img.width
        val h = img.height
        val ownAlpha = img.channels == 4 && minByte(img.channel(3)) < 255
        if (!ownAlpha) {
            val rgb = img.rgb()
            if (isFlat(rgb)) throw NoObjectException("the picture is one flat colour")
            img = rgb.withAlpha(alphaFn(rgb))
        }
        val alphaRaw = img.channel(3)
        if (maxByte(alphaRaw) < 128) throw NoObjectException("no alpha value reaches 128")
        val alpha = Pil.minFilter3(alphaRaw, w, h)
        val px = img.data
        for (i in 0 until w * h) px[i * 4 + 3] = alpha[i]
        var left = Int.MAX_VALUE
        var top = Int.MAX_VALUE
        var right = -1
        var bottom = -1
        for (y in 0 until h) for (x in 0 until w) {
            if (alpha[y * w + x].toInt() != 0) {
                if (x < left) left = x
                if (x > right) right = x
                if (y < top) top = y
                if (y > bottom) bottom = y
            }
        }
        if (right < 0) throw NoObjectException("the alpha channel is empty")
        val centerX = (left + right) / 2.0
        val centerY = (top + bottom) / 2.0
        val half = max(right - left, bottom - top) / 2.0 * margin
        val box = intArrayOf(
            (centerX - half).toInt(), (centerY - half).toInt(), (centerX + half).toInt(), (centerY + half).toInt(),
        )
        if (box[2] - box[0] < 1 || box[3] - box[1] < 1) throw NoObjectException("the object is under one pixel")
        val cropped = Pil.crop(img, box[0], box[1], box[2], box[3])
        val square = Pil.resize(cropped, IMAGE_SIZE, IMAGE_SIZE)
        return Preprocessed(Pil.compositeOnBlack(square), alphaRaw, w, h, box, ownAlpha)
    }

    /** pixels_of: [1, 3, 1024, 1024] float32 in [0, 1]. */
    fun pixels(composite: Image8): FloatArray {
        val n = composite.width * composite.height
        val out = FloatArray(3 * n)
        val s = composite.data
        for (i in 0 until n) for (c in 0 until 3) out[c * n + i] = (s[i * 3 + c].toInt() and 0xFF).toFloat() / 255.0f
        return out
    }

    private fun minByte(a: ByteArray): Int {
        var m = 255
        for (b in a) {
            val v = b.toInt() and 0xFF
            if (v < m) m = v
        }
        return m
    }

    private fun maxByte(a: ByteArray): Int {
        var m = 0
        for (b in a) {
            val v = b.toInt() and 0xFF
            if (v > m) m = v
        }
        return m
    }

    private fun isFlat(rgb: Image8): Boolean {
        val s = rgb.data
        for (i in 3 until s.size) if (s[i] != s[i % 3]) return false
        return true
    }
}
