/*
 * Ports smart_resize of the image processor of LiquidAI/d1-2B (revision 10e7671d, LFM Open License
 * v1.0) and PyTorch's uint8 antialiased bicubic resize (BSD-3-Clause) to Kotlin; see NOTICE and
 * licenses/.
 */

package com.d1_2b

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * One picture as the model's image processor leaves it: RGB resized so both sides are multiples of
 * 32 ([width] x [height]), cut into 16 x 16 patches ([patchRows] x [patchColumns]).
 */
class Picture(val width: Int, val height: Int, private val rgb: ByteArray) {
  val patchRows = height / PATCH
  val patchColumns = width / PATCH

  /** The picture's tokens in the prompt: one per 2 x 2 block of patches. */
  val tokens: Int
    get() = patchRows * patchColumns / 4

  private fun value(row: Int, column: Int, channel: Int): Float =
    ((rgb[(row * width + column) * 3 + channel].toInt() and 0xff).toFloat() - MEAN) / STD

  /**
   * The tower's inputs on a [canvasRows] x [canvasColumns] canvas: `patches` [rows * columns, 768]
   * (each patch's channel, row, column values; the picture at the top-left, raster order) and
   * `valid` [rows * columns] (1 on the picture's patches).
   */
  fun canvas(canvasRows: Int, canvasColumns: Int): Pair<FloatArray, FloatArray> {
    require(patchRows <= canvasRows && patchColumns <= canvasColumns) { "the picture does not fit the canvas" }
    val patches = FloatArray(canvasRows * canvasColumns * PATCH_VALUES)
    val valid = FloatArray(canvasRows * canvasColumns)
    for (r in 0 until patchRows) {
      for (c in 0 until patchColumns) {
        val cell = r * canvasColumns + c
        valid[cell] = 1f
        val base = cell * PATCH_VALUES
        for (channel in 0 until 3) {
          for (py in 0 until PATCH) {
            for (px in 0 until PATCH) {
              patches[base + channel * PATCH * PATCH + py * PATCH + px] = value(r * PATCH + py, c * PATCH + px, channel)
            }
          }
        }
      }
    }
    return patches to valid
  }

  /** The processor's `pixel_values` [patches, 1536] in merge-block order, both temporal copies (for checks). */
  fun pixelValues(): FloatArray {
    val out = FloatArray(patchRows * patchColumns * 2 * PATCH_VALUES)
    var n = 0
    for (bi in 0 until patchRows / 2) for (bj in 0 until patchColumns / 2) for (a in 0 until 2) for (b in 0 until 2) {
      val r = bi * 2 + a
      val c = bj * 2 + b
      val base = n * 2 * PATCH_VALUES
      for (channel in 0 until 3) for (t in 0 until 2) for (py in 0 until PATCH) for (px in 0 until PATCH) {
        out[base + (channel * 2 + t) * PATCH * PATCH + py * PATCH + px] = value(r * PATCH + py, c * PATCH + px, channel)
      }
      n++
    }
    return out
  }

  /** The tower's output on the canvas [rows/2 * columns/2, HIDDEN] (block raster) cut to the picture's tokens. */
  fun fromCanvas(features: FloatArray, canvasRows: Int, canvasColumns: Int): FloatArray {
    val out = FloatArray(tokens * HIDDEN)
    var n = 0
    for (bi in 0 until patchRows / 2) for (bj in 0 until patchColumns / 2) {
      System.arraycopy(features, (bi * (canvasColumns / 2) + bj) * HIDDEN, out, n * HIDDEN, HIDDEN)
      n++
    }
    return out
  }

  companion object {
    const val PATCH = 16
    const val PATCH_VALUES = 3 * PATCH * PATCH
    const val FACTOR = 32
    const val MIN_PIXELS = 65_536
    const val MAX_PIXELS = 16_777_216

    /** The model's pixel budget (`d1_vision_max_pixels`): a larger picture is shrunk first by the caller. */
    const val CAP_PIXELS = 1_638_400

    // The processor's mean and std, 0.5 each, fused with the rescale by 1 / 255 (float32 127.5).
    private const val MEAN = 127.5f
    private const val STD = 127.5f

    /** The processor's picture: [argb] pixels ([width] x [height], alpha ignored) resized as the model's processor does. */
    fun of(argb: IntArray, width: Int, height: Int): Picture {
      require(width * height <= CAP_PIXELS) { "a picture over ${CAP_PIXELS} pixels is shrunk before it reaches the model" }
      val rgb = ByteArray(width * height * 3)
      for (i in 0 until width * height) {
        val p = argb[i]
        rgb[i * 3] = (p shr 16).toByte()
        rgb[i * 3 + 1] = (p shr 8).toByte()
        rgb[i * 3 + 2] = p.toByte()
      }
      val (h, w) = smartResize(height, width)
      return Picture(w, h, resize(rgb, height, width, h, w))
    }

    /** smart_resize of the LFM3-VL image processor: both sides multiples of 32, the area inside the bounds. */
    fun smartResize(height: Int, width: Int): Pair<Int, Int> {
      require(max(height, width).toDouble() / min(height, width) <= 200) { "absolute aspect ratio must be smaller than 200" }
      var h = Math.rint(height.toDouble() / FACTOR).toInt() * FACTOR
      var w = Math.rint(width.toDouble() / FACTOR).toInt() * FACTOR
      if (h.toLong() * w > MAX_PIXELS) {
        val beta = sqrt(height.toDouble() * width / MAX_PIXELS)
        h = max(FACTOR, floor(height / beta / FACTOR).toInt() * FACTOR)
        w = max(FACTOR, floor(width / beta / FACTOR).toInt() * FACTOR)
      } else if (h.toLong() * w < MIN_PIXELS) {
        val beta = sqrt(MIN_PIXELS.toDouble() / (height.toDouble() * width))
        h = ceil(height * beta / FACTOR).toInt() * FACTOR
        w = ceil(width * beta / FACTOR).toInt() * FACTOR
      }
      return h to w
    }

    /** PyTorch's uint8 antialiased bicubic resize (torchvision `resize`, BICUBIC, antialias): width first, then height. */
    fun resize(rgb: ByteArray, height: Int, width: Int, outHeight: Int, outWidth: Int): ByteArray {
      var data = rgb
      var w = width
      if (outWidth != width) {
        data = resampleAxis(data, height, w, outWidth, horizontal = true)
        w = outWidth
      }
      if (outHeight != height) data = resampleAxis(data, height, w, outHeight, horizontal = false)
      return data
    }

    private fun cubic(xIn: Double): Double {
      val a = -0.5
      val x = abs(xIn)
      if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1
      if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
      return 0.0
    }

    /** `_compute_index_ranges_int16_weights`: per output index its first source index, taps, int16 weights and the precision. */
    private class Weights(val starts: IntArray, val sizes: IntArray, val weights: Array<IntArray>, val precision: Int)

    private fun weights(inSize: Int, outSize: Int): Weights {
      val scale = inSize.toDouble() / outSize
      val support = if (scale >= 1.0) 2.0 * scale else 2.0
      val maxTaps = ceil(support).toInt() * 2 + 1
      val starts = IntArray(outSize)
      val sizes = IntArray(outSize)
      val real = Array(outSize) { DoubleArray(maxTaps) }
      var wtMax = 0.0
      for (i in 0 until outSize) {
        val center = scale * (i + 0.5)
        val invscale = if (scale >= 1.0) 1.0 / scale else 1.0
        val xmin = max((center - support + 0.5).toInt(), 0)
        var xsize = min((center + support + 0.5).toInt(), inSize) - xmin
        xsize = min(max(xsize, 0), maxTaps)
        var total = 0.0
        for (j in 0 until xsize) {
          val w = cubic((j + xmin - center + 0.5) * invscale)
          real[i][j] = w
          total += w
        }
        var wmax = 0.0
        if (total != 0.0) {
          for (j in 0 until xsize) {
            real[i][j] = real[i][j] / total
            wmax = max(wmax, real[i][j])
          }
        }
        wtMax = max(wtMax, wmax)
        starts[i] = xmin
        sizes[i] = xsize
      }
      var precision = 0
      while (precision < 22) {
        if ((0.5 + wtMax * (1L shl (precision + 1))).toLong() >= (1L shl 15)) break
        precision++
      }
      val int16 =
        Array(outSize) { i ->
          IntArray(maxTaps) { j ->
            val scaled = real[i][j] * (1L shl precision)
            (if (scaled < 0) kotlin.math.truncate(-0.5 + scaled) else kotlin.math.truncate(0.5 + scaled)).toInt()
          }
        }
      return Weights(starts, sizes, int16, precision)
    }

    /** One separable pass in exact integer arithmetic (+ 2^(p-1), >> p, clamp 0..255). */
    private fun resampleAxis(data: ByteArray, height: Int, width: Int, outSize: Int, horizontal: Boolean): ByteArray {
      val inSize = if (horizontal) width else height
      val w = weights(inSize, outSize)
      val outHeight = if (horizontal) height else outSize
      val outWidth = if (horizontal) outSize else width
      val out = ByteArray(outHeight * outWidth * 3)
      val round = 1 shl (w.precision - 1)
      for (y in 0 until outHeight) {
        for (x in 0 until outWidth) {
          val i = if (horizontal) x else y
          for (channel in 0 until 3) {
            var acc = round
            for (t in 0 until w.sizes[i]) {
              val source = w.starts[i] + t
              val sy = if (horizontal) y else source
              val sx = if (horizontal) source else x
              acc += (data[(sy * width + sx) * 3 + channel].toInt() and 0xff) * w.weights[i][t]
            }
            out[(y * outWidth + x) * 3 + channel] = min(255, max(0, acc shr w.precision)).toByte()
          }
        }
      }
      return out
    }

    /** The canvases the tower files take ([rows, columns]); a picture runs on the smallest that holds its patch grid. */
    val CANVASES = listOf(24 to 24, 32 to 48, 48 to 32, 48 to 64, 64 to 48, 72 to 96, 96 to 72)

    fun canvasFor(picture: Picture, canvases: List<Pair<Int, Int>> = CANVASES): Pair<Int, Int> =
      canvases
        .filter { it.first >= picture.patchRows && it.second >= picture.patchColumns }
        .minWithOrNull(compareBy<Pair<Int, Int>> { it.first * it.second }.thenBy { it.first }.thenBy { it.second })
        ?: throw IllegalArgumentException("a picture of ${picture.patchRows} x ${picture.patchColumns} patches fits no tower canvas")
  }
}
