package com.d1omni

/** A picture as uint8 RGB, row-major [height, width, 3] (values 0..255 stored in bytes). */
class D1Rgb(val width: Int, val height: Int, val data: ByteArray) {
  init {
    require(width >= 1 && height >= 1) { "empty image ${width}x$height" }
    require(data.size.toLong() == width.toLong() * height * 3) {
      "${data.size} bytes for a ${width}x$height RGB image"
    }
  }
}

/** `layout()`'s plan: the tile grid ([gridCols] x [gridRows], used when [tiled]) and the thumbnail size. */
data class D1Layout(
  val gridCols: Int,
  val gridRows: Int,
  val thumbHeight: Int,
  val thumbWidth: Int,
  val tiled: Boolean,
)

/** One crop's tower inputs: `pixels` [1024 x 768], `mask` [1024] and the patch grid (rows, columns). */
class D1Patches(val pixels: FloatArray, val mask: FloatArray, val gridHeight: Int, val gridWidth: Int) {
  /** The real patches, ph * pw. */
  val patches: Int
    get() = gridHeight * gridWidth

  /** The crop's prefix rows after the unshuffle, (ph / 2)(pw / 2). */
  val cells: Int
    get() = (gridHeight / 2) * (gridWidth / 2)
}

/** `_linear_weights_f32`: per output index, the first input index, the tap count and the float32 weights. */
class D1ResampleWeights(val xmins: IntArray, val sizes: IntArray, val maxTaps: Int, val weights: FloatArray) {
  fun weight(index: Int, tap: Int): Float = weights[index * maxTaps + tap]
}

/**
 * The host steps of a request with a picture, Android-free: a port of the model repository's
 * `host/d1_vision_host.py` (the provider's `vision.py` layout / preprocess and transformers'
 * position-table resize), written line by line so that every value is the Python host's bit for bit
 * (`D1VisionTest` compares them with its dumps). The float32 / float64 casts follow the Python
 * function's order one by one: PyTorch's separable bilinear kernel with antialias, float32 weights,
 * each tap added as a float64 multiply-add rounded to float32 (the JVM never fuses a multiply and an
 * add, so this is numpy's two roundings exactly).
 */
object D1Vision {
  const val TILE = 512
  const val PATCH = 16
  const val MAX_PATCHES = 1024
  const val FACTOR = 32
  const val MAX_PIXELS = 256 * 1024
  const val MIN_PIXELS = 64 * 1024
  const val PROJECTOR_ROWS = 256
  const val HIDDEN = 768
  const val PATCH_VALUES = PATCH * PATCH * 3
  const val UNSHUFFLED = 4 * HIDDEN
  const val TABLE_SIDE = 16

  /**
   * `layout()`'s tile grids (columns, rows) in the order it walks them: the set of grids of 2..10
   * tiles sorted by tile count, equal counts in CPython's set iteration order (the dump's
   * `layout_cases.json` `ratios`; the order decides a tie).
   */
  val GRIDS: List<Pair<Int, Int>> =
    listOf(
      1 to 2, 2 to 1, 3 to 1, 1 to 3, 2 to 2, 4 to 1, 1 to 4, 5 to 1, 1 to 5, 1 to 6, 6 to 1, 3 to 2, 2 to 3,
      7 to 1, 1 to 7, 4 to 2, 2 to 4, 1 to 8, 8 to 1, 1 to 9, 3 to 3, 9 to 1, 2 to 5, 5 to 2, 10 to 1, 1 to 10,
    )

  /**
   * The provider's `layout(width, height)` (LFM2-VL's smart resize, the tile grid and the thumbnail):
   * Python's `round` (half to even) is [Math.rint], `math.floor` / `math.ceil` / `math.sqrt` are the
   * same double operations in the same order.
   */
  fun layout(width: Int, height: Int): D1Layout {
    require(minOf(width, height) >= 1) { "empty image" }
    var h = maxOf(FACTOR, round32(height))
    var w = maxOf(FACTOR, round32(width))
    if (h.toLong() * w > MAX_PIXELS) {
      val beta = Math.sqrt((height.toLong() * width).toDouble() / MAX_PIXELS)
      h = maxOf(FACTOR, Math.floor(height / beta / FACTOR).toInt() * FACTOR)
      w = maxOf(FACTOR, Math.floor(width / beta / FACTOR).toInt() * FACTOR)
    } else if (h.toLong() * w < MIN_PIXELS) {
      val beta = Math.sqrt(MIN_PIXELS / (height.toLong() * width).toDouble())
      h = Math.ceil(height * beta / FACTOR).toInt() * FACTOR
      w = Math.ceil(width * beta / FACTOR).toInt() * FACTOR
    }
    val large = maxOf(16, round32(height)).toLong() * maxOf(16, round32(width)) > MAX_PIXELS * 2L
    var grid = 1 to 1
    if (large) {
      var best = Double.POSITIVE_INFINITY
      val pixels = (width.toLong() * height).toDouble()
      for (ratio in GRIDS) {
        val diff = Math.abs(width.toDouble() / height - ratio.first.toDouble() / ratio.second)
        if (diff < best || (diff == best && pixels > 0.5 * TILE * TILE * ratio.first * ratio.second)) {
          grid = ratio
          best = diff
        }
      }
    }
    return D1Layout(grid.first, grid.second, h, w, large)
  }

  /** `round(side / 32) * 32` with Python's round (half to even). */
  private fun round32(side: Int): Int = Math.rint(side.toDouble() / FACTOR).toInt() * FACTOR

  /**
   * `_linear_weights_f32` (PyTorch's `compute_index_ranges_weights<float>`, antialias, align_corners
   * False): each cast as the Python function writes it.
   */
  fun linearWeightsF32(inSize: Int, outSize: Int): D1ResampleWeights {
    require(inSize >= 1 && outSize >= 1) { "resize $inSize -> $outSize" }
    val scale = inSize.toFloat() / outSize.toFloat()
    val support = if (scale >= 1.0f) (1.0 * scale.toDouble()).toFloat() else 1.0f
    val maxTaps = Math.ceil(support.toDouble()).toInt() * 2 + 1
    val xmins = IntArray(outSize)
    val sizes = IntArray(outSize)
    val weights = FloatArray(outSize * maxTaps)
    val invscale = if (scale >= 1.0f) (1.0 / scale.toDouble()).toFloat() else 1.0f
    val taps = FloatArray(maxTaps)
    for (i in 0 until outSize) {
      val center = (scale.toDouble() * (i + 0.5)).toFloat()
      val xmin = maxOf(((center - support).toDouble() + 0.5).toInt(), 0)
      var xsize = minOf(((center + support).toDouble() + 0.5).toInt(), inSize) - xmin
      xsize = minOf(maxOf(xsize, 0), maxTaps)
      var total = 0.0f
      for (j in 0 until xsize) {
        val arg = ((((j + xmin).toFloat() - center).toDouble() + 0.5) * invscale.toDouble()).toFloat()
        val x = Math.abs(arg)
        val w = if (x < 1.0f) (1.0 - x.toDouble()).toFloat() else 0.0f
        taps[j] = w
        total += w
      }
      for (j in 0 until xsize) {
        weights[i * maxTaps + j] = if (total != 0.0f) taps[j] / total else taps[j]
      }
      xmins[i] = xmin
      sizes[i] = xsize
    }
    return D1ResampleWeights(xmins, sizes, maxTaps, weights)
  }

  /**
   * `_resample_axis_f32` on a float32 array laid out as [outer, axis, inner] (row-major), along the
   * middle axis, to [outer, outSize, inner]: out = t0 * w0 (float32), then out = float32(float64(out)
   * + float64(t_j) * float64(w_j)) for each further tap.
   */
  fun resampleAxisF32(src: FloatArray, outer: Int, axisIn: Int, inner: Int, outSize: Int): FloatArray {
    require(src.size.toLong() == outer.toLong() * axisIn * inner) {
      "${src.size} values for [$outer, $axisIn, $inner]"
    }
    val plan = linearWeightsF32(axisIn, outSize)
    val out = FloatArray(outer * outSize * inner)
    for (o in 0 until outer) {
      val srcBase = o * axisIn * inner
      val outBase = o * outSize * inner
      for (i in 0 until outSize) {
        val first = srcBase + plan.xmins[i] * inner
        val w0 = plan.weight(i, 0)
        val row = outBase + i * inner
        for (c in 0 until inner) out[row + c] = src[first + c] * w0
        for (j in 1 until plan.sizes[i]) {
          val tap = srcBase + (plan.xmins[i] + j) * inner
          val w = plan.weight(i, j).toDouble()
          for (c in 0 until inner) {
            out[row + c] = (out[row + c].toDouble() + src[tap + c].toDouble() * w).toFloat()
          }
        }
      }
    }
    return out
  }

  /**
   * `resize_float`: uint8 -> float32, the width pass then the height pass (a side that keeps its
   * size is skipped), round half to even, clamp 0..255, uint8. An unchanged size returns [src].
   */
  fun resizeFloat(src: D1Rgb, outHeight: Int, outWidth: Int): D1Rgb {
    if (outHeight == src.height && outWidth == src.width) return src
    var x = FloatArray(src.data.size) { (src.data[it].toInt() and BYTE_MASK).toFloat() }
    var width = src.width
    if (outWidth != width) {
      x = resampleAxisF32(x, src.height, width, 3, outWidth)
      width = outWidth
    }
    if (outHeight != src.height) {
      x = resampleAxisF32(x, 1, src.height, width * 3, outHeight)
    }
    val out = ByteArray(x.size)
    for (index in x.indices) {
      val rounded = Math.rint(x[index].toDouble())
      out[index] = (if (rounded < 0.0) 0.0 else if (rounded > 255.0) 255.0 else rounded).toInt().toByte()
    }
    return D1Rgb(outWidth, outHeight, out)
  }

  /** Rows [top, top + height) and columns [left, left + width) of [src]. */
  fun cut(src: D1Rgb, top: Int, left: Int, height: Int, width: Int): D1Rgb {
    require(top >= 0 && left >= 0 && top + height <= src.height && left + width <= src.width) {
      "cut ${height}x$width at ($top, $left) of ${src.height}x${src.width}"
    }
    val out = ByteArray(height * width * 3)
    for (r in 0 until height) {
      System.arraycopy(src.data, ((top + r) * src.width + left) * 3, out, r * width * 3, width * 3)
    }
    return D1Rgb(width, height, out)
  }

  /**
   * `crops_of`: the crops in the provider's order (a tiled picture: resized to rows * 512 x cols *
   * 512 and cut into 512 x 512 tiles row-major; then the thumbnail at the layout size; a small
   * picture: the thumbnail alone) and the plan.
   */
  fun crops(rgb: D1Rgb): Pair<List<D1Rgb>, D1Layout> {
    val plan = layout(rgb.width, rgb.height)
    val crops = ArrayList<D1Rgb>()
    if (plan.tiled) {
      val big = resizeFloat(rgb, plan.gridRows * TILE, plan.gridCols * TILE)
      for (r in 0 until plan.gridRows) {
        for (c in 0 until plan.gridCols) crops.add(cut(big, r * TILE, c * TILE, TILE, TILE))
      }
    }
    crops.add(resizeFloat(rgb, plan.thumbHeight, plan.thumbWidth))
    return crops to plan
  }

  /**
   * `to_patches`: float32 (x − 127.5) / 127.5, 16 x 16 patches in raster order, each (row, column,
   * channel) = 768 values, zero rows after the real patches up to 1024; mask 1 for a real patch.
   */
  fun toPatches(crop: D1Rgb): D1Patches {
    val gridHeight = crop.height / PATCH
    val gridWidth = crop.width / PATCH
    val count = gridHeight * gridWidth
    require(count <= MAX_PATCHES) { "a crop of $count patches exceeds $MAX_PATCHES" }
    val pixels = FloatArray(MAX_PATCHES * PATCH_VALUES)
    val data = crop.data
    for (py in 0 until gridHeight) {
      for (px in 0 until gridWidth) {
        val base = (py * gridWidth + px) * PATCH_VALUES
        for (r in 0 until PATCH) {
          val rowStart = ((py * PATCH + r) * crop.width + px * PATCH) * 3
          val outStart = base + r * PATCH * 3
          for (v in 0 until PATCH * 3) {
            pixels[outStart + v] = ((data[rowStart + v].toInt() and BYTE_MASK).toFloat() - 127.5f) / 127.5f
          }
        }
      }
    }
    val mask = FloatArray(MAX_PATCHES) { if (it < count) 1.0f else 0.0f }
    return D1Patches(pixels, mask, gridHeight, gridWidth)
  }

  /**
   * `resize_positions`: the position table float32 [16, 16, 768] (row-major) resized to [h, w, 768]
   * with the same kernel (width, then height; an unchanged side skipped), rows in raster order.
   */
  fun resizePositions(table: FloatArray, height: Int, width: Int): FloatArray {
    require(table.size == TABLE_SIDE * TABLE_SIDE * HIDDEN) { "a position table of ${table.size} values" }
    var t = table
    var columns = TABLE_SIDE
    if (width != TABLE_SIDE) {
      t = resampleAxisF32(t, TABLE_SIDE, TABLE_SIDE, HIDDEN, width)
      columns = width
    }
    if (height != TABLE_SIDE) {
      t = resampleAxisF32(t, 1, TABLE_SIDE, columns * HIDDEN, height)
    }
    return if (t === table) table.copyOf() else t
  }

  /** `positions_padded`: [1024 x 768], the resized rows, then row 0 repeated. */
  fun positionsPadded(table: FloatArray, gridHeight: Int, gridWidth: Int): FloatArray {
    val count = gridHeight * gridWidth
    require(count in 1..MAX_PATCHES) { "a grid of $count patches" }
    val positions = resizePositions(table, gridHeight, gridWidth)
    val out = FloatArray(MAX_PATCHES * HIDDEN)
    System.arraycopy(positions, 0, out, 0, count * HIDDEN)
    for (row in count until MAX_PATCHES) System.arraycopy(positions, 0, out, row * HIDDEN, HIDDEN)
    return out
  }

  /**
   * `pixel_unshuffle`: the first h * w feature rows [h * w, c] as an (h, w, c) grid -> [(h/2)(w/2),
   * 4c]: channel j * 2c + k * c + i of cell (r, q) = input (2r + j, 2q + k, i), cells in raster order.
   */
  fun pixelUnshuffle(features: FloatArray, gridHeight: Int, gridWidth: Int, channels: Int = HIDDEN): FloatArray {
    require(gridHeight % 2 == 0 && gridWidth % 2 == 0) { "grid ($gridHeight, $gridWidth) is not divisible by 2" }
    require(features.size >= gridHeight * gridWidth * channels) { "features hold ${features.size} values" }
    val rows = gridHeight / 2
    val columns = gridWidth / 2
    val out = FloatArray(rows * columns * 4 * channels)
    for (r in 0 until rows) {
      for (q in 0 until columns) {
        val cell = (r * columns + q) * 4 * channels
        for (j in 0 until 2) {
          for (k in 0 until 2) {
            val source = ((2 * r + j) * gridWidth + (2 * q + k)) * channels
            System.arraycopy(features, source, out, cell + (j * 2 + k) * channels, channels)
          }
        }
      }
    }
    return out
  }

  /** `projector_input`: [256 x 3072] with zero rows after the [cells] rows of [unshuffled]. */
  fun projectorInput(unshuffled: FloatArray, cells: Int): FloatArray {
    require(cells <= PROJECTOR_ROWS) { "$cells cells exceed the projector graph's $PROJECTOR_ROWS rows" }
    require(unshuffled.size == cells * UNSHUFFLED) { "${unshuffled.size} values for $cells cells" }
    return FloatArray(PROJECTOR_ROWS * UNSHUFFLED).also { unshuffled.copyInto(it) }
  }

  /** P = the sum over crops of (ph / 2)(pw / 2). */
  fun prefixLength(patches: List<D1Patches>): Int = patches.sumOf { it.cells }

  private const val BYTE_MASK = 0xff
}
