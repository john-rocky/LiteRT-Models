package com.timestral

import kotlin.math.max
import kotlin.math.min

/**
 * One exported graph shape: [contextPatches] context patches and [horizonPatches] decoder patches,
 * so the graph always sees `contextPatches + horizonPatches` tokens of 32 steps each.
 */
data class Bucket(val name: String, val contextPatches: Int, val horizonPatches: Int) {
  /** Tokens of the graph (S = L + H). */
  val tokens: Int
    get() = contextPatches + horizonPatches
}

/**
 * The host tables installed next to the graph files: the checkpoint's `forecast_query` vector (768
 * floats, the decoder token) and the RoPE cos / sin tables (rows of 64 floats, row 0 at position
 * [firstPosition]), written with the official `SinCosRoPE` module by the conversion run.
 */
class HostTables(
  val forecastQuery: FloatArray,
  val ropeCos: FloatArray,
  val ropeSin: FloatArray,
  val firstPosition: Int,
) {
  init {
    require(forecastQuery.size == TimestralHost.D_MODEL) {
      "forecast_query has ${forecastQuery.size} values, expected ${TimestralHost.D_MODEL}"
    }
    require(ropeCos.size == ropeSin.size && ropeCos.size % TimestralHost.ROPE_DIM == 0) {
      "RoPE tables have ${ropeCos.size} and ${ropeSin.size} values"
    }
  }

  /** Rows of each RoPE table. */
  val rows: Int
    get() = ropeCos.size / TimestralHost.ROPE_DIM
}

/**
 * The five float32 inputs of the graph's `serving_default` signature, flattened row-major: patches
 * `[R, L, 64]`, dec `[R, H, 768]`, bias `[1, 1, S, S]`, cos and sin `[1, 1, S, 64]`.
 */
class GraphInputs(
  val patches: FloatArray,
  val dec: FloatArray,
  val bias: FloatArray,
  val cos: FloatArray,
  val sin: FloatArray,
  /** The first context patch (of the bucket's L) with an observation; 0 when none has one. */
  val firstObservedPatch: Int,
) {
  /** The inputs by signature name. */
  fun byName(): Map<String, FloatArray> =
    mapOf("patches" to patches, "dec" to dec, "bias" to bias, "cos" to cos, "sin" to sin)
}

/**
 * One series made ready for the graph: the context row (missing steps forward-filled, the last
 * [TimestralHost.MAX_CONTEXT] steps, left-padded to whole patches), its negation (the mirror row of
 * symmetric averaging), the causal statistics of both rows and their normalized values.
 */
class PreparedSeries(
  /** Forecast steps asked for. */
  val horizon: Int,
  /** Decoder patches the forecast needs (h = ceil(horizon / 32)). */
  val horizonPatches: Int,
  /** The whole series with missing steps set to 0 (the non-negative check reads it). */
  val inputs: FloatArray,
  /** Which steps of the whole series are observed (finite). */
  val observedAll: BooleanArray,
  /** The context row: T steps, T a multiple of 32. */
  val context: FloatArray,
  /** Which steps of [context] are observed. */
  val observed: BooleanArray,
  /** Per row (context, mirror) and step: the causal location of the step's patch. */
  val loc: Array<FloatArray>,
  /** Per row and step: the causal scale of the step's patch. */
  val scale: Array<FloatArray>,
  /** Per row and step: asinh((x - loc) / scale) on observed steps, 0 elsewhere. */
  val scaled: Array<FloatArray>,
) {
  /** Context patches (T / 32). */
  val contextPatches: Int
    get() = context.size / TimestralHost.PATCH

  /** The row values: the context, then its negation. */
  fun row(index: Int): FloatArray = if (index == 0) context else negate(context)

  private fun negate(values: FloatArray) = FloatArray(values.size) { -values[it] }
}

/** The host refused a forecast (the official forecaster raises the same error). */
class ForecastException(message: String) : Exception(message)

/**
 * The host side of Timestral-209M around one graph call, ported from the conversion run's
 * `scripts/host_forecast.py` with the official forecaster's defaults: symmetric averaging (the
 * context and its negation as the two rows of one call), non-negative clipping, quantile cap k =
 * 1e4, one decoder block of up to 32 patches with the advancing refine frame. Each stage keeps the
 * official dtype: the causal statistics and the refine frames are float64, normalization,
 * denormalization, the cap and the averaging are float32, asinh and sinh are evaluated in float64
 * and rounded to float32. Sums of 32 values follow NumPy's pairwise order and `max`, `min` and
 * `clip` follow NumPy's rules for NaN and signed zeros, so that the JVM tests can hold the results
 * to the Python reference value by value.
 *
 * Android-free: the JVM tests run it against vectors the Python reference wrote.
 */
object TimestralHost {
  /** Steps per patch. */
  const val PATCH = 32

  /** Width of the model and of the decoder token. */
  const val D_MODEL = 768

  /** Columns of each RoPE table row. */
  const val ROPE_DIM = 64

  /** Quantile levels 0.01, 0.05, 0.1, 0.15 … 0.9, 0.95, 0.99. */
  const val QUANTILES = 21

  /** Index of the 0.5 level. */
  const val MEDIAN = 10

  /** Index of the 0.1 level. */
  const val P10 = 2

  /** Index of the 0.9 level. */
  const val P90 = 18

  /** Rows of every graph call: the context and its negation. */
  const val ROWS = 2

  /** Decoder patches of one block. */
  const val BLOCK_PATCHES = 32

  /** Longest horizon of one block: 1,024 steps. */
  const val MAX_HORIZON = BLOCK_PATCHES * PATCH

  /**
   * Longest context: the last 8,192 steps (256 patches, the largest graph of this sample), the same
   * as the official forecaster with `max_context=8192`.
   */
  const val MAX_CONTEXT = 8192

  /** Floor of the causal standard deviation (float64). */
  private const val MIN_SCALE = 1e-6

  /** Quantile cap: context range ± k × the last scale. */
  private const val CAP_K = 1e4f

  private const val F32_MAX = Float.MAX_VALUE

  /** float32 machine epsilon, 2^-23. */
  private val F32_EPS = Math.ulp(1f)

  /** Smallest positive normal float64. */
  private const val F64_TINY = java.lang.Double.MIN_NORMAL

  /** The refine frame's clip, `float32 max ** 0.5 / 1e3` in Python (1.8446743e16). */
  private val FOLD_CLIP = Math.sqrt(F32_MAX.toDouble()) / 1e3

  /**
   * Builds the rows of one forecast from [series] (NaN or infinite = missing) for [horizon] steps
   * (1 to [MAX_HORIZON]).
   */
  fun prepare(series: FloatArray, horizon: Int): PreparedSeries {
    require(horizon in 1..MAX_HORIZON) { "horizon $horizon is outside 1..$MAX_HORIZON" }
    require(series.isNotEmpty()) { "empty series" }
    val observedAll = BooleanArray(series.size) { series[it].isFinite() }
    val inputs = FloatArray(series.size) { if (observedAll[it]) series[it] else 0f }
    var rows = forwardFill(inputs, observedAll)
    var observed = observedAll
    if (rows.size > MAX_CONTEXT) {
      rows = rows.copyOfRange(rows.size - MAX_CONTEXT, rows.size)
      observed = observed.copyOfRange(observed.size - MAX_CONTEXT, observed.size)
    }
    val padded = padToPatch(rows, observed)
    val context = padded.first
    val contextObserved = padded.second
    val both = arrayOf(context, FloatArray(context.size) { -context[it] })
    val loc = arrayOfNulls<FloatArray>(ROWS)
    val scale = arrayOfNulls<FloatArray>(ROWS)
    val scaled = arrayOfNulls<FloatArray>(ROWS)
    for (row in 0 until ROWS) {
      val stats = patchLocScale(both[row], contextObserved)
      loc[row] = stats.first
      scale[row] = stats.second
      scaled[row] = normalize(both[row], stats.first, stats.second, contextObserved)
    }
    return PreparedSeries(
      horizon,
      min(BLOCK_PATCHES, (horizon + PATCH - 1) / PATCH),
      inputs,
      observedAll,
      context,
      contextObserved,
      loc.requireNoNulls(),
      scale.requireNoNulls(),
      scaled.requireNoNulls(),
    )
  }

  /**
   * The graph inputs of [prepared] for [bucket] and the series rows [rows] (0 = the context, 1 =
   * its negation; both for a two-row graph, one per call for a one-row graph): the context patches
   * right-aligned in the L slots (the slots in front are padding: value 0, every step missing), the
   * decoder tokens, the additive attention mask (0 where a pair may attend, [mask] where it may
   * not) and the RoPE rows of the positions counted from the first observed patch. The mask and
   * the RoPE rows are the same for both rows.
   */
  fun buildInputs(
    prepared: PreparedSeries,
    bucket: Bucket,
    tables: HostTables,
    mask: Float,
    rows: IntArray = intArrayOf(0, 1),
  ): GraphInputs {
    val l = bucket.contextPatches
    val h = bucket.horizonPatches
    val lc = prepared.contextPatches
    require(lc <= l && prepared.horizonPatches <= h) { "the series does not fit ${bucket.name}" }
    require(rows.isNotEmpty() && rows.all { it in 0 until ROWS }) { "rows ${rows.toList()}" }
    val pad = l - lc
    val s = l + h
    val width = 2 * PATCH
    val patches = FloatArray(rows.size * l * width)
    for ((slot, row) in rows.withIndex()) {
      val base = slot * l * width
      for (p in 0 until pad) {
        for (j in 0 until PATCH) patches[base + p * width + PATCH + j] = 1f
      }
      val scaled = prepared.scaled[row]
      for (p in 0 until lc) {
        val slot = base + (pad + p) * width
        for (j in 0 until PATCH) {
          val t = p * PATCH + j
          patches[slot + j] = scaled[t]
          patches[slot + PATCH + j] = if (prepared.observed[t]) 0f else 1f
        }
      }
    }
    val observedPatch = BooleanArray(l)
    for (p in 0 until lc) {
      var any = false
      for (j in 0 until PATCH) any = any || prepared.observed[p * PATCH + j]
      observedPatch[pad + p] = any
    }
    // A key is valid once any patch at or before it has an observation.
    val keyValid = BooleanArray(l)
    var seen = false
    for (p in 0 until l) {
      seen = seen || observedPatch[p]
      keyValid[p] = seen
    }
    val firstObserved = observedPatch.indexOfFirst { it }.let { if (it < 0) 0 else it }
    val bias = FloatArray(s * s)
    for (q in 0 until s) {
      for (k in 0 until s) {
        val allow =
          when {
            q < l && k < l -> (k <= q && keyValid[k]) || k == q
            q >= l && k < l -> keyValid[k]
            q >= l -> k <= q
            else -> false
          }
        bias[q * s + k] = if (allow) 0f else mask
      }
    }
    val cos = FloatArray(s * ROPE_DIM)
    val sin = FloatArray(s * ROPE_DIM)
    for (t in 0 until s) {
      val tableRow = t - firstObserved - tables.firstPosition
      require(tableRow in 0 until tables.rows) { "RoPE table too small for position $tableRow" }
      System.arraycopy(tables.ropeCos, tableRow * ROPE_DIM, cos, t * ROPE_DIM, ROPE_DIM)
      System.arraycopy(tables.ropeSin, tableRow * ROPE_DIM, sin, t * ROPE_DIM, ROPE_DIM)
    }
    val dec = FloatArray(rows.size * h * D_MODEL)
    for (token in 0 until rows.size * h) {
      System.arraycopy(tables.forecastQuery, 0, dec, token * D_MODEL, D_MODEL)
    }
    return GraphInputs(patches, dec, bias, cos, sin, firstObserved)
  }

  /**
   * The forecast from the graph output [raw] (`[2, H, 32, 21]` normalized quantiles of [bucket],
   * the context's row then the negation's; a one-row graph's two outputs one after the other):
   * denormalize, refine, cap, average the two rows and clip; `[horizon × 21]` row-major, step
   * after step. Throws [ForecastException] when a quantile is NaN (the context's row is checked
   * first, as the official forecaster does).
   */
  fun finish(prepared: PreparedSeries, raw: FloatArray, bucket: Bucket): FloatArray {
    val h = prepared.horizonPatches
    val perRow = bucket.horizonPatches * PATCH * QUANTILES
    require(raw.size == ROWS * perRow) { "graph output has ${raw.size} values, expected ${ROWS * perRow}" }
    val steps = h * PATCH
    val last = prepared.context.size - 1
    val rows = Array(ROWS) { row ->
      val values = FloatArray(steps * QUANTILES)
      for (i in values.indices) {
        values[i] = clip(raw[row * perRow + i], -F32_MAX, F32_MAX)
      }
      var q = denormalize(values, prepared.loc[row][last], prepared.scale[row][last])
      q = refineBlock(q, prepared.row(row), prepared.observed, h)
      q.copyOf(prepared.horizon * QUANTILES)
    }
    val forward = cap(rows[0], prepared.row(0), prepared.observed, prepared.scale[0][last])
    checkNoNan(forward)
    val mirror = cap(rows[1], prepared.row(1), prepared.observed, prepared.scale[1][last])
    checkNoNan(mirror)
    val out = FloatArray(forward.size)
    for (t in 0 until prepared.horizon) {
      for (i in 0 until QUANTILES) {
        val a = forward[t * QUANTILES + i]
        val b = -mirror[t * QUANTILES + (QUANTILES - 1 - i)]
        out[t * QUANTILES + i] = 0.5f * (a + b)
      }
    }
    if (nonNegativeContext(prepared.inputs, prepared.observedAll)) {
      for (i in out.indices) out[i] = npMax(out[i], 0f)
    }
    return out
  }

  // ---------------------------------------------------------------- context

  /** Missing steps take the last observed value before them (0 before the first one). */
  internal fun forwardFill(inputs: FloatArray, observed: BooleanArray): FloatArray {
    if (observed.all { it }) return inputs
    var source = 0
    return FloatArray(inputs.size) { t ->
      if (observed[t]) source = t
      inputs[source]
    }
  }

  /** Left-pads with missing zeros to a whole number of patches. */
  internal fun padToPatch(
    values: FloatArray,
    observed: BooleanArray,
  ): Pair<FloatArray, BooleanArray> {
    val extra = values.size % PATCH
    if (extra == 0) return values to observed
    val pad = PATCH - extra
    val padded = FloatArray(pad + values.size)
    System.arraycopy(values, 0, padded, pad, values.size)
    val mask = BooleanArray(pad + observed.size)
    System.arraycopy(observed, 0, mask, pad, observed.size)
    return padded to mask
  }

  // ---------------------------------------------------------------- normalization

  /**
   * The official causal statistics: float64 running mean and Bessel standard deviation over the
   * observed steps (floored at 1e-6), rounded to float32. The cumulative sums run in order, as
   * NumPy's `cumsum` does (the first value starts the sum, so its signed zero is kept).
   */
  internal fun causalStatistics(
    x: FloatArray,
    observed: BooleanArray,
  ): Pair<FloatArray, FloatArray> {
    val n = x.size
    val means = FloatArray(n)
    val scales = FloatArray(n)
    var cumWeight = 0.0
    var cumValue = 0.0
    var m2 = 0.0
    var previousMean = 0.0
    for (t in 0 until n) {
      val value = x[t].toDouble()
      val weight = if (observed[t]) 1.0 else 0.0
      val weighted = weight * value
      if (t == 0) {
        cumWeight = weight
        cumValue = weighted
      } else {
        cumWeight += weight
        cumValue += weighted
      }
      val den = npMax(cumWeight, 1.0)
      val mean = cumValue / den
      val increment = (value - previousMean) * (value - mean) * weight
      m2 = if (t == 0) increment else m2 + increment
      val variance = m2 / npMax(den - 1.0, 1.0)
      means[t] = mean.toFloat()
      scales[t] = npMax(Math.sqrt(variance), MIN_SCALE).toFloat()
      previousMean = mean
    }
    return means to scales
  }

  /** Per step: the location and scale of its patch (the statistics at the patch's last step). */
  internal fun patchLocScale(
    x: FloatArray,
    observed: BooleanArray,
  ): Pair<FloatArray, FloatArray> {
    val (means, scales) = causalStatistics(x, observed)
    val loc = FloatArray(x.size)
    val scale = FloatArray(x.size)
    for (start in x.indices step PATCH) {
      val end = start + PATCH - 1
      loc.fill(means[end], start, start + PATCH)
      scale.fill(scales[end], start, start + PATCH)
    }
    return loc to scale
  }

  /** asinh((x - loc) / max(scale, eps)) in float32 on observed steps, 0 elsewhere. */
  internal fun normalize(
    x: FloatArray,
    loc: FloatArray,
    scale: FloatArray,
    observed: BooleanArray,
  ): FloatArray =
    FloatArray(x.size) { t ->
      if (observed[t]) {
        val y = (x[t] - loc[t]) / npMax(scale[t], F32_EPS)
        asinh(y.toDouble()).toFloat()
      } else {
        0f
      }
    }

  // ---------------------------------------------------------------- output

  /** clamp_nonfinite(sinh(preds) * scale + loc), all float32 but sinh. */
  internal fun denormalize(preds: FloatArray, loc: Float, scale: Float): FloatArray {
    val out = FloatArray(preds.size) { StrictMath.sinh(preds[it].toDouble()).toFloat() * scale + loc }
    clampNonFinite(out)
    return out
  }

  /**
   * In place, per step of 21 quantiles: +inf becomes the step's largest finite value, -inf its
   * smallest; an infinity left (no finite value in the step) becomes 0. NaN stays.
   */
  internal fun clampNonFinite(values: FloatArray) {
    for (start in values.indices step QUANTILES) {
      var hi = Float.NEGATIVE_INFINITY
      var lo = Float.POSITIVE_INFINITY
      for (i in start until start + QUANTILES) {
        val v = values[i]
        if (v.isFinite()) {
          hi = npMax(hi, v)
          lo = npMin(lo, v)
        }
      }
      for (i in start until start + QUANTILES) {
        var v = values[i]
        if (v == Float.POSITIVE_INFINITY) {
          v = hi
        } else if (v == Float.NEGATIVE_INFINITY) {
          v = lo
        }
        values[i] = if (v.isInfinite()) 0f else v
      }
    }
  }

  /**
   * The official `_refine_block` for one row of [h] decoder patches: the quantiles move from the
   * anchor frame (the last context patch's loc / scale) to the advancing frame of each patch.
   */
  internal fun refineBlock(
    quantiles: FloatArray,
    context: FloatArray,
    observed: BooleanArray,
    h: Int,
  ): FloatArray {
    if (h <= 1) return quantiles
    val (loc, scale) = patchLocScale(context, observed)
    val loc0 = loc[loc.size - 1]
    val scale0 = scale[scale.size - 1]
    val sinhMedian = DoubleArray(h * PATCH) { t ->
      (quantiles[t * QUANTILES + MEDIAN].toDouble() - loc0.toDouble()) / scale0.toDouble()
    }
    val frames = advancingBlockFrames(context, observed, sinhMedian, h)
    val out = FloatArray(quantiles.size)
    for (p in 0 until h) {
      val frameLoc = if (p == 0) loc0 else frames.first[p].toFloat()
      val frameScale = if (p == 0) scale0 else frames.second[p].toFloat()
      for (j in 0 until PATCH) {
        val base = (p * PATCH + j) * QUANTILES
        for (i in 0 until QUANTILES) {
          val sinhNorm = (quantiles[base + i] - loc0) / scale0
          out[base + i] = sinhNorm * frameScale + frameLoc
        }
      }
    }
    clampNonFinite(out)
    return out
  }

  /**
   * The official `advancing_block_frames` for one row: per decoder patch the float64 (loc, scale)
   * of a Welford state that starts from the whole context and takes in each decoded patch's median.
   */
  internal fun advancingBlockFrames(
    context: FloatArray,
    observed: BooleanArray,
    sinhMedian: DoubleArray,
    h: Int,
  ): Pair<DoubleArray, DoubleArray> {
    val contextPatches = context.size / PATCH
    val n = contextPatches + h
    val count = DoubleArray(n)
    val mean = DoubleArray(n)
    val m2 = DoubleArray(n)
    val products = DoubleArray(PATCH)
    val weights = DoubleArray(PATCH)
    for (p in 0 until contextPatches) {
      for (j in 0 until PATCH) {
        val t = p * PATCH + j
        weights[j] = if (observed[t]) 1.0 else 0.0
        products[j] = context[t].toDouble() * weights[j]
      }
      val c = reduceSum(weights)
      var m = reduceSum(products) / npMax(c, 1.0)
      for (j in 0 until PATCH) {
        val d = context[p * PATCH + j].toDouble() - m
        products[j] = weights[j] * (d * d)
      }
      var s = reduceSum(products)
      if (c == 0.0) {
        m = 0.0
        s = 0.0
      }
      count[p] = c
      mean[p] = m
      m2[p] = s
    }
    // The decoded patches start empty (count 0); only the scan's length depends on them.
    cumulativeWelford(count, mean, m2)
    var c = count[contextPatches - 1]
    var m = mean[contextPatches - 1]
    var s = m2[contextPatches - 1]
    val offset = m
    val locs = DoubleArray(h)
    val scales = DoubleArray(h)
    val folded = DoubleArray(PATCH)
    for (p in 0 until h) {
      val variance = npMax(s / npMax(c - 1.0, 1.0), 0.0)
      val sc = npMax(Math.sqrt(variance), MIN_SCALE)
      locs[p] = m
      scales[p] = sc
      for (j in 0 until PATCH) {
        val value = sinhMedian[p * PATCH + j] * sc + m - offset
        folded[j] = clip(value, -FOLD_CLIP, FOLD_CLIP) + offset
      }
      val blockMean = reduceSum(folded) / PATCH.toDouble()
      for (j in 0 until PATCH) {
        val d = folded[j] - blockMean
        products[j] = d * d
      }
      val blockM2 = reduceSum(products)
      val combined = welfordCombine(c, m, s, PATCH.toDouble(), blockMean, blockM2)
      c = combined[0]
      m = combined[1]
      s = combined[2]
    }
    return locs to scales
  }

  /** The official Welford merge of state a with state b: (w, mean, m2). */
  internal fun welfordCombine(
    wA: Double,
    meanA: Double,
    m2A: Double,
    wB: Double,
    meanB: Double,
    m2B: Double,
  ): DoubleArray {
    val w = wA + wB
    val wSafe = npMax(w, F64_TINY)
    val delta = meanB - meanA
    val mean = meanA + delta * wB / wSafe
    val m2 = m2A + m2B + delta * delta * wA * wB / wSafe
    return if (w == 0.0) doubleArrayOf(w, 0.0, 0.0) else doubleArrayOf(w, mean, m2)
  }

  /**
   * In place, the official Hillis-Steele scan of Welford states: after it, entry i holds the merge
   * of entries 0..i. Each round merges every entry with the one d before it (an empty state before
   * the start), d = 1, 2, 4 … while d < n.
   */
  internal fun cumulativeWelford(w: DoubleArray, mean: DoubleArray, m2: DoubleArray) {
    val n = w.size
    var d = 1
    while (d < n) {
      val w0 = w.copyOf()
      val mean0 = mean.copyOf()
      val m20 = m2.copyOf()
      for (i in 0 until n) {
        val wA = if (i >= d) w0[i - d] else 0.0
        val meanA = if (i >= d) mean0[i - d] else 0.0
        val m2A = if (i >= d) m20[i - d] else 0.0
        val merged = welfordCombine(wA, meanA, m2A, w0[i], mean0[i], m20[i])
        w[i] = merged[0]
        mean[i] = merged[1]
        m2[i] = merged[2]
      }
      d *= 2
    }
  }

  /**
   * Clamps a row's quantiles to [min - k s, max + k s] of its observed context, s the context's
   * last scale ([lastScale], float32).
   */
  internal fun cap(
    quantiles: FloatArray,
    rowValues: FloatArray,
    observed: BooleanArray,
    lastScale: Float,
  ): FloatArray {
    var contextMax = Float.NEGATIVE_INFINITY
    var contextMin = Float.POSITIVE_INFINITY
    for (t in rowValues.indices) {
      if (observed[t]) {
        contextMax = npMax(contextMax, rowValues[t])
        contextMin = npMin(contextMin, rowValues[t])
      }
    }
    if (contextMax == Float.NEGATIVE_INFINITY) contextMax = 0f
    if (contextMin == Float.POSITIVE_INFINITY) contextMin = 0f
    val anchor = lastScale * CAP_K
    val lo = contextMin - anchor
    val hi = contextMax + anchor
    return FloatArray(quantiles.size) { npMin(npMax(quantiles[it], lo), hi) }
  }

  private fun checkNoNan(values: FloatArray) {
    val nan = values.count { it.isNaN() }
    if (nan > 0) throw ForecastException("forecast produced $nan NaN values")
  }

  /** Whether every observed value is ≥ 0 (and at least one is observed). */
  internal fun nonNegativeContext(inputs: FloatArray, observed: BooleanArray): Boolean {
    var any = false
    for (t in inputs.indices) {
      if (observed[t] && inputs[t].isFinite()) {
        any = true
        if (!(inputs[t] >= 0f)) return false
      }
    }
    return any
  }

  // ---------------------------------------------------------------- NumPy arithmetic

  /**
   * NumPy's float64 sum of one axis of 8 to 128 values: eight running sums (values i, i + 8, …),
   * combined as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then the rest in order, added
   * to the reduction's start value 0.
   */
  internal fun reduceSum(values: DoubleArray): Double {
    val n = values.size
    require(n in 8..128) { "pairwise sum of $n values" }
    val r = DoubleArray(8) { values[it] }
    var i = 8
    while (i < n - n % 8) {
      for (j in 0 until 8) r[j] += values[i + j]
      i += 8
    }
    var sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    while (i < n) {
      sum += values[i]
      i++
    }
    return 0.0 + sum
  }

  /**
   * np.maximum as NumPy 2.5 computes it, measured on every broadcast pattern and on reductions:
   * IEEE 754 maximum, NaN wins and -0.0 < +0.0 — the rule of [Math.max].
   */
  internal fun npMax(a: Double, b: Double): Double = Math.max(a, b)

  internal fun npMax(a: Float, b: Float): Float = Math.max(a, b)

  /** np.minimum: IEEE 754 minimum, NaN wins and -0.0 < +0.0 — the rule of [Math.min]. */
  internal fun npMin(a: Float, b: Float): Float = Math.min(a, b)

  /** np.clip: the value stays unless it is strictly outside [lo, hi] (so NaN and -0.0 stay). */
  internal fun clip(v: Float, lo: Float, hi: Float): Float =
    if (v < lo) lo else if (v > hi) hi else v

  internal fun clip(v: Double, lo: Double, hi: Double): Double =
    if (v < lo) lo else if (v > hi) hi else v

  /**
   * asinh in float64, fdlibm's `s_asinh.c` on [StrictMath] (Java has no asinh): accurate to about
   * one ulp, like the C library NumPy calls.
   */
  internal fun asinh(x: Double): Double {
    val bits = java.lang.Double.doubleToRawLongBits(x)
    val high = (bits ushr 32).toInt()
    val magnitude = high and 0x7fffffff
    if (magnitude >= 0x7ff00000) return x + x
    if (magnitude < 0x3e300000) return x
    val absX = Math.abs(x)
    val w =
      when {
        magnitude > 0x41b00000 -> StrictMath.log(absX) + LN2
        magnitude > 0x40000000 ->
          StrictMath.log(2.0 * absX + 1.0 / (Math.sqrt(x * x + 1.0) + absX))
        else -> {
          val t = x * x
          StrictMath.log1p(absX + t / (1.0 + Math.sqrt(1.0 + t)))
        }
      }
    return if (high > 0) w else -w
  }

  private const val LN2 = 6.93147180559945286227e-01
}
