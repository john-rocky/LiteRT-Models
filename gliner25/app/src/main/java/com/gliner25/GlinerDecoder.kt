package com.gliner25

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln1p
import kotlin.math.sqrt
import org.json.JSONObject

/**
 * Float32 host continuation for the published GLiNER2.5 Small shared-pool graph.
 *
 * Ports gliner2 2.0.0 `models/boundary/pool.py:DocumentCandidatePool.forward` and
 * `SharedPoolScorer.forward`, `content.py:SpanContentPooler.pool`,
 * `engine.py:BoundaryExtractor._decode_entities` and `inference/overlap.py:resolve_overlaps`. The
 * checkpoint uses the shared pool, so the historical per-query proposer is not executed. Stable
 * top-k ordering, duplicate merging and float32 operations preserve its decisions.
 *
 * The graph returns `[1,1,1,1108*T+4574]` float32 values containing 17 logical tensors. Contracts
 * describe windows N/T = 128/48, 256/192 and 512/384; all five label queries share one pool.
 * Results use half-open Unicode code-point offsets, matching Python rather than UTF-16.
 */
class GlinerDecoder(
  private val hostAssets: File,
  configJson: String = File(hostAssets, "config.json").readText(),
) {
  /**
   * A `BoundaryExtractor._decode_entities` result: fixed label, original text, half-open Unicode
   * code-point offsets and sigmoid confidence after thresholding and flat overlap resolution. The
   * original text is retained because tokenizer normalization is lossy.
   */
  data class Span(
    val label: String,
    val text: String,
    /** Half-open Unicode code-point offsets, as in Python. */
    val start: Int,
    val end: Int,
    val confidence: Float,
  )

  /** One graph-contract tensor; offsets and lengths count floats, not bytes. */
  data class Slice(val name: String, val shape: IntArray, val offset: Int, val elements: Int)

  /**
   * A `DocumentCandidatePool.forward` entry with half-open word boundaries and its float32 prior.
   */
  data class PoolCandidate(val start: Int, val end: Int, val compatibility: Float)

  /** Ordered shared-pool candidates and `[candidateCount,5]` `SharedPoolScorer.forward` logits. */
  data class Trace(val candidates: List<PoolCandidate>, val logits: Array<FloatArray>)

  /**
   * Zero-copy view of the 17 graph-contract tensors consumed by the published
   * `host_decoder.py:decode` continuation of gliner2 2.0.0. Named slices avoid relying on
   * incidental FlatBuffer tensor order when reconstructing upstream sparse inputs.
   */
  class Packed internal constructor(val values: FloatArray, val slices: Map<String, Slice>) {
    /** Finds a logical tensor by its graph-contract name; absent tensors are contract errors. */
    fun slice(name: String): Slice = requireNotNull(slices[name]) { "Missing packed slice $name" }

    /**
     * Reads a flattened float32 element for the sparse continuation without reshaping or copying.
     */
    operator fun get(name: String, index: Int): Float {
      val slice = slice(name)
      require(index in 0 until slice.elements) { "$name index $index out of range" }
      return values[slice.offset + index]
    }

    /**
     * Copies one row-major feature row, bounded by its logical tensor rather than the full buffer.
     */
    fun row(name: String, row: Int, width: Int): FloatArray {
      val slice = slice(name)
      val from = slice.offset + row * width
      require(from >= slice.offset && from + width <= slice.offset + slice.elements)
      return values.copyOfRange(from, from + width)
    }
  }

  private data class Tensor(val shape: IntArray, val values: FloatArray)

  private data class RankedPair(val key: Int, val priority: Float, val valid: Boolean)

  private data class Scored(val score: Float, val start: Int, val end: Int, val index: Int)

  private data class Selection(val score: Double, val indices: List<Int>)

  private val parameters = readSafetensors(File(hostAssets, "sparse_decoder_fp32.safetensors"))
  private val contracts =
    listOf(128, 256, 512).associateWith { sequence ->
      JSONObject(File(hostAssets, "graph_contract_s$sequence.json").readText())
    }
  private val scorerPrefix = "boundary_head.shared_pool_scorer."
  private val poolBoundaryTopK: Int
  private val poolSize: Int
  private val minPoolPerQuery: Int

  init {
    val config = JSONObject(configJson).getJSONObject("boundary_head")
    require(config.getString("candidate_pool") == "shared")
    require(config.getInt("boundary_dim") == 128 && config.getInt("pair_dim") == 128)
    require(config.getInt("content_dim") == 64 && config.getBoolean("enable_span_content"))
    require(!config.getBoolean("content_soft_max_pool"))
    require(
      config.getInt("candidate_attention_layers") == 0 &&
        config.getInt("query_attention_layers") == 0
    )
    require(config.getBoolean("use_inside_evidence") && config.getBoolean("enable_abstention"))
    require(!config.getBoolean("adaptive_threshold"))
    require(
      config.getDouble("pair_temperature") == 1.0 && config.getDouble("abstention_threshold") == 0.5
    )
    require(config.getString("overlap_policy") == "flat")
    poolBoundaryTopK = config.getInt("pool_boundary_top_k")
    poolSize = config.getInt("pool_size")
    minPoolPerQuery = config.getInt("min_pool_per_query")
    val expected =
      mapOf(
        "boundary_head.candidate_encoder.bias" to intArrayOf(384),
        "boundary_head.candidate_encoder.weight" to intArrayOf(384, 256),
        "${scorerPrefix}candidate_norm.bias" to intArrayOf(128),
        "${scorerPrefix}candidate_norm.weight" to intArrayOf(128),
        "${scorerPrefix}content_pooler.layer_norm.bias" to intArrayOf(64),
        "${scorerPrefix}content_pooler.layer_norm.weight" to intArrayOf(64),
        "${scorerPrefix}content_projection.bias" to intArrayOf(128),
        "${scorerPrefix}content_projection.weight" to intArrayOf(128, 64),
        "${scorerPrefix}film_output.0.bias" to intArrayOf(64),
        "${scorerPrefix}film_output.0.weight" to intArrayOf(64, 128),
        "${scorerPrefix}film_output.3.bias" to intArrayOf(1),
        "${scorerPrefix}film_output.3.weight" to intArrayOf(1, 64),
        "${scorerPrefix}length_projection.bias" to intArrayOf(128),
        "${scorerPrefix}length_projection.weight" to intArrayOf(128, 3),
        "${scorerPrefix}prior_projection.bias" to intArrayOf(128),
        "${scorerPrefix}prior_projection.weight" to intArrayOf(128, 1),
      )
    require(parameters.keys == expected.keys) { "Unexpected sparse parameter inventory" }
    expected.forEach { (name, shape) ->
      require(parameters.getValue(name).shape.contentEquals(shape)) { name }
    }
  }

  /**
   * Reconstructs the published `host_decoder.py:decode` logical outputs from one flat buffer. Its
   * length selects T = 48, 192 or 384. All 17 slices must be contiguous float32 tensors; nonfinite
   * values are rejected before they can influence top-k ordering or confidence.
   */
  fun unpack(values: FloatArray): Packed {
    val contract =
      contracts.values.singleOrNull {
        it.getJSONObject("physical_output").getJSONArray("shape").getInt(3) == values.size
      } ?: error("No published graph contract for ${values.size} floats")
    require(values.all { it.isFinite() }) { "Packed graph output contains NaN or infinity" }
    val outputs = contract.getJSONArray("logical_outputs")
    val slices = linkedMapOf<String, Slice>()
    var next = 0
    for (i in 0 until outputs.length()) {
      val item = outputs.getJSONObject(i)
      val shapeJson = item.getJSONArray("shape")
      val shape = IntArray(shapeJson.length()) { shapeJson.getInt(it) }
      val slice =
        Slice(item.getString("name"), shape, item.getInt("offset"), item.getInt("elements"))
      require(item.getString("dtype") == "float32" && slice.offset == next)
      require(shape.fold(1) { a, b -> a * b } == slice.elements)
      require(slices.put(slice.name, slice) == null)
      next += slice.elements
    }
    require(slices.size == 17 && next == values.size)
    return Packed(values, slices)
  }

  /**
   * Continues `HostRuntime.decode` through gliner2 2.0.0's shared pool and
   * `BoundaryExtractor._decode_entities`, reusing prepared code-point mappings. Confidence defaults
   * to 0.5; upstream null abstention and flat overlap policy also apply.
   */
  fun decode(
    packed: FloatArray,
    input: GlinerInputs.Prepared,
    threshold: Float = 0.5f,
  ): List<Span> = decode(packed, input.text, input.words, threshold)

  /**
   * Ports gliner2 2.0.0 `BoundaryExtractor._decode_entities` and `resolve_overlaps` for a caller
   * retaining the original text and word map separately. `packed` is the single float32 output;
   * [words] supplies half-open Unicode code-point offsets, not UTF-16 indices. Stable pool
   * selection and the upstream flat interval policy are applied before returning spans.
   */
  fun decode(
    packed: FloatArray,
    text: String,
    words: List<GlinerInputs.Word>,
    threshold: Float = 0.5f,
  ): List<Span> {
    require(threshold.isFinite() && threshold in 0f..1f)
    val outputs = unpack(packed)
    val trace = trace(outputs, words.size)
    val result = ArrayList<Span>()
    for (query in LABELS.indices) {
      // Upstream abstains strictly above 0.5, after float32 sigmoid.
      if (sigmoid(outputs["null_logits", query]) > 0.5f) {
        continue
      }
      val scored = ArrayList<Scored>()
      trace.candidates.forEachIndexed { index, candidate ->
        val probability = sigmoid(trace.logits[index][query])
        if (probability >= threshold) {
          scored += Scored(probability, candidate.start, candidate.end, index)
        }
      }
      for (candidate in resolveFlat(scored)) {
        if (candidate.start < 0 || candidate.start >= candidate.end || candidate.end > words.size) {
          continue
        }
        val start = words[candidate.start].start
        val end = words[candidate.end - 1].end
        val surface =
          text.substring(text.offsetByCodePoints(0, start), text.offsetByCodePoints(0, end)).trim()
        if (surface.isNotEmpty()) {
          result += Span(LABELS[query], surface, start, end, candidate.score)
        }
      }
    }
    return result
  }

  /**
   * Exposes `DocumentCandidatePool.forward` ordering and `SharedPoolScorer.forward` logits before
   * sigmoid, abstention or overlap filtering. This preserves the intermediate evidence needed to
   * distinguish candidate-selection differences from final confidence differences.
   */
  fun trace(packed: FloatArray, wordCount: Int): Trace = trace(unpack(packed), wordCount)

  private fun trace(outputs: Packed, wordCount: Int): Trace {
    val boundaryCount = outputs.slice("pool_start").shape[1]
    require(wordCount in 0 until boundaryCount)
    val pool = buildPool(outputs, wordCount, boundaryCount)
    val logits = Array(pool.size) { FloatArray(LABELS.size) }
    val scale = sqrt(128f)
    pool.forEachIndexed { index, candidate ->
      val start = candidate.start
      val end = candidate.end
      val length = (end - start).coerceAtLeast(1).toFloat()
      val lengthFeatures =
        floatArrayOf(ln1p(length), length / wordCount.coerceAtLeast(1), 1f / sqrt(length))
      val lengthRep = linear("length_projection", lengthFeatures)
      val priorRep = linear("prior_projection", floatArrayOf(candidate.compatibility))
      val content =
        FloatArray(64) { channel ->
          (outputs["content_prefix", end * 64 + channel] -
            outputs["content_prefix", start * 64 + channel]) / length
        }
      val contentRep = linear("content_projection", layerNorm("content_pooler.layer_norm", content))
      val feature =
        FloatArray(128) { channel ->
          var value =
            outputs["score_start", start * 128 + channel] +
              outputs["score_end", end * 128 + channel]
          value += lengthRep[channel]
          value += priorRep[channel]
          value + contentRep[channel]
        }
      val normalized = layerNorm("candidate_norm", feature)
      for (query in LABELS.indices) {
        var score =
          dot(normalized, outputs.values, outputs.slice("score_query").offset + query * 128) / scale
        val conditioned =
          FloatArray(128) { channel ->
            normalized[channel] * (1f + outputs["film", query * 256 + channel]) +
              outputs["film", query * 256 + 128 + channel]
          }
        val hidden = linear("film_output.0", conditioned)
        for (channel in hidden.indices) {
          hidden[channel] = gelu(hidden[channel])
        }
        score += linear("film_output.3", hidden)[0]
        score += outputs["start_logits", query * boundaryCount + start]
        score += outputs["end_logits", query * boundaryCount + end]
        var interval =
          outputs["inside_prefix", query * boundaryCount + end] -
            outputs["inside_prefix", query * boundaryCount + start]
        interval += outputs["inside_prefix_mean", query] * (end - start).toFloat()
        score += interval / sqrt(length)
        require(score.isFinite()) { "Sparse decoder produced a nonfinite logit" }
        logits[index][query] = score
      }
    }
    return Trace(pool, logits)
  }

  private fun buildPool(outputs: Packed, wordCount: Int, n: Int): List<PoolCandidate> {
    val unionStart = FloatArray(n) { MASK_LOGIT }
    val unionEnd = FloatArray(n) { MASK_LOGIT }
    for (boundary in 0..wordCount) {
      unionStart[boundary] = LABELS.indices.maxOf { outputs["start_logits", it * n + boundary] }
      unionEnd[boundary] = LABELS.indices.maxOf { outputs["end_logits", it * n + boundary] }
    }
    // torch.sort(descending=True, stable=True): equal scores retain boundary index order.
    val starts =
      (0 until n).sortedWith(compareByDescending<Int> { unionStart[it] }).take(poolBoundaryTopK)
    val ends =
      (0 until n).sortedWith(compareByDescending<Int> { unionEnd[it] }).take(poolBoundaryTopK)
    val count = starts.size * ends.size
    val pairStarts = IntArray(count)
    val pairEnds = IntArray(count)
    val compatibility = FloatArray(count)
    val globalScores = FloatArray(count)
    val valid = BooleanArray(count)
    val poolStart = outputs.slice("pool_start").offset
    val poolEnd = outputs.slice("pool_end").offset
    val scale = sqrt(128f)
    for (s in starts.indices) {
      for (e in ends.indices) {
        val i = s * ends.size + e
        val sValid = starts[s] <= wordCount
        val eValid = ends[e] <= wordCount
        val start =
          if (sValid) {
            starts[s]
          } else {
            0
          }
        val end =
          if (eValid) {
            ends[e]
          } else {
            0
          }
        pairStarts[i] = start
        pairEnds[i] = end
        valid[i] = sValid && eValid && end > start
        compatibility[i] =
          dot(outputs.values, poolStart + start * 128, outputs.values, poolEnd + end * 128, 128) /
            scale
        globalScores[i] = (compatibility[i] + unionStart[start]) + unionEnd[end]
      }
    }
    val quota = minPoolPerQuery.coerceAtMost(count)
    val all = ArrayList<RankedPair>(LABELS.size * quota + count)
    for (query in LABELS.indices) {
      val perQuery =
        FloatArray(count) { i ->
          if (valid[i]) {
            (outputs["start_logits", query * n + pairStarts[i]] +
              outputs["end_logits", query * n + pairEnds[i]]) + compatibility[i]
          } else {
            MASK_LOGIT
          }
        }
      val ranked = (0 until count).sortedWith(compareByDescending<Int> { perQuery[it] }).take(quota)
      ranked.forEachIndexed { rank, i ->
        all +=
          RankedPair(pairStarts[i] * n + pairEnds[i], -MASK_LOGIT * 0.5f + (quota - rank), valid[i])
      }
    }
    for (i in 0 until count) {
      all += RankedPair(pairStarts[i] * n + pairEnds[i], globalScores[i], valid[i])
    }
    // _deduplicate_pool: stable priority sort, stable key sort, first occurrence,
    // then stable priority sort again. The last score ties therefore sort by key.
    val byScore =
      all
        .map {
          if (it.valid) {
            it
          } else {
            RankedPair(n * n, MASK_LOGIT, false)
          }
        }
        .sortedWith(compareByDescending<RankedPair> { it.priority })
    val byKey = byScore.sortedBy { it.key }
    val unique = ArrayList<RankedPair>(byKey.size)
    var previousKey = -1
    for (row in byKey) {
      val keep = row.valid && row.key != previousKey
      unique +=
        if (keep) {
          row
        } else {
          RankedPair(row.key, MASK_LOGIT, false)
        }
      previousKey = row.key
    }
    return unique
      .sortedWith(compareByDescending<RankedPair> { it.priority })
      .take(poolSize)
      .filter { it.valid }
      .map { row ->
        val start = row.key / n
        val end = row.key % n
        PoolCandidate(
          start,
          end,
          dot(outputs.values, poolStart + start * 128, outputs.values, poolEnd + end * 128, 128) /
            scale,
        )
      }
  }

  private fun linear(name: String, input: FloatArray): FloatArray {
    val weight = parameters.getValue("$scorerPrefix$name.weight")
    val bias = parameters.getValue("$scorerPrefix$name.bias").values
    require(weight.shape[1] == input.size)
    return FloatArray(weight.shape[0]) { row ->
      dot(input, weight.values, row * input.size) + bias[row]
    }
  }

  private fun layerNorm(name: String, input: FloatArray): FloatArray {
    val weight = parameters.getValue("$scorerPrefix$name.weight").values
    val bias = parameters.getValue("$scorerPrefix$name.bias").values
    var sum = 0f
    for (value in input) {
      sum += value
    }
    val mean = sum / input.size
    var variance = 0f
    for (value in input) {
      val delta = value - mean
      variance += delta * delta
    }
    val inverseStd = 1f / sqrt(variance / input.size + 1e-5f)
    return FloatArray(input.size) { (input[it] - mean) * inverseStd * weight[it] + bias[it] }
  }

  private fun resolveFlat(candidates: List<Scored>): List<Scored> {
    val rank =
      compareByDescending<Scored> { it.score }
        .thenBy { it.start }
        .thenBy { it.end }
        .thenBy { it.index }
    val ranked = candidates.sortedWith(rank)
    val seen = HashSet<Pair<Int, Int>>()
    val byEnd =
      ranked
        .filter { seen.add(it.start to it.end) }
        .sortedWith(
          compareBy<Scored> { it.end }
            .thenBy { it.start }
            .thenByDescending { it.score }
            .thenBy { it.index }
        )
    val best = ArrayList<Selection>(byEnd.size + 1)
    best += Selection(0.0, emptyList())
    for (i in byEnd.indices) {
      val item = byEnd[i]
      var predecessor = i - 1
      while (predecessor >= 0 && byEnd[predecessor].end > item.start) {
        predecessor--
      }
      val before = best[predecessor + 1]
      // Upstream overlap resolution uses Python floats (double) for DP totals.
      val withItem = Selection(before.score + item.score.toDouble(), before.indices + i)
      val withoutItem = best[i]
      val chooseWith =
        when {
          withItem.score != withoutItem.score -> withItem.score > withoutItem.score
          withItem.indices.size != withoutItem.indices.size ->
            withItem.indices.size > withoutItem.indices.size
          else -> {
            val a = withItem.indices.map { byEnd[it] }.sortedWith(rank)
            val b = withoutItem.indices.map { byEnd[it] }.sortedWith(rank)
            a.indices
              .firstOrNull { rank.compare(a[it], b[it]) != 0 }
              ?.let { rank.compare(a[it], b[it]) < 0 } ?: false
          }
        }
      best +=
        if (chooseWith) {
          withItem
        } else {
          withoutItem
        }
    }
    return best.last().indices.map { byEnd[it] }.sortedWith(rank)
  }

  companion object {
    val LABELS = listOf("person", "organization", "location", "product", "date")
    private const val MASK_LOGIT = -10000f

    /**
     * Reads the published host runtime's raw little-endian float32 output format for offline
     * decoding. The logical shape is recovered by [unpack], rather than stored in this file.
     */
    fun readPacked(file: File): FloatArray {
      val bytes = file.readBytes()
      require(bytes.size % 4 == 0)
      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun readSafetensors(file: File): Map<String, Tensor> {
      val bytes = file.readBytes()
      require(bytes.size >= 8)
      val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      val headerLength = buffer.long
      require(headerLength in 2..(bytes.size - 8).toLong())
      val header = JSONObject(String(bytes, 8, headerLength.toInt(), Charsets.UTF_8))
      val dataStart = 8 + headerLength.toInt()
      val result = linkedMapOf<String, Tensor>()
      for (name in header.keys()) {
        if (name == "__metadata__") {
          continue
        }
        val tensor = header.getJSONObject(name)
        require(tensor.getString("dtype") == "F32") { "$name is not float32" }
        val shapeJson = tensor.getJSONArray("shape")
        val shape = IntArray(shapeJson.length()) { shapeJson.getInt(it) }
        val offsets = tensor.getJSONArray("data_offsets")
        val start = offsets.getInt(0)
        val end = offsets.getInt(1)
        val size = shape.fold(1) { a, b -> Math.multiplyExact(a, b) }
        require(start >= 0 && end - start == size * 4 && dataStart + end <= bytes.size)
        buffer.position(dataStart + start)
        val values = FloatArray(size) { buffer.float }
        require(values.all { it.isFinite() }) { "$name contains nonfinite weights" }
        result[name] = Tensor(shape, values)
      }
      return result
    }

    private fun dot(a: FloatArray, b: FloatArray, bOffset: Int): Float =
      dot(a, 0, b, bOffset, a.size)

    private fun dot(a: FloatArray, aOffset: Int, b: FloatArray, bOffset: Int, count: Int): Float {
      // Four independent float32 accumulators limit scalar summation error.
      var s0 = 0f
      var s1 = 0f
      var s2 = 0f
      var s3 = 0f
      var i = 0
      while (i + 3 < count) {
        s0 += a[aOffset + i] * b[bOffset + i]
        s1 += a[aOffset + i + 1] * b[bOffset + i + 1]
        s2 += a[aOffset + i + 2] * b[bOffset + i + 2]
        s3 += a[aOffset + i + 3] * b[bOffset + i + 3]
        i += 4
      }
      var result = (s0 + s1) + (s2 + s3)
      while (i < count) {
        result += a[aOffset + i] * b[bOffset + i]
        i++
      }
      return result
    }

    private fun sigmoid(value: Float): Float = 1f / (1f + exp(-value))

    /** Exact erf GELU, not the tanh GELU approximation. */
    private fun gelu(value: Float): Float {
      val erf = erf((value * 0.7071067811865476f).toDouble()).toFloat()
      return (value * 0.5f) * (1f + erf)
    }

    /** Convergent erf power series; double evaluation supplies a rounded float special function. */
    private fun erf(value: Double): Double {
      val x = kotlin.math.abs(value)
      // Beyond 4, erf already rounds to 1 in float32.
      if (x >= 4.0) {
        return if (value < 0) {
          -1.0
        } else {
          1.0
        }
      }
      val square = x * x
      var powerOverFactorial = x
      var sum = x
      for (n in 1..100) {
        powerOverFactorial *= -square / n
        val term = powerOverFactorial / (2 * n + 1)
        sum += term
        if (kotlin.math.abs(term) < 1e-17) {
          break
        }
      }
      val result = sum * 1.1283791670955126
      return if (value < 0) {
        -result
      } else {
        result
      }
    }
  }
}
