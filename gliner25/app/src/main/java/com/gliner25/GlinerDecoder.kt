package com.gliner25

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
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
  private val profileEnabled: Boolean = false,
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

  private val stageNanos = LongArray(PROFILE_STAGES.size)

  /** Diagnostic stage durations for the last decode; empty unless explicitly enabled by a test. */
  fun profileMilliseconds(): Map<String, Double> =
    if (profileEnabled) {
      PROFILE_STAGES.indices.associate { PROFILE_STAGES[it] to stageNanos[it] / 1_000_000.0 }
    } else {
      emptyMap()
    }

  private fun tick(): Long =
    if (profileEnabled) {
      System.nanoTime()
    } else {
      0L
    }

  private fun record(stage: Int, start: Long) {
    if (profileEnabled) {
      stageNanos[stage] += System.nanoTime() - start
    }
  }

  private inline fun <T> measured(stage: Int, block: () -> T): T {
    val start = tick()
    val result = block()
    record(stage, start)
    return result
  }

  private data class Tensor(val shape: IntArray, val values: FloatArray)

  private data class Projection(
    val inputSize: Int,
    val outputSize: Int,
    val weight: FloatArray,
    val bias: FloatArray,
  )

  private data class Normalization(val weight: FloatArray, val bias: FloatArray)

  private data class Scored(val score: Float, val start: Int, val end: Int, val index: Int)

  private data class Selection(val score: Double, val indices: List<Int>)

  private val parameters = readSafetensors(File(hostAssets, "sparse_decoder_fp32.safetensors"))
  private val contracts =
    listOf(128, 256, 512).associateWith { sequence ->
      JSONObject(File(hostAssets, "graph_contract_s$sequence.json").readText())
    }
  private val scorerPrefix = "boundary_head.shared_pool_scorer."
  private val lengthProjection = projection("length_projection")
  private val priorProjection = projection("prior_projection")
  private val contentProjection = projection("content_projection")
  private val filmProjection = projection("film_output.0")
  private val filmOutput = projection("film_output.3")
  private val contentNorm = normalization("content_pooler.layer_norm")
  private val candidateNorm = normalization("candidate_norm")
  private val layouts =
    contracts.values.associate { contract ->
      contract.getJSONObject("physical_output").getJSONArray("shape").getInt(3) to layout(contract)
    }
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
    val slices =
      layouts[values.size] ?: error("No published graph contract for ${values.size} floats")
    // Primitive comparisons also reject NaN; avoid two library calls for every packed element.
    for (value in values) {
      require(value >= -Float.MAX_VALUE && value <= Float.MAX_VALUE) {
        "Packed graph output contains NaN or infinity"
      }
    }
    return Packed(values, slices)
  }

  private fun layout(contract: JSONObject): Map<String, Slice> {
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
    require(
      slices.size == 17 &&
        next == contract.getJSONObject("physical_output").getJSONArray("shape").getInt(3)
    )
    return java.util.Collections.unmodifiableMap(slices)
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
    if (profileEnabled) {
      stageNanos.fill(0L)
    }
    val outputs = measured(0) { unpack(packed) }
    val trace = trace(outputs, words.size)
    val result = ArrayList<Span>()
    for (query in LABELS.indices) {
      val thresholdStart = tick()
      // Upstream abstains strictly above 0.5, after float32 sigmoid.
      if (sigmoid(outputs["null_logits", query]) > 0.5f) {
        record(8, thresholdStart)
        continue
      }
      val scored = ArrayList<Scored>()
      trace.candidates.forEachIndexed { index, candidate ->
        val probability = sigmoid(trace.logits[index][query])
        if (probability >= threshold) {
          scored += Scored(probability, candidate.start, candidate.end, index)
        }
      }
      val selected = resolveFlat(scored)
      record(8, thresholdStart)
      val offsetStart = tick()
      for (candidate in selected) {
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
      record(9, offsetStart)
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
    val values = outputs.values
    val contentOffset = outputs.slice("content_prefix").offset
    val startOffset = outputs.slice("score_start").offset
    val endOffset = outputs.slice("score_end").offset
    val queryOffset = outputs.slice("score_query").offset
    val filmOffset = outputs.slice("film").offset
    val startLogits = outputs.slice("start_logits").offset
    val endLogits = outputs.slice("end_logits").offset
    val insideOffset = outputs.slice("inside_prefix").offset
    val insideMean = outputs.slice("inside_prefix_mean").offset
    val features = FloatArray(pool.size * 128)
    val hidden = FloatArray(pool.size * LABELS.size * 64)
    val scale = sqrt(128f)

    // Candidates are independent. Each worker keeps the original scalar reduction order, while
    // reusing its scratch arrays instead of allocating a projection result for every query.
    measured(3) {
      parallel(pool.size) { first, limit ->
        val lengthFeatures = FloatArray(3)
        val prior = FloatArray(1)
        val lengthRep = FloatArray(128)
        val priorRep = FloatArray(128)
        val content = FloatArray(64)
        val contentRep = FloatArray(128)
        for (index in first until limit) {
          val candidate = pool[index]
          val start = candidate.start
          val end = candidate.end
          val length = (end - start).coerceAtLeast(1).toFloat()
          lengthFeatures[0] = ln1p(length)
          lengthFeatures[1] = length / wordCount.coerceAtLeast(1)
          lengthFeatures[2] = 1f / sqrt(length)
          prior[0] = candidate.compatibility
          linearInto(lengthProjection, lengthFeatures, 0, lengthRep, 0)
          linearInto(priorProjection, prior, 0, priorRep, 0)
          for (channel in 0 until 64) {
            content[channel] =
              (values[contentOffset + end * 64 + channel] -
                values[contentOffset + start * 64 + channel]) / length
          }
          normalizeInPlace(contentNorm, content, 0, 64)
          linearInto(contentProjection, content, 0, contentRep, 0)
          for (channel in 0 until 128) {
            var value =
              values[startOffset + start * 128 + channel] + values[endOffset + end * 128 + channel]
            value += lengthRep[channel]
            value += priorRep[channel]
            features[index * 128 + channel] = value + contentRep[channel]
          }
          normalizeInPlace(candidateNorm, features, index * 128, 128)
        }
      }
    }
    measured(4) {
      parallel(pool.size * LABELS.size) { first, limit ->
        val conditioned = FloatArray(128)
        for (row in first until limit) {
          val candidate = row / LABELS.size
          val query = row % LABELS.size
          for (channel in 0 until 128) {
            conditioned[channel] =
              features[candidate * 128 + channel] *
                (1f + values[filmOffset + query * 256 + channel]) +
                values[filmOffset + query * 256 + 128 + channel]
          }
          linearInto(filmProjection, conditioned, 0, hidden, row * 64)
        }
      }
    }
    // Keep the exact existing erf series and its stopping criterion. Only independent elements
    // execute concurrently; no approximation, float16 conversion or cross-worker reduction occurs.
    measured(5) {
      parallel(hidden.size) { first, limit ->
        for (index in first until limit) {
          hidden[index] = gelu(hidden[index])
        }
      }
    }
    measured(7) {
      parallel(pool.size) { first, limit ->
        for (index in first until limit) {
          val candidate = pool[index]
          val start = candidate.start
          val end = candidate.end
          val length = (end - start).coerceAtLeast(1).toFloat()
          for (query in LABELS.indices) {
            var score = dot(features, index * 128, values, queryOffset + query * 128, 128) / scale
            score +=
              dot(hidden, (index * LABELS.size + query) * 64, filmOutput.weight, 0, 64) +
                filmOutput.bias[0]
            score += values[startLogits + query * boundaryCount + start]
            score += values[endLogits + query * boundaryCount + end]
            var interval =
              values[insideOffset + query * boundaryCount + end] -
                values[insideOffset + query * boundaryCount + start]
            interval += values[insideMean + query] * (end - start).toFloat()
            score += interval / sqrt(length)
            require(score.isFinite()) { "Sparse decoder produced a nonfinite logit" }
            logits[index][query] = score
          }
        }
      }
    }
    return Trace(pool, logits)
  }

  private fun buildPool(outputs: Packed, wordCount: Int, n: Int): List<PoolCandidate> {
    val poolStartTime = tick()
    val values = outputs.values
    val startLogits = outputs.slice("start_logits").offset
    val endLogits = outputs.slice("end_logits").offset
    val unionStart = FloatArray(n) { MASK_LOGIT }
    val unionEnd = FloatArray(n) { MASK_LOGIT }
    for (boundary in 0..wordCount) {
      var start = values[startLogits + boundary]
      var end = values[endLogits + boundary]
      for (query in 1 until LABELS.size) {
        start = maxOf(start, values[startLogits + query * n + boundary])
        end = maxOf(end, values[endLogits + query * n + boundary])
      }
      unionStart[boundary] = start
      unionEnd[boundary] = end
    }
    // Stable descending order retains boundary indices on ties, including the masked tail.
    val starts = stableByScore(IntArray(n) { it }, unionStart).copyOf(minOf(n, poolBoundaryTopK))
    val ends = stableByScore(IntArray(n) { it }, unionEnd).copyOf(minOf(n, poolBoundaryTopK))
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
          dot(values, poolStart + start * 128, values, poolEnd + end * 128, 128) / scale
        globalScores[i] = (compatibility[i] + unionStart[start]) + unionEnd[end]
      }
    }
    record(1, poolStartTime)
    val proposerStart = tick()
    val quota = minPoolPerQuery.coerceAtMost(count)
    val total = LABELS.size * quota + count
    val keys = IntArray(total)
    val priorities = FloatArray(total)
    val keep = BooleanArray(total)
    var next = 0
    val perQuery = FloatArray(count)
    for (query in LABELS.indices) {
      for (i in 0 until count) {
        perQuery[i] =
          if (valid[i]) {
            (values[startLogits + query * n + pairStarts[i]] +
              values[endLogits + query * n + pairEnds[i]]) + compatibility[i]
          } else {
            MASK_LOGIT
          }
      }
      val ranked = stableByScore(IntArray(count) { it }, perQuery)
      for (rank in 0 until quota) {
        val i = ranked[rank]
        keys[next] = pairStarts[i] * n + pairEnds[i]
        priorities[next] = -MASK_LOGIT * 0.5f + (quota - rank)
        keep[next] = valid[i]
        next++
      }
    }
    for (i in 0 until count) {
      keys[next] = pairStarts[i] * n + pairEnds[i]
      priorities[next] = globalScores[i]
      keep[next] = valid[i]
      next++
    }
    for (i in 0 until total) {
      if (!keep[i]) {
        keys[i] = n * n
        priorities[i] = MASK_LOGIT
      }
    }
    // Preserve all three upstream stable sorts and invalid rows. Dropping invalid rows before
    // taking poolSize would change masked-score ties, even though ordinary fixtures rarely hit one.
    val byScore = stableByScore(IntArray(total) { it }, priorities)
    val byKey = stableByKey(byScore, keys)
    var previousKey = -1
    for (i in byKey) {
      val unique = keep[i] && keys[i] != previousKey
      if (!unique) {
        priorities[i] = MASK_LOGIT
        keep[i] = false
      }
      previousKey = keys[i]
    }
    val selected = stableByScore(byKey, priorities)
    val result = ArrayList<PoolCandidate>(poolSize)
    for (rank in 0 until minOf(poolSize, total)) {
      val i = selected[rank]
      if (keep[i]) {
        val start = keys[i] / n
        val end = keys[i] % n
        result.add(
          PoolCandidate(
            start,
            end,
            dot(values, poolStart + start * 128, values, poolEnd + end * 128, 128) / scale,
          )
        )
      }
    }
    record(6, proposerStart)
    return result
  }

  private fun stableByScore(indices: IntArray, scores: FloatArray): IntArray {
    var source = indices.copyOf()
    var target = IntArray(source.size)
    var width = 1
    while (width < source.size) {
      var first = 0
      while (first < source.size) {
        val middle = minOf(first + width, source.size)
        val limit = minOf(first + 2 * width, source.size)
        var left = first
        var right = middle
        var out = first
        while (left < middle && right < limit) {
          // Float.compare preserves the original comparator's signed-zero and infinity order.
          if (java.lang.Float.compare(scores[source[left]], scores[source[right]]) >= 0) {
            target[out++] = source[left++]
          } else {
            target[out++] = source[right++]
          }
        }
        while (left < middle) {
          target[out++] = source[left++]
        }
        while (right < limit) {
          target[out++] = source[right++]
        }
        first = limit
      }
      val swap = source
      source = target
      target = swap
      width *= 2
    }
    return source
  }

  private fun stableByKey(indices: IntArray, keys: IntArray): IntArray {
    var source = indices.copyOf()
    var target = IntArray(source.size)
    var width = 1
    while (width < source.size) {
      var first = 0
      while (first < source.size) {
        val middle = minOf(first + width, source.size)
        val limit = minOf(first + 2 * width, source.size)
        var left = first
        var right = middle
        var out = first
        while (left < middle && right < limit) {
          if (keys[source[left]] <= keys[source[right]]) {
            target[out++] = source[left++]
          } else {
            target[out++] = source[right++]
          }
        }
        while (left < middle) {
          target[out++] = source[left++]
        }
        while (right < limit) {
          target[out++] = source[right++]
        }
        first = limit
      }
      val swap = source
      source = target
      target = swap
      width *= 2
    }
    return source
  }

  private fun projection(name: String): Projection {
    val weight = parameters.getValue("$scorerPrefix$name.weight")
    return Projection(
      weight.shape[1],
      weight.shape[0],
      weight.values,
      parameters.getValue("$scorerPrefix$name.bias").values,
    )
  }

  private fun normalization(name: String) =
    Normalization(
      parameters.getValue("$scorerPrefix$name.weight").values,
      parameters.getValue("$scorerPrefix$name.bias").values,
    )

  private fun linearInto(
    projection: Projection,
    input: FloatArray,
    inputOffset: Int,
    output: FloatArray,
    outputOffset: Int,
  ) {
    val width = projection.inputSize
    val weight = projection.weight
    val bias = projection.bias
    for (row in 0 until projection.outputSize) {
      output[outputOffset + row] = dot(input, inputOffset, weight, row * width, width) + bias[row]
    }
  }

  private fun normalizeInPlace(norm: Normalization, input: FloatArray, offset: Int, size: Int) {
    val weight = norm.weight
    val bias = norm.bias
    var sum = 0f
    for (i in 0 until size) {
      sum += input[offset + i]
    }
    val mean = sum / size
    var variance = 0f
    for (i in 0 until size) {
      val delta = input[offset + i] - mean
      variance += delta * delta
    }
    val inverseStd = 1f / sqrt(variance / size + 1e-5f)
    for (i in 0 until size) {
      input[offset + i] = (input[offset + i] - mean) * inverseStd * weight[i] + bias[i]
    }
  }

  private fun parallel(size: Int, block: (Int, Int) -> Unit) {
    if (size < 32) {
      block(0, size)
      return
    }
    val futures =
      (1 until 4).map { worker ->
        scorerWorkers.submit(Callable { block(size * worker / 4, size * (worker + 1) / 4) })
      }
    block(0, size / 4)
    for (future in futures) {
      try {
        future.get()
      } catch (failure: ExecutionException) {
        throw failure.cause ?: failure
      }
    }
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
    // One bounded process-wide pool: three workers plus the calling thread. Shared lifetime
    // avoids creating threads per input or per ViewModel; completed tasks retain no model data.
    private val scorerWorkers by lazy {
      Executors.newFixedThreadPool(3) { task ->
        Thread(task, "Gliner-Sparse").apply { isDaemon = true }
      }
    }
    // The shared-pool checkpoint does not execute its retained candidate-encoder weights.
    private val PROFILE_STAGES =
      listOf(
        "unpack",
        "candidate_pool_build",
        "candidate_encoder_256_384",
        "pool_content",
        "film_linear",
        "film_gelu",
        "proposer_topk_merge_unique",
        "pair_scoring",
        "threshold_overlap",
        "offsets",
      )
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
      val negativeSquare = -(x * x)
      var powerOverFactorial = x
      var sum = x
      var n = 1.0
      var denominator = 3.0
      while (n <= 100.0) {
        powerOverFactorial *= negativeSquare / n
        val term = powerOverFactorial / denominator
        sum += term
        // These two comparisons are exactly abs(term) < epsilon, without a library call.
        if (term > -1e-17 && term < 1e-17) {
          break
        }
        // Integers through 201 are exact doubles, so the original denominators are unchanged.
        n += 1.0
        denominator += 2.0
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
