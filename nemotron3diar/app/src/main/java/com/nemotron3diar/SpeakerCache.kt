package com.nemotron3diar

import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln

/**
 * Streaming state of Nemotron-3-Diarization (batch 1): the Arrival-Order Speaker Cache and the FIFO of the most
 * recent encoder frames, a port of transformers' `Nemotron3DiarizationSpeakerCache` (update, _pool_probs,
 * _get_frame_scores, _boost_scores, _compress).
 *
 * Every row carries the absolute encoder-frame index it came from (-1 for a silence slot), so a run can be compared
 * with the reference cache contents frame by frame.
 *
 * The fp32 arithmetic follows torch's CPU order where it decides exact ties: sigmoid = 1 / (1 + exp(-x)),
 * avg_pool = sequential sum of 8 then / 8, the sum over 8 speakers = four pair accumulators (s, s + 4) combined
 * left to right. torch.topk breaks exact ties in an unspecified order; here the lower index wins. Differences left
 * (torch's vectorized exp / log round a few % of values 1 ulp apart) can flip a selection only at a 1-ulp tie;
 * [compressions] records the boundary margins so such a step can be recognized.
 */
class SpeakerCache(
  private val fifoLength: Int,
  private val updatePeriod: Int,
  private val silence: FloatArray,
  private val cacheLength: Int = CACHE_LENGTH,
) {
  private val hidden = silence.size
  private val budget = cacheLength / NUM_SPEAKERS - NUM_SILENCE
  private val minPositiveScores = floor(budget * MIN_POSITIVE_SCORES_RATE).toInt()
  private val numStrong = floor(budget * STRONG_BOOST_RATE).toInt()
  private val numWeak = floor(budget * WEAK_BOOST_RATE).toInt()

  private var embeds = FloatArray(0)
  private var probs = FloatArray(0)
  private var ids = IntArray(0)
  private var fifo = FloatArray(0)
  private var fifoIds = IntArray(0)

  var numCacheFrames = 0
    private set

  var numFifoFrames = 0
    private set

  var isCompressed = false
    private set

  /** One entry per compression: what was kept and how close the top-k boundaries were. */
  val compressions = mutableListOf<Compression>()

  class Compression(
    val step: Int,
    val numCandidates: Int,
    /** Absolute frame ids of the kept rows in cache order (-1 = silence slot). */
    val keptIds: IntArray,
    /** Per boost pass (strong, weak) and speaker: scores just inside and outside the top-k, or null. */
    val boostBoundaries: List<FloatArray?>,
    /** The final top-k over speaker-major scores: the last kept and the first dropped score. */
    val selectBoundary: FloatArray,
  )

  /** Cached rows followed by FIFO rows: the first [numCacheFrames] + [numFifoFrames] rows of the next input. */
  fun rows(out: FloatArray, offset: Int = 0): Int {
    embeds.copyInto(out, offset, 0, numCacheFrames * hidden)
    fifo.copyInto(out, offset + numCacheFrames * hidden, 0, numFifoFrames * hidden)
    return numCacheFrames + numFifoFrames
  }

  /** Frame ids of [rows], in the same order. */
  fun rowIds(): IntArray = ids.copyOf(numCacheFrames) + fifoIds.copyOf(numFifoFrames)

  /**
   * Pushes a processed step. [input] holds the step's encoder input rows ([numRows] x hidden: the rows of [rows],
   * then the chunk and its look-ahead), [logits] its logits ([numRows] * 8 x 8), [numChunkFrames] the chunk rows
   * that join the FIFO (the look-ahead rows are fed again next step), [chunkFirstId] the absolute frame index of
   * the first chunk row, [rowValid] (optional) the valid rows, whose padding rows get zero probabilities.
   */
  fun update(
    step: Int,
    input: FloatArray,
    numRows: Int,
    logits: FloatArray,
    numChunkFrames: Int,
    chunkFirstId: Int,
    rowValid: BooleanArray? = null,
  ) {
    val nc = numCacheFrames
    val nf = numFifoFrames
    check(numRows >= nc + nf + numChunkFrames) { "input rows $numRows < ${nc + nf + numChunkFrames}" }
    val stepProbs = poolProbs(logits, numRows, rowValid)

    // fifo_embeds = [fifo | chunk frames]
    val n = nf + numChunkFrames
    val fifoEmb = FloatArray(n * hidden)
    val fifoId = IntArray(n)
    fifo.copyInto(fifoEmb, 0, 0, nf * hidden)
    fifoIds.copyInto(fifoId, 0, 0, nf)
    input.copyInto(fifoEmb, nf * hidden, (nc + nf) * hidden, (nc + nf + numChunkFrames) * hidden)
    for (j in 0 until numChunkFrames) fifoId[nf + j] = chunkFirstId + j

    val popped = numPopped(n)
    if (popped > 0) {
      // an uncompressed cache still holds plain chunk frames, whose probabilities this step re-estimates;
      // a compressed one is out of order, so the stored probabilities are the only ones
      val m = nc + popped
      var candEmb = FloatArray(m * hidden)
      var candProbs = FloatArray(m * NUM_SPEAKERS)
      var candIds = IntArray(m)
      embeds.copyInto(candEmb, 0, 0, nc * hidden)
      fifoEmb.copyInto(candEmb, nc * hidden, 0, popped * hidden)
      (if (isCompressed) probs else stepProbs).copyInto(candProbs, 0, 0, nc * NUM_SPEAKERS)
      // fifo_probs = probs[nc : nc + n]
      stepProbs.copyInto(candProbs, nc * NUM_SPEAKERS, nc * NUM_SPEAKERS, (nc + popped) * NUM_SPEAKERS)
      ids.copyInto(candIds, 0, 0, nc)
      fifoId.copyInto(candIds, nc, 0, popped)
      var kept = m
      if (m > cacheLength) {
        val sel = compress(step, candProbs, candIds, m)
        val e = FloatArray(cacheLength * hidden)
        val p = FloatArray(cacheLength * NUM_SPEAKERS)
        val d = IntArray(cacheLength)
        for ((r, f) in sel.withIndex()) {
          if (f == m) {
            silence.copyInto(e, r * hidden)
            d[r] = -1 // probabilities stay 0
          } else {
            candEmb.copyInto(e, r * hidden, f * hidden, (f + 1) * hidden)
            candProbs.copyInto(p, r * NUM_SPEAKERS, f * NUM_SPEAKERS, (f + 1) * NUM_SPEAKERS)
            d[r] = candIds[f]
          }
        }
        candEmb = e
        candProbs = p
        candIds = d
        kept = cacheLength
        isCompressed = true
      }
      embeds = candEmb
      probs = candProbs
      ids = candIds
      numCacheFrames = kept
    }
    val left = n - popped
    fifo = fifoEmb.copyOfRange(popped * hidden, n * hidden)
    fifoIds = fifoId.copyOfRange(popped, n)
    numFifoFrames = left
  }

  private fun numPopped(n: Int): Int {
    if (n <= fifoLength) return 0
    return minOf(maxOf(updatePeriod, n - fifoLength), n)
  }

  /** sigmoid, then the mean of every 8 logit rows: [rows * 8, 8] -> [rows, 8]; padding rows get 0. */
  private fun poolProbs(logits: FloatArray, rows: Int, rowValid: BooleanArray?): FloatArray {
    val out = FloatArray(rows * NUM_SPEAKERS)
    for (r in 0 until rows) {
      if (rowValid != null && !rowValid[r]) continue
      for (s in 0 until NUM_SPEAKERS) {
        var acc = 0f
        for (j in 0 until SUBSAMPLING) acc += sigmoid(logits[((r * SUBSAMPLING) + j) * NUM_SPEAKERS + s])
        out[r * NUM_SPEAKERS + s] = acc / SUBSAMPLING
      }
    }
    return out
  }

  /** Frame scores [frames, 8]: log-likelihood ratio of "only this speaker", -inf for non-speech / weak frames. */
  private fun frameScores(p: FloatArray, frames: Int): FloatArray {
    val scores = FloatArray(frames * NUM_SPEAKERS)
    val lp = FloatArray(NUM_SPEAKERS)
    val lc = FloatArray(NUM_SPEAKERS)
    for (f in 0 until frames) {
      for (s in 0 until NUM_SPEAKERS) {
        val v = p[f * NUM_SPEAKERS + s]
        lp[s] = ln(maxOf(v, THRESHOLD))
        lc[s] = ln(maxOf(1f - v, THRESHOLD))
      }
      // torch's sum over the 8 speakers: pair accumulators (s, s + 4), combined left to right
      val sum = (((lc[0] + lc[4]) + (lc[1] + lc[5])) + (lc[2] + lc[6])) + (lc[3] + lc[7])
      for (s in 0 until NUM_SPEAKERS) {
        val speech = p[f * NUM_SPEAKERS + s] > 0.5f
        scores[f * NUM_SPEAKERS + s] = if (speech) ((lp[s] - lc[s]) + sum) - LOG_HALF else NEG_INF
      }
    }
    for (s in 0 until NUM_SPEAKERS) {
      var positive = 0
      for (f in 0 until frames) if (scores[f * NUM_SPEAKERS + s] > 0f) positive++
      if (positive < minPositiveScores) continue
      for (f in 0 until frames) {
        val i = f * NUM_SPEAKERS + s
        // not positive but speech -> dropped (non-speech is -inf already)
        if (!(scores[i] > 0f) && p[i] > 0.5f) scores[i] = NEG_INF
      }
    }
    return scores
  }

  /** Adds [boost] to the [k] best frames of every speaker; returns the boundary (k-th, k+1-th) per speaker. */
  private fun boost(scores: FloatArray, frames: Int, k: Int, boost: Float, boundaries: MutableList<FloatArray?>) {
    val col = FloatArray(frames)
    for (s in 0 until NUM_SPEAKERS) {
      for (f in 0 until frames) col[f] = scores[f * NUM_SPEAKERS + s]
      val order = topK(col, frames, minOf(k, frames))
      boundaries += if (k < frames) floatArrayOf(col[order[k - 1]], col[kthDropped(col, frames, order, k)]) else null
      for (j in 0 until minOf(k, frames)) {
        val i = order[j] * NUM_SPEAKERS + s
        scores[i] = scores[i] + boost
      }
    }
  }

  /** Keeps [cacheLength] frames grouped by speaker; returns the candidate index per slot ([m] = silence). */
  private fun compress(step: Int, p: FloatArray, candIds: IntArray, m: Int): IntArray {
    val scores = frameScores(p, m)
    // frames beyond the cache capacity are the ones popped from the FIFO
    for (f in cacheLength until m) for (s in 0 until NUM_SPEAKERS) {
      val i = f * NUM_SPEAKERS + s
      scores[i] = scores[i] + LATEST_BOOST
    }
    val boundaries = mutableListOf<FloatArray?>()
    boost(scores, m, numStrong, STRONG_BOOST, boundaries)
    boost(scores, m, numWeak, WEAK_BOOST, boundaries)
    // speaker-major flat scores, NUM_SILENCE (+inf) silence slots after every speaker's frames
    val scored = m + NUM_SILENCE
    val flat = FloatArray(NUM_SPEAKERS * scored)
    for (s in 0 until NUM_SPEAKERS) {
      for (f in 0 until m) flat[s * scored + f] = scores[f * NUM_SPEAKERS + s]
      for (f in m until scored) flat[s * scored + f] = Float.POSITIVE_INFINITY
    }
    val order = topK(flat, flat.size, cacheLength)
    val sentinel = flat.size
    val picked = IntArray(cacheLength) { if (flat[order[it]] == NEG_INF) sentinel else order[it] }
    picked.sort()
    val frame = IntArray(cacheLength) { if (picked[it] == sentinel) m else minOf(picked[it] % scored, m) }
    val selectBoundary = floatArrayOf(flat[order[cacheLength - 1]], flat[kthDropped(flat, flat.size, order, cacheLength)])
    compressions +=
      Compression(step, m, IntArray(cacheLength) { if (frame[it] == m) -1 else candIds[frame[it]] }, boundaries,
        selectBoundary)
    return frame
  }

  companion object {
    const val NUM_SPEAKERS = 8
    const val SUBSAMPLING = 8
    const val CACHE_LENGTH = 264
    const val NUM_SILENCE = 1
    const val THRESHOLD = 0.25f
    const val MIN_POSITIVE_SCORES_RATE = 0.5
    const val STRONG_BOOST_RATE = 0.75
    const val WEAK_BOOST_RATE = 1.5
    const val LATEST_BOOST = 0.05f
    val LOG_HALF = ln(0.5).toFloat()
    val STRONG_BOOST = (-2.0 * ln(0.5)).toFloat()
    val WEAK_BOOST = (-ln(0.5)).toFloat()
    const val NEG_INF = Float.NEGATIVE_INFINITY

    fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

    /** Indices of the [k] largest of values[0, n), best first; equal values keep the lower index first. */
    fun topK(values: FloatArray, n: Int, k: Int): IntArray {
      val idx = (0 until n).sortedWith { a, b ->
        val c = values[b].compareTo(values[a])
        if (c != 0) c else a.compareTo(b)
      }
      return IntArray(k) { idx[it] }
    }

    /** The first index after the top-[k] in the same order (the best value left out). */
    private fun kthDropped(values: FloatArray, n: Int, top: IntArray, k: Int): Int {
      val inTop = BooleanArray(n)
      for (j in 0 until k) inTop[top[j]] = true
      var best = -1
      for (i in 0 until n) {
        if (inTop[i]) continue
        if (best < 0 || values[i] > values[best]) best = i
      }
      return best
    }
  }
}
