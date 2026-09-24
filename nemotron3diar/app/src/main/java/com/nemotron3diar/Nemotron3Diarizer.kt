package com.nemotron3diar

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/** The two graphs of a Nemotron-3-Diarization build (LiteRT on the device, a replay of reference data in tests). */
interface Engine : AutoCloseable {
  /** graph A: mel [104 x 128] -> chunk_embeds [13 x 512] (8-frame stacking + projection). */
  fun frontend(mel: FloatArray): FloatArray

  /** graph B: packed_embeds [T x 512], attn_bias [T], rope_cos / rope_sin [T x 64] -> logits [8T x 8]. */
  fun encoder(packed: FloatArray, bias: FloatArray, cos: FloatArray, sin: FloatArray): FloatArray
}

/**
 * Chunking of one graph-B build, in encoder frames (1 encoder frame = 8 mel frames = 80 ms).
 *
 * [maxRows] is graph B's fixed T = speaker cache (264) + FIFO + chunk + look-ahead. [padBias] marks the rows past
 * the step's length (the graph zeroes them before the head's k=3 convolution); [maskedBias] marks a real row whose
 * key is masked (the offline pass masks the frame after the last full hop), which only the offline graph separates
 * from padding.
 */
data class StreamConfig(
  val name: String,
  val chunkFrames: Int,
  val rightContext: Int,
  val fifoLength: Int,
  val updatePeriod: Int,
  val maxRows: Int,
  val padBias: Float,
  val maskedBias: Float,
) {
  /** Mel frames of one streaming chunk (chunk + look-ahead). */
  val melFramesPerChunk = (chunkFrames + rightContext) * SUBSAMPLING

  /** Mel frames the frame cursor advances per streaming step. */
  val melFramesPerStep = chunkFrames * SUBSAMPLING

  companion object {
    const val SUBSAMPLING = 8

    /** processor streaming_modes["low_latency"] = (9, 4) and config.streaming_config; graph B T = 541. */
    val LOW_LATENCY = StreamConfig("low_latency", 9, 4, 264, 222, 541, -3.0e4f, -3.0e4f)

    /** Offline pass of the whole file: config.chunk_length 340 / chunk_right_context 40 / fifo 40 / period 300. */
    val OFFLINE = StreamConfig("offline", 340, 40, 40, 300, 684, -32768f, -16384f)
  }
}

/** One graph-B step and the logits it emits. */
class Step(
  val index: Int,
  /** First mel frame (10 ms) of the chunk and of [logits]. */
  val firstFrame: Int,
  val melFrames: Int,
  val lookahead: Int,
  /** Encoder input rows: cache + FIFO + chunk + look-ahead. */
  val length: Int,
  /** (cache frames, FIFO frames, compressed) before and after the step. */
  val pre: IntArray,
  val post: IntArray,
  /** Logits of the frames this step emits, [numFrames x 8] (sigmoid gives the speaker activity). */
  val logits: FloatArray,
  val numFrames: Int,
  val melMs: Double,
  val frontendMs: Double,
  val encoderMs: Double,
  val cacheMs: Double,
  val totalMs: Double,
)

/**
 * Streaming Sortformer host of Nemotron-3-Diarization: 16 kHz audio in, per-frame speaker logits out.
 *
 * Per chunk: log-mel ([MelFrontend]) -> graph A (mel -> chunk_embeds) -> packed = [speaker cache | FIFO | chunk +
 * look-ahead] (zero tail to T), attn_bias (0 valid, [StreamConfig.padBias] past the length), RoPE tables for
 * positions 0..T-1 -> graph B -> logits of the L rows -> [SpeakerCache.update] -> the chunk's own logit rows.
 *
 * Streaming ([push] / [finish]) follows the processor's low_latency schedule: the first chunk takes mel frames
 * [0, 104) once 16680 samples have arrived (centered windows), chunk k >= 1 takes [72k, 72k + 104) once the audio
 * reaches 72k * 160 - 256 + 17040, and the last chunk takes the rest with no look-ahead. [runFile] is the offline
 * pass over a whole file (StreamConfig.OFFLINE): centered frames 0..N/160, chunks of 340 encoder frames with up to
 * 40 look-ahead frames.
 */
class Nemotron3Diarizer(
  private val engine: Engine,
  private val mel: MelFrontend,
  silence: FloatArray,
  val config: StreamConfig = StreamConfig.LOW_LATENCY,
) {
  val cache = SpeakerCache(config.fifoLength, config.updatePeriod, silence)
  private val t = config.maxRows
  private val ropeCos: FloatArray
  private val ropeSin: FloatArray
  private val packed = FloatArray(t * HIDDEN)
  private val bias = FloatArray(t)
  private val melBuf = FloatArray(FRONTEND_FRAMES * MelFrontend.N_MELS)
  private var nextChunk = 0
  private var finished = false

  /** Steps run so far. */
  val numSteps: Int
    get() = nextChunk

  init {
    val tables = ropeTables(t)
    ropeCos = tables.first
    ropeSin = tables.second
  }

  // ------------------------------------------------------------------------------------------ streaming

  private fun chunkStartSample(k: Int) = k * config.melFramesPerStep * MelFrontend.HOP - MelFrontend.N_FFT / 2

  private val firstChunkSamples = (config.melFramesPerChunk - 1) * MelFrontend.HOP + MelFrontend.WIN_LENGTH / 2
  private val chunkSamples = config.melFramesPerChunk * MelFrontend.HOP + MelFrontend.WIN_LENGTH

  /** Appends audio and runs every chunk it completes. */
  fun push(samples: FloatArray, offset: Int = 0, count: Int = samples.size - offset): List<Step> {
    check(!finished) { "stream finished" }
    mel.append(samples, offset, count)
    val out = mutableListOf<Step>()
    while (true) {
      val k = nextChunk
      val ready = if (k == 0) mel.numSamples >= firstChunkSamples
      else chunkStartSample(k) + chunkSamples <= mel.numSamples
      if (!ready) break
      out += runStep(k, k * config.melFramesPerStep, config.melFramesPerChunk, config.rightContext)
    }
    return out
  }

  /** Ends the stream: the rest of the audio as the last chunk, every frame scored (no look-ahead). */
  fun finish(): List<Step> {
    check(!finished) { "stream finished" }
    finished = true
    mel.close() // a centered single chunk reads zeros past the end, as the processor pads it
    val n = mel.numSamples
    val k = nextChunk
    val frames = if (k == 0) n / MelFrontend.HOP // a single first-and-last chunk, centered
    else (n - chunkStartSample(k) - MelFrontend.N_FFT) / MelFrontend.HOP + 1
    if (frames < 1) return emptyList()
    return listOf(runStep(k, k * config.melFramesPerStep, frames, 0))
  }

  private fun runStep(k: Int, g0: Int, frames: Int, lookahead: Int): Step {
    check(frames <= FRONTEND_FRAMES) { "chunk of $frames mel frames" }
    val t0 = System.nanoTime()
    melBuf.fill(0f)
    mel.frames(g0, frames, melBuf)
    val t1 = System.nanoTime()
    val emb = engine.frontend(melBuf)
    val t2 = System.nanoTime()
    val numEmbeds = (frames + SUBSAMPLING - 1) / SUBSAMPLING
    val numChunk = numEmbeds - lookahead
    val step =
      encode(k, g0, frames, lookahead, emb, 0, numEmbeds, numChunk, g0 / SUBSAMPLING, null, t0, t1, t2)
    nextChunk = k + 1
    return step
  }

  /**
   * Packs [cache rows | embeds[from, from + count)], runs graph B, updates the cache and returns the step with the
   * logits of its [numChunk] chunk rows (at most [frames] mel frames).
   */
  private fun encode(
    k: Int,
    g0: Int,
    frames: Int,
    lookahead: Int,
    embeds: FloatArray,
    from: Int,
    count: Int,
    numChunk: Int,
    firstId: Int,
    embedValid: BooleanArray?,
    t0: Long,
    t1: Long,
    t2: Long,
  ): Step {
    val pre = state()
    packed.fill(0f)
    val cached = cache.rows(packed)
    val length = cached + count
    check(length <= t) { "step $k: $length rows > T=$t" }
    embeds.copyInto(packed, cached * HIDDEN, from * HIDDEN, (from + count) * HIDDEN)
    var rowValid: BooleanArray? = null
    for (r in 0 until t) bias[r] = if (r < length) 0f else config.padBias
    if (embedValid != null) {
      rowValid = BooleanArray(length) { r -> r < cached || embedValid[from + r - cached] }
      for (r in 0 until length) if (!rowValid[r]) bias[r] = config.maskedBias
    }
    val t3 = System.nanoTime()
    val logits = engine.encoder(packed, bias, ropeCos, ropeSin)
    val t4 = System.nanoTime()
    cache.update(k, packed, length, logits, numChunk, firstId, rowValid)
    val numOut = minOf(numChunk * SUBSAMPLING, frames)
    val out = logits.copyOfRange(cached * SUBSAMPLING * NUM_SPEAKERS, (cached * SUBSAMPLING + numOut) * NUM_SPEAKERS)
    val t5 = System.nanoTime()
    return Step(
      index = k,
      firstFrame = g0,
      melFrames = frames,
      lookahead = lookahead,
      length = length,
      pre = pre,
      post = state(),
      logits = out,
      numFrames = numOut,
      melMs = (t1 - t0) / 1e6,
      frontendMs = (t2 - t1) / 1e6,
      encoderMs = (t4 - t3) / 1e6,
      cacheMs = ((t3 - t2) + (t5 - t4)) / 1e6,
      totalMs = (t5 - t0) / 1e6,
    )
  }

  private fun state() = intArrayOf(cache.numCacheFrames, cache.numFifoFrames, if (cache.isCompressed) 1 else 0)

  // ------------------------------------------------------------------------------------------ offline file mode

  /**
   * The offline pass over a whole file (use StreamConfig.OFFLINE): centered mel frames 0..N/160 of which the last
   * (N/160) is masked and zero, as the processor returns them; graph A over blocks of 104 frames; chunks of
   * [StreamConfig.chunkFrames] encoder frames plus up to [StreamConfig.rightContext] following ones. Returns one
   * [Step] per chunk; their logits concatenated are the N/160 + 1 frames of the file.
   */
  fun runFile(samples: FloatArray): List<Step> {
    check(nextChunk == 0 && !finished) { "runFile needs a fresh diarizer" }
    finished = true
    mel.append(samples)
    mel.close()
    val valid = samples.size / MelFrontend.HOP
    val frames = valid + 1
    val numEmbeds = (frames + SUBSAMPLING - 1) / SUBSAMPLING
    val blockEmbeds = FRONTEND_FRAMES / SUBSAMPLING
    val embeds = FloatArray(numEmbeds * HIDDEN)
    val embedValid = BooleanArray(numEmbeds) { it * SUBSAMPLING < valid }
    var melMs = 0.0
    var frontendMs = 0.0
    var b = 0
    while (b * blockEmbeds < numEmbeds) {
      val f0 = b * FRONTEND_FRAMES
      val nf = minOf(FRONTEND_FRAMES, frames - f0)
      val ta = System.nanoTime()
      melBuf.fill(0f)
      val computed = minOf(nf, valid - f0) // the masked frame stays zero
      if (computed > 0) mel.frames(f0, computed, melBuf)
      val tb = System.nanoTime()
      val emb = engine.frontend(melBuf)
      val tc = System.nanoTime()
      val e0 = b * blockEmbeds
      val ne = minOf(blockEmbeds, numEmbeds - e0)
      emb.copyInto(embeds, e0 * HIDDEN, 0, ne * HIDDEN)
      melMs += (tb - ta) / 1e6
      frontendMs += (tc - tb) / 1e6
      b++
    }
    val steps = mutableListOf<Step>()
    var start = 0
    var k = 0
    while (start < numEmbeds) {
      val end = minOf(start + config.chunkFrames, numEmbeds)
      val hi = minOf(end + config.rightContext, numEmbeds)
      val t0 = System.nanoTime()
      val step =
        encode(k, start * SUBSAMPLING, minOf(frames - start * SUBSAMPLING, (hi - start) * SUBSAMPLING), hi - end,
          embeds, start, hi - start, end - start, start, embedValid, t0, t0, t0)
      // the file-level mel / graph-A time goes to the first step
      steps += if (k == 0) step.withFrontTime(melMs, frontendMs) else step
      start = end
      k++
    }
    nextChunk = k
    return steps
  }

  private fun Step.withFrontTime(mel: Double, frontend: Double) =
    Step(index, firstFrame, melFrames, lookahead, length, pre, post, logits, numFrames, mel, frontend, encoderMs,
      cacheMs, totalMs + mel + frontend)

  override fun toString() = "Nemotron3Diarizer(${config.name}, T=$t)"

  companion object {
    const val HIDDEN = 512
    const val HEAD_DIM = 64
    const val NUM_SPEAKERS = 8
    const val SUBSAMPLING = 8
    const val FRONTEND_FRAMES = 104
    const val ROPE_THETA = 10000f

    /**
     * RoPE cos / sin [T x 64] for positions 0..T-1, as transformers builds them in fp32: inv_freq = 1 / theta^(2j/64),
     * freqs = position * inv_freq (one fp32 product), emb = [freqs, freqs].
     */
    fun ropeTables(t: Int): Pair<FloatArray, FloatArray> {
      val half = HEAD_DIM / 2
      val invFreq = FloatArray(half) { 1f / ROPE_THETA.pow((2 * it).toFloat() / HEAD_DIM) }
      val c = FloatArray(t * HEAD_DIM)
      val s = FloatArray(t * HEAD_DIM)
      for (p in 0 until t) for (j in 0 until half) {
        val f = p.toFloat() * invFreq[j]
        val cf = cos(f)
        val sf = sin(f)
        c[p * HEAD_DIM + j] = cf
        c[p * HEAD_DIM + half + j] = cf
        s[p * HEAD_DIM + j] = sf
        s[p * HEAD_DIM + half + j] = sf
      }
      return c to s
    }

    /** Speech segments of [probability > threshold] per speaker: (start frame, end frame exclusive, speaker). */
    fun segments(logits: FloatArray, frames: Int, threshold: Float = 0.5f): List<IntArray> {
      val out = mutableListOf<IntArray>()
      for (s in 0 until NUM_SPEAKERS) {
        var start = -1
        for (f in 0..frames) {
          val active = f < frames && SpeakerCache.sigmoid(logits[f * NUM_SPEAKERS + s]) > threshold
          if (active && start < 0) start = f
          if (!active && start >= 0) {
            out += intArrayOf(start, f, s)
            start = -1
          }
        }
      }
      out.sortWith(compareBy<IntArray>({ it[0] }, { it[2] }))
      return out
    }
  }
}
