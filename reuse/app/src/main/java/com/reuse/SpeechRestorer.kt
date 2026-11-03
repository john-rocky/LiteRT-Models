package com.reuse

// [card-snippet-begin]
import android.content.Context
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max

// NVIDIA NemotronLabs RE-USE speech restoration on LiteRT: one fixed-length chunk graph (CompiledModel), the STFT and
// iSTFT on the host (Spectral.kt), chunks joined by a Hann-weighted overlap-add as in NVIDIA's inference_chunk.py.
//
// Usage (16 kHz mono float PCM in, the same length out):
//   val env = Environment.create()
//   SpeechRestorer(context, "reuse.tflite", CompiledModel.Options(Accelerator.GPU), env).use { restorer ->
//     val restored = restorer.enhance(pcm)
//   }
class SpeechRestorer(
  context: Context,
  modelAsset: String,
  options: CompiledModel.Options,
  env: Environment,
  hopPortion: Double? = null, // the step between chunks as a fraction of the chunk; null = the default below
) : AutoCloseable {

  init {
    require(hopPortion == null || (hopPortion > 0.0 && hopPortion <= 1.0)) {
      "hopPortion must be in (0, 1]: $hopPortion"
    }
  }

  private val model = CompiledModel.create(context.assets, modelAsset, options, env)
  private val inputs = model.createInputBuffers()
  private val outputs = model.createOutputBuffers()

  // The input shape [1, 2, T, 321] fixes the chunk: T STFT frames, hop 80 samples at 16 kHz.
  private val dims =
    checkNotNull(model.getInputTensorType("args_0", "serving_default").layout) { "No input shape" }.dimensions

  init {
    require(dims.size == 4 && dims[1] == 2 && dims[3] == Spectral.BINS) { "Unexpected input shape $dims" }
  }

  val frames = dims[2]
  val chunk = Spectral.HOP * (frames - 1)
  // 2 s chunks overlap by 1/8 (hop 0.875), shorter chunks by half, unless hopPortion is given.
  val hop =
    when {
      hopPortion != null -> (hopPortion * chunk).toInt()
      chunk >= 2 * SAMPLE_RATE -> chunk * 7 / 8
      else -> chunk / 2
    }

  private val spectral = Spectral()
  private val window = FloatArray(chunk) { max(0.5 - 0.5 * cos(2.0 * PI * it / chunk), 1e-7).toFloat() }

  /** Per chunk of the last enhance(), in ms: STFT, writeFloat, run, readFloat, iSTFT + overlap-add. */
  val chunkTimes = ArrayList<DoubleArray>()

  /** The final division by the window sum in the last enhance(), in ms. */
  var finalizeMs = 0.0

  fun enhance(pcm: FloatArray): FloatArray {
    val x = if (pcm.size >= chunk) pcm else pcm.copyOf(chunk) // zero-pad a clip shorter than one chunk
    val n = x.size
    val out = FloatArray(n)
    val wsum = FloatArray(n)
    val spec = FloatArray(2 * frames * Spectral.BINS)
    val wav = FloatArray(chunk)
    chunkTimes.clear()
    for (start in chunkStarts(n, chunk, hop)) {
      val t0 = System.nanoTime()
      spectral.analyze(x, start, chunk, spec) // [log1p|X|, angle X]
      val t1 = System.nanoTime()
      inputs[0].writeFloat(spec)
      val t2 = System.nanoTime()
      model.run(inputs, outputs) // enqueues the graph
      val t3 = System.nanoTime()
      val y = outputs[0].readFloat() // waits for it: [m, x_r, x_i]
      val t4 = System.nanoTime()
      spectral.synthesize(y, frames, wav)
      for (i in 0 until chunk) {
        out[start + i] += wav[i] * window[i]
        wsum[start + i] += window[i]
      }
      val t5 = System.nanoTime()
      chunkTimes.add(doubleArrayOf(ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5)))
    }
    val t6 = System.nanoTime()
    for (i in 0 until n) out[i] /= wsum[i]
    finalizeMs = ms(t6, System.nanoTime())
    return out.copyOf(pcm.size)
  }

  override fun close() {
    inputs.forEach { it.close() }
    outputs.forEach { it.close() }
    model.close()
  }

  companion object {
    const val SAMPLE_RATE = 16000

    /** Chunk starts as in inference_chunk.py: every [hop] samples, the last chunk aligned to the end. */
    fun chunkStarts(n: Int, chunk: Int, hop: Int): IntArray {
      val count = if (n <= chunk) 1 else (n - chunk + hop - 1) / hop + 1
      return IntArray(count) { i -> if (i < count - 1) i * hop else n - chunk }
    }

    private fun ms(from: Long, to: Long) = (to - from) / 1e6
  }
}
// [card-snippet-end]
