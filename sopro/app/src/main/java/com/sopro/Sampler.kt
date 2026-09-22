package com.sopro

import java.util.SplittableRandom
import kotlin.math.exp

fun interface UniformSource {
  fun next(): Double
}

class RandomUniform(seed: Long = System.nanoTime()) : UniformSource {
  private val random = SplittableRandom(seed)

  override fun next() = random.nextDouble()
}

/** Float32 source distribution; the inverse CDF deliberately accumulates in float64. */
object Sampler {
  const val BOS = 4375
  const val EOS = 4376
  const val MAX_STEPS = 704

  data class Generation(
    val tokens: IntArray,
    val drawCount: Int,
    val stepCalls: Int,
    val stopReason: String,
  )

  // NumPy's contiguous float reduction uses eight partial sums and blocks of 128.
  private fun sum(a: FloatArray, start: Int = 0, length: Int = a.size): Float {
    if (length < 8) {
      var value = -0.0f
      for (i in start until start + length) value += a[i]
      return value
    }
    if (length <= 128) {
      val partial = FloatArray(8) { a[start + it] }
      var i = 8
      while (i < length - length % 8) {
        for (j in 0..7) partial[j] += a[start + i + j]
        i += 8
      }
      var value =
        ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
          ((partial[4] + partial[5]) + (partial[6] + partial[7]))
      while (i < length) value += a[start + i++]
      return value
    }
    var half = length / 2
    half -= half % 8
    return sum(a, start, half) + sum(a, start + half, length - half)
  }

  fun probabilities(
    logits: FloatArray,
    allowEos: Boolean = false,
    temperature: Float = .8f,
    topP: Float = .9f,
    topK: Int = 25,
  ): FloatArray {
    require(logits.size == 4377)
    val x = logits.copyOf()
    x[BOS] = -1e9f
    if (!allowEos) x[EOS] = -1e9f
    if (temperature <= 0f) {
      val index = x.indices.maxBy { x[it] }
      return FloatArray(x.size).also { it[index] = 1f }
    }
    for (i in x.indices) x[i] /= maxOf(1e-5f, temperature)
    val maximum = x.max()
    val p = FloatArray(x.size) { exp((x[it] - maximum).toDouble()).toFloat() }
    var total = sum(p)
    for (i in p.indices) p[i] /= total
    if (topK > 0 && topK < p.size) {
      val sorted = p.copyOf().also { it.sort() }
      val kth = sorted[p.size - topK]
      for (i in p.indices) if (p[i] < kth) p[i] = 0f
      total = maxOf(sum(p), 1e-8f)
      for (i in p.indices) p[i] /= total
    }
    if (topP < 1f) {
      // Kotlin object sorting is stable; explicitly compare indices on ties.
      val order = p.indices.sortedWith(compareByDescending<Int> { p[it] }.thenBy { it })
      var cumulative = 0f
      var remove = false
      val threshold = topP.coerceIn(0f, 1f)
      for (index in order) {
        val previousRemove = remove
        cumulative += p[index]
        remove = cumulative > threshold
        if (previousRemove) p[index] = 0f
      }
      total = maxOf(sum(p), 1e-8f)
      for (i in p.indices) p[i] /= total
    }
    return p
  }

  fun inverseCdf(probabilities: FloatArray, uniform: Double): Int {
    require(uniform >= 0.0 && uniform < 1.0)
    val cdf = DoubleArray(probabilities.size)
    var total = 0.0
    for (i in probabilities.indices) {
      require(probabilities[i].isFinite() && probabilities[i] >= 0f)
      total += probabilities[i].toDouble()
      cdf[i] = total
    }
    require(total > 0.0)
    for (i in cdf.indices) cdf[i] /= total
    var low = 0
    var high = cdf.size
    while (low < high) {
      val middle = (low + high) ushr 1
      if (uniform < cdf[middle]) high = middle else low = middle + 1
    }
    // Preserve EOS here; the source clamps speech IDs only after checking EOS.
    return minOf(low, cdf.lastIndex)
  }

  fun generate(
    initialLogits: FloatArray,
    advance: (Int) -> FloatArray,
    uniform: UniformSource,
    maxSteps: Int = MAX_STEPS,
    minSteps: Int = 10,
  ): Generation {
    require(maxSteps in 1..MAX_STEPS)
    var logits = initialLogits
    val tokens = ArrayList<Int>()
    var calls = 0
    for (step in 0 until maxSteps) {
      val allowEos = step + 1 >= maxOf(1, minSteps)
      val picked = inverseCdf(probabilities(logits, allowEos), uniform.next())
      if (allowEos && picked == EOS) return Generation(tokens.toIntArray(), step + 1, calls, "eos")
      val token = picked.coerceIn(0, 4374)
      tokens += token
      if (step + 1 < maxSteps) {
        logits = advance(token)
        calls++
      }
    }
    return Generation(tokens.toIntArray(), maxSteps, calls, "max_steps")
  }
}
