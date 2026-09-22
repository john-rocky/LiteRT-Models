// SPDX-License-Identifier: Apache-2.0
package com.sopro

import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sin

/** torchaudio functional.resample float32 sinc_interp_hann, width 6 and rolloff .99. */
object GenericResampler {
  private data class Kernel(
    val source: Int,
    val target: Int,
    val width: Int,
    val phases: Array<FloatArray>,
  )

  private val kernels = mutableMapOf<Pair<Int, Int>, Kernel>()

  private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

  @Synchronized
  private fun kernel(sourceRate: Int, targetRate: Int): Kernel =
    kernels.getOrPut(sourceRate to targetRate) {
      val divisor = gcd(sourceRate, targetRate)
      val source = sourceRate / divisor
      val target = targetRate / divisor
      val base = minOf(source, target) * .99
      val width = ceil(6.0 * source / base).toInt()
      val scale = (base / source).toFloat()
      val phases =
        Array(target) { phase ->
          FloatArray(2 * width + source) { index ->
            // Preserve the source's float32 division, addition, multiply and clamp sequence.
            var t =
              (-phase.toFloat() / target.toFloat() + (index - width).toFloat() / source.toFloat()) *
                base.toFloat()
            t = t.coerceIn(-6f, 6f)
            val windowCos = cos(t * PI.toFloat() / 6f / 2f)
            val window = windowCos * windowCos
            t *= PI.toFloat()
            val sinc = if (t == 0f) 1f else sin(t) / t
            sinc * (window * scale)
          }
        }
      Kernel(source, target, width, phases)
    }

  fun resample(wave: FloatArray, sourceRate: Int, targetRate: Int = 24000): FloatArray {
    require(sourceRate > 0 && targetRate > 0)
    require(wave.all { it.isFinite() }) { "Reference audio contains non-finite samples" }
    if (wave.isEmpty() || sourceRate == targetRate) return wave.copyOf()
    val k = kernel(sourceRate, targetRate)
    val length = ((wave.size.toLong() * k.target + k.source - 1) / k.source).toInt()
    return FloatArray(length) { index ->
      val start = (index / k.target) * k.source - k.width
      val filter = k.phases[index % k.target]
      var total = 0f
      val first = maxOf(0, -start)
      val end = minOf(filter.size, wave.size - start)
      for (j in first until end) total += wave[start + j] * filter[j]
      total
    }
  }
}
