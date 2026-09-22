package com.sopro

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin

/** Stateful mono mirror of conversion/host_stream_istft.py, n_fft=1024/hop=256. */
class StreamingIstft(private val window: FloatArray) {
  data class State(
    var processedFrames: Int = 0,
    var emittedSamples: Int = 0,
    var tailStart: Int = 0,
    var ola: FloatArray = FloatArray(0),
    var env: FloatArray = FloatArray(0),
  )

  private val fft = Fft(1024)
  var state = State()
    private set

  init {
    require(window.size == 1024)
  }

  fun reset() {
    state = State()
  }

  /** Features are row-major [frames, 1026]: 513 log magnitudes then 513 phases. */
  fun overlapAdd(features: FloatArray): Pair<FloatArray, FloatArray> {
    require(features.size % 1026 == 0)
    val frames = features.size / 1026
    if (frames == 0) return FloatArray(0) to FloatArray(0)
    val out = FloatArray((frames - 1) * 256 + 1024)
    val env = FloatArray(out.size)
    val real = FloatArray(513)
    val imag = FloatArray(513)
    val clamp = ln(100.0).toFloat()
    for (frame in 0 until frames) {
      val base = frame * 1026
      for (bin in 0..512) {
        val magnitude = exp(minOf(features[base + bin], clamp))
        val phase = features[base + 513 + bin]
        real[bin] = magnitude * cos(phase)
        imag[bin] = magnitude * sin(phase)
      }
      val samples = fft.inverseReal(real, imag)
      val start = frame * 256
      for (j in samples.indices) {
        out[start + j] += samples[j] * window[j]
        env[start + j] += window[j] * window[j]
      }
    }
    return out to env
  }

  fun process(features: FloatArray, flush: Boolean = false): FloatArray {
    val st = state
    val (chunk, chunkEnv) = overlapAdd(features)
    val offset = st.processedFrames * 256 - st.tailStart
    val required = offset + chunk.size
    val length = max(st.ola.size, required)
    val ola = st.ola.copyOf(length)
    val env = st.env.copyOf(length)
    for (j in chunk.indices) {
      ola[offset + j] += chunk[j]
      env[offset + j] += chunkEnv[j]
    }
    st.processedFrames += features.size / 1026
    var target = max(0, st.processedFrames * 256 - 512)
    if (flush && st.processedFrames > 0) target = max(target, (st.processedFrames - 1) * 256)
    val count = max(0, target - st.emittedSamples)
    val start = st.emittedSamples + 512 - st.tailStart
    val out = FloatArray(count) { ola[start + it] / max(env[start + it], 1e-8f) }
    st.emittedSamples = target
    val trim = st.emittedSamples + 512 - st.tailStart
    st.ola = ola.copyOfRange(trim, ola.size)
    st.env = env.copyOfRange(trim, env.size)
    st.tailStart += trim
    if (flush) state = State()
    return out
  }
}

/** Fixed graph driver retaining only real frames, including partial-tail replay. */
object StaticStreamFeatures {
  fun run(
    mel: FloatArray,
    frames: Int,
    invoke: (String, List<FloatArray>) -> List<FloatArray>,
    emit: (FloatArray, Boolean) -> Unit,
  ) {
    require(frames >= 128) { "Static stream host requires at least 128 real mel frames" }
    require(mel.size == 100 * frames)
    fun chunk(start: Int): FloatArray {
      val out = FloatArray(100 * 64)
      for (c in 0 until 100) mel.copyInto(out, c * 64, c * frames + start, c * frames + start + 64)
      return out
    }
    var outputs = invoke("start", listOf(chunk(0)))
    emit(outputs[0], false)
    var state = outputs.drop(1)
    for (i in 1 until frames / 64) {
      outputs = invoke("step", listOf(chunk(i * 64)) + state)
      emit(outputs[0], false)
      state = outputs.drop(1)
    }
    val remainder = frames % 64
    if (remainder != 0) {
      val restart = invoke("start", listOf(chunk(frames - 128)))
      val tail = invoke("step", listOf(chunk(frames - 64)) + restart.drop(1))
      emit(tail[0].copyOfRange(tail[0].size - remainder * 1026, tail[0].size), false)
      state = tail.drop(1)
    }
    emit(invoke("flush", state)[0], true)
  }
}
