package com.sopro

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt
import org.json.JSONObject

/** Frozen fp32 front ends. Input is mono; output tensors are channel-major [1,C,T]. */
class HostDsp(val c: Map<String, FloatArray>, val resampleWidth: Int) {
  data class NormalizedReference(val wav: FloatArray, val levelDb: Double)

  private val fft1024 = Fft(1024)
  private val fft400 = Fft(400)

  companion object {
    fun fromAssets(directory: File): HostDsp {
      val descriptor = JSONObject(File(directory, "host_assets.json").readText())
      val tensors = descriptor.getJSONArray("tensors")
      val constants = mutableMapOf<String, FloatArray>()
      val files = mutableMapOf<String, ByteArray>()
      var width = -1
      for (i in 0 until tensors.length()) {
        val tensor = tensors.getJSONObject(i)
        if (tensor.getString("file") != "dsp_constants_fp32.bin") continue
        val file = tensor.getString("file")
        val all = files.getOrPut(file) { File(directory, file).readBytes() }
        val offset = tensor.getInt("offset_bytes")
        val bytes = all.copyOfRange(offset, offset + tensor.getInt("size_bytes"))
        val digest =
          MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(digest == tensor.getString("sha256")) {
          "Invalid DSP tensor SHA-256: ${tensor.getString("name")}"
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val name = tensor.getString("name")
        if (name == "resample_24_16_width") width = buffer.int
        else {
          require(tensor.getString("dtype") == "float32")
          constants[name] = FloatArray(bytes.size / 4) { buffer.float }
        }
      }
      require(width >= 0)
      return HostDsp(constants, width)
    }
  }

  /** sopro.audio.normalize_reference: full-clip level first, before bucket cropping. */
  fun normalizeReference(wav: FloatArray, sampleRate: Int = 24000): NormalizedReference {
    require(wav.isNotEmpty()) { "Reference audio must contain samples" }
    require(wav.all { it.isFinite() }) { "Reference contains non-finite samples" }
    val win = (sampleRate * .025).toInt()
    val hop = (sampleRate * .010).toInt()
    val levelDb: Double
    if (wav.size < win) {
      val squares = FloatArray(wav.size) { wav[it] * wav[it] }
      val rms = sqrt(PostProcess.pairwiseSum(squares) / wav.size.toFloat())
      levelDb = 20.0 * log10(maxOf(rms.toDouble(), 1e-6))
    } else {
      val rms =
        FloatArray((wav.size - win) / hop + 1) { frame ->
          val squares = FloatArray(win) { wav[frame * hop + it] * wav[frame * hop + it] }
          maxOf(sqrt(PostProcess.pairwiseSum(squares) / win.toFloat()), 1e-6f)
        }
      val sorted = rms.sortedArray()
      val position = (sorted.size - 1).toFloat() * .2f
      val lo = position.toInt()
      val fraction = position - lo.toFloat()
      val a = sorted[lo]
      val b = sorted[minOf(lo + 1, sorted.lastIndex)]
      val quantile = if (fraction < .5f) a + (b - a) * fraction else b - (b - a) * (1f - fraction)
      val threshold = quantile * 1.5f
      val active = rms.filter { it > threshold }.sorted()
      val median =
        if (active.isEmpty()) sorted[(sorted.size - 1) / 2] else active[(active.size - 1) / 2]
      levelDb = (20f * log10(median)).toDouble()
    }
    var gainDb = (-19.8 - levelDb).coerceIn(0.0, 30.0)
    var peak = 0f
    for (v in wav) peak = maxOf(peak, abs(v))
    if (peak > 0f) gainDb = minOf(gainDb, maxOf(0.0, 20.0 * log10(.95 / peak)))
    val gain = 10.0.pow(gainDb / 20.0).toFloat()
    return NormalizedReference(FloatArray(wav.size) { wav[it] * gain }, levelDb + gainDb)
  }

  fun fixedReference(wav: FloatArray): NormalizedReference {
    val normalized = normalizeReference(wav)
    return normalized.copy(wav = normalized.wav.copyOf(240000))
  }

  fun resample24to16(wav: FloatArray): FloatArray {
    val kernel = c.getValue("resample_24_16_kernel")
    val kernelWidth = kernel.size / 2
    val out = FloatArray((2 * wav.size + 2) / 3)
    for (j in out.indices) {
      val inputStart = (j / 2) * 3 - resampleWidth
      val kernelStart = (j % 2) * kernelWidth
      var sum = 0f
      for (k in 0 until kernelWidth) {
        val p = inputStart + k
        if (p >= 0 && p < wav.size) sum += wav[p] * kernel[kernelStart + k]
      }
      out[j] = sum
    }
    return out
  }

  fun speakerMel(wav16: FloatArray): FloatArray = mel(wav16, "speaker")

  fun semanticMel(wav16: FloatArray): FloatArray = mel(wav16, "semantic")

  fun acousticMel(wav24: FloatArray): FloatArray = mel(wav24, "acoustic")

  fun acousticMelNormalized(wav24: FloatArray): FloatArray {
    val mel = acousticMel(wav24)
    val frames = mel.size / 100
    val mean = c.getValue("mel_mean")
    val std = c.getValue("mel_std")
    for (channel in 0 until 100) for (t in 0 until frames) {
      val p = channel * frames + t
      mel[p] = (mel[p] - mean[channel]) / std[channel]
    }
    return mel
  }

  fun mel(wav: FloatArray, kind: String): FloatArray {
    require(wav.size > 1)
    val semantic = kind == "semantic"
    val speaker = kind == "speaker"
    require(semantic || speaker || kind == "acoustic")
    val nfft = if (semantic) 400 else 1024
    val hop = if (kind == "acoustic") 256 else 160
    val channels = if (kind == "acoustic") 100 else 80
    val bins = nfft / 2 + 1
    val originalFrames = (wav.size + hop - 1) / hop
    val paddedLength = wav.size + if (semantic) nfft else 0
    val frameCount =
      minOf(paddedLength / hop + 1, if (semantic) originalFrames + 2 else Int.MAX_VALUE)
    val shortWindow = c.getValue(kind + "_window")
    val window = FloatArray(nfft)
    shortWindow.copyInto(window, (nfft - shortWindow.size) / 2)
    val bank = c.getValue(kind + "_melbank")
    require(bank.size == bins * channels)
    val output = FloatArray(channels * frameCount)
    val block = FloatArray(nfft)
    val fft = if (nfft == 400) fft400 else fft1024
    fun sample(index: Int): Float {
      var p = index
      while (p < 0 || p >= paddedLength) {
        p = if (p < 0) -p else 2 * paddedLength - 2 - p
      }
      return if (p < wav.size) wav[p] else 0f
    }
    for (t in 0 until frameCount) {
      for (j in 0 until nfft) block[j] = sample(t * hop + j - nfft / 2) * window[j]
      val spec = fft.realForward(block)
      val power =
        FloatArray(bins) { k ->
          val magnitude = sqrt(spec.real[k] * spec.real[k] + spec.imag[k] * spec.imag[k])
          if (kind == "acoustic") magnitude else magnitude * magnitude
        }
      for (channel in 0 until channels) {
        var value = 0f
        for (k in 0 until bins) value += power[k] * bank[k * channels + channel]
        val floor = if (speaker) 1e-5f else if (semantic) 1e-10f else 1e-7f
        output[channel * frameCount + t] =
          if (semantic) log10(maxOf(value, floor)) else ln(maxOf(value, floor))
      }
    }
    if (speaker) {
      for (t in 0 until frameCount) {
        // The channel reduction has non-contiguous rows in the Python array.
        var mean = 0f
        for (channel in 0 until channels) mean += output[channel * frameCount + t]
        mean /= channels.toFloat()
        var variance = 0f
        for (channel in 0 until channels) {
          val p = channel * frameCount + t
          val delta = output[p] - mean
          output[p] = delta
          variance += delta * delta
        }
        val scale = sqrt(variance / channels.toFloat() + 1e-5f)
        for (channel in 0 until channels) output[channel * frameCount + t] /= scale
      }
    } else if (semantic) {
      val floor = output.maxOrNull()!! - 8f
      for (i in output.indices) output[i] = (maxOf(output[i], floor) + 4f) / 4f
    }
    return output
  }
}
