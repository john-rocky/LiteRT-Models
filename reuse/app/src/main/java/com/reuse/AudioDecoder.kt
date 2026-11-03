package com.reuse

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/** Decodes an audio Uri (WAV directly, other formats with MediaCodec) to mono float PCM at [rate] Hz. */
object AudioDecoder {

  fun decode(context: Context, uri: Uri, rate: Int, maxSeconds: Int): FloatArray {
    val head = ByteArray(12)
    context.contentResolver.openInputStream(uri)!!.use { s ->
      var n = 0
      while (n < head.size) {
        val r = s.read(head, n, head.size - n)
        if (r < 0) break
        n += r
      }
    }
    val audio =
      if (WavFile.isWav(head)) {
        WavFile.read(context.contentResolver.openInputStream(uri)!!.use { it.readBytes() })
      } else {
        decodeWithCodec(context, uri, maxSeconds)
      }
    val x = resample(audio.samples, audio.sampleRate, rate)
    return if (x.size <= maxSeconds * rate) x else x.copyOf(maxSeconds * rate)
  }

  private fun decodeWithCodec(context: Context, uri: Uri, maxSeconds: Int): WavFile.Audio {
    val extractor = MediaExtractor()
    extractor.setDataSource(context, uri, null)
    val track = (0 until extractor.trackCount).firstOrNull {
      (extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME) ?: "").startsWith("audio/")
    } ?: error("No audio track in the selected file.")
    extractor.selectTrack(track)
    val format = extractor.getTrackFormat(track)
    val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
    codec.configure(format, null, null, 0)
    codec.start()
    var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
    var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
    var encoding = AudioFormat.ENCODING_PCM_16BIT
    val pcm = ByteArrayOutputStream()
    val info = MediaCodec.BufferInfo()
    var inputDone = false
    var outputDone = false
    try {
      while (!outputDone && pcm.size().toLong() < maxSeconds.toLong() * rate * channels * 4 + (1 shl 20)) {
        if (!inputDone) {
          val i = codec.dequeueInputBuffer(10_000)
          if (i >= 0) {
            val n = extractor.readSampleData(codec.getInputBuffer(i)!!, 0)
            if (n < 0) {
              codec.queueInputBuffer(i, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
              inputDone = true
            } else {
              codec.queueInputBuffer(i, 0, n, extractor.sampleTime, 0)
              extractor.advance()
            }
          }
        }
        val o = codec.dequeueOutputBuffer(info, 10_000)
        if (o >= 0) {
          val buf = codec.getOutputBuffer(o)!!
          val bytes = ByteArray(info.size)
          buf.position(info.offset)
          buf.get(bytes)
          pcm.write(bytes)
          codec.releaseOutputBuffer(o, false)
          if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
        } else if (o == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
          val f = codec.outputFormat
          rate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
          channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
          if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
        }
      }
    } finally {
      codec.stop()
      codec.release()
      extractor.release()
    }
    val b = ByteBuffer.wrap(pcm.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
    val isFloat = encoding == AudioFormat.ENCODING_PCM_FLOAT
    val frames = b.remaining() / (if (isFloat) 4 else 2) / channels
    val mono = FloatArray(frames)
    for (i in 0 until frames) {
      var acc = 0f
      for (c in 0 until channels) acc += if (isFloat) b.float else b.short / 32768f
      mono[i] = acc / channels
    }
    return WavFile.Audio(mono, rate)
  }

  /** Band-limited resampling: a Hann-windowed sinc with 16 zero crossings on each side, one table per phase. */
  fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
    if (from == to) return x
    val g = gcd(from, to)
    val up = to / g
    val down = from / g
    val cutoff = 0.95 * min(1.0, to.toDouble() / from)
    val half = ceil(16 / cutoff).toInt()
    val taps = 2 * half + 1
    val table = Array(up) { p ->
      DoubleArray(taps) { k ->
        val d = (k - half) - p.toDouble() / up // input samples from the output instant
        val arg = cutoff * d
        val sinc = if (abs(arg) < 1e-12) 1.0 else sin(PI * arg) / (PI * arg)
        if (abs(d) >= half) 0.0 else cutoff * sinc * (0.5 + 0.5 * cos(PI * d / half))
      }
    }
    val n = (x.size.toLong() * up / down).toInt()
    val out = FloatArray(n)
    for (i in 0 until n) {
      val pos = i.toLong() * down
      val base = (pos / up).toInt()
      val h = table[(pos % up).toInt()]
      var acc = 0.0
      for (k in 0 until taps) {
        val j = base + k - half
        if (j >= 0 && j < x.size) acc += x[j] * h[k]
      }
      out[i] = acc.toFloat()
    }
    return out
  }

  private tailrec fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
}
