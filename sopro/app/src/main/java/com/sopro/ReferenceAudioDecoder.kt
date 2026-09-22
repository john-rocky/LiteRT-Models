// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder

/** Platform decoder, interleaved PCM downmix and the shared sinc frontend. */
object ReferenceAudioDecoder {
  fun decodeBundled(context: Context): Decoded {
    val file = java.io.File(context.filesDir, "reference_import/demo_voice.wav")
    file.parentFile?.mkdirs()
    context.assets.open("demo_voice.wav").use { source ->
      file.outputStream().use { source.copyTo(it) }
    }
    return decode(context, Uri.fromFile(file))
  }

  data class Decoded(
    val wave: FloatArray,
    val sampleRate: Int,
    val originalRate: Int,
    val channels: Int,
  )

  fun decode(context: Context, uri: Uri, shouldStop: () -> Boolean = { false }): Decoded {
    val extractor = MediaExtractor()
    var codec: MediaCodec? = null
    try {
      extractor.setDataSource(context, uri, null)
      val track =
        (0 until extractor.trackCount).firstOrNull {
          extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: error("Selected file has no audio track")
      extractor.selectTrack(track)
      val format = extractor.getTrackFormat(track)
      val mime = requireNotNull(format.getString(MediaFormat.KEY_MIME))
      var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
      var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
      var encoding = AudioFormat.ENCODING_PCM_16BIT
      codec = MediaCodec.createDecoderByType(mime)
      codec.configure(format, null, null, 0)
      codec.start()
      val bufferInfo = MediaCodec.BufferInfo()
      val chunks = mutableListOf<FloatArray>()
      var inputDone = false
      var outputDone = false
      var lastOutput = System.nanoTime()
      while (!outputDone) {
        if (shouldStop())
          throw java.util.concurrent.CancellationException("Reference decoding stopped")
        check(System.nanoTime() - lastOutput < 30_000_000_000L) {
          "Audio decoder produced no output for 30 seconds"
        }
        if (!inputDone) {
          val input = codec.dequeueInputBuffer(10000)
          if (input >= 0) {
            val buffer = requireNotNull(codec.getInputBuffer(input))
            buffer.clear()
            val size = extractor.readSampleData(buffer, 0)
            if (size < 0) {
              codec.queueInputBuffer(input, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
              inputDone = true
            } else {
              codec.queueInputBuffer(input, 0, size, extractor.sampleTime, 0)
              extractor.advance()
            }
          }
        }
        when (val index = codec.dequeueOutputBuffer(bufferInfo, 10000)) {
          MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
            val outputFormat = codec.outputFormat
            rate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            channels = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            encoding =
              if (outputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING))
                outputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
              else AudioFormat.ENCODING_PCM_16BIT
          }
          else ->
            if (index >= 0) {
              if (bufferInfo.size > 0) {
                val buffer =
                  requireNotNull(codec.getOutputBuffer(index))
                    .duplicate()
                    .order(ByteOrder.LITTLE_ENDIAN)
                buffer.position(bufferInfo.offset)
                buffer.limit(bufferInfo.offset + bufferInfo.size)
                val bytesPerSample =
                  when (encoding) {
                    AudioFormat.ENCODING_PCM_FLOAT,
                    AudioFormat.ENCODING_PCM_32BIT -> 4
                    AudioFormat.ENCODING_PCM_24BIT_PACKED -> 3
                    AudioFormat.ENCODING_PCM_8BIT -> 1
                    AudioFormat.ENCODING_PCM_16BIT -> 2
                    else -> error("Unsupported decoded PCM encoding $encoding")
                  }
                require(channels > 0 && buffer.remaining() % (channels * bytesPerSample) == 0)
                val mono =
                  FloatArray(buffer.remaining() / (channels * bytesPerSample)) {
                    var sum = 0f
                    repeat(channels) {
                      sum +=
                        when (encoding) {
                          AudioFormat.ENCODING_PCM_FLOAT -> buffer.float
                          AudioFormat.ENCODING_PCM_32BIT -> buffer.int / 2147483648f
                          AudioFormat.ENCODING_PCM_24BIT_PACKED -> {
                            val value =
                              (buffer.get().toInt() and 255) or
                                ((buffer.get().toInt() and 255) shl 8) or
                                (buffer.get().toInt() shl 16)
                            value / 8388608f
                          }
                          AudioFormat.ENCODING_PCM_8BIT ->
                            ((buffer.get().toInt() and 255) - 128) / 128f
                          else -> buffer.short / 32768f
                        }
                    }
                    sum / channels
                  }
                chunks += mono
                lastOutput = System.nanoTime()
              }
              outputDone = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
              codec.releaseOutputBuffer(index, false)
            }
        }
      }
      val mono = FloatArray(chunks.sumOf { it.size })
      var offset = 0
      chunks.forEach {
        it.copyInto(mono, offset)
        offset += it.size
      }
      require(mono.isNotEmpty()) { "Selected file has no decoded audio samples" }
      return Decoded(GenericResampler.resample(mono, rate), 24000, rate, channels)
    } finally {
      codec?.let {
        try {
          it.stop()
        } catch (_: IllegalStateException) {}
        it.release()
      }
      extractor.release()
    }
  }
}
