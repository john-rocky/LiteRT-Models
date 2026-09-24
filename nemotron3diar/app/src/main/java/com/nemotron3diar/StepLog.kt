package com.nemotron3diar

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Writes a diarization run for the gates (scripts/gate_kotlin_parity.py, scripts/gate_closed_loop.py):
 * steps.json (per step: chunk, length, cache state, the frame ids of the cached rows, ms; per compression: the kept
 * frame ids and the top-k boundary scores) and float32 little-endian bins.
 */
object StepLog {

  fun writeFloats(file: File, data: FloatArray, count: Int = data.size) {
    val bb = ByteBuffer.allocate(count * 4).order(ByteOrder.LITTLE_ENDIAN)
    bb.asFloatBuffer().put(data, 0, count)
    file.writeBytes(bb.array())
  }

  fun readFloats(file: File): FloatArray = floats(file.readBytes())

  /** float32 little-endian from a stream (an app asset); closes it. */
  fun floatsFromStream(input: InputStream): FloatArray = input.use { floats(it.readBytes()) }

  private fun floats(bytes: ByteArray): FloatArray {
    val out = FloatArray(bytes.size / 4)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
  }

  /** One step entry; [rowIds] are the frame ids of the cache + FIFO rows the step was fed. */
  fun stepJson(s: Step, rowIds: IntArray): String =
    buildString {
      append("{\"k\":").append(s.index)
      append(",\"g0\":").append(s.firstFrame)
      append(",\"nf\":").append(s.melFrames)
      append(",\"la\":").append(s.lookahead)
      append(",\"L\":").append(s.length)
      append(",\"pre\":").append(ints(s.pre))
      append(",\"post\":").append(ints(s.post))
      append(",\"nout\":").append(s.numFrames)
      append(",\"ms\":{\"mel\":").append(s.melMs).append(",\"frontend\":").append(s.frontendMs)
      append(",\"encoder\":").append(s.encoderMs).append(",\"cache\":").append(s.cacheMs)
      append(",\"total\":").append(s.totalMs).append('}')
      append(",\"row_ids\":").append(ints(rowIds))
      append('}')
    }

  fun compressionJson(c: SpeakerCache.Compression): String =
    buildString {
      append("{\"step\":").append(c.step)
      append(",\"candidates\":").append(c.numCandidates)
      append(",\"kept_ids\":").append(ints(c.keptIds))
      append(",\"boost_boundaries\":[")
      c.boostBoundaries.forEachIndexed { i, b ->
        if (i > 0) append(',')
        append(if (b == null) "null" else floatList(b))
      }
      append("],\"select_boundary\":").append(floatList(c.selectBoundary))
      append('}')
    }

  fun runJson(meta: String, steps: List<String>, compressions: List<SpeakerCache.Compression>): String =
    buildString {
      append("{\"meta\":").append(meta)
      append(",\"steps\":[\n")
      append(steps.joinToString(",\n"))
      append("\n],\"compressions\":[\n")
      append(compressions.joinToString(",\n") { compressionJson(it) })
      append("\n]}\n")
    }

  private fun ints(a: IntArray) = a.joinToString(",", "[", "]")

  /** Shortest round-trip decimal; +-Infinity as Python's json reads them. */
  private fun floatList(a: FloatArray) = a.joinToString(",", "[", "]") { it.toString() }
}
