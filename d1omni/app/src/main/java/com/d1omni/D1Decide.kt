package com.d1omni

import kotlin.math.max
import kotlin.math.roundToLong

/** One question of a Decide: its row, its probabilities, what the screen shows and its times. */
class D1RunQuestion(
  val qid: String,
  val question: D1Question,
  /** The probabilities in option order ([yes, no] for a noul), float64, unrounded. */
  val probabilities: DoubleArray,
  val shown: D1Shown,
  /** `prompt.answer()` of [probabilities]. */
  val answer: Map<String, Any?>,
  val ids: IntArray,
  val markers: IntArray,
  val prefixRows: Int,
  val bucket: Int,
  /** The decision call: input writes + `run()` + read-back, whole ms. */
  val inferMs: Long,
  /** The question from its inputs to its answer (inputs, call, read-out, answer), whole ms. */
  val totalMs: Long,
  val backend: D1Backend,
  val precision: D1Precision?,
)

/**
 * One Decide's work and its times: the host steps' and media graphs' whole ms, the media rows P, the questions, the
 * work in nanoseconds (host steps + media graphs + decision calls, from the input's samples / bytes / text to the last
 * answer) and its wall-clock start and end, and the media's details for the run JSON.
 */
class D1Work(
  val totalNanos: Long,
  val startedWall: Long,
  val endedWall: Long,
  val hostMs: LinkedHashMap<String, Long>,
  val mediaGraphMs: LinkedHashMap<String, Long>,
  val prefixRows: Int,
  val questions: List<D1RunQuestion>,
  val media: LinkedHashMap<String, Any?>,
) {
  /** The ms the screen shows under the answers (the work rounded once). */
  val itemMs: Long
    get() = D1Text.itemMs(totalNanos)

  val buckets: List<Int>
    get() = questions.map { it.bucket }
}

/**
 * What Decide runs for each input, on [D1Runtime.dispatcher]: a voice note's samples through the mel and the audio
 * graph, a photo's bytes through the decoder, the shrink to [D1Photo.MAX_SIDE] and the vision graphs, a message's text
 * as it is; then one decision call per question on the smallest resident graph that holds its row. [onAnswer] runs after
 * each question's call (the screen fills the answers in as they come).
 */
object D1Decide {
  fun audio(
    engine: D1AppEngine,
    samples: ShortArray,
    questions: LinkedHashMap<String, D1Question>,
    onAnswer: (D1RunQuestion) -> Unit = {},
  ): D1Work {
    val startedWall = System.currentTimeMillis()
    val start = System.nanoTime()
    val hostMs = LinkedHashMap<String, Long>()
    val mediaGraphMs = LinkedHashMap<String, Long>()
    val media = LinkedHashMap<String, Any?>()
    val audio = engine.decide.audio.audioPrefix(samples)
    hostMs["mel"] = ms(audio.waveformNanos + audio.melNanos)
    hostMs["inputs"] = ms(audio.inputsNanos + audio.rowsNanos)
    mediaGraphMs["audio"] = audio.call.totalMs.roundToLong()
    media["info"] = audio.info.toJson()
    media["steps_ms"] = audio.times()
    media["audio_backend"] = audio.backend.wireName
    media["audio_precision"] = audio.precision?.wireName
    return finish(engine, null, questions, audio.prefix, audio.info.prefixRows, D1Kind.AUDIO, start, startedWall, hostMs,
      mediaGraphMs, media, onAnswer)
  }

  fun image(
    engine: D1AppEngine,
    bytes: ByteArray,
    questions: LinkedHashMap<String, D1Question>,
    onAnswer: (D1RunQuestion) -> Unit = {},
  ): D1Work {
    val startedWall = System.currentTimeMillis()
    val start = System.nanoTime()
    val hostMs = LinkedHashMap<String, Long>()
    val mediaGraphMs = LinkedHashMap<String, Long>()
    val media = LinkedHashMap<String, Any?>()
    val decoded = D1Image.decode(bytes)
    val decodedNanos = System.nanoTime() - start
    val t1 = System.nanoTime()
    val rgb = D1Photo.shrink(decoded.rgb)
    val shrinkNanos = System.nanoTime() - t1
    val run = engine.vision.imagePrefix(rgb)
    hostMs["decode"] = ms(decodedNanos)
    hostMs["shrink"] = ms(shrinkNanos)
    hostMs["resize"] = ms(run.resizeNanos)
    hostMs["patches"] = ms(run.patchesNanos)
    hostMs["pos"] = ms(run.positionsNanos)
    hostMs["unshuffle"] = ms(run.unshuffleNanos)
    mediaGraphMs["tower"] = ms(run.towerNanos)
    mediaGraphMs["projector"] = ms(run.projectorNanos)
    media["decode"] =
      linkedMapOf(
        "format" to D1Photo.format(bytes),
        "exif_orientation" to decoded.orientation,
        "bitmap_color_space" to decoded.colorSpace,
        "png_chunks_stripped" to decoded.strippedChunks,
        "decoded_hw" to listOf(decoded.rgb.height, decoded.rgb.width),
        "model_hw" to listOf(rgb.height, rgb.width),
        "shrunk" to (rgb !== decoded.rgb),
        "rgb_sha256" to D1Answers.sha256(rgb.data),
      )
    media["prefix"] = run.toJson()
    media["tower_backend"] = engine.vision.towerBackend.wireName
    media["projector_backend"] = engine.vision.projectorBackend.wireName
    return finish(engine, null, questions, run.prefix, run.rows, D1Kind.IMAGE, start, startedWall, hostMs, mediaGraphMs,
      media, onAnswer)
  }

  fun text(
    engine: D1AppEngine,
    text: String,
    questions: LinkedHashMap<String, D1Question>,
    onAnswer: (D1RunQuestion) -> Unit = {},
  ): D1Work {
    val startedWall = System.currentTimeMillis()
    val start = System.nanoTime()
    return finish(engine, text, questions, null, 0, D1Kind.TEXT, start, startedWall, LinkedHashMap(), LinkedHashMap(),
      LinkedHashMap(), onAnswer)
  }

  private fun finish(
    engine: D1AppEngine,
    state: Any?,
    questions: LinkedHashMap<String, D1Question>,
    prefix: FloatArray?,
    prefixRows: Int,
    kind: D1Kind,
    start: Long,
    startedWall: Long,
    hostMs: LinkedHashMap<String, Long>,
    mediaGraphMs: LinkedHashMap<String, Long>,
    media: LinkedHashMap<String, Any?>,
    onAnswer: (D1RunQuestion) -> Unit,
  ): D1Work {
    val t0 = System.nanoTime()
    val rows = engine.rows(state, questions.values, prefixRows, kind)
    hostMs["encode"] = ms(System.nanoTime() - t0)
    val out = ArrayList<D1RunQuestion>()
    for ((name, row) in questions.keys.zip(rows)) {
      val call = engine.question(row, prefix)
      val result =
        D1RunQuestion(
          name,
          row.question,
          call.probabilities,
          D1Answers.shown(row.question, call.probabilities),
          call.answer,
          row.ids,
          row.markers,
          row.prefixRows,
          call.bucket,
          call.call.totalMs.roundToLong(),
          ms(call.totalNanos),
          call.backend,
          call.precision,
        )
      out.add(result)
      onAnswer(result)
    }
    return D1Work(System.nanoTime() - start, startedWall, System.currentTimeMillis(), hostMs, mediaGraphMs, prefixRows,
      out, media)
  }

  private fun ms(nanos: Long): Long = (nanos / 1e6).roundToLong()
}

/**
 * The photo step before the model's own preprocessing, Android-free: a picture longer than [MAX_SIDE] px on its long
 * side is shrunk to it (the model repository's check-set pictures are that size) with the same float bilinear
 * antialias resize the provider's preprocessing uses (`D1Vision.resizeFloat`, `d1_vision_host.resize_float`), so that a
 * phone photo becomes one crop whose rows fit the 128- and 256-position graphs (a 12 MP photo would otherwise be cut into
 * ten tiles and need the 4,096-position graph). A smaller picture goes in unchanged.
 */
object D1Photo {
  const val MAX_SIDE = 384

  /** The picture's size for the model: [MAX_SIDE] on the long side, the short side scaled and rounded half to even. */
  fun shrunkSize(width: Int, height: Int): Pair<Int, Int> {
    if (max(width, height) <= MAX_SIDE) return width to height
    return if (width >= height) {
      MAX_SIDE to max(1, Math.rint(height.toDouble() * MAX_SIDE / width).toInt())
    } else {
      max(1, Math.rint(width.toDouble() * MAX_SIDE / height).toInt()) to MAX_SIDE
    }
  }

  fun shrink(rgb: D1Rgb): D1Rgb {
    val (width, height) = shrunkSize(rgb.width, rgb.height)
    return if (width == rgb.width && height == rgb.height) rgb else D1Vision.resizeFloat(rgb, height, width)
  }

  /** "png", "jpeg", "webp", "gif", "heif" or "unknown" from the file's first bytes. */
  fun format(bytes: ByteArray): String {
    fun at(offset: Int, text: String) =
      bytes.size >= offset + text.length && text.indices.all { bytes[offset + it] == text[it].code.toByte() }
    return when {
      D1ImageOps.isPng(bytes) -> "png"
      bytes.size >= 3 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() && bytes[2] == 0xFF.toByte() -> "jpeg"
      at(0, "RIFF") && at(8, "WEBP") -> "webp"
      at(0, "GIF8") -> "gif"
      at(4, "ftyp") -> "heif"
      else -> "unknown"
    }
  }
}
