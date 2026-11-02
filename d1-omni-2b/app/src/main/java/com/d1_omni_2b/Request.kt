package com.d1_omni_2b

import java.io.File
import java.security.MessageDigest

/** The question types the editor offers, and how each is written in the model's question JSON. */
enum class QType(val label: String) {
  NOUL("Yes / no"),
  CHOICE("Choice"),
  MULTIPLE("Several choices"),
  SCORE("Score"),
  TEXT("Text"),
  NUMBER("Number"),
  ESTIMATE("Number in a range"),
  DATE("Date"),
  BOX("Box"),
  BOXES("Boxes"),
}

data class OptionDraft(val name: String, val text: String)

/** A clip in the editor: the WAV file it was read from (in the app's folder), and what the reader made of it. */
class ClipDraft(val key: Long, val file: File, val label: String, val clip: AudioClip, val sha256: String) {
  val seconds: Double
    get() = clip.seconds
}

/** One question card of the editor; [spoken] is its recorded question, which replaces [instructions]. */
data class QuestionDraft(
  val key: Long,
  val name: String,
  val type: QType,
  val instructions: String,
  val options: List<OptionDraft> = listOf(OptionDraft("yes", "Yes"), OptionDraft("no", "No")),
  val levels: List<String> = listOf("Low", "Medium", "High"),
  val low: String = "0",
  val high: String = "10",
  val step: String = "1",
  val unit: String = "",
  val max: String = "",
  val spoken: ClipDraft? = null,
) {
  /** The question in the model's JSON (type, instructions, criteria, ...); a spoken one's instructions are its clip. */
  fun toJson(): LinkedHashMap<String, Any?> {
    val out = LinkedHashMap<String, Any?>()
    out["type"] =
      when (type) {
        QType.NOUL -> "noul"
        QType.CHOICE,
        QType.MULTIPLE -> "choice"
        QType.SCORE -> "score"
        QType.TEXT -> "text"
        QType.NUMBER,
        QType.ESTIMATE -> "number"
        QType.DATE -> "date"
        QType.BOX -> "box"
        QType.BOXES -> "boxes"
      }
    if (type == QType.MULTIPLE) out["multiple"] = true
    out["instructions"] = spoken?.clip ?: instructions
    when (type) {
      QType.CHOICE,
      QType.MULTIPLE ->
        out["criteria"] = LinkedHashMap<String, Any?>().apply { options.forEach { put(it.name.trim(), it.text.trim().ifEmpty { null }) } }
      QType.SCORE -> out["criteria"] = levels.map { it.trim() }
      QType.ESTIMATE -> {
        out["range"] = listOf(number("low", low), number("high", high))
        if (step.isNotBlank()) out["step"] = number("step", step)
      }
      QType.BOXES -> if (max.isNotBlank()) out["max"] = number("max", max)
      else -> Unit
    }
    if ((type == QType.NUMBER || type == QType.ESTIMATE) && unit.isNotBlank()) out["unit"] = unit.trim()
    return out
  }

  private fun number(field: String, text: String): JsonNumber {
    val trimmed = text.trim()
    require(NUMBER.matches(trimmed)) { "question '$name': $field \"$text\" is not a number" }
    return JsonNumber(trimmed)
  }

  companion object {
    private val NUMBER = Regex("-?(0|[1-9]\\d*)(\\.\\d+)?([eE][-+]?\\d+)?")

    /** A card from the question JSON of a bundled example or a run file; [spoken] reads a spoken one's clip. */
    fun fromJson(key: Long, name: String, raw: Map<*, *>, spoken: (Map<*, *>) -> ClipDraft): QuestionDraft {
      val kind = raw["type"] as String
      val multiple = raw["multiple"] == true
      val type =
        when {
          kind == "choice" && multiple -> QType.MULTIPLE
          kind == "number" && raw.containsKey("range") -> QType.ESTIMATE
          else -> QType.entries.first { it.name == kind.uppercase() || (it == QType.NOUL && kind == "noul") }
        }
      val said = raw["instructions"]
      var draft =
        if (said is Map<*, *>) QuestionDraft(key, name, type, "", spoken = spoken(said))
        else QuestionDraft(key, name, type, said as String)
      when (val criteria = raw["criteria"]) {
        is Map<*, *> -> draft = draft.copy(options = criteria.map { (k, v) -> OptionDraft(k as String, (v as String?) ?: "") })
        is List<*> -> draft = draft.copy(levels = criteria.map { it as String })
      }
      (raw["range"] as List<*>?)?.let { draft = draft.copy(low = text(it[0]), high = text(it[1]), step = raw["step"]?.let { s -> text(s) } ?: "") }
      (raw["unit"] as String?)?.let { draft = draft.copy(unit = it) }
      raw["max"]?.let { draft = draft.copy(max = text(it)) }
      return draft
    }

    private fun text(value: Any?): String = (value as? JsonNumber)?.literal ?: value.toString()
  }
}

/** A request as the editor holds it, checked and turned into the host's call before anything runs. */
object Requests {
  /** The longest clip the app records, in seconds (the largest audio graph takes 30.015 s). */
  const val RECORD_SECONDS = 30

  /**
   * (state, named questions, media) of the cards, or an [IllegalArgumentException] that says what to change: an
   * invalid JSON state, a request with nothing to read, a question without a name, a duplicate name, or a question
   * that is neither typed nor recorded.
   */
  fun build(
    stateText: String,
    stateIsJson: Boolean,
    picture: Picture?,
    clips: List<ClipDraft>,
    questions: List<QuestionDraft>,
  ): Triple<Any?, LinkedHashMap<String, Any?>, List<Any>> {
    val state: Any? =
      when {
        stateIsJson ->
          try {
            D1Json.parse(stateText)
          } catch (failure: Exception) {
            throw IllegalArgumentException("The state is not valid JSON: ${failure.message}")
          }
        stateText.isBlank() ->
          if (picture != null || clips.isNotEmpty()) null else throw IllegalArgumentException("Write a state, add a recording, or add a photo.")
        else -> stateText
      }
    require(questions.isNotEmpty()) { "Add a question." }
    val named = LinkedHashMap<String, Any?>()
    for (q in questions) {
      require(q.name.isNotBlank()) { "Every question needs a name." }
      require(q.name !in named) { "Two questions are named '${q.name}'." }
      require(q.spoken != null || q.instructions.isNotBlank()) {
        "Question '${q.name}' is empty: type the question, or record it."
      }
      named[q.name] = q.toJson()
    }
    val media = ArrayList<Any>()
    if (picture != null) media.add(picture)
    clips.forEach { media.add(it.clip) }
    return Triple(state, named, media)
  }

  /**
   * A clip the app takes, or an [IllegalArgumentException] that says why not: longer than the largest audio graph
   * ([maxFrames] mel frames), or at a rate the app does not resample.
   */
  fun checkClip(clip: AudioClip, extractor: FeatureExtractor, maxFrames: Int) {
    Resample.shape(clip.rate, extractor.samplingRate)?.let { (phases, taps) ->
      require(phases.toLong() * taps <= Resample.MAX_KERNEL) {
        "${clip.name}: a clip at ${clip.rate} Hz needs a resampling filter too large for this app; convert it to " +
          "${extractor.samplingRate} Hz first"
      }
    }
    val frames = extractor.numFrames(Resample.length(clip.mono.size, clip.rate, extractor.samplingRate))
    require(frames <= maxFrames) {
      "${clip.name} is ${String.format(java.util.Locale.ROOT, "%.2f", clip.seconds)} s; the audio encoder takes at most " +
        "${PyFormat.g(extractor.seconds(maxFrames))} s ($maxFrames frames). The app never cuts a clip: record or pick a shorter one."
    }
  }

  fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  /** The SHA-256 of float32 values as little-endian bytes (what numpy's `tobytes()` hashes). */
  fun sha256(values: FloatArray): String {
    val digest = MessageDigest.getInstance("SHA-256")
    val buffer = java.nio.ByteBuffer.allocate(4096 * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    var at = 0
    while (at < values.size) {
      buffer.clear()
      val n = minOf(4096, values.size - at)
      for (i in 0 until n) buffer.putFloat(values[at + i])
      digest.update(buffer.array(), 0, n * 4)
      at += n
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }
}

/** The model files a stage needs, named when missing (before anything loads). */
object ModelFiles {
  /** What every request needs: the provider's files and the embedding table. */
  val HOST = listOf(
    "provider/tokenizer.json",
    "provider/config.json",
    "provider/calibration.json",
    "provider/processor_config.json",
    "provider/chat_template.jinja",
    "tables/embed_table.npy",
  )

  /** The files of [names] missing from [dir]. */
  fun missing(dir: File, names: List<String>): List<String> = names.filter { !File(dir, it).isFile }

  /** Throws an [IllegalArgumentException] naming the files of [names] missing from [dir], and what to do. */
  fun require(dir: File, names: List<String>, what: String) {
    val gone = missing(dir, names)
    require(gone.isEmpty()) {
      "Missing model file${if (gone.size > 1) "s" else ""} for $what in ${dir.absolutePath}: ${gone.joinToString()}. " +
        "Copy them with scripts/push_model.sh (README.md); nothing was loaded."
    }
  }
}
