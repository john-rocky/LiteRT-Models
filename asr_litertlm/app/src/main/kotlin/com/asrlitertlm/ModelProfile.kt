package com.asrlitertlm

/** How a model writes its answer. */
enum class AnswerFormat {
  /** Qwen3-ASR and its fine-tunes: `language <name><asr_text><transcript>`. */
  LANGUAGE_TAGGED,

  /** Fun-ASR-Nano: the transcript alone. */
  PLAIN,
}

/** A transcript, split into the language the model named (empty when it named none) and the text. */
data class Answer(val language: String, val text: String)

/**
 * One bundle's contract with this app. All three bundles go through the same calls: one Engine, a new Conversation per
 * clip, one user message holding only the audio (each bundle's own template adds the rest: Fun-ASR's default
 * instruction `语音转写：`, Qwen3-ASR's audio markers), greedy sampling and [maxOutputTokens]. Only the shape of the
 * answer differs ([answerFormat]).
 */
data class ModelProfile(
  /** Stable key for launch extras and the device check. */
  val id: String,
  val displayName: String,
  /** The file name the app looks for in its external files directory. */
  val fileName: String,
  val hubRepo: String,
  /** Size of the file on the Hub, shown before the file is on the phone. */
  val hubBytes: Long,
  val license: String,
  val languages: String,
  val answerFormat: AnswerFormat,
  val maxOutputTokens: Int = 512,
) {
  /** Splits a raw answer into language and text. */
  fun parse(raw: String): Answer =
    when (answerFormat) {
      AnswerFormat.PLAIN -> Answer(language = "", text = raw.trim())
      AnswerFormat.LANGUAGE_TAGGED -> parseTagged(raw)
    }

  companion object {
    private const val ASR_TEXT = "<asr_text>"
    private const val LANGUAGE_PREFIX = "language "

    val QWEN3_ASR_1_7B =
      ModelProfile(
        id = "qwen3-asr-1.7b",
        displayName = "Qwen3-ASR-1.7B",
        fileName = "Qwen3-ASR-1.7B.litertlm",
        hubRepo = "litert-community/Qwen3-ASR-1.7B",
        hubBytes = 2_693_064_192L,
        license = "Apache-2.0",
        languages = "30 languages",
        answerFormat = AnswerFormat.LANGUAGE_TAGGED,
      )

    val FUN_ASR_NANO_2512 =
      ModelProfile(
        id = "fun-asr-nano-2512",
        displayName = "Fun-ASR-Nano-2512",
        fileName = "Fun-ASR-Nano-2512.litertlm",
        hubRepo = "litert-community/Fun-ASR-Nano-2512",
        hubBytes = 1_255_894_736L,
        license = "Apache-2.0",
        languages = "zh / en / ja",
        answerFormat = AnswerFormat.PLAIN,
      )

    val CONFUCIUS4_R2T2 =
      ModelProfile(
        id = "confucius4-r2t2",
        displayName = "Confucius4-R2T2",
        fileName = "Confucius4-R2T2.litertlm",
        hubRepo = "mlboydaisuke/Confucius4-R2T2-LiteRT",
        hubBytes = 2_693_064_192L,
        license = "NetEase Youdao Model Use License",
        languages = "zh / en (Qwen3-ASR fine-tune)",
        answerFormat = AnswerFormat.LANGUAGE_TAGGED,
      )

    /** The picker's rows, in screen order. */
    val ALL = listOf(QWEN3_ASR_1_7B, FUN_ASR_NANO_2512, CONFUCIUS4_R2T2)

    fun byId(id: String?): ModelProfile? = ALL.firstOrNull { it.id == id }

    /**
     * `language <name><asr_text><text>` -> (name, text), as qwen-asr's parse_asr_output does without a forced
     * language; "language None" gives an empty name, and an answer without the tag is all text.
     */
    private fun parseTagged(raw: String): Answer {
      val answer = raw.trim()
      val tag = answer.indexOf(ASR_TEXT)
      if (tag < 0) return Answer(language = "", text = answer)
      val meta = answer.substring(0, tag)
      val name =
        meta
          .lineSequence()
          .map { it.trim() }
          .firstOrNull { it.lowercase().startsWith(LANGUAGE_PREFIX) }
          ?.substring(LANGUAGE_PREFIX.length)
          ?.trim()
          .orEmpty()
      val language = if (name.equals("None", ignoreCase = true)) "" else name
      return Answer(language = language, text = answer.substring(tag + ASR_TEXT.length).trim())
    }
  }
}
