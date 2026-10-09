package com.d1omni

import java.security.MessageDigest
import kotlin.math.roundToLong

/**
 * One unread item of the inbox demo: a voice note, a photo or a message ([kind]), the media file it
 * carries (audio and image) and its typed questions about the item; [header] is the card's title
 * ("Voice note", "Photo", "Message").
 */
class D1InboxItem(
  val item: String,
  val kind: D1Kind,
  val header: String,
  val mediaFile: String?,
  val mediaSha256: String?,
  val mediaBytes: Long?,
  /** The request's state as parsed JSON: a string, any JSON value, or null. */
  val state: Any?,
  val questions: LinkedHashMap<String, D1Question>,
)

/**
 * The inbox demo's fixture (`res/raw/inbox_demo.json`, `demo/fixtures/demo/inbox_demo.json` in the
 * conversion run): a title and the items in the order the demo answers them. Android-free.
 */
class D1InboxFixture(val id: String, val title: String, val items: List<D1InboxItem>) {
  /** Every question of every item, in order: (item, question name). */
  val questionCount: Int
    get() = items.sumOf { it.questions.size }

  companion object {
    const val FORMAT = "d1omni-inbox-demo/1"

    private val SHA256 = Regex("[0-9a-f]{64}")

    /** Parses and checks a fixture; throws [IllegalArgumentException] with the reason. */
    fun parse(bytes: ByteArray): D1InboxFixture {
      val root = D1Json.parse(bytes)
      require(root is Map<*, *>) { "a fixture is a JSON object" }
      require(root["fixture"] == FORMAT) { "fixture format ${root["fixture"]}, this app reads $FORMAT" }
      val id = root["id"] as? String ?: throw IllegalArgumentException("a fixture needs an id")
      val title = root["title"] as? String ?: throw IllegalArgumentException("a fixture needs a title")
      val list = root["items"] as? List<*> ?: throw IllegalArgumentException("a fixture needs items")
      require(list.isNotEmpty()) { "a fixture without items" }
      val items = list.map { item(it) }
      require(items.map { it.item }.toSet().size == items.size) { "two items share an id" }
      return D1InboxFixture(id, title, items)
    }

    private fun item(value: Any?): D1InboxItem {
      require(value is Map<*, *>) { "an item is a JSON object" }
      val id = value["item"] as? String ?: throw IllegalArgumentException("an item needs an id")
      require(D1Launch.fileNameValid(id)) { "item id $id: letters, digits, dot, underscore and hyphen" }
      val kind =
        D1Kind.entries.firstOrNull { it.wireName == value["kind"] }
          ?: throw IllegalArgumentException("$id: kind ${value["kind"]} is not text, image or audio")
      val header = value["header"] as? String ?: throw IllegalArgumentException("$id: no header")
      val media = value["media"] as? Map<*, *>
      if (kind == D1Kind.TEXT) {
        require(media == null) { "$id: a text item carries no media" }
      } else {
        require(media != null) { "$id: an $kind item needs media {file, sha256}" }
        val file = media["file"] as? String
        require(file != null && D1Launch.fileNameValid(file)) { "$id: media file $file is not a plain file name" }
        require((media["sha256"] as? String)?.matches(SHA256) == true) { "$id: media sha256 is not 64 hex digits" }
      }
      val questions = value["questions"] as? Map<*, *>
      require(questions != null && questions.isNotEmpty()) { "$id: no questions" }
      val named = LinkedHashMap<String, D1Question>()
      for ((name, question) in questions) {
        named[name as String] =
          try {
            D1Prompt.asQuestion(question)
          } catch (failure: IllegalArgumentException) {
            throw IllegalArgumentException("$id/$name: ${failure.message}", failure)
          }
      }
      return D1InboxItem(
        id,
        kind,
        header,
        media?.get("file") as String?,
        media?.get("sha256") as String?,
        (media?.get("bytes") as JsonNumber?)?.literal?.toLong(),
        value["state"],
        named,
      )
    }
  }
}

/** What an answer row shows: the answer's word, its probability to three decimals, the bar's length. */
data class D1Shown(val answer: String, val prob: String, val fraction: Double)

/**
 * The answer row of a question, from its probabilities in option order ([yes, no] for a noul),
 * Android-free: noul = "yes" and P(yes) when P(yes) >= 0.5, else "no" and 1 − P(yes); choice = the
 * name of the first most likely option and its probability; score = the text of the first most
 * likely level and its probability. The probability is shown to three decimals of its exact value,
 * half to even (`D1Readout.shown`), the rule `demo/check_take.py` checks.
 */
object D1InboxAnswer {
  const val DECIMALS = 3

  /** The keys of the probabilities in option order: yes / no, the option names, or the levels. */
  fun keys(question: D1Question): List<String> =
    when (question.type) {
      QuestionType.NOUL -> listOf("yes", "no")
      QuestionType.CHOICE -> question.choiceNames
      QuestionType.SCORE -> (0 until question.options).map { it.toString() }
    }

  fun shown(question: D1Question, probabilities: DoubleArray): D1Shown {
    if (question.type == QuestionType.NOUL) {
      val yes = probabilities[0]
      return if (yes >= 0.5) {
        D1Shown("yes", D1Readout.shown(yes, DECIMALS), yes)
      } else {
        D1Shown("no", D1Readout.shown(1.0 - yes, DECIMALS), 1.0 - yes)
      }
    }
    val best = D1Prompt.firstArgmax(probabilities)
    val word =
      if (question.type == QuestionType.CHOICE) question.choiceNames[best]
      else D1Prompt.criterion((question.criteria as List<*>)[best])
    return D1Shown(word, D1Readout.shown(probabilities[best], DECIMALS), probabilities[best])
  }

  /** Every word a question's row can show (the layout measures the widest before drawing). */
  fun words(question: D1Question): List<String> =
    when (question.type) {
      QuestionType.NOUL -> listOf("yes", "no")
      QuestionType.CHOICE -> question.choiceNames
      QuestionType.SCORE -> (question.criteria as List<*>).map { D1Prompt.criterion(it) }
    }

  /** The key of the first most likely option. */
  fun argmaxKey(question: D1Question, probabilities: DoubleArray): String =
    keys(question)[D1Prompt.firstArgmax(probabilities)]

  /** sha256 of the IDs as int32 little-endian, lower-case hex (the run JSON's `ids_sha256`). */
  fun idsSha256(ids: IntArray): String {
    val bytes = ByteArray(ids.size * Int.SIZE_BYTES)
    for ((index, id) in ids.withIndex()) {
      for (shift in 0 until Int.SIZE_BYTES) bytes[index * Int.SIZE_BYTES + shift] = (id ushr (8 * shift)).toByte()
    }
    return MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
  }

}

/** The words on the inbox screens, Android-free (the JVM tests and the run JSON use the same). */
object D1InboxText {
  const val ICON_AUDIO = "🎤"
  const val ICON_IMAGE = "📷"
  const val ICON_TEXT = "✉️"

  fun icon(kind: D1Kind): String =
    when (kind) {
      D1Kind.AUDIO -> ICON_AUDIO
      D1Kind.IMAGE -> ICON_IMAGE
      D1Kind.TEXT -> ICON_TEXT
    }

  /** "Voice note · 8.7 s" for a clip of [samples] at 16 kHz, else the header itself. */
  fun header(item: D1InboxItem, samples: Int?): String =
    if (item.kind == D1Kind.AUDIO && samples != null) {
      "${item.header} · ${"%.1f".format(java.util.Locale.ROOT, samples / D1Audio.SAMPLE_RATE.toDouble())} s"
    } else {
      item.header
    }

  /** The header's right end: the media graph's ms ("audio 32 ms", "vision 140 ms"), none for text. */
  fun mediaMs(kind: D1Kind, ms: Long?): String? =
    when {
      ms == null || kind == D1Kind.TEXT -> null
      kind == D1Kind.AUDIO -> "audio $ms ms"
      else -> "vision $ms ms"
    }

  /** The reserved width of [mediaMs]. */
  fun mediaMsWidest(kind: D1Kind): String? = mediaMs(kind, 888)

  fun rowMs(ms: Long): String = "$ms ms"

  const val ROW_MS_WIDEST = "888 ms"
  const val PROB_WIDEST = "0.000"

  fun itemTotal(answers: Int, ms: Long): String = "$answers ${if (answers == 1) "answer" else "answers"} · $ms ms"

  fun itemTotalWidest(answers: Int): String = itemTotal(answers, 8888)

  /** An item's work in whole ms, as its card shows it (each item rounded on its own). */
  fun itemMs(nanos: Long): Long = (nanos / 1_000_000.0).roundToLong()

  /** The footer's total: the cards' totals as shown, added up (so the screen's numbers add up). */
  fun requestTotalMs(itemNanos: List<Long>): Long = itemNanos.sumOf { itemMs(it) }

  /** The pill's labels by state. */
  val PILL = linkedMapOf("ready" to "READY", "playing" to "● PLAYING", "deciding" to "DECIDING", "done" to "DONE")

  /**
   * "GPU FP32" or "GPU FP16 (FP32 accum)" when every graph runs on the GPU at that precision, "GPU"
   * when the GPU graphs differ in precision, "CPU 4 threads" when every graph runs on the CPU, and
   * "GPU + CPU" when some graph fell back to the CPU.
   */
  fun accelerator(graphs: List<Pair<D1Backend, D1Precision>>): String {
    val backends = graphs.map { it.first }.distinct()
    if (backends == listOf(D1Backend.CPU)) return "CPU ${D1Decider.CPU_THREADS} threads"
    if (D1Backend.CPU in backends) return "GPU + CPU"
    val precisions = graphs.map { it.second }.distinct()
    return when (precisions) {
      listOf(D1Precision.FP32) -> "GPU FP32"
      listOf(D1Precision.FP16_FP32_ACCUM) -> "GPU FP16 (FP32 accum)"
      else -> "GPU"
    }
  }

  /** "d1-omni-600M fp16 · L128 + L256 · audio T1001 · vision tower" from the resident graphs. */
  fun graphsLine(decision: List<Int>, audio: List<Int>, vision: Boolean): String =
    listOfNotNull(
        "d1-omni-600M fp16",
        decision.sorted().joinToString(" + ") { "L$it" }.ifEmpty { null },
        audio.sorted().joinToString(" + ") { "audio T$it" }.ifEmpty { null },
        if (vision) "vision tower" else null,
      )
      .joinToString(" · ")

  /** The footer: device, LiteRT and accelerator; the graphs; the total (empty until the end). */
  fun footer(device: String, accelerator: String, graphs: String, totalMs: Long?, airplane: Boolean): List<String> =
    listOf(
      "$device · LiteRT ${D1Decider.LITERT_VERSION} · $accelerator",
      graphs,
      if (totalMs == null) "" else "total $totalMs ms · airplane mode ${if (airplane) "on" else "off"}",
    )

  /** The widest the footer's last line gets (its room is kept from the start). */
  fun footerTotalWidest(): String = "total 8888 ms · airplane mode off"
}
