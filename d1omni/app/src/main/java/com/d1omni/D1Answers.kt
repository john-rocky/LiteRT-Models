package com.d1omni

import java.security.MessageDigest
import kotlin.math.roundToLong

/** What an answer shows: the answer's word and its probability to three decimals. */
data class D1Shown(val answer: String, val prob: String)

/**
 * The answer of a question as the screen shows it, from its probabilities in option order ([yes, no] for a noul),
 * Android-free: noul = "yes" and P(yes) when P(yes) >= 0.5, else "no" and 1 − P(yes); choice = the name of the first
 * most likely option and its probability; score = the text of the first most likely level and its probability. The
 * probability is shown to three decimals of its exact value, half to even (`D1Readout.shown`), the rule
 * `demo/check_take.py` checks.
 */
object D1Answers {
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
      return if (yes >= 0.5) D1Shown("yes", D1Readout.shown(yes, DECIMALS)) else D1Shown("no", D1Readout.shown(1.0 - yes, DECIMALS))
    }
    val best = D1Prompt.firstArgmax(probabilities)
    val word =
      if (question.type == QuestionType.CHOICE) question.choiceNames[best]
      else D1Prompt.criterion((question.criteria as List<*>)[best])
    return D1Shown(word, D1Readout.shown(probabilities[best], DECIMALS))
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
    return sha256(bytes)
  }

  fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}

/** The words and numbers on the screen, Android-free (the JVM tests and the run JSON use the same). */
object D1Text {
  /** The pill's labels by state. */
  val PILL = linkedMapOf(
    "loading" to "LOADING",
    "ready" to "READY",
    "recording" to "RECORDING",
    "deciding" to "DECIDING",
    "done" to "DONE",
  )

  /** Pill colours by state (READY grey, RECORDING red, DECIDING blue, DONE green: round 4's palette). */
  val PILL_PALETTE: Map<String, String> =
    linkedMapOf(
      "loading" to "#5F6368",
      "ready" to "#5F6368",
      "recording" to "#E53935",
      "deciding" to "#1565C0",
      "done" to "#2E7D32",
    )

  /** An item's work in whole ms, as the screen shows it (each input rounded on its own). */
  fun itemMs(nanos: Long): Long = (nanos / 1_000_000.0).roundToLong()

  /** "L128", or "L128 + L256" when the questions ran on different graphs. */
  fun buckets(buckets: List<Int>): String = buckets.distinct().sorted().joinToString(" + ") { "L$it" }

  /** The small line under the answers: "312 ms · L256". */
  fun msLine(ms: Long, buckets: List<Int>): String = "$ms ms · ${buckets(buckets)}"

  /** "Voice note · 7.4 s" style seconds of [samples] at 16 kHz, one decimal. */
  fun seconds(samples: Int): String = "%.1f s".format(java.util.Locale.ROOT, samples / D1Audio.SAMPLE_RATE.toDouble())

  /**
   * The summary's headline: "3 inputs · 4 answers · 760 ms · airplane mode on" — the inputs answered, their answers and
   * the sum of the ms each input showed (so the screen's numbers add up).
   */
  fun summary(inputs: Int, answers: Int, totalMs: Long, airplane: Boolean): String =
    "${summaryCounts(inputs, answers)} · $totalMs ms · ${airplane(airplane)}"

  /** The summary's first line: "3 inputs · 4 answers". */
  fun summaryCounts(inputs: Int, answers: Int): String =
    "$inputs ${if (inputs == 1) "input" else "inputs"} · $answers ${if (answers == 1) "answer" else "answers"}"

  /** The summary's last line: "airplane mode on". */
  fun airplane(on: Boolean): String = "airplane mode ${if (on) "on" else "off"}"

  /** One input's line on the summary: "Message: yes 0.999 · billing 0.939". */
  fun summaryLine(label: String, shown: List<D1Shown>): String =
    "$label: ${shown.joinToString(" · ") { "${it.answer} ${it.prob}" }}"

  /**
   * "GPU FP32" or "GPU FP16 (FP32 accum)" when every graph runs on the GPU at that precision, "GPU" when the GPU graphs
   * differ in precision, "CPU 4 threads" when every graph runs on the CPU, and "GPU + CPU" when some graph fell back to
   * the CPU.
   */
  fun accelerator(graphs: List<Pair<D1Backend, D1Precision>>): String {
    val backends = graphs.map { it.first }.distinct()
    if (backends == listOf(D1Backend.CPU)) return "CPU ${D1Decider.CPU_THREADS} threads"
    if (D1Backend.CPU in backends) return "GPU + CPU"
    return when (graphs.map { it.second }.distinct()) {
      listOf(D1Precision.FP32) -> "GPU FP32"
      listOf(D1Precision.FP16_FP32_ACCUM) -> "GPU FP16 (FP32 accum)"
      else -> "GPU"
    }
  }

  /** "Galaxy S26 · LiteRT 2.2.0 · GPU · d1-omni-600M": the summary's last line. */
  fun deviceLine(device: String, accelerator: String): String =
    "$device · LiteRT ${D1Decider.LITERT_VERSION} · $accelerator · d1-omni-600M"
}
