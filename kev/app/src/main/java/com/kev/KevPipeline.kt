package com.kev

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * One graph call: a padded row in, the whole `hidden` output out. The app runs the LiteRT graph
 * ([KevDecider]); the JVM tests put the oracle's hidden states at the readout positions instead.
 */
interface RowRunner {
  /** Graph window L: [run] takes `ids` and `valid` with L entries each. */
  val length: Int

  /** Runs one row and returns `hidden`, L × [KevPointerHead.HIDDEN_SIZE] floats, row-major. */
  fun run(ids: IntArray, valid: FloatArray): FloatArray
}

/**
 * A request with a name, as the demo fixtures and the bundled examples store it: `{"id", "state",
 * "questions"}`. The request part follows the author's `SystemOneRequest`, which ignores keys it
 * does not define (here `id`).
 */
class KevFixture(val id: String, val request: KevRequest) {
  companion object {
    fun parse(text: String): KevFixture = fromJson(KevJson.parse(text))

    fun fromJson(value: Any?): KevFixture {
      val fixture =
        value as? Map<*, *> ?: throw IllegalArgumentException("The fixture is not a JSON object")
      val id = fixture["id"] as? String
      require(!id.isNullOrEmpty()) { "The fixture has no id" }
      return KevFixture(id, KevRequest.fromJson(fixture))
    }
  }
}

/** A question row that the resident graph window cannot take. */
class KevWindowException(val questionId: String, val rowTokens: Int, val window: Int) :
  IllegalArgumentException(
    "Question $questionId: the row is $rowTokens tokens, the graph takes $window"
  )

/** The graph returned NaN or infinity on a question's real positions; no answer is produced. */
class KevNonFiniteException(val questionId: String, val count: Int) :
  IllegalStateException("Question $questionId: the graph returned $count non-finite values")

/**
 * A request tokenized once: rows in question order, the metadata that maps probabilities to
 * answers, and the tokenizer time of the state ([stateMs]) and of each question's branch.
 */
class KevPrepared(
  val request: KevRequest,
  val meta: List<QuestionMeta>,
  val encoded: KevEncoded,
  val stateMs: Double,
  val branchMs: DoubleArray,
) {
  /** Every question's row, in question order. */
  val rows: List<KevRow> = encoded.rows()

  /** Tokenizer time of the whole request (state and every branch), in milliseconds. */
  val tokenizeMs: Double
    get() = stateMs + branchMs.sum()

  /** Real tokens of every row, in question order. */
  val rowLengths: List<Int>
    get() = rows.map { it.length }

  /** Real tokens of the longest row. */
  val longestRow: Int
    get() = rows.maxOf { it.length }

  /** The smallest of [windows] that holds every row, or null when a row is over the largest. */
  fun window(windows: List<Int>): Int? = KevWindows.smallestHolding(windows, longestRow)

  /** `usage.input_tokens`. */
  val inputTokens: Int
    get() = encoded.inputTokens

  /** Tokenizer time attributed to question [index]: its branch, plus the state for question 0. */
  fun questionTokenizeMs(index: Int): Double = branchMs[index] + if (index == 0) stateMs else 0.0
}

/** One question through the graph and the head, with its wall-clock phases in milliseconds. */
class KevQuestionResult(
  val index: Int,
  val meta: QuestionMeta,
  val row: KevRow,
  /** The graph window the row ran in. */
  val window: Int,
  val scores: KevScores,
  /** The float32 softmax widened exactly to double, as `to_answers` receives it. */
  val probabilities: DoubleArray,
  /**
   * Input writes, `run()` and the read-back of `hidden`: `run()` alone returns before the work
   * ends.
   */
  val inferMs: Double,
  /** Readout rows, both projections, the dot products and the softmax. */
  val headMs: Double,
)

/**
 * The Android-free decision path: request → `to_record` → rows (tokenizer) → one graph call per
 * question ([RowRunner]) → the decide and option rows of `hidden` → pointer head → `to_answers`.
 */
class KevPipeline(val tokenizer: KevTokenizer, val head: KevPointerHead) {
  val encoder = KevEncoder(tokenizer)

  /** Renders and tokenizes [request] once, timing the state and each branch. */
  fun prepare(request: KevRequest): KevPrepared {
    val (record, meta) = KevRecords.toRecord(request)
    val stateStart = System.nanoTime()
    val stateIds = encoder.stateIds(record.state)
    val stateEnd = System.nanoTime()
    val branchMs = DoubleArray(record.questions.size)
    val branches =
      record.questions.mapIndexed { index, question ->
        val start = System.nanoTime()
        encoder.branch(question).also { branchMs[index] = millis(System.nanoTime() - start) }
      }
    return KevPrepared(
      request,
      meta,
      KevEncoded(stateIds, branches),
      millis(stateEnd - stateStart),
      branchMs,
    )
  }

  /**
   * Runs question [index] of [prepared] on [runner]: pads its row to the runner's window, reads
   * `hidden` at the decide token and at each option's closing token, and applies the head.
   */
  fun run(prepared: KevPrepared, index: Int, runner: RowRunner): KevQuestionResult {
    val row = prepared.rows[index]
    val meta = prepared.meta[index]
    val window = runner.length
    if (row.length > window) throw KevWindowException(meta.id, row.length, window)
    val padded = row.padded(window)
    val inferStart = System.nanoTime()
    val hidden = runner.run(padded.ids, padded.valid)
    val inferEnd = System.nanoTime()
    check(hidden.size == window * HIDDEN) {
      "hidden has ${hidden.size} values, expected ${window * HIDDEN}"
    }
    val nonFinite = nonFiniteCount(hidden, 0, row.length * HIDDEN)
    if (nonFinite > 0) throw KevNonFiniteException(meta.id, nonFinite)
    val scores =
      head.score(
        hiddenRow(hidden, row.decideIndex),
        row.optionIndices.map { hiddenRow(hidden, it) },
      )
    val probabilities =
      DoubleArray(scores.probabilities.size) { scores.probabilities[it].toDouble() }
    val headEnd = System.nanoTime()
    return KevQuestionResult(
      index,
      meta,
      row,
      window,
      scores,
      probabilities,
      millis(inferEnd - inferStart),
      millis(headEnd - inferEnd),
    )
  }

  /** `to_answers` over finished questions, keyed by question ID in question order. */
  fun answers(
    prepared: KevPrepared,
    results: List<KevQuestionResult>,
  ): LinkedHashMap<String, Any?> =
    KevAnswers.toAnswers(results.map { it.probabilities }, results.map { prepared.meta[it.index] })

  /** The response body: model, answers and `usage.input_tokens`. */
  fun response(prepared: KevPrepared, answers: Map<String, Any?>): LinkedHashMap<String, Any?> =
    KevAnswers.response(answers, prepared.inputTokens, prepared.request.model)

  companion object {
    private const val HIDDEN = KevPointerHead.HIDDEN_SIZE

    /** One position of a row-major [L, 1024] `hidden`. */
    fun hiddenRow(hidden: FloatArray, position: Int): FloatArray =
      hidden.copyOfRange(position * HIDDEN, (position + 1) * HIDDEN)

    /** NaN and infinite values in `values[from until to]`. */
    fun nonFiniteCount(values: FloatArray, from: Int, to: Int): Int {
      var count = 0
      for (index in from until to) {
        if (!values[index].isFinite()) count++
      }
      return count
    }

    /** sha256 of the IDs as int32 little-endian bytes, lowercase hex. */
    fun idsSha256(ids: IntArray): String {
      val bytes = ByteBuffer.allocate(ids.size * Int.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
      bytes.asIntBuffer().put(ids)
      return MessageDigest.getInstance("SHA-256").digest(bytes.array()).joinToString("") {
        (it.toInt() and BYTE_MASK).toString(HEX_RADIX).padStart(2, '0')
      }
    }

    fun millis(nanoseconds: Long): Double = nanoseconds / NANOS_PER_MILLI

    private const val NANOS_PER_MILLI = 1_000_000.0
    private const val BYTE_MASK = 0xff
    private const val HEX_RADIX = 16
  }
}
