package com.kev

/**
 * One question's causal row: the state followed by the question's branch, positions 0..n-1. The
 * graph reads `hidden` at [decideIndex] (the decide token, the row's last token) and at each
 * [optionIndices] entry (the option's closing token), in option order.
 */
class KevRow(val ids: IntArray, val decideIndex: Int, val optionIndices: IntArray) {
  /** Real tokens in the row. */
  val length: Int
    get() = ids.size

  /** The smallest graph window that holds the row, or null when it is over the largest one. */
  val window: Int?
    get() = KevEncoder.WINDOWS.firstOrNull { length <= it }

  /** [window], or a rejection that says how long the row is. */
  fun requireWindow(): Int =
    window
      ?: throw IllegalArgumentException(
        "The question row is $length tokens; the largest graph takes ${KevEncoder.WINDOWS.last()}. " +
          "Shorten the state or the question."
      )

  /** The graph inputs for [window]: IDs right-padded with the pad ID and the 1/0 valid mask. */
  fun padded(window: Int): KevPaddedRow {
    require(length <= window) { "$length tokens > L=$window" }
    val paddedIds = IntArray(window) { KevEncoder.PAD_ID }
    ids.copyInto(paddedIds)
    return KevPaddedRow(paddedIds, FloatArray(window) { if (it < length) 1f else 0f })
  }
}

/** Graph inputs of one row: `ids` int32 `[1, L]` and `valid` float32 `[1, L]`. */
class KevPaddedRow(val ids: IntArray, val valid: FloatArray)

/** One question's branch, with readout offsets inside the branch (`rows_of`). */
class KevBranch(val ids: IntArray, val decide: Int, val options: IntArray)

/** A record encoded as the author's `encode` + `rows_of`: the state IDs and one branch per question. */
class KevEncoded(val stateIds: IntArray, val branches: List<KevBranch>) {
  /** `usage.input_tokens`: the state once plus every branch (`len(enc["ids"])`). */
  val inputTokens: Int
    get() = stateIds.size + branches.sumOf { it.ids.size }

  /** The causal row of question [question]: state + branch, with row-level readout indices. */
  fun row(question: Int): KevRow {
    val branch = branches[question]
    val offset = stateIds.size
    val ids = stateIds + branch.ids
    val row = KevRow(ids, offset + branch.decide, IntArray(branch.options.size) { offset + branch.options[it] })
    check(row.ids[row.decideIndex] == KevEncoder.DECIDE_ID && row.decideIndex == ids.size - 1) {
      "Decide token is not the row's last token"
    }
    check(row.optionIndices.all { row.ids[it] == KevEncoder.OPTION_END_ID }) {
      "An option index is not an option end token"
    }
    return row
  }

  /** Every question's row, in question order. */
  fun rows(): List<KevRow> = branches.indices.map { row(it) }
}

/**
 * Port of the author's `user_tokens`, `encode` and `rows_of` (`kev/model.py`) in the serving form
 * the graph runs: one causal row per question, `[state] + state tokens` then `[question] +
 * instructions + ([option] + option + [/option])… + [decide]`. Nothing is truncated: a row is
 * matched to the smallest graph window (512 / 1024 / 2048) or rejected.
 */
class KevEncoder(private val tokenizer: KevTokenizer) {
  init {
    for ((token, id) in DELIMITERS) {
      check(tokenizer.tokenId(token) == id) { "tokenizer.json gives $token ${tokenizer.tokenId(token)}, not $id" }
    }
  }

  /**
   * `user_tokens`: tokenizes caller text so that it can never produce a delimiter. The tokenizer
   * would cut `<|name|>` out of the text as an added token, so it is rewritten to `<¦name¦>` first.
   */
  fun userTokens(text: String): IntArray =
    tokenizer.encode(SPECIAL_TOKEN_PATTERN.replace(text) { "<¦${it.groupValues[1]}¦>" })

  /** `encode` followed by `rows_of`, without truncation. */
  fun encode(record: KevRecord): KevEncoded {
    val stateIds = intArrayOf(STATE_ID) + userTokens(record.state)
    val branches =
      record.questions.map { question ->
        val ids = ArrayList<Int>()
        ids.add(QUESTION_ID)
        userTokens(question.instructions).forEach { ids.add(it) }
        val options = IntArray(question.options.size)
        for ((index, option) in question.options.withIndex()) {
          ids.add(OPTION_START_ID)
          userTokens(option).forEach { ids.add(it) }
          ids.add(OPTION_END_ID)
          options[index] = ids.size - 1
        }
        ids.add(DECIDE_ID)
        KevBranch(ids.toIntArray(), ids.size - 1, options)
      }
    return KevEncoded(stateIds, branches)
  }

  companion object {
    /** `<|fim_prefix|>` opens the state. */
    const val STATE_ID = 248060

    /** `<|fim_middle|>` opens a question's branch. */
    const val QUESTION_ID = 248061

    /** `<|box_start|>` opens an option. */
    const val OPTION_START_ID = 248049

    /** `<|box_end|>` closes an option; its hidden state is the option's readout. */
    const val OPTION_END_ID = 248050

    /** `<|fim_suffix|>` ends the branch; its hidden state is the question's readout. */
    const val DECIDE_ID = 248062

    /** `<|endoftext|>` right-pads rows (pad = eos; there is no BOS). */
    const val PAD_ID = 248044

    /** Graph windows (sequence lengths), smallest first. */
    val WINDOWS = listOf(512, 1024, 2048)

    private val DELIMITERS =
      listOf(
        "<|fim_prefix|>" to STATE_ID,
        "<|fim_middle|>" to QUESTION_ID,
        "<|box_start|>" to OPTION_START_ID,
        "<|box_end|>" to OPTION_END_ID,
        "<|fim_suffix|>" to DECIDE_ID,
        "<|endoftext|>" to PAD_ID,
      )

    /** The author's `_SPECIAL_RE`. */
    private val SPECIAL_TOKEN_PATTERN = Regex("<\\|([A-Za-z0-9_]+)\\|>")
  }
}
