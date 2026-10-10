package com.d1omni

/** The three request kinds; [wireName] is their key in `contract.json` `modes`. */
enum class D1Kind(val wireName: String) {
  TEXT("text"),
  IMAGE("image"),
  AUDIO("audio"),
}

/**
 * One question's row: the encoded [ids] (they sit at positions P … P+n−1 of the bucket), the
 * [markers] (positions inside [ids]), [prefixRows] = P media rows before them, and whether the
 * read-out divides by the temperature ([calibrate], text requests only).
 */
class D1Row(
  val question: D1Question,
  val ids: IntArray,
  val markers: IntArray,
  val prefixRows: Int,
  val calibrate: Boolean,
) {
  /** The positions the graph reads: P + n. */
  val positions: Int
    get() = prefixRows + ids.size
}

/**
 * The six inputs of `decide_<L>` for one row (`d1_host.build_inputs` + `qtype_onehot`): `ids` int32
 * [1, L], `prefix` float32 [1, L, 1024] (row-major; null = all zeros, a text row), `media` / `pad`
 * / `keep_right` float32 [1, L], `qtype_onehot` float32 [1, 3].
 */
class D1Inputs(
  val length: Int,
  val ids: IntArray,
  val prefix: FloatArray?,
  val media: FloatArray,
  val pad: FloatArray,
  val keepRight: FloatArray,
  val qtypeOneHot: FloatArray,
)

/**
 * The host side of a request (`D1Omni.rows` and `d1_host.build_inputs` of the model repository's
 * Python host): each request kind's settings from the contract, `prompt.encode` per question, the
 * bucket, and the six graph inputs.
 */
object D1Rows {
  /** Width of a prefix row (the trunk's hidden size). */
  const val PREFIX_WIDTH = 1024

  /** The fewest positions an image or audio request leaves for text (`max_len < 64` refuses). */
  private const val MIN_TEXT_POSITIONS = 64

  /**
   * One row per question for [kind] with [prefixRows] media rows before the text: the mode's
   * position limit, its noul default and its audio option form; a null [state] becomes `""` (or
   * the mode's `state_none_becomes`, `{}` for audio). [truncateTo] (the largest installed bucket)
   * cuts the state with encode's own rule so that every row fits it.
   */
  fun rows(
    tokenizer: D1Tokenizer,
    contract: D1Contract,
    state: Any?,
    questions: List<D1Question>,
    prefixRows: Int,
    kind: D1Kind,
    truncateTo: Int? = null,
  ): List<D1Row> {
    val mode = requireNotNull(contract.modes[kind.wireName]) { "contract.json has no mode $kind" }
    require((kind == D1Kind.TEXT) == (prefixRows == 0)) {
      "a text request has no prefix; an image or audio request has one"
    }
    var requestState = state
    if (kind == D1Kind.AUDIO && state == null && mode.stateNoneBecomes !== D1Contract.NOT_SET) {
      requestState = copyJson(mode.stateNoneBecomes)
    }
    var maxLength = minOf(mode.maxLength, contract.maxLength - prefixRows)
    if (truncateTo != null) maxLength = minOf(maxLength, truncateTo - prefixRows)
    require(maxLength >= MIN_TEXT_POSITIONS) {
      "the media take $prefixRows of the ${contract.maxLength} positions; send fewer images"
    }
    val finalState = requestState ?: ""
    return questions.map { question ->
      val encoded =
        D1Prompt.encode(tokenizer, finalState, question, maxLength, mode.noulDefault, mode.audio)
      D1Row(question, encoded.ids, encoded.markers, prefixRows, mode.calibrate)
    }
  }

  /** `usage.input_tokens`: every position the trunk reads, P + n per question. */
  fun inputTokens(rows: List<D1Row>): Int = rows.sumOf { it.positions }

  /**
   * `build_inputs` for a bucket of [length] positions: [ids] at P … P+n−1 (0 elsewhere), [prefix]
   * (P × 1024 row-major values; null for a text row, whose prefix input is all zeros) at 0 … P−1,
   * `media` 1 on the prefix, `pad` 1 on the real positions, `keep_right` 0 only at P−1; then the
   * question type's one-hot.
   */
  fun buildInputs(
    ids: IntArray,
    prefix: FloatArray?,
    prefixRows: Int,
    length: Int,
    type: QuestionType,
  ): D1Inputs {
    val count = ids.size
    require(prefixRows >= 0 && prefixRows + count <= length) {
      "row of $prefixRows + $count positions does not fit L = $length"
    }
    val idsInput = IntArray(length)
    ids.copyInto(idsInput, prefixRows)
    val prefixInput =
      if (prefixRows == 0) {
        require(prefix == null || prefix.isEmpty()) { "P = 0 but prefix rows given" }
        null
      } else {
        val rows = requireNotNull(prefix) { "P = $prefixRows but no prefix rows" }
        require(rows.size == prefixRows * PREFIX_WIDTH) {
          "prefix holds ${rows.size} values, P x 1024 = ${prefixRows * PREFIX_WIDTH}"
        }
        FloatArray(length * PREFIX_WIDTH).also { rows.copyInto(it) }
      }
    return D1Inputs(
      length,
      idsInput,
      prefixInput,
      FloatArray(length) { if (it < prefixRows) 1f else 0f },
      FloatArray(length) { if (it < prefixRows + count) 1f else 0f },
      FloatArray(length) { if (it == prefixRows - 1) 0f else 1f },
      qtypeOneHot(type),
    )
  }

  /** `qtype_onehot`: [choice, score, noul]. */
  fun qtypeOneHot(type: QuestionType): FloatArray =
    FloatArray(3) { if (it == type.index) 1f else 0f }

  /** A fresh copy of a parsed JSON value (the contract's `{}` must not be shared between requests). */
  private fun copyJson(value: Any?): Any? =
    when (value) {
      is Map<*, *> -> LinkedHashMap<String, Any?>().apply {
        for ((key, member) in value) put(key as String, copyJson(member))
      }
      is List<*> -> value.map { copyJson(it) }
      else -> value
    }
}
