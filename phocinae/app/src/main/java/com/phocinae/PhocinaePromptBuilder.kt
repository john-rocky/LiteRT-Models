// SPDX-License-Identifier: Apache-2.0
// Port of laya 0.3.4 common.py and PhocinaeHost._to_internal (Convai Innovations).
package com.phocinae

/** Normalized upstream question schema; criteria order determines marker order. */
data class PhocinaeQuestion(val type: String, val instructions: String, val criteria: Any?) {
  /** Graph channel: choice=0, score=1, noul=2. */
  val qtype: Int
    get() =
      when (type) {
        "choice" -> 0
        "score" -> 1
        "noul" -> 2
        else -> throw IllegalArgumentException("Unknown question type: $type")
      }

  /** Number of marker logits required by the decoder. */
  val optionCount: Int
    get() =
      when (type) {
        "choice" -> PhocinaeJson.asObject(criteria).size
        "score" -> PhocinaeJson.asArray(criteria).size
        else -> 2
      }

  /** Serializes normalized fields in the host contract's insertion order. */
  fun asMap(): Map<String, Any?> =
    linkedMapOf("type" to type, "instructions" to instructions, "criteria" to criteria)
}

/** Unpadded token IDs and surviving option-marker positions for one graph invocation. */
data class PhocinaeSequence(val ids: IntArray, val markers: IntArray, val question: PhocinaeQuestion) {
  /** Unpadded IDs; graph wrappers append PAD rows and an attention mask. */
  val sequenceIds: IntArray
    get() = ids

  /** Positions to gather from token_logits. */
  val markerPositions: IntArray
    get() = markers

  /** Upstream question type string. */
  val qtype: String
    get() = question.type

  /** Rendered criteria in the same order as markers. */
  val options: List<String>
    get() = PhocinaePromptBuilder.renderOptions(question)
}

/** One question per graph call. Every encode disables tokenizer post-processing. */
class PhocinaePromptBuilder(
  private val tokenizer: PhocinaeTokenizer,
  val maxLen: Int = 256,
  val headMaxLen: Int = 256,
) {
  init {
    require(maxLen > 0 && headMaxLen > 0)
  }

  /** Accepts upstream map schemas before building their prompt. */
  fun normalize(question: Map<String, Any?>): PhocinaeQuestion = Companion.normalize(question)

  /** Builds one row using the exact option cap, squeeze rule, and right truncation. */
  fun build(state: Any?, question: Map<String, Any?>, questionId: String = ""): PhocinaeSequence =
    build(state, normalize(question), questionId)

  /** Builds one row using the exact option cap, squeeze rule, and right truncation. */
  fun build(state: Any?, question: PhocinaeQuestion, questionId: String = ""): PhocinaeSequence {
    val options = renderOptions(question)
    val instructions = question.instructions.replace(MASK_LITERAL, " ")
    val head =
      tokenizer.encode("${question.type} question: $instructions", addSpecialTokens = false)
    var encodedOptions =
      options.map { option ->
        val text =
          tokenizer.encode(" " + option.replace(MASK_LITERAL, " "), addSpecialTokens = false)
        intArrayOf(MASK) + text.copyOf(minOf(text.size, 48))
      }
    var optionBudget = headMaxLen - encodedOptions.sumOf { it.size }
    if (optionBudget < 16) {
      val per = maxOf(4, Math.floorDiv(headMaxLen - 16, maxOf(1, encodedOptions.size)))
      encodedOptions = encodedOptions.map { it.copyOf(minOf(it.size, per)) }
      optionBudget = headMaxLen - encodedOptions.sumOf { it.size }
    }
    val ids = ArrayList<Int>()
    ids.add(CLS)
    ids.addAll(head.take(maxOf(8, optionBudget)))
    ids.add(SEP)
    val markers = ArrayList<Int>()
    for (option in encodedOptions) {
      markers.add(ids.size)
      ids.addAll(option.asList())
    }
    ids.add(SEP)
    val room = maxOf(0, maxLen - ids.size - 1)
    val stateIds =
      tokenizer.encode(serializeState(state).replace(MASK_LITERAL, " "), addSpecialTokens = false)
    ids.addAll(stateIds.take(room))
    ids.add(SEP)
    val retainedMarkers = markers.filter { it < maxLen }.toIntArray()
    // This is upstream's actual rejection rule: a squeezed option is valid while its marker
    // survives.
    require(retainedMarkers.size == options.size) {
      "question '$questionId' options exceed head_max_len=$headMaxLen"
    }
    return PhocinaeSequence(ids.take(maxLen).toIntArray(), retainedMarkers, question)
  }

  companion object {
    const val PAD = 0
    const val SEP = 1
    const val CLS = 2
    const val MASK = 4
    const val MASK_LITERAL = "<mask>"

    /** Converts choice lists and non-string instructions as the Python host does. */
    fun normalize(question: Map<String, Any?>): PhocinaeQuestion {
      val type = question["type"] as? String ?: error("Question requires string type")
      require(type == "choice" || type == "score" || type == "noul") {
        "Unknown question type: $type"
      }
      var criteria = question["criteria"]
      if (type == "choice" && criteria is List<*>) {
        val labels = criteria
        criteria =
          LinkedHashMap<String, Any?>().apply {
            labels.forEach { label ->
              put(label as? String ?: error("Choice label must be a string"), null)
            }
          }
      }
      require(question.containsKey("instructions")) { "Question requires instructions" }
      val instructions =
        question["instructions"].let {
          if (it is String) it else PhocinaeJson.stringify(it, ensureAscii = true)
        }
      return PhocinaeQuestion(type, instructions, criteria)
    }

    /** Strings stay literal; other states use Python-compatible JSON serialization. */
    fun serializeState(state: Any?): String =
      if (state is String) state else PhocinaeJson.stringify(state)

    /** Renders one criterion with the host's JSON and fallback-string behavior. */
    fun renderCriterion(value: Any?): String =
      when (value) {
        is String -> value
        null,
        is Number,
        is Boolean,
        is Map<*, *>,
        is List<*> -> PhocinaeJson.stringify(value)
        else -> PhocinaeJson.stringify(value.toString()) // json.dumps(default=str)
      }

    /** Preserves choice labels, score levels, and false/true noul ordering. */
    fun renderOptions(question: PhocinaeQuestion): List<String> =
      when (question.type) {
        "choice" ->
          PhocinaeJson.asObject(question.criteria).map { (label, value) ->
            if (value == null || value == "") label else "$label: ${renderCriterion(value)}"
          }
        "score" ->
          PhocinaeJson.asArray(question.criteria).mapIndexed { index, value ->
            "level $index: ${renderCriterion(value)}"
          }
        "noul" -> {
          // Source uses `crit = crit or {}` before looking up the two named criteria.
          val criteria =
            if (pythonFalsey(question.criteria)) emptyMap()
            else PhocinaeJson.asObject(question.criteria)
          val falseValue = criteria["false"]
          val trueValue = criteria["true"]
          listOf(
            "false: " +
              if (falseValue == null || falseValue == "") "no, the statement does not hold"
              else renderCriterion(falseValue),
            "true: " +
              if (trueValue == null || trueValue == "") "yes, the statement holds"
              else renderCriterion(trueValue),
          )
        }
        else -> error("Unknown question type: ${question.type}")
      }

    private fun pythonFalsey(value: Any?): Boolean =
      when (value) {
        null -> true
        is Boolean -> !value
        is Number -> value.toDouble() == 0.0
        is String -> value.isEmpty()
        is Collection<*> -> value.isEmpty()
        is Map<*, *> -> value.isEmpty()
        else -> false
      }
  }
}
