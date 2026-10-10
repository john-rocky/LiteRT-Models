package com.d1omni

import kotlin.math.abs

/** The three question types and their index in the graph's `qtype_onehot` (`prompt.QTYPES`). */
enum class QuestionType(val wireName: String, val index: Int) {
  CHOICE("choice", 0),
  SCORE("score", 1),
  NOUL("noul", 2);

  companion object {
    fun of(name: String): QuestionType? = entries.firstOrNull { it.wireName == name }
  }
}

/**
 * One question in the Decision Index schema (`prompt.Question`): a [type], its [instructions] and
 * its [criteria] as parsed JSON (a choice: an object of option name to description; a score: a list
 * of 2 to 10 level descriptions, lowest first; a noul: null or an object with `true` / `false`,
 * `yes` / `no`).
 */
class D1Question(val type: QuestionType, val instructions: String, val criteria: Any?) {
  init {
    when (type) {
      QuestionType.CHOICE ->
        require(criteria is Map<*, *> && criteria.size >= 2) {
          "a choice needs criteria {name: description} with at least two options"
        }
      QuestionType.SCORE ->
        require(criteria is List<*> && criteria.size in 2..MAX_LEVELS) {
          "a score needs criteria: a list of 2 to 10 level descriptions, lowest first"
        }
      QuestionType.NOUL ->
        require(criteria == null || criteria is Map<*, *>) {
          "noul criteria are optional: {\"true\": \"...\", \"false\": \"...\"} (or \"yes\", \"no\")"
        }
    }
  }

  /** The number of options: 2 for a noul, else the number of criteria. */
  val options: Int
    get() =
      when (type) {
        QuestionType.NOUL -> 2
        QuestionType.CHOICE -> (criteria as Map<*, *>).size
        QuestionType.SCORE -> (criteria as List<*>).size
      }

  /** The option names of a choice, in criteria order. */
  val choiceNames: List<String>
    get() = (criteria as Map<*, *>).keys.map { it as String }

  private companion object {
    const val MAX_LEVELS = 10
  }
}

/** One question's encoded row: [ids] (BOS first) and the position of each option's marker. */
class D1Encoded(val ids: IntArray, val markers: IntArray)

/**
 * Port of the provider's `prompt.py` (LiquidAI/d1-omni-600M, revision 414f8d64): `as_question`,
 * `escape`, `serialize`, `_criterion`, `render_options`, `encode`, `answer` and `temperature_key`,
 * in the same order of operations. A question becomes one sequence:
 *
 *     <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> … <decide>
 */
object D1Prompt {
  /** `prompt.DELIM` and `prompt.MARKER`: the tokens the encoder writes by name. */
  const val STATE_TOKEN = "<|reserved_7|>"
  const val QUESTION_TOKEN = "<|reserved_8|>"
  const val OPTION_TOKEN = "<|reserved_9|>"
  const val OPTION_END_TOKEN = "<|reserved_10|>"
  const val DECIDE_TOKEN = "<|reserved_11|>"
  const val MARKER_TOKEN = "<|mask|>"
  const val BOS_TOKEN = "<|startoftext|>"

  /** Option tokens per option before the block is shared out (`encode(per_option=24)`). */
  const val PER_OPTION = 24

  private val SPECIAL = Regex("<\\|([A-Za-z0-9_]+)\\|>")

  /** `as_question`: a question object as parsed JSON -> [D1Question], or a rejection. */
  fun asQuestion(value: Any?): D1Question {
    require(value is Map<*, *> && value.containsKey("type") && value.containsKey("instructions")) {
      "a question is a dict with `type`, `instructions` and, for choice and score, `criteria`"
    }
    val typeName = value["type"]
    val type =
      (typeName as? String)?.let { QuestionType.of(it) }
        ?: throw IllegalArgumentException(
          "question type must be one of ['choice', 'noul', 'score'], got ${PythonText.repr(typeName)}"
        )
    return D1Question(type, PythonText.str(value["instructions"]), value["criteria"])
  }

  /** `escape`: `<|name|>` -> `<¦name¦>`, so caller text cannot emit a delimiter or marker token. */
  fun escape(text: String): String = SPECIAL.replace(text) { "<¦${it.groupValues[1]}¦>" }

  /** `serialize`: a string as it is, anything else as `json.dumps(state, ensure_ascii=False)`. */
  fun serialize(state: Any?): String = if (state is String) state else D1Json.dumps(state)

  /** `_criterion`: a string as it is, anything else as JSON with `", "` and `": "`. */
  fun criterion(value: Any?): String =
    if (value is String) value else D1Json.dumps(value, ", ", ": ")

  /**
   * `render_options`: the option texts in the model's order. A noul is read as [false, true]. After
   * an audio prefix, options are written as the audio questions were trained: `option_000: text`,
   * and a noul as `false: no`, `true: yes`.
   */
  fun renderOptions(
    question: D1Question,
    noulDefault: Map<*, *>? = null,
    audio: Boolean = false,
  ): List<String> =
    when (question.type) {
      QuestionType.CHOICE -> {
        val criteria = question.criteria as Map<*, *>
        if (audio) {
          criteria.entries.mapIndexed { index, (key, value) ->
            val text = if (value == null || value == "") key else value
            "option_${index.toString().padStart(3, '0')}: ${criterion(text)}"
          }
        } else {
          criteria.entries.map { (key, value) ->
            if (value == null || value == "") key as String else "$key: ${criterion(value)}"
          }
        }
      }
      QuestionType.SCORE ->
        (question.criteria as List<*>).mapIndexed { index, level ->
          "level $index: ${criterion(level)}"
        }
      QuestionType.NOUL ->
        if (audio) {
          listOf("false: no", "true: yes")
        } else {
          // `q.criteria or noul_default or {}`: an empty object counts as none.
          val criteria =
            (question.criteria as Map<*, *>?)?.takeIf { it.isNotEmpty() }
              ?: noulDefault?.takeIf { it.isNotEmpty() }
              ?: emptyMap<String, Any?>()
          val no = if (criteria.containsKey("false")) criteria["false"] else criteria["no"]
          val yes = if (criteria.containsKey("true")) criteria["true"] else criteria["yes"]
          listOf(
            "false: " +
              (if (no != null && no != "") criterion(no) else "no, the statement does not hold"),
            "true: " + (if (yes != null && yes != "") criterion(yes) else "yes, the statement holds"),
          )
        }
    }

  /**
   * `encode`: the token IDs of one question over one state, and the position of each option's
   * marker. The option block gets max(96, min(24k + 32, maxLen / 2)) tokens, shared out evenly; the
   * state is truncated on the right to the room that is left.
   */
  fun encode(
    tokenizer: D1Tokenizer,
    state: Any?,
    question: D1Question,
    maxLength: Int,
    noulDefault: Map<*, *>? = null,
    audio: Boolean = false,
    perOption: Int = PER_OPTION,
  ): D1Encoded {
    fun id(token: String): Int =
      requireNotNull(tokenizer.tokenId(token)) { "tokenizer.json has no $token" }
    fun enc(text: String): IntArray = tokenizer.encode(escape(text))
    val options = renderOptions(question, noulDefault, audio)
    val count = options.size
    val budget = maxOf(96, minOf(count * perOption + 32, Math.floorDiv(maxLength, 2)))
    val per = maxOf(2, Math.floorDiv(budget - 3 * count, count))
    val questionIds = ArrayList<Int>()
    questionIds.add(id(QUESTION_TOKEN))
    enc(question.instructions).forEach { questionIds.add(it) }
    truncate(questionIds, maxOf(16, budget))
    val markers = ArrayList<Int>(count)
    for (text in options) {
      markers.add(questionIds.size + 1)
      questionIds.add(id(OPTION_TOKEN))
      questionIds.add(id(MARKER_TOKEN))
      val optionIds = enc(" $text")
      for (index in 0 until minOf(per, optionIds.size)) questionIds.add(optionIds[index])
      questionIds.add(id(OPTION_END_TOKEN))
    }
    questionIds.add(id(DECIDE_TOKEN))
    val room = maxOf(0, maxLength - questionIds.size - 2)
    val stateText = enc(serialize(state))
    val stateIds = ArrayList<Int>(1 + minOf(room, stateText.size))
    stateIds.add(id(STATE_TOKEN))
    for (index in 0 until minOf(room, stateText.size)) stateIds.add(stateText[index])
    val ids = ArrayList<Int>(1 + stateIds.size + questionIds.size)
    ids.add(id(BOS_TOKEN))
    ids.addAll(stateIds)
    ids.addAll(questionIds)
    truncate(ids, maxLength)
    val shifted = IntArray(markers.size) { markers[it] + 1 + stateIds.size }
    require(shifted.last() < maxLength) { "the options do not fit in the context" }
    return D1Encoded(ids.toIntArray(), shifted)
  }

  /**
   * `answer`: a noul's P(yes) (its probabilities are [yes, no]); a choice's pick and its
   * probabilities; a score's expected level (CPython 3.12's `sum`), its probabilities and legend.
   */
  fun answer(question: D1Question, probabilities: DoubleArray): LinkedHashMap<String, Any?> {
    if (question.type == QuestionType.NOUL) {
      return linkedMapOf("type" to "noul", "noul" to probabilities[0])
    }
    val best = firstArgmax(probabilities)
    if (question.type == QuestionType.CHOICE) {
      val names = question.choiceNames
      val table = LinkedHashMap<String, Any?>()
      for ((index, name) in names.withIndex()) {
        if (index < probabilities.size) table[name] = probabilities[index]
      }
      return linkedMapOf(
        "type" to "choice",
        "choice" to names[best],
        "confidence" to probabilities[best],
        "probabilities" to table,
      )
    }
    val levels = question.criteria as List<*>
    return linkedMapOf(
      "type" to "score",
      "score" to pythonSum(DoubleArray(probabilities.size) { it * probabilities[it] }),
      "confidence" to probabilities[best],
      "probabilities" to
        LinkedHashMap<String, Any?>().apply {
          probabilities.forEachIndexed { index, p -> put(index.toString(), p) }
        },
      "legend" to
        LinkedHashMap<String, Any?>().apply {
          levels.forEachIndexed { index, level -> put(index.toString(), criterion(level)) }
        },
    )
  }

  /** `temperature_key`: `<type>:2`, `:3-5`, `:6-10` or `:11+` by the option count. */
  fun temperatureKey(question: D1Question): String {
    val count = question.options
    val band =
      when {
        count <= 2 -> "2"
        count <= 5 -> "3-5"
        count <= 10 -> "6-10"
        else -> "11+"
      }
    return "${question.type.wireName}:$band"
  }

  /** Index of the first maximum, as Python's `max(range(n), key=p.__getitem__)`. */
  fun firstArgmax(values: DoubleArray): Int {
    var best = 0
    for (index in 1 until values.size) {
      if (values[index] > values[best]) best = index
    }
    return best
  }

  /**
   * CPython 3.12's built-in `sum` of floats from the int start 0: the first value is taken as it
   * is, the rest are added with Neumaier's compensation, and the compensation is added at the end.
   */
  fun pythonSum(values: DoubleArray): Double {
    if (values.isEmpty()) return 0.0
    var total = 0.0 + values[0]
    var compensation = 0.0
    for (index in 1 until values.size) {
      val value = values[index]
      val sum = total + value
      compensation += if (abs(total) >= abs(value)) (total - sum) + value else (value - sum) + total
      total = sum
    }
    if (compensation != 0.0 && compensation.isFinite()) {
      total += compensation
    }
    return total
  }

  /** Python's `list[:n]` in place (n >= 0). */
  private fun truncate(list: ArrayList<Int>, length: Int) {
    while (list.size > length) list.removeAt(list.lastIndex)
  }
}

/**
 * Python's `str()` and `repr()` of the values `json.loads` makes (str, int, float, bool, None,
 * list, dict): `as_question` passes the instructions through `str()`, so a number or a list there
 * becomes the text Python prints for it.
 */
object PythonText {
  fun str(value: Any?): String = if (value is String) value else repr(value)

  fun repr(value: Any?): String =
    when (value) {
      null -> "None"
      is Boolean -> if (value) "True" else "False"
      is String -> stringRepr(value)
      is JsonNumber -> value.pythonString()
      is Int,
      is Long -> value.toString()
      is Double -> PythonFloat.repr(value)
      is Map<*, *> ->
        value.entries.joinToString(", ", "{", "}") { (key, member) ->
          "${repr(key)}: ${repr(member)}"
        }
      is List<*> -> value.joinToString(", ", "[", "]") { repr(it) }
      else -> throw IllegalArgumentException("Not a JSON value: ${value::class.java.name}")
    }

  /** CPython's `unicode_repr`: single quotes unless the text has `'` and no `"`. */
  private fun stringRepr(text: String): String {
    val quote = if (text.contains('\'') && !text.contains('"')) '"' else '\''
    val out = StringBuilder().append(quote)
    var index = 0
    while (index < text.length) {
      val code = text.codePointAt(index)
      index += Character.charCount(code)
      when {
        code == quote.code || code == '\\'.code -> out.append('\\').appendCodePoint(code)
        code == '\t'.code -> out.append("\\t")
        code == '\n'.code -> out.append("\\n")
        code == '\r'.code -> out.append("\\r")
        code < ' '.code || code == DELETE -> out.append("\\x").append(hex(code, 2))
        code < DELETE -> out.appendCodePoint(code)
        printable(code) -> out.appendCodePoint(code)
        code <= 0xff -> out.append("\\x").append(hex(code, 2))
        code <= 0xffff -> out.append("\\u").append(hex(code, 4))
        else -> out.append("\\U").append(hex(code, 8))
      }
    }
    return out.append(quote).toString()
  }

  /** `str.isprintable()` of one code point: not in Cc, Cf, Cs, Co, Cn, Zl, Zp or Zs. */
  private fun printable(code: Int): Boolean =
    when (Character.getType(code).toByte()) {
      Character.CONTROL,
      Character.FORMAT,
      Character.SURROGATE,
      Character.PRIVATE_USE,
      Character.UNASSIGNED,
      Character.LINE_SEPARATOR,
      Character.PARAGRAPH_SEPARATOR,
      Character.SPACE_SEPARATOR -> false
      else -> true
    }

  private fun hex(code: Int, digits: Int) = code.toString(16).padStart(digits, '0')

  private const val DELETE = 0x7f
}
