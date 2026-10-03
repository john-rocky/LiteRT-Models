package com.kev

/** Question types of the `/v1/systemone` request. */
enum class QuestionType(val wireName: String) {
  /** Yes/no: two options, "no" then "yes"; the answer is p(yes). */
  NOUL("noul"),

  /** One of the criteria names; the answer is the most likely name. */
  CHOICE("choice"),

  /** Ordered levels; the answer is the expected level index. */
  SCORE("score");

  companion object {
    fun fromWireName(name: Any?): QuestionType =
      entries.firstOrNull { it.wireName == name }
        ?: throw IllegalArgumentException("Unknown question type: $name")
  }
}

/**
 * One question of a request. [instructions] and [criteria] hold parsed JSON values ([KevJson]):
 * criteria is an object for noul (optional) and choice, an array of levels for score.
 */
class KevQuestion(
  val id: String,
  val type: QuestionType,
  val instructions: Any?,
  val criteria: Any?,
)

/**
 * A `/v1/systemone` request (the author's `SystemOneRequest`): a state (text or JSON) and typed
 * questions in request order. [parse] applies the same validation as the author's pydantic model.
 */
class KevRequest(val state: Any?, val questions: List<KevQuestion>, val model: String = DEFAULT_MODEL) {
  companion object {
    /** `SystemOneRequest.model`'s default. */
    const val DEFAULT_MODEL = "kev-latest"

    /** Most options a choice or score question may have (the author's `MAX_OPTIONS`). */
    const val MAX_OPTIONS = 255

    fun parse(json: String): KevRequest = fromJson(KevJson.parse(json))

    /** Validates a parsed request object the way the author's pydantic models do. */
    fun fromJson(value: Any?): KevRequest {
      val request = value as? Map<*, *> ?: throw IllegalArgumentException("Request is not an object")
      require(request.containsKey("state")) { "Request has no state" }
      // pydantic: an absent model takes the default; a present one must be a string (not null).
      val model = if (request.containsKey("model")) request["model"] else DEFAULT_MODEL
      require(model is String) { "model is not a string" }
      val questions = request["questions"] as? Map<*, *>
      require(questions != null && questions.isNotEmpty()) { "questions must be a non-empty object" }
      return KevRequest(
        request["state"],
        questions.map { (id, question) -> parseQuestion(id as String, question) },
        model,
      )
    }

    private fun parseQuestion(id: String, value: Any?): KevQuestion {
      val question = value as? Map<*, *> ?: throw IllegalArgumentException("Question $id is not an object")
      val type = QuestionType.fromWireName(question["type"])
      val criteria = question["criteria"]
      when (type) {
        QuestionType.NOUL ->
          require(criteria == null || criteria is Map<*, *>) { "Question $id: noul criteria must be an object" }
        QuestionType.CHOICE ->
          require(criteria is Map<*, *> && criteria.size in 1..MAX_OPTIONS) {
            "Question $id: choice criteria must have 1..$MAX_OPTIONS options"
          }
        QuestionType.SCORE ->
          require(criteria is List<*> && criteria.size in 1..MAX_OPTIONS) {
            "Question $id: score criteria must have 1..$MAX_OPTIONS levels"
          }
      }
      return KevQuestion(id, type, question["instructions"], criteria)
    }
  }
}

/** One question as the encoder reads it: rendered instructions and option texts. */
class KevRecordQuestion(val instructions: String, val options: List<String>)

/** The author's internal record: the rendered state and the rendered questions. */
class KevRecord(val state: String, val questions: List<KevRecordQuestion>)

/**
 * What maps a question's probabilities back to its answer: the question ID, its type, the keys the
 * probabilities are reported under, and for score questions the legend (level index → text).
 */
class QuestionMeta(
  val id: String,
  val type: QuestionType,
  val keys: List<String>,
  val legend: Map<String, String>?,
)

/** Ports of the author's `render`, `option_text`, `question_keys` and `to_record` (`kev/api.py`). */
object KevRecords {
  /**
   * Flattens a JSON value into the text the model sees (`render`): strings as they are, scalars as
   * Python's `str()` (`True`, `1`, `64.9`, `1840.0`), arrays as `- item` lines, objects as
   * `key: value` lines, nested containers indented by two spaces per level.
   */
  fun render(value: Any?, indent: Int = 0): String {
    val pad = INDENT.repeat(indent)
    return when (value) {
      null -> ""
      is String -> value
      is Boolean -> if (value) "True" else "False"
      is JsonNumber -> value.pythonString()
      is Int,
      is Long -> value.toString()
      is Double -> PythonFloat.repr(value)
      is List<*> ->
        value.joinToString("\n") { item -> "$pad- ${pythonLeftStrip(render(item, indent + 1))}" }
      is Map<*, *> ->
        value.entries.joinToString("\n") { (key, item) ->
          if (item is Map<*, *> || item is List<*>) {
            "$pad$key:\n${render(item, indent + 1)}"
          } else {
            "$pad$key: ${render(item)}"
          }
        }
      else -> throw IllegalArgumentException("Not a JSON value: ${value::class.java.name}")
    }
  }

  /** `option_text`: the name alone when the description is null or empty, else `name: text`. */
  fun optionText(name: String, description: Any?): String =
    if (description == null || description == "") name else "$name: ${render(description)}"

  /** `question_keys`: criteria names (choice), `false` / `true` (noul), level indices (score). */
  fun questionKeys(type: QuestionType, criteria: Any?): List<String> =
    when (type) {
      QuestionType.CHOICE -> (criteria as Map<*, *>).keys.map { it as String }
      QuestionType.NOUL -> listOf("false", "true")
      QuestionType.SCORE -> List((criteria as List<*>).size) { it.toString() }
    }

  /** `to_record`: the record the encoder reads and the metadata that maps answers back. */
  fun toRecord(request: KevRequest): Pair<KevRecord, List<QuestionMeta>> {
    val questions = ArrayList<KevRecordQuestion>(request.questions.size)
    val meta = ArrayList<QuestionMeta>(request.questions.size)
    for (question in request.questions) {
      val keys = questionKeys(question.type, question.criteria)
      var legend: Map<String, String>? = null
      val options =
        when (question.type) {
          QuestionType.NOUL -> {
            val criteria = question.criteria as Map<*, *>? ?: emptyMap<String, Any?>()
            listOf(optionText("no", criteria["false"]), optionText("yes", criteria["true"]))
          }
          QuestionType.CHOICE ->
            (question.criteria as Map<*, *>).map { (name, description) ->
              optionText(name as String, description)
            }
          QuestionType.SCORE ->
            (question.criteria as List<*>).map { render(it) }.also { levels ->
              legend = LinkedHashMap<String, String>().apply { keys.zip(levels).toMap(this) }
            }
        }
      questions.add(KevRecordQuestion(render(question.instructions), options))
      meta.add(QuestionMeta(question.id, question.type, keys, legend))
    }
    return KevRecord(render(request.state), questions) to meta
  }

  /** Python's `str.lstrip()`: drops leading characters for which `str.isspace()` is true. */
  internal fun pythonLeftStrip(text: String): String {
    var start = 0
    while (start < text.length && isPythonSpace(text[start])) {
      start++
    }
    return text.substring(start)
  }

  /** `str.isspace()` (bidirectional class WS, B or S, or category Zs); all are in the BMP. */
  private fun isPythonSpace(character: Char): Boolean {
    val code = character.code
    return code in 0x09..0x0d ||
      code in 0x1c..0x20 ||
      code == 0x85 ||
      code == 0xa0 ||
      code == 0x1680 ||
      code in 0x2000..0x200a ||
      code == 0x2028 ||
      code == 0x2029 ||
      code == 0x202f ||
      code == 0x205f ||
      code == 0x3000
  }

  private const val INDENT = "  "
}
