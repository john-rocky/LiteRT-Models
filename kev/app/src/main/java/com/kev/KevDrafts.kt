package com.kev

/** One question as the editor holds it; [options] has one option per line (see [KevDrafts]). */
data class QuestionDraft(
  /** Stable identity for the screen; not part of the request. */
  val key: Long,
  val id: String,
  val type: QuestionType,
  val instructions: String,
  val options: String,
)

/** The editor's request: the state text and the questions in order. */
data class RequestDraft(val state: String, val questions: List<QuestionDraft>)

/** How the editor reads a state text. */
enum class StateFormat {
  /** A JSON object or array: the model sees it rendered as `key: value` / `- item` lines. */
  JSON,

  /** Plain text, given to the model as it is. */
  TEXT,

  /** Starts like JSON but does not parse, so it is given to the model as plain text. */
  TEXT_INVALID_JSON,
}

/** Why the editor cannot build a request. */
enum class DraftProblem {
  NO_QUESTIONS,
  EMPTY_ID,
  DUPLICATE_ID,
  NO_OPTIONS,
  TOO_MANY_OPTIONS,
  EMPTY_OPTION_NAME,
  DUPLICATE_OPTION,
  BAD_NOUL_OPTION,
}

/** An editor request that cannot be built; [question] is 1-based, [detail] names the culprit. */
class KevDraftException(val problem: DraftProblem, val question: Int, val detail: String) :
  IllegalArgumentException("$problem at question $question: $detail")

/**
 * Converts between requests and the editor. Options are one per line: choice `key: description` or
 * `key`, noul `true: description` / `false: description` (both optional), score one level per line.
 * Descriptions are plain text. The state is JSON when the whole text parses as a JSON object or
 * array, plain text otherwise.
 */
object KevDrafts {
  /** The editor form of [request]; question keys count up from [firstKey]. */
  fun fromRequest(request: KevRequest, firstKey: Long = 0): RequestDraft =
    RequestDraft(
      stateText(request.state),
      request.questions.mapIndexed { index, question ->
        QuestionDraft(
          firstKey + index,
          question.id,
          question.type,
          plain(question.instructions),
          optionLines(question).joinToString("\n"),
        )
      },
    )

  /** Reads [draft] as a request, validated like the author's pydantic model. */
  fun toRequest(draft: RequestDraft): KevRequest {
    if (draft.questions.isEmpty()) throw KevDraftException(DraftProblem.NO_QUESTIONS, 0, "")
    val questions = LinkedHashMap<String, Any?>()
    for ((index, question) in draft.questions.withIndex()) {
      val number = index + 1
      val id = question.id.trim()
      if (id.isEmpty()) throw KevDraftException(DraftProblem.EMPTY_ID, number, "")
      if (id in questions) throw KevDraftException(DraftProblem.DUPLICATE_ID, number, id)
      questions[id] =
        linkedMapOf(
          "type" to question.type.wireName,
          "instructions" to question.instructions.trim(),
          "criteria" to criteria(question, number),
        )
    }
    return KevRequest.fromJson(
      linkedMapOf("state" to stateValue(draft.state), "questions" to questions)
    )
  }

  /** The request state the editor text stands for: a parsed JSON object or array, or the text. */
  fun stateValue(text: String): Any? =
    if (stateFormat(text) == StateFormat.JSON) KevJson.parse(text) else text

  fun stateFormat(text: String): StateFormat {
    val start = text.trimStart()
    if (!start.startsWith("{") && !start.startsWith("[")) return StateFormat.TEXT
    val parsed = runCatching { KevJson.parse(text) }.getOrNull()
    return if (parsed is Map<*, *> || parsed is List<*>) StateFormat.JSON
    else StateFormat.TEXT_INVALID_JSON
  }

  private fun criteria(question: QuestionDraft, number: Int): Any? {
    val lines = question.options.lines().map { it.trim() }.filter { it.isNotEmpty() }
    return when (question.type) {
      QuestionType.SCORE -> {
        requireOptionCount(lines.size, number)
        lines
      }
      QuestionType.CHOICE -> {
        requireOptionCount(lines.size, number)
        namedOptions(lines, number)
      }
      QuestionType.NOUL -> {
        val options = namedOptions(lines, number)
        for (name in options.keys) {
          if (name != "true" && name != "false")
            throw KevDraftException(DraftProblem.BAD_NOUL_OPTION, number, name)
        }
        options.takeIf { it.isNotEmpty() }
      }
    }
  }

  private fun requireOptionCount(count: Int, number: Int) {
    if (count == 0) throw KevDraftException(DraftProblem.NO_OPTIONS, number, "")
    if (count > KevRequest.MAX_OPTIONS) {
      throw KevDraftException(DraftProblem.TOO_MANY_OPTIONS, number, count.toString())
    }
  }

  /** `name: description` or `name` lines; the name ends at the leftmost colon. */
  private fun namedOptions(lines: List<String>, number: Int): LinkedHashMap<String, String?> {
    val options = LinkedHashMap<String, String?>()
    for (line in lines) {
      val colon = line.indexOf(':')
      val name = (if (colon < 0) line else line.substring(0, colon)).trim()
      val description = if (colon < 0) null else line.substring(colon + 1).trim().ifEmpty { null }
      if (name.isEmpty()) throw KevDraftException(DraftProblem.EMPTY_OPTION_NAME, number, line)
      if (name in options) throw KevDraftException(DraftProblem.DUPLICATE_OPTION, number, name)
      options[name] = description
    }
    return options
  }

  private fun stateText(state: Any?): String =
    when (state) {
      is Map<*, *>,
      is List<*> -> KevJson.writeIndented(state)
      else -> plain(state)
    }

  private fun optionLines(question: KevQuestion): List<String> =
    when (question.type) {
      QuestionType.SCORE -> (question.criteria as List<*>).map { oneLine(it) }
      QuestionType.CHOICE ->
        (question.criteria as Map<*, *>).map { (name, value) -> optionLine(name as String, value) }
      // to_record reads only "false" and "true".
      QuestionType.NOUL ->
        (question.criteria as Map<*, *>? ?: emptyMap<String, Any?>())
          .filterKeys { it == "true" || it == "false" }
          .map { (name, value) -> optionLine(name as String, value) }
    }

  private fun optionLine(name: String, description: Any?): String =
    if (description == null || description == "") name else "$name: ${oneLine(description)}"

  /** A JSON value as one editor line: strings as they are, other values rendered by `render`. */
  private fun oneLine(value: Any?): String = plain(value).replace('\n', ' ')

  private fun plain(value: Any?): String = value as? String ?: KevRecords.render(value)
}
