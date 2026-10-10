package com.d1omni

/**
 * One question as the editor holds it: [key] identifies it on the screen (not part of the request), [id] names it in
 * the request and the run JSON, [options] has one option per line (see [D1Drafts]).
 */
data class QuestionDraft(
  val key: Long,
  val id: String,
  val type: QuestionType,
  val instructions: String,
  val options: String,
)

/** Why the editor cannot build the questions. */
enum class D1DraftProblem {
  NO_QUESTIONS,
  EMPTY_ID,
  DUPLICATE_ID,
  EMPTY_INSTRUCTIONS,
  TOO_FEW_OPTIONS,
  TOO_MANY_LEVELS,
  EMPTY_OPTION_NAME,
  DUPLICATE_OPTION,
  BAD_NOUL_OPTION,
}

/** Questions that cannot be built; [question] is 1-based (0 = the list as a whole), [detail] names the culprit. */
class D1DraftException(val problem: D1DraftProblem, val question: Int, val detail: String) :
  IllegalArgumentException("$problem at question $question: $detail")

/**
 * Converts between the provider's questions and the editor (the Kev Decide sample's syntax), Android-free. Options are
 * one per line: a choice `name: description` or `name` (at least two), a score one level per line from the lowest (2
 * to 10), a noul optionally `true: what yes means` / `false: what no means` (`yes` / `no` are read as the same keys,
 * as the provider reads them). Descriptions are plain text; the name ends at the first colon.
 */
object D1Drafts {
  const val MIN_CHOICES = 2
  const val MIN_LEVELS = 2
  const val MAX_LEVELS = 10

  /** The editor form of [questions]; keys count up from [firstKey]. */
  fun fromQuestions(questions: Map<String, D1Question>, firstKey: Long = 0): List<QuestionDraft> =
    questions.entries.mapIndexed { index, (id, question) ->
      QuestionDraft(firstKey + index, id, question.type, question.instructions, optionLines(question).joinToString("\n"))
    }

  /** The questions [drafts] stand for, in order, validated as the provider's `as_question` validates them. */
  fun toQuestions(drafts: List<QuestionDraft>): LinkedHashMap<String, D1Question> {
    if (drafts.isEmpty()) throw D1DraftException(D1DraftProblem.NO_QUESTIONS, 0, "")
    val questions = LinkedHashMap<String, D1Question>()
    for ((index, draft) in drafts.withIndex()) {
      val number = index + 1
      val id = draft.id.trim()
      if (id.isEmpty()) throw D1DraftException(D1DraftProblem.EMPTY_ID, number, "")
      if (id in questions) throw D1DraftException(D1DraftProblem.DUPLICATE_ID, number, id)
      val instructions = draft.instructions.trim()
      if (instructions.isEmpty()) throw D1DraftException(D1DraftProblem.EMPTY_INSTRUCTIONS, number, id)
      questions[id] = D1Question(draft.type, instructions, criteria(draft, number))
    }
    return questions
  }

  /** The question as the request's JSON holds it (`{type, instructions, criteria}`; no criteria key for a bare noul). */
  fun toJson(question: D1Question): LinkedHashMap<String, Any?> =
    linkedMapOf<String, Any?>("type" to question.type.wireName, "instructions" to question.instructions).apply {
      if (question.criteria != null) put("criteria", question.criteria)
    }

  /** "choice · booking, cancel, prices, complaint", "noul · yes or no", "score · 3 levels": the collapsed editor's line. */
  fun summary(question: D1Question): String =
    when (question.type) {
      QuestionType.CHOICE -> "choice · ${question.choiceNames.joinToString(", ")}"
      QuestionType.NOUL -> "noul · yes or no"
      QuestionType.SCORE -> "score · ${question.options} levels"
    }

  /** The screen's sentence for [failure]. */
  fun message(failure: D1DraftException): String {
    val n = failure.question
    return when (failure.problem) {
      D1DraftProblem.NO_QUESTIONS -> "Add a question first."
      D1DraftProblem.EMPTY_ID -> "Question $n needs a name."
      D1DraftProblem.DUPLICATE_ID -> "Question $n: the name ${failure.detail} is used twice."
      D1DraftProblem.EMPTY_INSTRUCTIONS -> "Question $n: write the question."
      D1DraftProblem.TOO_FEW_OPTIONS -> "Question $n: a choice needs at least $MIN_CHOICES options and a score at least $MIN_LEVELS levels, one per line."
      D1DraftProblem.TOO_MANY_LEVELS -> "Question $n: a score has at most $MAX_LEVELS levels."
      D1DraftProblem.EMPTY_OPTION_NAME -> "Question $n: the line \"${failure.detail}\" has no name before its colon."
      D1DraftProblem.DUPLICATE_OPTION -> "Question $n: the option ${failure.detail} is used twice."
      D1DraftProblem.BAD_NOUL_OPTION -> "Question $n: a yes-or-no question takes only true: and false: lines, not ${failure.detail}."
    }
  }

  private fun criteria(draft: QuestionDraft, number: Int): Any? {
    val lines = draft.options.lines().map { it.trim() }.filter { it.isNotEmpty() }
    return when (draft.type) {
      QuestionType.SCORE -> {
        if (lines.size < MIN_LEVELS) throw D1DraftException(D1DraftProblem.TOO_FEW_OPTIONS, number, lines.size.toString())
        if (lines.size > MAX_LEVELS) throw D1DraftException(D1DraftProblem.TOO_MANY_LEVELS, number, lines.size.toString())
        lines
      }
      QuestionType.CHOICE -> {
        val options = namedOptions(lines, number)
        if (options.size < MIN_CHOICES) throw D1DraftException(D1DraftProblem.TOO_FEW_OPTIONS, number, options.size.toString())
        options
      }
      QuestionType.NOUL -> {
        val options = namedOptions(lines, number)
        for (name in options.keys) {
          if (name !in NOUL_NAMES) throw D1DraftException(D1DraftProblem.BAD_NOUL_OPTION, number, name)
        }
        options.takeIf { it.isNotEmpty() }
      }
    }
  }

  /** `name: description` or `name` lines; the name ends at the leftmost colon; no description = null. */
  private fun namedOptions(lines: List<String>, number: Int): LinkedHashMap<String, Any?> {
    val options = LinkedHashMap<String, Any?>()
    for (line in lines) {
      val colon = line.indexOf(':')
      val name = (if (colon < 0) line else line.substring(0, colon)).trim()
      val description = if (colon < 0) null else line.substring(colon + 1).trim().ifEmpty { null }
      if (name.isEmpty()) throw D1DraftException(D1DraftProblem.EMPTY_OPTION_NAME, number, line)
      if (name in options) throw D1DraftException(D1DraftProblem.DUPLICATE_OPTION, number, name)
      options[name] = description
    }
    return options
  }

  private fun optionLines(question: D1Question): List<String> =
    when (question.type) {
      QuestionType.SCORE -> (question.criteria as List<*>).map { oneLine(D1Prompt.criterion(it)) }
      QuestionType.CHOICE -> (question.criteria as Map<*, *>).map { (name, value) -> optionLine(name as String, value) }
      QuestionType.NOUL ->
        (question.criteria as Map<*, *>? ?: emptyMap<String, Any?>()).map { (name, value) -> optionLine(name as String, value) }
    }

  private fun optionLine(name: String, description: Any?): String =
    if (description == null || description == "") name else "$name: ${oneLine(D1Prompt.criterion(description))}"

  private fun oneLine(text: String): String = text.replace('\n', ' ')

  private val NOUL_NAMES = setOf("true", "false", "yes", "no")
}
