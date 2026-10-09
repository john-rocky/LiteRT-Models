package com.d1omni

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test

/**
 * The questions editor (the Kev Decide sample's syntax): the sample's questions survive the editor unchanged, each
 * option line becomes the provider's criteria (choice `name: description` / `name`, score levels, noul true / false),
 * and every rule of the provider's `as_question` (and the editor's own: a name, a question) is refused with its reason.
 */
class D1DraftsTest {
  private fun draft(type: QuestionType, options: String, id: String = "q", instructions: String = "Is it?") =
    QuestionDraft(1, id, type, instructions, options)

  private fun json(questions: Map<String, D1Question>): String =
    D1Json.write(LinkedHashMap(questions.mapValues { D1Drafts.toJson(it.value) }))

  @Test
  fun theSampleQuestionsRoundTrip() {
    val sample = D1Sample.parse(File("src/main/res/raw/sample.json").readBytes())
    for (input in sample.inputs) {
      val drafts = D1Drafts.fromQuestions(input.questions, 10)
      assertEquals(listOf(10L, 11L).take(drafts.size), drafts.map { it.key })
      assertEquals(json(input.questions), json(D1Drafts.toQuestions(drafts)))
    }
    val voice = D1Drafts.fromQuestions(sample.input(D1Input.VOICE).questions).single()
    assertEquals("topic", voice.id)
    assertEquals(QuestionType.CHOICE, voice.type)
    assertEquals(
      "booking: Booking or changing an appointment\ncancel: Cancelling an appointment\n" +
        "prices: A question about prices or opening hours\ncomplaint: A complaint",
      voice.options,
    )
    val refund = D1Drafts.fromQuestions(sample.input(D1Input.MESSAGE).questions).first()
    assertEquals(QuestionType.NOUL, refund.type)
    assertEquals("", refund.options)
  }

  @Test
  fun optionLinesBecomeTheProvidersCriteria() {
    val choice = D1Drafts.toQuestions(listOf(draft(QuestionType.CHOICE, " dog: A dog \n\ncat\nbird:  ")))
    assertEquals("{\"q\":{\"type\":\"choice\",\"instructions\":\"Is it?\",\"criteria\":{\"dog\":\"A dog\",\"cat\":null,\"bird\":null}}}", json(choice))
    // A name without a description is written as the name alone (prompt.render_options).
    assertEquals(listOf("dog: A dog", "cat", "bird"), D1Prompt.renderOptions(choice.getValue("q")))
    // The name ends at the first colon; the description keeps the rest.
    val colon = D1Drafts.toQuestions(listOf(draft(QuestionType.CHOICE, "time: 9:30 or later\nearly: before 9")))
    assertEquals("9:30 or later", (colon.getValue("q").criteria as Map<*, *>)["time"])
    val score = D1Drafts.toQuestions(listOf(draft(QuestionType.SCORE, "Can wait\nToday\nRight now")))
    assertEquals(listOf("Can wait", "Today", "Right now"), score.getValue("q").criteria)
    val noul = D1Drafts.toQuestions(listOf(draft(QuestionType.NOUL, "true: they want money back\nfalse: anything else")))
    assertEquals(mapOf("true" to "they want money back", "false" to "anything else"), noul.getValue("q").criteria)
    assertNull(D1Drafts.toQuestions(listOf(draft(QuestionType.NOUL, "  "))).getValue("q").criteria)
    // yes / no lines are read as the provider reads them
    assertEquals(mapOf("yes" to "y", "no" to "n"), D1Drafts.toQuestions(listOf(draft(QuestionType.NOUL, "yes: y\nno: n"))).getValue("q").criteria)
    // trimmed name and question
    val trimmed = D1Drafts.toQuestions(listOf(QuestionDraft(1, "  animal ", QuestionType.NOUL, "  What is it?  ", "")))
    assertEquals(listOf("animal"), trimmed.keys.toList())
    assertEquals("What is it?", trimmed.getValue("animal").instructions)
  }

  @Test
  fun refusals() {
    fun problem(drafts: List<QuestionDraft>): D1DraftProblem =
      try {
        D1Drafts.toQuestions(drafts)
        fail("accepted $drafts")
        throw IllegalStateException()
      } catch (failure: D1DraftException) {
        assertEquals(failure.problem.name, D1Drafts.message(failure).isNotBlank(), true)
        failure.problem
      }
    assertEquals(D1DraftProblem.NO_QUESTIONS, problem(emptyList()))
    assertEquals(D1DraftProblem.EMPTY_ID, problem(listOf(draft(QuestionType.NOUL, "", id = " "))))
    assertEquals(D1DraftProblem.DUPLICATE_ID, problem(listOf(draft(QuestionType.NOUL, ""), draft(QuestionType.NOUL, ""))))
    assertEquals(D1DraftProblem.EMPTY_INSTRUCTIONS, problem(listOf(draft(QuestionType.NOUL, "", instructions = "  "))))
    assertEquals(D1DraftProblem.TOO_FEW_OPTIONS, problem(listOf(draft(QuestionType.CHOICE, "only: one"))))
    assertEquals(D1DraftProblem.TOO_FEW_OPTIONS, problem(listOf(draft(QuestionType.SCORE, "one level"))))
    assertEquals(D1DraftProblem.TOO_MANY_LEVELS, problem(listOf(draft(QuestionType.SCORE, (1..11).joinToString("\n") { "level $it" }))))
    assertEquals(D1DraftProblem.EMPTY_OPTION_NAME, problem(listOf(draft(QuestionType.CHOICE, ": nameless\nb"))))
    assertEquals(D1DraftProblem.DUPLICATE_OPTION, problem(listOf(draft(QuestionType.CHOICE, "a\na: again"))))
    assertEquals(D1DraftProblem.BAD_NOUL_OPTION, problem(listOf(draft(QuestionType.NOUL, "maybe: not sure"))))
    assertEquals(
      "Question 2: the name q is used twice.",
      D1Drafts.message(D1DraftException(D1DraftProblem.DUPLICATE_ID, 2, "q")),
    )
  }

  @Test
  fun foldedLines() {
    val sample = D1Sample.parse(File("src/main/res/raw/sample.json").readBytes())
    assertEquals(
      "choice · booking, cancel, prices, complaint",
      D1Drafts.summary(sample.input(D1Input.VOICE).questions.getValue("topic")),
    )
    assertEquals("noul · yes or no", D1Drafts.summary(sample.input(D1Input.MESSAGE).questions.getValue("refund")))
    assertEquals("score · 3 levels", D1Drafts.summary(D1Question(QuestionType.SCORE, "q", listOf("a", "b", "c"))))
  }
}
