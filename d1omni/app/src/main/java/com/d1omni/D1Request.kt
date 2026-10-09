package com.d1omni

/**
 * One request in the provider's `system_one` shape: a [state] (a string, any JSON value, or null)
 * and named [questions] in order; [id] names it in logs and reports.
 */
class D1Request(val id: String?, val state: Any?, val questions: LinkedHashMap<String, D1Question>) {
  companion object {
    /**
     * `{"id"?, "state"?, "questions": {name: {type, instructions, criteria?}}}` as parsed JSON; a
     * missing state is null. Throws [IllegalArgumentException] with the reason for anything else.
     */
    fun fromJson(value: Any?): D1Request {
      require(value is Map<*, *>) { "a request is a JSON object with state and questions" }
      val questions = value["questions"]
      require(questions is Map<*, *> && questions.isNotEmpty()) {
        "a request needs questions: {name: question}"
      }
      val named = LinkedHashMap<String, D1Question>()
      for ((name, question) in questions) {
        named[name as String] =
          try {
            D1Prompt.asQuestion(question)
          } catch (failure: IllegalArgumentException) {
            throw IllegalArgumentException("question $name: ${failure.message}", failure)
          }
      }
      return D1Request(value["id"] as? String, value["state"], named)
    }

    fun parse(text: String): D1Request = fromJson(D1Json.parse(text))
  }
}
