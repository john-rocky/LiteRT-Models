package com.d1omni

/** The app's three inputs, one screen each: [kind] is the request kind the model is asked with. */
enum class D1Input(val wireName: String, val kind: D1Kind, val label: String) {
  VOICE("voice", D1Kind.AUDIO, "Voice"),
  PHOTO("photo", D1Kind.IMAGE, "Photo"),
  MESSAGE("message", D1Kind.TEXT, "Message");

  companion object {
    fun of(name: String): D1Input? = entries.firstOrNull { it.wireName == name }
  }
}

/**
 * One input of the bundled sample: its media file in `res/raw` (voice and photo) with its sha256 and size, the text of
 * a message (the request's state), and the input's default questions in order.
 */
class D1SampleInput(
  val input: D1Input,
  val mediaFile: String?,
  val mediaSha256: String?,
  val mediaBytes: Long?,
  /** The request's state as parsed JSON: the message's text, or null for a voice note and a photo. */
  val state: Any?,
  val questions: LinkedHashMap<String, D1Question>,
  /** What the voice note says (for the README and the record; the model never reads it). */
  val transcript: String?,
)

/**
 * The bundled sample (`res/raw/sample.json`, `demo/fixtures/story/sample_r7.json` in the conversion run): one voice
 * note, one photo and one message of an invented pet-grooming shop's customer, and the default question of each input.
 * Load sample puts them in the three screens; the questions are also each screen's default. Android-free.
 */
class D1Sample(val id: String, val title: String, val inputs: List<D1SampleInput>) {
  fun input(which: D1Input): D1SampleInput = inputs.first { it.input == which }

  companion object {
    const val FORMAT = "d1omni-sample/1"

    private val SHA256 = Regex("[0-9a-f]{64}")

    /** Parses and checks a sample file; throws [IllegalArgumentException] with the reason. */
    fun parse(bytes: ByteArray): D1Sample {
      val root = D1Json.parse(bytes)
      require(root is Map<*, *>) { "a sample is a JSON object" }
      require(root["sample"] == FORMAT) { "sample format ${root["sample"]}, this app reads $FORMAT" }
      val id = root["id"] as? String ?: throw IllegalArgumentException("a sample needs an id")
      val title = root["title"] as? String ?: throw IllegalArgumentException("a sample needs a title")
      val list = root["inputs"] as? List<*> ?: throw IllegalArgumentException("a sample needs inputs")
      val inputs = list.map { input(it) }
      require(inputs.map { it.input } == D1Input.entries) {
        "a sample holds one voice, one photo and one message, in that order"
      }
      return D1Sample(id, title, inputs)
    }

    private fun input(value: Any?): D1SampleInput {
      require(value is Map<*, *>) { "an input is a JSON object" }
      val input =
        D1Input.of(value["input"] as? String ?: "")
          ?: throw IllegalArgumentException("input ${value["input"]} is not voice, photo or message")
      require(value["kind"] == input.kind.wireName) { "${input.wireName}: kind ${value["kind"]}, expected ${input.kind.wireName}" }
      val media = value["media"] as? Map<*, *>
      if (input == D1Input.MESSAGE) {
        require(media == null) { "a message carries no media" }
        require(value["state"] is String) { "a message's state is its text" }
      } else {
        require(media != null) { "${input.wireName} needs media {file, sha256, bytes}" }
        val file = media["file"] as? String
        require(file != null && D1Launch.fileNameValid(file)) { "${input.wireName}: media file $file is not a plain file name" }
        require((media["sha256"] as? String)?.matches(SHA256) == true) { "${input.wireName}: media sha256 is not 64 hex digits" }
        require(value["state"] == null) { "${input.wireName}: the state of a voice note or a photo is null" }
      }
      val questions = value["questions"] as? Map<*, *>
      require(questions != null && questions.isNotEmpty()) { "${input.wireName}: no questions" }
      val named = LinkedHashMap<String, D1Question>()
      for ((name, question) in questions) {
        named[name as String] =
          try {
            D1Prompt.asQuestion(question)
          } catch (failure: IllegalArgumentException) {
            throw IllegalArgumentException("${input.wireName}/$name: ${failure.message}", failure)
          }
      }
      return D1SampleInput(
        input,
        media?.get("file") as String?,
        media?.get("sha256") as String?,
        (media?.get("bytes") as JsonNumber?)?.literal?.toLong(),
        value["state"],
        named,
        media?.get("text") as String?,
      )
    }
  }
}
