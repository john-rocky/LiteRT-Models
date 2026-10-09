package com.d1omni

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The bundled sample and its expected answers against the conversion run's files (`demo/fixtures/story/sample_r7.json`,
 * `demo/oracle/expected_story.json`: the provider's float32 probabilities and the Python host's rows and probabilities
 * of the four questions): the bundled sample is the run's file byte for byte, its media are the files the oracle
 * scored; this app encodes every question to the host's ids and markers after the host's P; the screen's strings of
 * the provider's and the host's probabilities equal Python's; `answer()` of the host's probabilities equals Python's;
 * the int32 sha256 of the ids equals the host's.
 */
class D1SampleTest {
  private fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  private val raw = File("src/main/res/raw")

  @Test
  fun bundledSampleAndMediaAreTheRunsFiles() {
    val bundled = File(raw, "sample.json").readBytes()
    assertEquals(sha256(ExternalTestData.demoFile("fixtures/story/sample_r7.json").readBytes()), sha256(bundled))
    val sample = D1Sample.parse(bundled)
    assertEquals("d1-omni", sample.title)
    assertEquals(D1Input.entries, sample.inputs.map { it.input })
    assertEquals(listOf(1, 1, 2), sample.inputs.map { it.questions.size })
    val sources = mapOf(D1Input.VOICE to "fixtures/story/voice_c.wav", D1Input.PHOTO to null)
    for (input in sample.inputs) {
      val name = input.mediaFile ?: continue
      val bytes = File(raw, name).readBytes()
      assertEquals(name, input.mediaSha256, sha256(bytes))
      assertEquals(name, input.mediaBytes, bytes.size.toLong())
      val source = sources[input.input]
      val original =
        if (source != null) ExternalTestData.demoFile(source) else ExternalTestData.repoFile("fixtures/media/$name")
      assertEquals(name, sha256(original.readBytes()), sha256(bytes))
    }
    val samples = D1Wav.parse(File(raw, "sample_voice_note.wav").readBytes())
    assertEquals(117600, samples.size)
    assertEquals("7.4 s", D1Text.seconds(samples.size))
    assertEquals("You charged me twice for the bath and trim on Saturday, please refund one of them.", sample.input(D1Input.MESSAGE).state)
  }

  @Test
  fun rowsAnswersAndShownStringsMatchTheExpectedFile() {
    val sample = D1Sample.parse(File(raw, "sample.json").readBytes())
    val expected = ExternalTestData.json(ExternalTestData.demoFile("oracle/expected_story.json"))
    assertEquals(true, (expected["summary"] as Map<*, *>)["pass"])
    val inputs = (expected["inputs"] as List<*>).map { it as Map<*, *> }.associateBy { it["input"] as String }
    val tokenizer = ExternalTestData.tokenizer()
    val contract = ExternalTestData.contract()
    var checked = 0
    for (input in sample.inputs) {
      val e = inputs.getValue(input.input.wireName)
      val questions = (e["questions"] as List<*>).map { it as Map<*, *> }
      val prefixRows = (e["prefix_rows"] as JsonNumber).toInt()
      val rows =
        D1Rows.rows(tokenizer, contract, input.state, input.questions.values.toList(), prefixRows, input.input.kind)
      for ((row, q) in rows.zip(questions)) {
        val qid = q["qid"] as String
        val host = q["mac_host"] as Map<*, *>
        val key = "${input.input.wireName}/$qid"
        assertTrue(key, row.ids.contentEquals(ExternalTestData.ints(host["ids"])))
        assertTrue(key, row.markers.contentEquals(ExternalTestData.ints(host["markers"])))
        assertEquals(key, (host["P"] as JsonNumber).toInt(), row.prefixRows)
        assertEquals(key, host["ids_sha256"], D1Answers.idsSha256(row.ids))
        assertEquals(key, D1Contract.bucketFor(row.positions, listOf(128, 256)), (host["bucket"] as JsonNumber).toInt())
        val question = input.questions.getValue(qid)
        assertEquals(key, q["keys"] as List<*>, D1Answers.keys(question))
        assertEquals(key, q["instructions"], question.instructions)
        for ((which, probsKey) in listOf("shown_provider" to "provider", "shown_mac_host" to "mac_host")) {
          val probs = ExternalTestData.doubles((q[probsKey] as Map<*, *>)["probs"])
          val shown = D1Answers.shown(question, probs)
          val want = q[which] as Map<*, *>
          assertEquals("$key $which", want["answer"], shown.answer)
          assertEquals("$key $which", want["prob"], shown.prob)
          assertEquals("$key argmax", (q[probsKey] as Map<*, *>)["argmax_key"], D1Answers.argmaxKey(question, probs))
        }
        val hostProbs = ExternalTestData.doubles(host["probs"])
        assertEquals(key, D1Json.write(host["answer"]), D1Json.write(D1Prompt.answer(question, hostProbs)))
        checked++
      }
    }
    assertEquals(4, checked)
  }

  @Test
  fun samplesThatCannotBeRead() {
    val good = String(File(raw, "sample.json").readBytes(), Charsets.UTF_8)
    val bad =
      listOf(
        good.replace("d1omni-sample/1", "d1omni-sample/2"),
        good.replace("\"input\": \"photo\"", "\"input\": \"video\""),
        good.replace("\"file\": \"img_dogs_01.png\"", "\"file\": \"../img_dogs_01.png\""),
        good.replace("\"kind\": \"audio\"", "\"kind\": \"text\""),
        good.replace("\"type\": \"noul\"", "\"type\": \"rank\""),
      )
    for (text in bad) {
      assertTrue(text.take(80), text != good)
      val failure = runCatching { D1Sample.parse(text.toByteArray()) }.exceptionOrNull()
      assertTrue(text.take(80), failure is IllegalArgumentException)
    }
  }
}
