package com.d1omni

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The inbox demo's fixture and answers against the conversion run's expected file
 * (`demo/oracle/expected_demo.json`: the provider's float32 probabilities and the Python host's rows,
 * probabilities and screen strings for the eight questions): the bundled fixture is the run's
 * fixture byte for byte and its media are the model repository's check-set files; this app encodes
 * every question to the host's ids and markers after the host's P; the screen's strings (answer word
 * and three decimals) of both the provider's and the host's probabilities equal Python's; `answer()`
 * of the host's probabilities equals Python's; the int32 sha256 of the ids equals the host's.
 */
class D1InboxTest {
  private fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  private val raw = File("src/main/res/raw")

  @Test
  fun bundledFixtureAndMediaAreTheRunsFiles() {
    val bundled = File(raw, "inbox_demo.json").readBytes()
    assertEquals(sha256(ExternalTestData.demoFile("fixtures/demo/inbox_demo.json").readBytes()), sha256(bundled))
    val fixture = D1InboxFixture.parse(bundled)
    assertEquals("d1-omni Inbox", fixture.title)
    assertEquals(listOf("aud_food_03", "img_dogs_01", "card_text"), fixture.items.map { it.item })
    assertEquals(listOf(D1Kind.AUDIO, D1Kind.IMAGE, D1Kind.TEXT), fixture.items.map { it.kind })
    assertEquals(8, fixture.questionCount)
    for (item in fixture.items) {
      val name = item.mediaFile ?: continue
      val bytes = File(raw, name).readBytes()
      assertEquals(name, item.mediaSha256, sha256(bytes))
      assertEquals(name, item.mediaBytes, bytes.size.toLong())
      assertEquals(name, sha256(ExternalTestData.repoFile("fixtures/media/$name").readBytes()), sha256(bytes))
    }
    val samples = D1Wav.parse(File(raw, "aud_food_03.wav").readBytes())
    assertEquals(139200, samples.size)
    assertEquals("Voice note · 8.7 s", D1InboxText.header(fixture.items[0], samples.size))
  }

  @Test
  fun rowsAnswersAndShownStringsMatchTheExpectedFile() {
    val fixture = D1InboxFixture.parse(File(raw, "inbox_demo.json").readBytes())
    val expected = ExternalTestData.json(ExternalTestData.demoFile("oracle/expected_demo.json"))
    val items = (expected["items"] as List<*>).map { it as Map<*, *> }.associateBy { it["item"] as String }
    val tokenizer = ExternalTestData.tokenizer()
    val contract = ExternalTestData.contract()
    var checked = 0
    for (item in fixture.items) {
      val e = items.getValue(item.item)
      val questions = (e["questions"] as List<*>).map { it as Map<*, *> }
      val prefixRows = (e["prefix_rows"] as JsonNumber).toInt()
      val rows = D1Rows.rows(tokenizer, contract, item.state, item.questions.values.toList(), prefixRows, item.kind)
      for ((row, q) in rows.zip(questions)) {
        val qid = q["qid"] as String
        val host = q["mac_host"] as Map<*, *>
        val key = "${item.item}/$qid"
        assertTrue(key, row.ids.contentEquals(ExternalTestData.ints(host["ids"])))
        assertTrue(key, row.markers.contentEquals(ExternalTestData.ints(host["markers"])))
        assertEquals(key, (host["P"] as JsonNumber).toInt(), row.prefixRows)
        assertEquals(key, host["ids_sha256"], D1InboxAnswer.idsSha256(row.ids))
        assertEquals(key, D1Contract.bucketFor(row.positions, listOf(128, 256)), (host["bucket"] as JsonNumber).toInt())
        val question = item.questions.getValue(qid)
        assertEquals(key, (q["keys"] as List<*>), D1InboxAnswer.keys(question))
        for ((which, probsKey) in listOf("shown_provider" to "provider", "shown_mac_host" to "mac_host")) {
          val probs = ExternalTestData.doubles((q[probsKey] as Map<*, *>)["probs"])
          val shown = D1InboxAnswer.shown(question, probs)
          val want = q[which] as Map<*, *>
          assertEquals("$key $which", want["answer"], shown.answer)
          assertEquals("$key $which", want["prob"], shown.prob)
          assertEquals("$key argmax", (q[probsKey] as Map<*, *>)["argmax_key"], D1InboxAnswer.argmaxKey(question, probs))
        }
        val hostProbs = ExternalTestData.doubles(host["probs"])
        assertEquals(key, D1Json.write(host["answer"]), D1Json.write(D1Prompt.answer(question, hostProbs)))
        checked++
      }
    }
    assertEquals(8, checked)
  }

  @Test
  fun shownRule() {
    val noul = D1Question(QuestionType.NOUL, "q", null)
    assertEquals(D1Shown("yes", "0.891", 0.8908922076225281), D1InboxAnswer.shown(noul, doubleArrayOf(0.8908922076225281, 0.1091077923774719)))
    val no = D1InboxAnswer.shown(noul, doubleArrayOf(0.25, 0.75))
    assertEquals("no", no.answer)
    assertEquals("0.750", no.prob)
    // P(yes) exactly 0.5 shows yes
    assertEquals("yes", D1InboxAnswer.shown(noul, doubleArrayOf(0.5, 0.5)).answer)
    val choice = D1Question(QuestionType.CHOICE, "q", linkedMapOf("a" to "A", "b" to "B"))
    assertEquals(D1Shown("a", "0.500", 0.5), D1InboxAnswer.shown(choice, doubleArrayOf(0.5, 0.5)))
    val score = D1Question(QuestionType.SCORE, "q", listOf("Low", "High"))
    assertEquals("High", D1InboxAnswer.shown(score, doubleArrayOf(0.0625, 0.9375)).answer)
    assertEquals("0.938", D1InboxAnswer.shown(score, doubleArrayOf(0.0625, 0.9375)).prob)
    assertEquals(listOf("Low", "High"), D1InboxAnswer.words(score))
    assertEquals(listOf("0", "1"), D1InboxAnswer.keys(score))
    // int32 little-endian: [1] -> 01 00 00 00
    assertEquals(sha256(byteArrayOf(1, 0, 0, 0)), D1InboxAnswer.idsSha256(intArrayOf(1)))
  }

  @Test
  fun screenTexts() {
    val gpu = D1Backend.GPU
    assertEquals("GPU FP32", D1InboxText.accelerator(listOf(gpu to D1Precision.FP32, gpu to D1Precision.FP32)))
    assertEquals(
      "GPU FP16 (FP32 accum)",
      D1InboxText.accelerator(listOf(gpu to D1Precision.FP16_FP32_ACCUM)),
    )
    assertEquals("GPU", D1InboxText.accelerator(listOf(gpu to D1Precision.FP32, gpu to D1Precision.FP16_FP32_ACCUM)))
    assertEquals("CPU 4 threads", D1InboxText.accelerator(listOf(D1Backend.CPU to D1Precision.FP32)))
    assertEquals("GPU + CPU", D1InboxText.accelerator(listOf(gpu to D1Precision.FP32, D1Backend.CPU to D1Precision.FP32)))
    val graphs = D1InboxText.graphsLine(listOf(256, 128), listOf(1001), vision = true)
    assertEquals("d1-omni-600M fp16 · L128 + L256 · audio T1001 · vision tower", graphs)
    assertEquals(
      listOf("Galaxy S26 · LiteRT 2.2.0 · GPU", graphs, "total 712 ms · airplane mode on"),
      D1InboxText.footer("Galaxy S26", "GPU", graphs, 712, true),
    )
    assertEquals("", D1InboxText.footer("Galaxy S26", "GPU", graphs, null, false)[2])
    assertEquals("audio 32 ms", D1InboxText.mediaMs(D1Kind.AUDIO, 32))
    assertEquals("vision 140 ms", D1InboxText.mediaMs(D1Kind.IMAGE, 140))
    assertEquals(null, D1InboxText.mediaMs(D1Kind.TEXT, 3))
    assertEquals("3 answers · 302 ms", D1InboxText.itemTotal(3, 302))
    assertEquals("Galaxy S26", D1Device.marketName("SM-S942Q"))
    assertEquals("Pixel 9", D1Device.marketName("Pixel 9"))
  }

  @Test
  fun fixturesThatCannotBeRead() {
    val good = String(File(raw, "inbox_demo.json").readBytes(), Charsets.UTF_8)
    val bad =
      listOf(
        good.replace("d1omni-inbox-demo/1", "d1omni-inbox-demo/2"),
        good.replace("\"kind\": \"audio\"", "\"kind\": \"video\""),
        good.replace("\"file\": \"aud_food_03.wav\"", "\"file\": \"../aud_food_03.wav\""),
        good.replace("\"item\": \"card_text\"", "\"item\": \"img_dogs_01\""),
        good.replace("\"type\": \"score\"", "\"type\": \"rank\""),
      )
    for (text in bad) {
      val failure = runCatching { D1InboxFixture.parse(text.toByteArray()) }.exceptionOrNull()
      assertTrue(text.take(80), failure is IllegalArgumentException)
    }
  }
}
