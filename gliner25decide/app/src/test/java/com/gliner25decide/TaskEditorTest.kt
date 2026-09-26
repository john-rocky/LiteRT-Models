package com.gliner25decide

import javax.xml.parsers.DocumentBuilderFactory
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.w3c.dom.Element

/** The app's one-line-per-task editor format, the bundled example, and schema validation. */
class TaskEditorTest {
  @Test
  fun parsesLabelsMultiLabelThresholdsAndPrompts() {
    val tasks =
      DecideSchema.parseTaskLines(
        "intent: a, b ,c\n\n" +
          "topics: x, y | multi 0.4\n" +
          "flags: p, q | multi\n" +
          "answer: yes, no | prompt: Is it late?\n"
      )
    assertEquals(4, tasks.size)
    assertEquals(Task("intent", listOf("a", "b", "c")), tasks[0])
    assertEquals(Task("topics", listOf("x", "y"), multiLabel = true, clsThreshold = 0.4), tasks[1])
    assertEquals(Task("flags", listOf("p", "q"), multiLabel = true), tasks[2])
    assertEquals(Task("answer", listOf("yes", "no"), prompt = "Is it late?"), tasks[3])
    assertEquals("answer: Is it late?", DecideSchema.promptString(tasks[3]))
    assertThrows(IllegalArgumentException::class.java) { DecideSchema.parseTaskLines("no colon") }
    assertThrows(IllegalArgumentException::class.java) {
      DecideSchema.parseTaskLines("t: a | multi 2")
    }
    assertThrows(IllegalArgumentException::class.java) { DecideSchema.parseTaskLines("t: a | wat") }
  }

  @Test
  fun validationRejectsWhatGliner2CannotDecodeAsWritten() {
    val slots = DecideInputs.LABEL_SLOTS
    DecideSchema.validate(
      listOf(Task("urgency", listOf("a")), Task("urgency level", listOf("b"))),
      slots,
    )
    assertThrows(IllegalArgumentException::class.java) { DecideSchema.validate(emptyList(), slots) }
    assertThrows(IllegalArgumentException::class.java) {
      DecideSchema.validate(listOf(Task("t", emptyList())), slots)
    }
    assertThrows(IllegalArgumentException::class.java) {
      DecideSchema.validate(listOf(Task("t", listOf("a")), Task("t", listOf("b"))), slots)
    }
    assertThrows(IllegalArgumentException::class.java) {
      DecideSchema.validate(listOf(Task("t", List(33) { "l$it" })), slots)
    }
    // "a: b" (prompt "b" on task "a") resolves to a task literally named "a: b" in gliner2.
    val ambiguous = listOf(Task("a", listOf("x"), prompt = "b"), Task("a: b", listOf("y")))
    assertEquals(
      1,
      DecideSchema.resolveConfigIndex(DecideSchema.promptString(ambiguous[0]), ambiguous),
    )
    assertThrows(IllegalArgumentException::class.java) { DecideSchema.validate(ambiguous, slots) }
    assertNull(DecideSchema.resolveConfigIndex("zzz", ambiguous))
  }

  @Test
  fun bundledExampleParsesAndFitsTheSmallestWindow() {
    val strings = androidStrings()
    val tasks = DecideSchema.parseTaskLines(strings.getValue("example_tasks"))
    assertEquals(listOf("intent", "urgency", "sentiment", "topics"), tasks.map { it.name })
    assertTrue(tasks.last().multiLabel)
    assertEquals(0.4, tasks.last().clsThreshold, 0.0)
    DecideSchema.validate(tasks, DecideInputs.LABEL_SLOTS)
    val root = ExternalTestData.resolve()
    val inputs = DecideInputs(GlinerTokenizer(ExternalTestData.tokenizer(root)))
    val prepared = inputs.prepare(strings.getValue("example_text"), tasks)
    assertEquals(128, prepared.window)
    ExternalTestData.reportFile("example.json")
      .writeText(
        JSONObject()
          .put("text", strings.getValue("example_text"))
          .put("tasks", strings.getValue("example_tasks"))
          .put("encoded_tokens", prepared.encoded.encodedLength)
          .put("labels", prepared.encoded.labelCount)
          .put("window", prepared.window)
          .toString(2) + "\n"
      )
  }

  /** Reads single-line `strings.xml` values, applying the only escape they use (`\n`). */
  private fun androidStrings(): Map<String, String> {
    val file = ExternalTestData.moduleFile("app/src/main/res/values/strings.xml")
    val document = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file)
    val nodes = document.getElementsByTagName("string")
    return (0 until nodes.length).associate {
      val element = nodes.item(it) as Element
      element.getAttribute("name") to element.textContent.trim().replace("\\n", "\n")
    }
  }
}
