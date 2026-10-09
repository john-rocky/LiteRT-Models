package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The demo run JSON's shape (the keys `demo/check_take.py`, `pill_track.py` and `card_track.py`
 * read) on a made-up run: every key at the top level, per item, per question and in the layout, the
 * whole-ms numbers, the shown strings, the palettes, and a round trip through the JSON writer.
 */
class D1DemoRunTest {
  private fun run(): LinkedHashMap<String, Any?> {
    val noul = D1Question(QuestionType.NOUL, "Is it?", null)
    val probs = doubleArrayOf(0.8908922076225281, 0.10910782217979431)
    val question =
      D1DemoQuestion("request", noul, probs, D1InboxAnswer.shown(noul, probs), "84 ms", D1Prompt.answer(noul, probs),
        intArrayOf(1, 17, 18), intArrayOf(1, 2), 109, 256, 84, 85, D1Backend.GPU, D1Precision.FP32)
    val item =
      D1DemoItem("aud_food_03", D1Kind.AUDIO, "aud_food_03.wav", "455c", "bundled",
        D1DemoPlayback(970L, 1_000L, 1_200L, 8700, 139200, "timestamp", "played", 8950.0),
        linkedMapOf("wav" to 1L, "mel" to 13L), linkedMapOf("audio" to 33L), 109, "Voice note · 8.7 s", 302, 10_000L,
        10_302L, listOf(question), emptyMap())
    val layout =
      D1DemoLayout(1080, 2340, 222, 2118, intArrayOf(900, 230, 160, 72), 30,
        linkedMapOf("aud_food_03" to intArrayOf(60, 400, 48, 48)), 48f, 3.0f, 1.0f, null)
    return D1DemoRun.build(
      D1DemoRunInput("inbox_demo", "files/inbox_demo.json", "8044", "SM-S942Q", "samsung", "Galaxy S26", "16", "GPU",
        linkedMapOf("decide" to "fp32", "audio" to "fp16acc", "vision" to "fp16acc"),
        linkedMapOf("precision" to null, "precision_audio" to null, "precision_vision" to null),
        listOf(D1DemoGraph("decide_L256", "d1-omni-600M_decide_L256_fp16.tflite", 896315712, D1Backend.GPU, D1Precision.FP32, 2990.0, null)),
        linkedMapOf("avail_mem_bytes" to 3_400_000_000L, "proc_mem_available_kb" to 3_500_000L), 15_000, 900,
        "d1-omni Inbox", listOf("a", "b", "total 302 ms · airplane mode on"), 1000, 1500, 302, 301_600_000L, listOf(item), true,
        "6:cpuset:/top-app", "6:cpuset:/top-app", layout, listOf(mapOf("event" to "presentation")), emptyMap(), emptyMap())
    )
  }

  @Test
  fun keysAndValues() {
    val run = run()
    assertEquals(D1DemoRun.KEYS, run.keys.toList())
    val item = (run["items"] as List<*>).single() as Map<*, *>
    assertEquals(D1DemoRun.ITEM_KEYS, item.keys.toList())
    val question = (item["questions"] as List<*>).single() as Map<*, *>
    assertEquals(D1DemoRun.QUESTION_KEYS, question.keys.toList())
    assertEquals(linkedMapOf("answer" to "yes", "prob" to "0.891"), question["shown"])
    assertEquals("84 ms", question["shown_ms"])
    assertEquals(84L, question["infer_ms"])
    assertEquals(302L, run["request_total_ms"])
    assertEquals(301_600_000L, run["request_total_ns"])
    assertEquals(listOf("yes", "no"), question["keys"])
    val layout = run["layout"] as Map<*, *>
    assertEquals(D1DemoRun.LAYOUT_KEYS, layout.keys.toList())
    assertEquals(linkedMapOf("left" to 900, "top" to 230, "width" to 160, "height" to 72, "pad_left" to 30), layout["pill_px"])
    assertEquals("#E53935", (layout["pill_palette"] as Map<*, *>)["playing"])
    assertEquals("#1565C0", (layout["palette"] as Map<*, *>)["running"])
    val runtime = run["runtime"] as Map<*, *>
    assertEquals("GPU", runtime["accelerator"])
    assertEquals("2.2.0", runtime["litert"])
    val graphs = run["graphs"] as Map<*, *>
    assertEquals(listOf("resident", "memory_at_ready"), graphs.keys.toList())
    // the writer's output reads back with the same keys and strings
    val again = D1Json.parse(D1Json.writeIndented(run, 1)) as Map<*, *>
    assertEquals(run.keys.toList(), again.keys.toList())
    val againQuestion = (((again["items"] as List<*>)[0] as Map<*, *>)["questions"] as List<*>)[0] as Map<*, *>
    assertEquals("0.891", (againQuestion["shown"] as Map<*, *>)["prob"])
    assertEquals(0.8908922076225281, ((againQuestion["probs"] as List<*>)[0] as JsonNumber).toDouble(), 0.0)
  }

  @Test
  fun footerTotalIsTheCardsSum() {
    // 318.6 + 258.6 + 159.6 ms: every card rounds up, while the unrounded sum (736.8 ms) would round to 737.
    val itemNanos = listOf(318_600_000L, 258_600_000L, 159_600_000L)
    val cards = itemNanos.map { D1InboxText.itemMs(it) }
    assertEquals(listOf(319L, 259L, 160L), cards)
    val total = D1InboxText.requestTotalMs(itemNanos)
    assertEquals(cards.sum(), total)
    assertEquals(738L, total)
    assertEquals("3 answers · 319 ms", D1InboxText.itemTotal(3, cards[0]))
    assertEquals("total 738 ms · airplane mode on", D1InboxText.footer("Galaxy S26", "GPU", "graphs", total, true).last())
  }
}
