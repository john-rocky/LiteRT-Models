package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * One Decide's run JSON (the keys `demo/check_take.py` reads) on a made-up recorded voice note: every key at the top
 * level, per question and in the layout, the whole-ms numbers (the screen's ms = the work rounded once), the shown
 * strings, the palette, and a round trip through the JSON writer.
 */
class D1RunTest {
  private fun work(): D1Work {
    val choice = D1Question(QuestionType.CHOICE, "What is the customer asking for?", linkedMapOf("booking" to "Booking", "cancel" to "Cancel"))
    val probs = doubleArrayOf(0.9993660449981689, 0.0006339550018310547)
    val question =
      D1RunQuestion("topic", choice, probs, D1Answers.shown(choice, probs), D1Prompt.answer(choice, probs),
        intArrayOf(1, 17, 18), intArrayOf(1, 2), 92, 256, 81, 83, D1Backend.GPU, D1Precision.FP32)
    return D1Work(311_500_001L, 10_000L, 10_312L, linkedMapOf("mel" to 13L, "inputs" to 1L, "encode" to 2L),
      linkedMapOf("audio" to 33L), 92, listOf(question), linkedMapOf("info" to linkedMapOf("P" to 92)))
  }

  private fun run(): LinkedHashMap<String, Any?> {
    val work = work()
    val layout =
      D1RunLayout(1080, 2340, intArrayOf(870, 120, 170, 90), 36,
        listOf(D1AnswerLayout("topic", "answer", "booking", 34f, 102f, intArrayOf(90, 1500, 400, 120), 1, false)), 3.0f, 1.0f)
    return D1Run.build(
      D1RunInput(
        input = D1Input.VOICE,
        source = D1Source.RECORDED,
        media = linkedMapOf("name" to "recorded-1.wav", "sha256" to "ab", "bytes" to 44, "samples" to 0),
        state = null,
        work = work,
        shownMs = D1Text.msLine(work.itemMs, work.buckets),
        deviceModel = "SM-S942Q",
        deviceManufacturer = "samsung",
        deviceShownAs = "Galaxy S26",
        androidRelease = "16",
        accelerator = "GPU",
        precision = linkedMapOf("decide" to "fp32", "audio" to "fp16acc", "vision" to "fp16acc"),
        precisionRequested = linkedMapOf("precision" to null, "precision_audio" to null, "precision_vision" to null),
        graphs = listOf(D1RunGraph("decide_L256", "d1-omni-600M_decide_L256_fp16.tflite", 896315712, D1Backend.GPU, D1Precision.FP32, 2990.0, null)),
        memoryAtReady = linkedMapOf("avail_mem_bytes" to 3_400_000_000L, "proc_mem_available_kb" to 3_500_000L),
        engineLoadMs = 9_000,
        warmupMs = 800,
        airplaneMode = true,
        cgroup = "6:cpuset:/top-app",
        cgroupEnd = "6:cpuset:/top-app",
        layout = layout,
        events = listOf(mapOf("event" to "decide_start")),
        stateStart = emptyMap(),
        stateEnd = emptyMap(),
      )
    )
  }

  @Test
  fun keysAndValues() {
    val run = run()
    assertEquals(D1Run.KEYS, run.keys.toList())
    assertEquals("d1omni-run/1", run["run"])
    assertEquals("voice", run["input"])
    assertEquals("audio", run["kind"])
    assertEquals("recorded", run["source"])
    val question = (run["questions"] as List<*>).single() as Map<*, *>
    assertEquals(D1Run.QUESTION_KEYS, question.keys.toList())
    assertEquals(linkedMapOf("answer" to "booking", "prob" to "0.999"), question["shown"])
    assertEquals(
      linkedMapOf("type" to "choice", "instructions" to "What is the customer asking for?",
        "criteria" to linkedMapOf("booking" to "Booking", "cancel" to "Cancel")),
      question["question"],
    )
    assertEquals(listOf("booking", "cancel"), question["keys"])
    assertEquals(81L, question["infer_ms"])
    // the screen's ms is the work rounded once: 311.500001 ms -> 312
    assertEquals(312L, run["item_total_ms"])
    assertEquals(311_500_001L, run["item_total_ns"])
    assertEquals("312 ms · L256", run["shown_ms"])
    val layout = run["layout"] as Map<*, *>
    assertEquals(D1Run.LAYOUT_KEYS, layout.keys.toList())
    assertEquals(linkedMapOf("left" to 870, "top" to 120, "width" to 170, "height" to 90, "pad_left" to 36), layout["pill_px"])
    assertEquals("#E53935", (layout["pill_palette"] as Map<*, *>)["recording"])
    val answer = (layout["answers"] as List<*>).single() as Map<*, *>
    assertEquals(34f, answer["sp"])
    assertEquals(false, answer["overflow"])
    val runtime = run["runtime"] as Map<*, *>
    assertEquals("2.2.0", runtime["litert"])
    assertEquals(listOf("resident", "memory_at_ready"), (run["graphs"] as Map<*, *>).keys.toList())
    // the writer's output reads back with the same keys and strings
    val again = D1Json.parse(D1Json.writeIndented(run, 1)) as Map<*, *>
    assertEquals(run.keys.toList(), again.keys.toList())
    val againQuestion = (again["questions"] as List<*>)[0] as Map<*, *>
    assertEquals("0.999", (againQuestion["shown"] as Map<*, *>)["prob"])
    assertEquals(0.9993660449981689, ((againQuestion["probs"] as List<*>)[0] as JsonNumber).toDouble(), 0.0)
    assertTrue((again["layout"] as Map<*, *>)["answers"] is List<*>)
  }

  @Test
  fun aMessageHasItsTextAsTheState() {
    val text = "You charged me twice."
    val work = D1Work(1_000_000L, 1L, 2L, linkedMapOf("encode" to 1L), linkedMapOf(), 0, emptyList(), linkedMapOf())
    val run =
      D1Run.build(
        D1RunInput(D1Input.MESSAGE, D1Source.TYPED, linkedMapOf("chars" to text.length), text, work, "1 ms · ", "m", "s", "d",
          "16", "GPU", linkedMapOf(), linkedMapOf(), emptyList(), emptyMap(), 0, 0, false, "c", "c", null, emptyList(),
          emptyMap(), emptyMap())
      )
    assertEquals(text, run["state"])
    assertEquals("typed", run["source"])
    assertEquals(0, run["prefix_rows"])
    assertEquals(null, (run["layout"] as Map<*, *>)["pill_px"])
  }
}
