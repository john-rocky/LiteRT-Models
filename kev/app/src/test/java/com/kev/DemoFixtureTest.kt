package com.kev

import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The demo recording's fixtures and run JSON. Fixtures are `{"id", "state", "questions"}` files
 * (the demo lane's `kev_work/demo/fixtures/<id>.json`); they must parse and encode into rows of the
 * lengths the demo lane measured with the author's code (`token_lengths.json`) and, when its oracle
 * is there, into the oracle's exact row IDs. Without those files the bundled `example_ticket.json`
 * (the same `demo_ticket_01`, byte for byte) stands in, against the oracle rows bundled in
 * `examples_oracle.json`. The run JSON builder must write every key the recording scripts read,
 * with the card's ms equal to the question's `infer_ms`.
 */
class DemoFixtureTest {
  private val demo by lazy { File(ExternalTestData.root(), "demo") }

  @Test
  fun fixturesEncodeToTheMeasuredRows() {
    val encoder = KevEncoder(ExternalTestData.tokenizer())
    val fixtures =
      File(demo, "fixtures")
        .listFiles { file ->
          file.name.endsWith(".json") && file.name !in setOf("requests.json", "token_lengths.json")
        }
        ?.sortedBy { it.name }
        .orEmpty()
    val lengthsFile = File(demo, "fixtures/token_lengths.json")
    if (fixtures.isEmpty() || !lengthsFile.isFile) {
      // The bundled copy of demo_ticket_01: rows equal the bundled demo oracle rows.
      val example =
        KevFixture.parse(
          ExternalTestData.moduleFile("app/src/main/res/raw/example_ticket.json").readText()
        )
      assertEquals("demo_ticket_01", example.id)
      val expected =
        (KevJson.parse(ExternalTestData.resource("examples_oracle.json").readBytes()) as Map<*, *>)[
          "examples"]
          as List<*>
      val questions =
        (expected.single { (it as Map<*, *>)["id"] == example.id } as Map<*, *>)["questions"]
          as List<*>
      val rows = encoder.encode(KevRecords.toRecord(example.request).first).rows()
      assertEquals(questions.size, rows.size)
      for ((row, question) in rows.zip(questions)) {
        assertArrayEquals(OracleFixtures.ints((question as Map<*, *>)["row_ids"]), row.ids)
      }
      assertEquals(listOf(131, 101, 93), rows.map { it.length })
      println("KEV_DEMO_FIXTURE fallback demo_ticket_01 rows=${rows.map { it.length }}")
      return
    }
    val lengths = (KevJson.parse(lengthsFile.readBytes()) as Map<*, *>)["records"] as Map<*, *>
    val oracleFile = File(demo, "oracle/oracle_demo.json")
    val oracleRows =
      if (oracleFile.isFile) {
        ((KevJson.parse(oracleFile.readBytes()) as Map<*, *>)["questions"] as List<*>).associate {
          val question = it as Map<*, *>
          "${question["id"]}/${question["qid"]}" to OracleFixtures.ints(question["row_ids"])
        }
      } else {
        emptyMap()
      }
    val report = LinkedHashMap<String, Any?>()
    var idsCompared = 0
    for (file in fixtures) {
      val fixture = KevFixture.parse(file.readText())
      assertEquals(file.name, "${fixture.id}.json")
      val (record, meta) = KevRecords.toRecord(fixture.request)
      val encoded = encoder.encode(record)
      val rows = encoded.rows()
      val measured = lengths[fixture.id] as Map<*, *>
      val measuredRows =
        (measured["rows"] as Map<*, *>).mapValues { (it.value as JsonNumber).toInt() }
      assertEquals(fixture.id, measuredRows, meta.map { it.id }.zip(rows.map { it.length }).toMap())
      assertEquals(
        fixture.id,
        (measured["input_tokens"] as JsonNumber).toInt(),
        encoded.inputTokens,
      )
      for ((question, row) in meta.zip(rows)) {
        oracleRows["${fixture.id}/${question.id}"]?.let {
          assertArrayEquals("${fixture.id}/${question.id}", it, row.ids)
          idsCompared++
        }
        assertEquals(256, row.window(KevFiles.DEFAULT_INSTALL))
      }
      report[fixture.id] =
        linkedMapOf(
          "rows" to meta.map { it.id }.zip(rows.map { it.length }).toMap(),
          "input_tokens" to encoded.inputTokens,
        )
    }
    val ticket = report["demo_ticket_01"] as Map<*, *>
    assertEquals(mapOf("team" to 131, "refund" to 101, "mood" to 93), ticket["rows"])
    ExternalTestData.writeReport(
      "demo_fixtures.json",
      linkedMapOf(
        "test" to "DemoFixtureTest",
        "fixtures" to report,
        "row_ids_equal_demo_oracle" to idsCompared,
      ),
    )
    println(
      "KEV_DEMO_FIXTURES ${fixtures.size} fixtures, rows = token_lengths.json, row ids equal to the demo oracle: $idsCompared"
    )
  }

  @Test
  fun runJsonHasEveryKeyTheRecordingReads() {
    val meta =
      listOf(
        QuestionMeta("team", QuestionType.CHOICE, listOf("billing", "shipping"), null),
        QuestionMeta("refund", QuestionType.NOUL, listOf("false", "true"), null),
        QuestionMeta(
          "mood",
          QuestionType.SCORE,
          listOf("0", "1", "2"),
          mapOf("0" to "Calm", "1" to "Annoyed", "2" to "Angry"),
        ),
      )
    val probabilities =
      listOf(
        doubleArrayOf(0.8205, 0.1795),
        doubleArrayOf(0.054, 0.946),
        doubleArrayOf(0.2, 0.6579, 0.1421),
      )
    val answers = KevAnswers.toAnswers(probabilities, meta)
    val questions =
      meta.indices.map { index ->
        val row =
          KevRow(intArrayOf(248060, 11, 248061, 248049, 12, 248050, 248062), 6, intArrayOf(5))
        val probs = probabilities[index]
        val scores =
          KevScores(
            FloatArray(probs.size),
            FloatArray(probs.size),
            FloatArray(probs.size) { probs[it].toFloat() },
          )
        val result =
          KevQuestionResult(
            index,
            meta[index],
            row,
            KevForm.ROW,
            256,
            5,
            scores,
            probs,
            640.4 + index,
            0.6,
          )
        val answer = answers.getValue(meta[index].id) as Map<*, *>
        val view = KevAnswerView.of(answer, meta[index], probs)
        val inferMs = KevAnswerView.wholeMillis(result.inferMs)
        KevDemoQuestion(
          result,
          answer,
          view.shownCompact(),
          "$inferMs ms",
          3,
          inferMs,
          1,
          inferMs + 4,
        )
      }
    for (layout in
      listOf(
        null,
        KevDemoLayout(
          1080,
          2340,
          222,
          2118,
          listOf(KevDemoLayout.Card("team", 63, 1400, 77, 77, 73.1f)),
          68.25f,
          2.625f,
          1f,
        ),
      )) {
      val l256 = KevGraphFile(KevGraphKey.Window(256), 1_262_377_184L)
      val run =
        KevDemoRun.build(
          KevDemoRunInput(
            fixtureId = "demo_ticket_01",
            fixturePath = "/data/user/0/com.kev/files/demo_ticket_01.json",
            deviceModel = "SM-S942Q",
            deviceManufacturer = "samsung",
            deviceShownAs = "Galaxy S26",
            deviceAndroidRelease = "16",
            litert = "2.2.0",
            accelerator = "GPU FP32",
            precision = "fp32",
            graph = l256,
            form = KevForm.ROW,
            windowsUsed = listOf(256),
            resident = listOf(l256),
            compiled = listOf(KevGraphKey.Window(256)),
            availableBytesBeforeCompile = listOf(4_876_628_000L),
            closed = listOf(KevGraphKey.Window(512)),
            secondRefused = false,
            plan =
              KevDemoPlan(
                KevGraphMode.AUTO,
                KevForm.ROW,
                KevPrediction(969.0, null),
                KevPlanInputs(7_000_000_000L, listOf(KevGraphKey.Window(512))),
              ),
            state = null,
            engineLoadMs = 16_200,
            warmupMs = 700,
            title = "Kev Decide",
            footerLines = listOf("line 1", "line 2", "line 3"),
            delayMs = 1500,
            gapMs = 800,
            tokenizeMs = 9,
            requestTotalMs = 1950,
            questions = questions,
            airplaneMode = true,
            cgroup = "6:cpuset:/top-app",
            cgroupEnd = "6:cpuset:/top-app",
            layout = layout,
          )
        )
      // The JSON text is valid and keeps every key.
      val parsed = KevJson.parse(KevJson.write(run)) as Map<*, *>
      assertEquals(KevDemoRun.KEYS, parsed.keys.toList())
      assertEquals("demo_ticket_01", (parsed["fixture"] as Map<*, *>)["id"])
      assertEquals(
        listOf("model", "manufacturer", "shown_as", "android_release"),
        (parsed["device"] as Map<*, *>).keys.toList(),
      )
      assertEquals("Galaxy S26", (parsed["device"] as Map<*, *>)["shown_as"])
      // `graph` stays an object with file / L / bytes (the recording scripts read graph.file).
      val graph = parsed["graph"] as Map<*, *>
      assertEquals(GRAPH_KEYS, graph.keys.toList())
      assertEquals("kev-0.8b_rowprefill_L256_fp16fc_i8emb.tflite", graph["file"])
      assertEquals(256, (graph["L"] as JsonNumber).toInt())
      assertEquals("row", graph["form"])
      assertEquals("fp32", graph["precision"])
      assertNull(graph["pair"])
      val compiled = (graph["compiled"] as List<*>).single() as Map<*, *>
      assertEquals(256, (compiled["L"] as JsonNumber).toInt())
      assertEquals("4876628000", (compiled["avail_mem_bytes"] as JsonNumber).literal)
      assertEquals(
        listOf(512),
        (graph["closed"] as List<*>).map { ((it as Map<*, *>)["L"] as JsonNumber).toInt() },
      )
      val plan = parsed["plan"] as Map<*, *>
      assertEquals(
        listOf("requested", "form", "predicted_ms", "avail_mem_bytes", "resident"),
        plan.keys.toList(),
      )
      assertEquals(listOf("L512"), plan["resident"])
      assertEquals("row", plan["form"])
      assertNull((plan["predicted_ms"] as Map<*, *>)["pair"])
      assertNull(parsed["state"])
      assertEquals("fp32", (parsed["runtime"] as Map<*, *>)["precision"])
      assertEquals(listOf("line 1", "line 2", "line 3"), parsed["footer_lines"])
      for (question in parsed["questions"] as List<*>) {
        val entry = question as Map<*, *>
        assertEquals(KevDemoRun.QUESTION_KEYS, entry.keys.toList())
        assertEquals("row", entry["form"])
        assertEquals(5, (entry["branch_len"] as JsonNumber).toInt())
        // The recording's check: the card's ms (shown_ms) is the question's infer_ms.
        assertEquals(
          (entry["infer_ms"] as JsonNumber).toInt(),
          (entry["shown_ms"] as String).substringBefore(" ").toInt(),
        )
        assertTrue(entry["shown"] is Map<*, *>)
        val ids = OracleFixtures.ints(entry["row_ids"])
        assertEquals(KevPipeline.idsSha256(ids), entry["ids_sha256"])
        assertEquals(ids.size, (entry["row_len"] as JsonNumber).toInt())
      }
      val shown = (parsed["questions"] as List<*>).map { (it as Map<*, *>)["shown"] }
      assertEquals(
        listOf(
          mapOf("choice" to "billing", "probabilities" to mapOf("billing" to "0.8205")),
          mapOf("noul" to "0.9460"),
          mapOf("probabilities" to mapOf("1" to "0.6579")),
        ),
        shown,
      )
      val layoutJson = parsed["layout"] as Map<*, *>
      assertEquals(KevDemoRun.LAYOUT_KEYS, layoutJson.keys.toList())
      assertEquals(KevDemoRun.PALETTE, layoutJson["palette"])
      if (layout == null) {
        assertNull(layoutJson["screen_px"])
      } else {
        assertEquals(
          listOf(1080, 2340),
          (layoutJson["screen_px"] as List<*>).map { (it as JsonNumber).toInt() },
        )
        val card = (layoutJson["cards"] as List<*>).single() as Map<*, *>
        assertEquals(
          listOf("left", "top", "width", "height"),
          (card["indicator_px"] as Map<*, *>).keys.toList(),
        )
        assertEquals(listOf("top", "bottom"), (layoutJson["content_px"] as Map<*, *>).keys.toList())
      }
    }
  }

  @Test
  fun aPairRunKeepsTheRowCoordinatesAndAddsTheState() {
    val meta = QuestionMeta("refund", QuestionType.NOUL, listOf("false", "true"), null)
    val probs = doubleArrayOf(0.054, 0.946)
    val answer = KevAnswers.toAnswers(listOf(probs), listOf(meta)).getValue("refund") as Map<*, *>
    // State [state] + 11 (2 tokens), branch [question] [option] 12 [/option] [decide] (5 tokens).
    val row = KevRow(intArrayOf(248060, 11, 248061, 248049, 12, 248050, 248062), 6, intArrayOf(5))
    val result =
      KevQuestionResult(
        0,
        meta,
        row,
        KevForm.PAIR,
        64,
        5,
        KevScores(FloatArray(2), FloatArray(2), FloatArray(2) { probs[it].toFloat() }),
        probs,
        187.4,
        0.6,
      )
    val view = KevAnswerView.of(answer, meta, probs)
    val pair = KevGraphFile(KevGraphKey.Pair(KevPairShape(128, 64)), 1_264_544_720L)
    val run =
      KevDemoRun.build(
        KevDemoRunInput(
          fixtureId = "own_fiveq_09",
          fixturePath = "/data/user/0/com.kev/files/own_fiveq_09.json",
          deviceModel = "SM-S942Q",
          deviceManufacturer = "samsung",
          deviceShownAs = "Galaxy S26",
          deviceAndroidRelease = "16",
          litert = "2.2.0",
          accelerator = "GPU FP32",
          precision = "fp32",
          graph = pair,
          form = KevForm.PAIR,
          windowsUsed = emptyList(),
          resident = listOf(pair),
          compiled = listOf(KevGraphKey.Pair(KevPairShape(128, 64))),
          availableBytesBeforeCompile = listOf(7_000_000_000L),
          closed = listOf(KevGraphKey.Window(128), KevGraphKey.Window(256)),
          secondRefused = false,
          plan =
            KevDemoPlan(
              KevGraphMode.AUTO,
              KevForm.PAIR,
              KevPrediction(1321.0, 1201.0),
              KevPlanInputs(
                5_000_000_000L,
                listOf(KevGraphKey.Window(128), KevGraphKey.Window(256)),
              ),
            ),
          state = KevStateResult(2, 128, 266.4),
          engineLoadMs = 21_000,
          warmupMs = 500,
          title = "Kev Decide",
          footerLines = listOf("line 1", "line 2", "line 3"),
          delayMs = 1500,
          gapMs = 800,
          tokenizeMs = 9,
          requestTotalMs = 480,
          questions =
            listOf(KevDemoQuestion(result, answer, view.shownCompact(), "187 ms", 3, 187, 1, 190)),
          airplaneMode = false,
          cgroup = "6:cpuset:/top-app",
          cgroupEnd = "6:cpuset:/top-app",
          layout = null,
        )
      )
    val parsed = KevJson.parse(KevJson.write(run)) as Map<*, *>
    assertEquals(KevDemoRun.KEYS, parsed.keys.toList())
    val graph = parsed["graph"] as Map<*, *>
    assertEquals(GRAPH_KEYS, graph.keys.toList())
    assertEquals("kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite", graph["file"])
    assertNull(graph["L"])
    assertEquals("pair", graph["form"])
    assertEquals(
      mapOf("Ls" to 128, "Lq" to 64),
      (graph["pair"] as Map<*, *>).mapValues { (it.value as JsonNumber).toInt() },
    )
    val compiled = (graph["compiled"] as List<*>).single() as Map<*, *>
    assertEquals(listOf("Ls", "Lq", "avail_mem_bytes"), compiled.keys.toList())
    assertEquals(
      listOf(128, 256),
      (graph["closed"] as List<*>).map { ((it as Map<*, *>)["L"] as JsonNumber).toInt() },
    )
    val resident = (graph["resident"] as List<*>).single() as Map<*, *>
    assertEquals(listOf("file", "Ls", "Lq", "bytes"), resident.keys.toList())
    val plan = parsed["plan"] as Map<*, *>
    assertEquals("pair", plan["form"])
    assertEquals(listOf("L128", "L256"), plan["resident"])
    assertEquals("5000000000", (plan["avail_mem_bytes"] as JsonNumber).literal)
    assertEquals(
      listOf(1321.0, 1201.0),
      (plan["predicted_ms"] as Map<*, *>).values.map { (it as JsonNumber).toDouble() },
    )
    val state = parsed["state"] as Map<*, *>
    assertEquals(listOf("tokens", "window", "ms"), state.keys.toList())
    assertEquals(266, (state["ms"] as JsonNumber).toInt())
    val question = (parsed["questions"] as List<*>).single() as Map<*, *>
    assertEquals(KevDemoRun.QUESTION_KEYS, question.keys.toList())
    assertEquals("pair", question["form"])
    assertEquals(64, (question["window"] as JsonNumber).toInt())
    assertEquals(5, (question["branch_len"] as JsonNumber).toInt())
    // The row coordinates the recording checks: the whole row, decide = its last token.
    assertArrayEquals(row.ids, OracleFixtures.ints(question["row_ids"]))
    assertEquals(7, (question["row_len"] as JsonNumber).toInt())
    assertEquals(6, (question["decide_idx"] as JsonNumber).toInt())
    assertEquals(listOf(5), OracleFixtures.ints(question["opt_idx"]).toList())
    assertEquals(187, (question["infer_ms"] as JsonNumber).toInt())
    assertEquals("187 ms", question["shown_ms"])
  }

  private companion object {
    val GRAPH_KEYS =
      listOf(
        "file",
        "L",
        "bytes",
        "form",
        "precision",
        "pair",
        "windows",
        "resident",
        "compiled",
        "closed",
        "second_refused",
      )
  }
}
