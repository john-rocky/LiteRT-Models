package com.kev

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The bundled debug gate asset (`app/src/debug/assets/gate_fixtures.json`: SemIf 144 + 12 invented
 * records with the oracle per question): every request encodes to the asset's rows and readout
 * indices, its answers follow from its probabilities, the declared counts hold, and every question
 * equals the external oracle.
 */
class GateFixturesTest {
  private val asset =
    KevJson.parse(
      ExternalTestData.moduleFile("app/src/debug/assets/gate_fixtures.json").readBytes()
    ) as Map<*, *>
  private val items = (asset["items"] as List<*>).map { it as Map<*, *> }

  @Test
  fun declaredCountsHold() {
    val questions = items.sumOf { (it["questions"] as List<*>).size }
    assertEquals((asset["records"] as JsonNumber).toInt(), items.size)
    assertEquals((asset["questions"] as JsonNumber).toInt(), questions)
    assertEquals(156, items.size)
    assertEquals(181, questions)
    assertEquals(listOf("semif", "own"), items.map { it["source"] as String }.distinct())
    assertEquals(144, items.count { it["source"] == "semif" })
    assertEquals(12, items.count { it["source"] == "own" })
    assertEquals(items.size, items.map { it["id"] }.toSet().size)
    assertEquals(emptyList<Any?>(), asset["excluded"])
  }

  @Test
  fun answersFollowFromTheAssetProbabilities() {
    for (item in items) {
      val (_, meta) = KevRecords.toRecord(KevRequest.fromJson(item["request"]))
      val questions =
        (item["questions"] as List<*>).map {
          OracleFixtures.question(it as Map<*, *> + ("id" to item["id"]))
        }
      val answers = KevAnswers.toAnswers(questions.map { it.probabilities }, meta)
      for (question in questions) {
        assertNull(
          "${item["id"]}/${question.qid}",
          OracleFixtures.jsonDifference(question.answer, answers[question.qid]),
        )
      }
    }
  }

  @Test
  fun requestsEncodeToTheAssetRows() {
    val encoder = KevEncoder(ExternalTestData.tokenizer())
    val windows = linkedMapOf(512 to 0, 1024 to 0, 2048 to 0)
    var rows = 0
    for (item in items) {
      val (record, meta) = KevRecords.toRecord(KevRequest.fromJson(item["request"]))
      val encoded = encoder.encode(record)
      assertEquals(
        item["id"].toString(),
        (item["input_tokens"] as JsonNumber).toInt(),
        encoded.inputTokens,
      )
      val questions = item["questions"] as List<*>
      assertEquals(questions.size, meta.size)
      for ((index, question) in questions.withIndex()) {
        val expected = OracleFixtures.question(question as Map<*, *> + ("id" to item["id"]))
        assertEquals(expected.qid, meta[index].id)
        val row = encoded.row(index)
        assertArrayEquals("${item["id"]}/${expected.qid}", expected.rowIds, row.ids)
        assertEquals(expected.decideIndex, row.decideIndex)
        assertArrayEquals(expected.optionIndices, row.optionIndices)
        val window = row.requireWindow()
        windows[window] = windows.getValue(window) + 1
        rows++
      }
    }
    val declared =
      (asset["windows"] as Map<*, *>).entries.associate { (k, v) ->
        (k as String).toInt() to (v as JsonNumber).toInt()
      }
    assertEquals(declared, windows)
    ExternalTestData.writeReport(
      "gate_fixtures.json",
      linkedMapOf(
        "test" to "GateFixturesTest",
        "records" to items.size,
        "rows_identical" to rows,
        "windows" to windows.mapKeys { it.key.toString() },
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println("KEV_GATE_ASSET records=${items.size} rows_identical=$rows windows=$windows")
  }

  @Test
  fun assetEqualsTheOracle() {
    val oracle = OracleFixtures.loadOracle().questions.associateBy { it.key }
    var compared = 0
    for (item in items) {
      for (question in item["questions"] as List<*>) {
        val fromAsset = OracleFixtures.question(question as Map<*, *> + ("id" to item["id"]))
        val expected = requireNotNull(oracle[fromAsset.key]) { fromAsset.key }
        assertArrayEquals(expected.rowIds, fromAsset.rowIds)
        assertEquals(expected.decideIndex, fromAsset.decideIndex)
        assertArrayEquals(expected.optionIndices, fromAsset.optionIndices)
        assertArrayEquals(expected.probabilities, fromAsset.probabilities, 0.0)
        assertNull(OracleFixtures.jsonDifference(expected.answer, fromAsset.answer))
        compared++
      }
    }
    assertEquals(181, compared)
  }
}
