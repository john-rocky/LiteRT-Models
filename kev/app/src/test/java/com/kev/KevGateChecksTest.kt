package com.kev

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Android-free parts of the on-device gate and timing runs: the bundled assets they read, the
 * oracle's near-tie gap recomputed in float32, the median definition, and the conversion run's
 * timing rows.
 */
class KevGateChecksTest {
  @Test
  fun bundledAssetsParse() {
    val items =
      KevGateChecks.parseAsset(
        ExternalTestData.moduleFile("app/src/debug/assets/gate_fixtures.json").readBytes()
      )
    assertEquals(156, items.size)
    assertEquals(181, items.sumOf { it.questions.size })
    assertTrue(items.any { it.id == KevTimingCore.REQUEST_PATH_RECORD && it.questions.size == 5 })
    val asset =
      ExternalTestData.moduleFile("app/src/debug/assets/tokenizer_probes.json").readBytes()
    // The device gate's probes are the JVM test's probes, byte for byte.
    assertArrayEquals(ExternalTestData.resource("tokenizer_probes.json").readBytes(), asset)
    assertEquals(54, KevGateChecks.parseProbes(asset).size)
  }

  @Test
  fun deviceTokenizerProbesPassOnTheJvm() {
    val tokenizer = ExternalTestData.tokenizer()
    val encoder = KevEncoder(tokenizer)
    val probes =
      KevGateChecks.parseProbes(
        ExternalTestData.moduleFile("app/src/debug/assets/tokenizer_probes.json").readBytes()
      )
    for (probe in probes) {
      assertArrayEquals(probe.id, probe.rawIds, tokenizer.encode(probe.text))
      assertArrayEquals(probe.id, probe.userIds, encoder.userTokens(probe.text))
    }
  }

  @Test
  fun nearTieGapIsTheOraclesFloat32Gap() {
    val oracle =
      KevJson.parse(ExternalTestData.file(ExternalTestData.ORACLE).readBytes()) as Map<*, *>
    var nearTies = 0
    for (entry in oracle["questions"] as List<*>) {
      val json = entry as Map<*, *>
      val question =
        KevGateQuestion(
          json["qid"] as String,
          json["type"] as String,
          (json["keys"] as List<*>).map { it as String },
          OracleFixtures.ints(json["row_ids"]),
          (json["decide_idx"] as JsonNumber).toInt(),
          OracleFixtures.ints(json["opt_idx"]),
          OracleFixtures.doubles(json["probs"]),
          json["answer"] as Map<*, *>,
        )
      assertEquals(
        "${json["id"]}/${question.qid}",
        (json["top2_gap"] as JsonNumber).toDouble(),
        question.top2Gap,
        0.0,
      )
      assertEquals(json["near_tie"], question.nearTie)
      if (question.nearTie) nearTies++
    }
    assertEquals(15, nearTies)
  }

  @Test
  fun jsonIntegersStayInsideInt() {
    assertEquals(Int.MAX_VALUE, JsonNumber("2147483647").toInt())
    assertEquals(Int.MIN_VALUE, JsonNumber("-2147483648").toInt())
    assertEquals(0, JsonNumber("-0").toInt())
    assertEquals(248076, JsonNumber("248076").toInt())
    assertTrue(
      runCatching { JsonNumber("2147483648").toInt() }.exceptionOrNull() is IllegalArgumentException
    )
    assertTrue(
      runCatching { JsonNumber("-2147483649").toInt() }.exceptionOrNull()
        is IllegalArgumentException
    )
    assertTrue(
      runCatching { JsonNumber("1.0").toInt() }.exceptionOrNull() is IllegalArgumentException
    )
  }

  @Test
  fun medianIsNumpys() {
    assertEquals(2.0, KevStats.of(listOf(3.0, 1.0, 2.0))!!.median, 0.0)
    assertEquals(2.5, KevStats.of(listOf(4.0, 1.0, 3.0, 2.0))!!.median, 0.0)
    val stats = KevStats.of(listOf(5.0, 1.0))!!
    assertEquals(listOf(3.0, 1.0, 5.0, 2), listOf(stats.median, stats.min, stats.max, stats.count))
    assertEquals(null, KevStats.of(emptyList()))
  }

  @Test
  fun conversionTimingRowsParse() {
    val rows = KevTimingRows.parse(ExternalTestData.file("device/timing_rows.json").readBytes())
    assertEquals(listOf("fiveq", "T300", "T1000"), rows.sets.map { it.name })
    assertEquals(listOf(512, 512, 1024), rows.sets.map { it.window })
    assertEquals(listOf("request", "single", "single"), rows.sets.map { it.kind })
    assertEquals(listOf(132, 142, 132, 128, 128), rows.sets[0].rows.map { it.ids.size })
    assertEquals(listOf(300), rows.sets[1].rows.map { it.ids.size })
    assertEquals(listOf(false, true, true), rows.sets.map { it.synthetic })
    // The five-question set is the bundled own_fiveq_09 request's rows.
    val items =
      KevGateChecks.parseAsset(
        ExternalTestData.moduleFile("app/src/debug/assets/gate_fixtures.json").readBytes()
      )
    val fiveq = items.first { it.id == KevTimingCore.REQUEST_PATH_RECORD }
    for ((row, question) in rows.sets[0].rows.zip(fiveq.questions)) assertArrayEquals(
      question.rowIds,
      row.ids,
    )
  }
}
