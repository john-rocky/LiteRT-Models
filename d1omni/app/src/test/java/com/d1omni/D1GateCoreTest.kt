package com.d1omni

import kotlin.math.abs
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The device gate's work without a device: the phone's rows files (`device/r1/rows_L128.json`
 * 210 rows, `rows_L256.json` 43, `timing_rows.json`) parse, each row's request encodes again to
 * its ids and markers and its state serializes to its hash (what `ids_match` / `state_match`
 * report on the phone), the inputs are the row's layout, and a stand-in graph that returns the
 * desktop CPU's scores (`fixtures/scores_probe.json`) gives the Python host's probabilities.
 */
class D1GateCoreTest {
  @Test
  fun rowsFilesEncodeAgainOnThisTokenizer() {
    val core = D1GateCore(ExternalTestData.tokenizer(), ExternalTestData.contract())
    var rows = 0
    var matched = 0
    val counts = LinkedHashMap<String, Int>()
    for ((name, length) in listOf(ExternalTestData.ROWS_L128 to 128, ExternalTestData.ROWS_L256 to 256)) {
      val file = D1GateRows.parse(ExternalTestData.demoFile(name).readBytes())
      assertEquals(length, file.length)
      counts[name] = file.rows.size
      for (row in file.rows) {
        rows++
        val recheck = core.recheck(row)
        if (recheck.idsMatch == true && recheck.markersMatch == true && recheck.stateMatch == true) {
          matched++
        } else {
          println("D1_GATE recheck ${row.key}: ${recheck.idsMatch} ${recheck.markersMatch} ${recheck.stateMatch}")
        }
        val inputs = core.inputs(row, length)
        assertArrayEquals(row.ids, inputs.ids.copyOf(row.ids.size))
        assertTrue(inputs.ids.drop(row.ids.size).all { it == 0 })
        assertEquals(row.ids.size, inputs.pad.count { it == 1f })
        assertTrue(inputs.media.all { it == 0f } && inputs.keepRight.all { it == 1f })
        assertEquals(1f, inputs.qtypeOneHot[row.type.index])
        assertTrue(row.ids.size <= length && (length == 128 || row.ids.size > 128))
      }
    }
    println("D1_GATE recheck $matched/$rows $counts")
    assertEquals(mapOf(ExternalTestData.ROWS_L128 to 210, ExternalTestData.ROWS_L256 to 43), counts)
    assertEquals(253, matched)
    val sets = D1TimingSet.parse(ExternalTestData.demoFile(ExternalTestData.TIMING_ROWS).readBytes())
    assertEquals(listOf("card3", "one"), sets.map { it.name })
    assertEquals(listOf(3, 1), sets.map { it.rows.size })
    assertTrue(sets.all { it.length == 128 })
  }

  @Test
  fun aStandInGraphGivesThePythonProbabilities() {
    val core = D1GateCore(ExternalTestData.tokenizer(), ExternalTestData.contract())
    val scores =
      (ExternalTestData.json(ExternalTestData.demoFile(ExternalTestData.SCORES))["rows"] as List<*>)
        .map { it as Map<*, *> }
        .filter { it["kind"] == "text" }
        .associateBy { it["key"] as String }
    var compared = 0
    var worst = 0.0
    for ((name, length) in listOf(ExternalTestData.ROWS_L128 to 128, ExternalTestData.ROWS_L256 to 256)) {
      for (row in D1GateRows.parse(ExternalTestData.demoFile(name).readBytes()).rows) {
        val probe = scores[row.key] ?: continue
        assertEquals(length, (probe["L"] as JsonNumber).toInt())
        val output = ExternalTestData.doubles(probe["scores"]).map { it.toFloat() }.toFloatArray()
        val call = D1Call(output, 1_000_000, 2_000_000, 3_000_000)
        val record = core.record(row, call, 0L, core.recheck(row))
        val probs = (record["probs"] as List<*>).map { it as Double }
        val expected = ExternalTestData.doubles(probe["probs"])
        for (index in probs.indices) worst = maxOf(worst, abs(probs[index] - expected[index]))
        assertEquals(true, record["finite"])
        assertEquals(true, record["ids_match"])
        assertEquals(6.0, record["write_run_read_ms"] as Double, 1e-9)
        compared++
      }
    }
    println("D1_GATE stand-in rows=$compared max|dp|=$worst")
    assertEquals(14, compared)
    assertTrue("max |dp| $worst", worst <= 1e-12)
  }
}
