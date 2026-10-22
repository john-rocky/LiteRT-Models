package com.timestral

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The Kotlin host against `scripts/host_forecast.py` on the vectors `scripts/make_test_vectors.py`
 * wrote: synthetic series through both hosts, the graph replaced by a float32 formula both sides
 * compute the same way. Each vector names its graph form (L, H, R): two-row graphs take one call,
 * one-row graphs one call for the context and one for its negation (the Python reference's
 * `mirror_in_one_call=False` path).
 *
 * Bars (registered before the first run): the graph inputs of every call are bit-equal (sha256 of
 * their float32 bytes; `dec`, `cos` and `sin` only with `-Ptimestral.host=<dir>`, the tables
 * installed with the model), the anchor statistics are bit-equal, and every final quantile is
 * within a relative difference of 1e-6 of the Python value (or the same error message).
 */
class HostVectorsTest {
  @Test
  fun everyVectorMatchesThePythonHost() {
    val names = String(TestData.resource("vectors/index.txt")).lines().filter { it.isNotBlank() }
    assertTrue("too few vectors: ${names.size}", names.size >= 19)
    val tables = TestData.hostTables()
    val problems = ArrayList<String>()
    val report = StringBuilder("vector\tgraph\trows\th\tinputs\tquantiles\n")
    for (name in names) {
      val outcome = runCase(name, tables, problems)
      report.append(name).append('\t').append(outcome).append('\n')
    }
    println(report)
    System.getProperty("timestral.out")?.let { directory ->
      File(directory).mkdirs()
      File(directory, "host_vectors.tsv").writeText(report.toString())
    }
    println("tables ${if (tables == null) "absent: dec / cos / sin not compared" else "present"}")
    if (problems.isNotEmpty()) fail(problems.joinToString("\n"))
  }

  private fun runCase(name: String, tables: HostTables?, problems: MutableList<String>): String {
    val base = "vectors/$name"
    val props = TestData.properties("$base/case.properties")
    val series = TestData.floats(TestData.resource("$base/series.f32"))
    val horizon = props.getProperty("horizon").toInt()
    val mask = props.getProperty("mask").toDouble().toFloat()
    val rows = props.getProperty("form_rows").toInt()
    val bucket =
      Bucket(
        props.getProperty("bucket"),
        props.getProperty("form_l").toInt(),
        props.getProperty("form_h").toInt(),
      )
    val prepared = TimestralHost.prepare(series, horizon)
    assertEquals(name, props.getProperty("context_patches").toInt(), prepared.contextPatches)
    assertEquals(name, props.getProperty("horizon_patches").toInt(), prepared.horizonPatches)
    if (rows == 2 && bucket.horizonPatches == 32) {
      // The first vectors were written on the smallest two-row graph that holds them.
      val chosen =
        GraphSpec.choose(TestData.TWO_ROW_H32, Backend.CPU, prepared.contextPatches, prepared.horizonPatches)
      assertEquals(name, bucket.name, chosen?.name)
    }
    val last = prepared.context.size - 1
    val loc = TestData.hexFloats(props.getProperty("anchor_loc"))
    val scale = TestData.hexFloats(props.getProperty("anchor_scale"))
    for (row in 0 until TimestralHost.ROWS) {
      if (prepared.loc[row][last].toRawBits() != loc[row].toRawBits() ||
        prepared.scale[row][last].toRawBits() != scale[row].toRawBits()) {
        problems.add(
          "$name row $row anchor: Kotlin ${prepared.loc[row][last]} / ${prepared.scale[row][last]}, " +
            "Python ${loc[row]} / ${scale[row]}"
        )
      }
    }
    val tablesOrPlaceholder = tables ?: TestData.placeholderTables
    val calls =
      if (rows == 2) listOf(intArrayOf(0, 1)) else listOf(intArrayOf(0), intArrayOf(1))
    val inputs = calls.map { TimestralHost.buildInputs(prepared, bucket, tablesOrPlaceholder, mask, it) }
    assertEquals(name, props.getProperty("first_observed_patch").toInt(), inputs[0].firstObservedPatch)
    val compared = if (tables == null) listOf("patches", "bias") else inputs[0].byName().keys.toList()
    var equal = 0
    var total = 0
    for ((index, call) in inputs.withIndex()) {
      for (key in compared) {
        val wanted =
          if (key == "patches" && index == 1) props.getProperty("sha256_patches_row1")
          else props.getProperty("sha256_$key")
        if (wanted.isNullOrEmpty()) continue // the negation's call never ran in Python (an error first)
        total++
        val actual = TestData.sha256(call.byName().getValue(key))
        if (actual == wanted) {
          equal++
        } else {
          problems.add("$name: call $index input $key differs from Python (sha256 $actual)")
        }
      }
    }
    val raw = formula(props, bucket)
    val error = props.getProperty("error").orEmpty()
    val quantiles: String
    if (error.isNotEmpty()) {
      val message =
        try {
          TimestralHost.finish(prepared, raw, bucket)
          "no error"
        } catch (failure: ForecastException) {
          failure.message
        }
      if (message != error) problems.add("$name: error \"$message\", Python \"$error\"")
      quantiles = "error: $message"
    } else {
      val actual = TimestralHost.finish(prepared, raw, bucket)
      val expected = TestData.floats(TestData.resource("$base/quantiles.f32"))
      val difference = TestData.difference(actual, expected)
      if (!(difference.maxRelative <= MAX_RELATIVE)) problems.add("$name: quantiles $difference")
      quantiles = difference.toString()
    }
    return "${bucket.name}\t$rows\t${prepared.horizonPatches}\t$equal/$total bit-equal\t$quantiles"
  }

  /**
   * The Python runner's formula for both rows (`[2, H, 32, 21]`): base + slope t + z (width +
   * growth t), float32 step by step; a one-row graph's two calls give the same two rows.
   */
  private fun formula(props: java.util.Properties, bucket: Bucket): FloatArray {
    val base = TestData.hexFloats(props.getProperty("raw_base"))
    val slope = TestData.hexFloats(props.getProperty("raw_slope"))
    val width = TestData.hexFloat(props.getProperty("raw_width"))
    val growth = TestData.hexFloat(props.getProperty("raw_growth"))
    val z = TestData.hexFloats(props.getProperty("raw_z"))
    val n = TimestralHost.QUANTILES
    val steps = bucket.horizonPatches * TimestralHost.PATCH
    val out = FloatArray(TimestralHost.ROWS * steps * n)
    for (row in 0 until TimestralHost.ROWS) {
      for (t in 0 until steps) {
        val time = t.toFloat()
        for (i in 0 until n) {
          out[(row * steps + t) * n + i] = base[row] + slope[row] * time + z[i] * (width + growth * time)
        }
      }
    }
    if (props.getProperty("raw_nan") == "true") out[TimestralHost.MEDIAN] = Float.NaN
    if (props.getProperty("raw_inf") == "true") {
      out[n + n - 1] = Float.POSITIVE_INFINITY
      out[(steps + 1) * n] = Float.NEGATIVE_INFINITY
    }
    return out
  }

  private companion object {
    const val MAX_RELATIVE = 1e-6
  }
}
