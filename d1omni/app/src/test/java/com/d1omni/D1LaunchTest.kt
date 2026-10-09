package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The launch extras: normal, gate and timing launches, and the ones that cannot be followed. */
class D1LaunchTest {
  private class Extras(private val values: Map<String, Any>) : D1Extras {
    override fun has(name: String) = values.containsKey(name)

    override fun string(name: String) = values[name] as String?

    override fun int(name: String, default: Int) = values[name] as Int? ?: default

    override fun boolean(name: String, default: Boolean) = values[name] as Boolean? ?: default
  }

  private fun parse(debug: Boolean = true, vararg pairs: Pair<String, Any>) =
    D1Launch.parse(Extras(pairs.toMap()), debug)

  @Test
  fun normalLaunch() {
    assertEquals(D1Launch.Normal(D1Backend.GPU, D1Precision.FP32), parse())
    assertEquals(
      D1Launch.Normal(D1Backend.GPU, D1Precision.FP32),
      parse(true, "precision" to "fp32"),
    )
    assertEquals(D1Launch.Normal(D1Backend.CPU, D1Precision.FP32), parse(true, "backend" to "cpu"))
  }

  @Test
  fun gateLaunch() {
    assertEquals(
      D1Launch.Gate("rows_L256.json", "G.json", 0, D1Backend.GPU, D1Precision.FP32, 128),
      parse(true, "gate" to true, "fixture" to "rows_L256.json", "report" to "G.json", "resident" to 128),
    )
    assertEquals(
      D1Launch.Gate("rows_L128.json", "F.json", 60, D1Backend.GPU, D1Precision.FP32, null),
      parse(true, "gate" to true, "fixture" to "rows_L128.json", "report" to "F.json", "limit" to 60, "precision" to "fp32"),
    )
  }

  @Test
  fun timingLaunch() {
    assertEquals(
      D1Launch.Timing("timing_rows.json", "T.json", 5, 20, 120000L, listOf("card3", "one"), D1Backend.GPU, D1Precision.FP16_FP32_ACCUM),
      parse(
        true,
        "timing" to true,
        "rows" to "timing_rows.json",
        "report" to "T.json",
        "cool_ms" to 120000,
        "sets" to "card3, one",
        "precision" to "fp16acc",
      ),
    )
  }

  @Test
  fun launchesThatCannotBeFollowed() {
    val invalid =
      listOf(
        parse(false, "gate" to true, "fixture" to "r.json", "report" to "g.json"),
        parse(true, "gate" to true, "report" to "g.json"),
        parse(true, "gate" to true, "fixture" to "../r.json", "report" to "g.json"),
        parse(true, "gate" to true, "fixture" to "r.json", "report" to "g.json.partial"),
        parse(true, "gate" to true, "fixture" to "r.json", "report" to "g.json", "resident" to 100),
        parse(true, "timing" to true, "report" to "t.json"),
        parse(true, "timing" to true, "rows" to "r.json", "report" to "t.json", "reps" to 0),
        parse(true, "gate" to true, "timing" to true, "fixture" to "r.json", "rows" to "r.json", "report" to "t.json"),
        parse(true, "precision" to "fp16"),
        parse(true, "backend" to "npu"),
      )
    for (launch in invalid) assertTrue("$launch", launch is D1Launch.Invalid)
  }
}
