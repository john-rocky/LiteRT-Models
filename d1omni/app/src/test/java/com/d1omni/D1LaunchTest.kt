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
    val defaults = parse() as D1Launch.Normal
    assertEquals(D1Backend.GPU, defaults.backend)
    assertEquals(D1Precision.FP32, defaults.precision)
    assertEquals(D1Precision.FP16_FP32_ACCUM, defaults.audioPrecision)
    assertEquals(D1Precision.FP16_FP32_ACCUM, defaults.precisions.vision)
    // `precision` sets every kind of graph; `precision_audio` / `precision_vision` one kind.
    val all = parse(true, "precision" to "fp32") as D1Launch.Normal
    assertEquals(listOf(D1Precision.FP32, D1Precision.FP32, D1Precision.FP32), listOf(all.precisions.decide, all.precisions.audio, all.precisions.vision))
    assertEquals("fp32", all.precisions.requested["precision"])
    val mixed = parse(true, "precision" to "fp16acc", "precision_vision" to "fp32") as D1Launch.Normal
    assertEquals(
      listOf(D1Precision.FP16_FP32_ACCUM, D1Precision.FP16_FP32_ACCUM, D1Precision.FP32),
      listOf(mixed.precisions.decide, mixed.precisions.audio, mixed.precisions.vision),
    )
    assertEquals(D1Backend.CPU, (parse(true, "backend" to "cpu") as D1Launch.Normal).backend)
    assertTrue(parse(true, "precision_vision" to "fp8") is D1Launch.Invalid)
  }

  @Test
  fun autoplayLaunch() {
    assertEquals(
      D1Launch.Autoplay("inbox_demo.json", 1000, 1500),
      parse(true, "autoplay" to true, "fixture" to "inbox_demo.json"),
    )
    val long = parse(true, "autoplay" to true, "fixture" to "inbox_demo.json", "delay_ms" to 4000, "gap_ms" to 6000, "precision" to "fp16acc", "backend" to "gpu")
    assertTrue(long is D1Launch.Autoplay)
    long as D1Launch.Autoplay
    assertEquals(4000L, long.delayMs)
    assertEquals(6000L, long.gapMs)
    assertEquals(D1Precision.FP16_FP32_ACCUM, long.precisions.decide)
    assertEquals(
      D1Launch.Autoplay("/data/user/0/com.d1omni/files/inbox_demo.json", 0, 0),
      parse(true, "autoplay" to true, "fixture" to "/data/user/0/com.d1omni/files/inbox_demo.json", "delay_ms" to 0, "gap_ms" to 0),
    )
    // a release build runs the autoplay too (it is the demo, not a debug run)
    assertTrue(parse(false, "autoplay" to true, "fixture" to "inbox_demo.json") is D1Launch.Autoplay)
    val invalid =
      listOf(
        parse(true, "autoplay" to true),
        parse(true, "autoplay" to true, "fixture" to "../x.json"),
        parse(true, "autoplay" to true, "fixture" to "/data/user/0/com.d1omni/files/../x.json"),
        parse(true, "autoplay" to true, "fixture" to "inbox_demo.json", "delay_ms" to -1),
        parse(true, "autoplay" to true, "fixture" to "inbox_demo.json", "precision" to "fp16"),
        parse(true, "autoplay" to true, "gate" to true, "fixture" to "inbox_demo.json", "report" to "g.json"),
      )
    for (launch in invalid) assertTrue("$launch", launch is D1Launch.Invalid)
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
