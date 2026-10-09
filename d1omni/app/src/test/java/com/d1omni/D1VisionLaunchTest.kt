package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The picture runs' launch extras ([D1Launch.Vision]); every other launch goes to [D1Launch.parse]. */
class D1VisionLaunchTest {
  private class Extras(private val values: Map<String, Any>) : D1Extras {
    override fun has(name: String) = values.containsKey(name)

    override fun string(name: String) = values[name] as String?

    override fun int(name: String, default: Int) = values[name] as Int? ?: default

    override fun boolean(name: String, default: Boolean) = values[name] as Boolean? ?: default
  }

  private fun parse(debug: Boolean = true, vararg pairs: Pair<String, Any>) =
    D1Launch.Vision.parse(Extras(pairs.toMap()), debug)

  @Test
  fun pictureGate() {
    assertEquals(
      D1Launch.VGate("rows_image_small.json", "V.json", 0, D1Backend.GPU, D1Precision.FP32, D1Precision.FP16_FP32_ACCUM),
      parse(true, "vgate" to true, "fixture" to "rows_image_small.json", "report" to "V.json"),
    )
    assertEquals(
      D1Launch.VGate("rows_image.json", "W.json", 3, D1Backend.GPU, D1Precision.FP16_FP32_ACCUM, D1Precision.FP32),
      parse(
        true,
        "vgate" to true,
        "fixture" to "rows_image.json",
        "report" to "W.json",
        "limit" to 3,
        "precision" to "fp16acc",
        "precision_vision" to "fp32",
      ),
    )
  }

  @Test
  fun pictureTiming() {
    assertEquals(
      D1Launch.VTiming("timing_image.json", "T.json", 5, 20, 120000L, listOf("dogs2"), D1Backend.GPU, D1Precision.FP32,
        D1Precision.FP16_FP32_ACCUM),
      parse(true, "vtiming" to true, "rows" to "timing_image.json", "report" to "T.json", "cool_ms" to 120000,
        "sets" to "dogs2"),
    )
  }

  @Test
  fun otherLaunchesAreNotPictureRuns() {
    assertNull(parse(true))
    assertNull(parse(true, "gate" to true, "fixture" to "rows_L128.json", "report" to "G.json"))
    assertNull(parse(true, "precision" to "fp32"))
  }

  @Test
  fun launchesThatCannotBeFollowed() {
    val invalid =
      listOf(
        parse(false, "vgate" to true, "fixture" to "r.json", "report" to "v.json"),
        parse(true, "vgate" to true, "report" to "v.json"),
        parse(true, "vgate" to true, "fixture" to "../r.json", "report" to "v.json"),
        parse(true, "vgate" to true, "fixture" to "r.json", "report" to "v.json.partial"),
        parse(true, "vgate" to true, "fixture" to "r.json", "report" to "v.json", "precision_vision" to "fp16"),
        parse(true, "vgate" to true, "fixture" to "r.json", "report" to "v.json", "precision" to "int8"),
        parse(true, "vgate" to true, "fixture" to "r.json", "report" to "v.json", "backend" to "npu"),
        parse(true, "vgate" to true, "gate" to true, "fixture" to "r.json", "report" to "v.json"),
        parse(true, "vgate" to true, "vtiming" to true, "fixture" to "r.json", "rows" to "t.json", "report" to "v.json"),
        parse(true, "vtiming" to true, "report" to "t.json"),
        parse(true, "vtiming" to true, "rows" to "t.json", "report" to "t.json", "reps" to 0),
        parse(true, "vtiming" to true, "timing" to true, "rows" to "t.json", "report" to "t.json"),
      )
    for (launch in invalid) assertTrue("$launch", launch is D1Launch.Invalid)
  }
}
