package com.wevisone

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The host steps around the graphs against the references of the conversion run: Pillow 12.2.0's bicubic resize
 * (byte for byte), torch's box Fourier features, torch's sigmoid + bilinear alpha, and the fp16 table read.
 */
class HostMathTest {
    @Test
    fun pillowBicubicByteForByte() {
        val index = Golden.file("resize_index.json")
        assumeTrue("needs -Pwevisone.golden", index != null)
        val images = JSONObject(index!!.readText()).getJSONObject("images")
        var equal = 0
        var total = 0
        for (stem in images.keys()) {
            val rec = images.getJSONObject(stem)
            val w = rec.getInt("width")
            val h = rec.getInt("height")
            val src = File(Golden.dir, "img_$stem.rgb").readBytes()
            for (size in intArrayOf(512, 1024)) {
                val want = File(Golden.dir, "img_${stem}_$size.rgb").readBytes()
                val got = PilBicubic.resize(src, w, h, size, size)
                total++
                if (got.contentEquals(want)) {
                    equal++
                } else {
                    val diff = got.indices.count { got[it] != want[it] }
                    println("RESIZE DIFF $stem $size: $diff bytes differ")
                }
            }
        }
        println("RESIZE $equal/$total resizes byte-equal to Pillow 12.2.0 (${images.length()} images x 512, 1024)")
        assertEquals(total, equal)
    }

    private fun table(): BoxFeatures =
        File(Golden.moduleRoot, "app/src/main/assets/box_log_table.bin").inputStream().use { BoxFeatures.read(it) }

    @Test
    fun boxFeaturesAgainstTorch() {
        assumeTrue("needs -Pwevisone.golden", Golden.file("box_index.json") != null)
        val box = table()
        val widths = Golden.floats(Golden.file("box_widths.f32")!!)
        val ew = Golden.floats(Golden.file("box_ew.f32")!!)
        var exact = 0
        for (i in widths.indices) if (java.lang.Float.floatToRawIntBits(box.encodeSize(widths[i])) == java.lang.Float.floatToRawIntBits(ew[i])) exact++
        println("BOX encoded sizes bit-equal to torch: $exact/${widths.size}")
        assertEquals(widths.size, exact)

        fun maxUlps(values: FloatArray, want: FloatArray): Long {
            var worst = 0L
            val pe = FloatArray(BoxFeatures.PE)
            for (i in values.indices) {
                BoxFeatures.fourier(values[i], pe, 0)
                for (k in 0 until BoxFeatures.PE) worst = maxOf(worst, Golden.ulps(pe[k], want[i * BoxFeatures.PE + k]))
            }
            return worst
        }
        val bins = Golden.floats(Golden.file("box_bins.f32")!!)
        val uBins = maxUlps(bins, Golden.floats(Golden.file("box_f_bins.f32")!!))
        val uEw = maxUlps(ew, Golden.floats(Golden.file("box_f_ew.f32")!!))
        println("BOX Fourier features vs torch: max $uBins ulp over ${bins.size} bins, $uEw ulp over ${ew.size} sizes")
        assertTrue(uBins <= 1 && uEw <= 1)

        val rows = JSONObject(Golden.file("box_index.json")!!.readText()).getJSONObject("rows")
        for (row in rows.keys()) {
            val bins4 = rows.getJSONObject(row).getJSONArray("boxes_bins").getJSONArray(0).let { a -> IntArray(4) { a.getInt(it) } }
            val got = box.features(BoxFeatures.normalized(bins4))
            val want = Golden.floats(File(Golden.dir, "box_row_$row.f32"))
            val worst = got.indices.maxOf { Golden.ulps(got[it], want[it]) }
            println("BOX row $row ${bins4.toList()}: max $worst ulp over 256 features")
            assertTrue(worst <= 1)
        }
    }

    @Test
    fun alphaAgainstTorch() {
        val index = Golden.file("alpha_index.json")
        assumeTrue("needs -Pwevisone.golden", index != null)
        val rows = JSONObject(index!!.readText()).getJSONObject("rows")
        for (row in rows.keys()) {
            val r = rows.getJSONObject(row)
            val w = r.getInt("width")
            val h = r.getInt("height")
            val logits = Golden.floats(File(Golden.dir, "alpha_${row}_logits.f32"))
            val want = Golden.floats(File(Golden.dir, "alpha_${row}_ref.f32"))
            val got = AlphaMath.alpha(logits, 1024, h, w)
            var maxAbs = 0.0
            var sum = 0.0
            for (i in got.indices) {
                val d = kotlin.math.abs(got[i].toDouble() - want[i])
                maxAbs = maxOf(maxAbs, d)
                sum += d
            }
            println("ALPHA $row ${w}x$h: max |d| %.3g, mean |d| %.3g vs torch".format(maxAbs, sum / got.size))
            assertTrue(maxAbs < 1e-4)
        }
    }

    @Test
    fun halfToFloatAllValues() {
        // every finite fp16 value against its definition (-1)^s * 2^(e - 15) * (1 + m / 1024), subnormals m * 2^-24
        for (bits in 0 until 0x10000) {
            val s = if (bits and 0x8000 != 0) -1.0 else 1.0
            val e = (bits ushr 10) and 0x1F
            val m = bits and 0x3FF
            if (e == 0x1F) continue
            val want = if (e == 0) s * m * Math.pow(2.0, -24.0) else s * Math.pow(2.0, (e - 15).toDouble()) * (1 + m / 1024.0)
            assertEquals("bits $bits", want.toFloat(), Half.toFloat(bits), 0f)
        }
        assertTrue(Half.toFloat(0x7C00) == Float.POSITIVE_INFINITY && Half.toFloat(0x7E00).isNaN())
    }

    @Test
    fun promptsMatchFixtureInstructions() {
        // build_prompt for the app's tasks on the fixture queries (the public rows' instructions)
        assertEquals(
            "Locate all the instances that match the following description: red rectangle. blue circle. green circle.",
            Prompts.instruction(Task.DETECT, "red rectangle, blue circle, green circle"),
        )
        assertEquals(
            "Locate the instances that match the following description and provide the alpha mattes: the green circle.",
            Prompts.instruction(Task.CUT_OUT, "the green circle"),
        )
        assertEquals("point to: airplane. windsock", Prompts.instruction(Task.POINT, "airplane, windsock"))
        assertEquals("How many red rectangles in the image?", Prompts.instruction(Task.COUNT_POINTS, "red rectangles"))
        assertEquals(
            "How many blue circles in the image? Please count them with boxes.",
            Prompts.instruction(Task.COUNT_BOXES, "blue circles"),
        )
        assertEquals("detect all the word in the box format", Prompts.instruction(Task.READ_TEXT, ""))
        assertEquals("Locate the following text in the image: EXIT 12", Prompts.instruction(Task.FIND_TEXT, "EXIT 12"))
        assertEquals(
            "Locate the instances that match the following description: the yellow rectangle.",
            Prompts.instruction(Task.FIND, "the yellow rectangle"),
        )
    }
}
