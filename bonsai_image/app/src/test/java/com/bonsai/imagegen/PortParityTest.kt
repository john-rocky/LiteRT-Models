// Copyright 2026 Daisuke Majima. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// JVM parity tests for the Kotlin ports, against the same artifacts that
// validated the Swift ports: the 26-case Python-tokenizer golden set
// (src/test/resources/tok_golden.json, run against the tokenizer tables that
// prep_assets.sh copies into src/main/assets), the per-size graph selection
// against that pipeline_meta.json, and the Swift noise stream. The recorded
// Mac pipeline fixtures (device_fixtures, fixtures256: written from
// generate.py and the app's noise stream in numpy) live outside this
// repository; tests that need an absent artifact are skipped, not failed, so
// the app still builds from a bare checkout.

package com.bonsai.imagegen

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files

class PortParityTest {
    private val home = System.getProperty("user.home")
    private val golden = javaClass.classLoader?.getResource("tok_golden.json")
    // the module directory is the working directory of unit tests
    private val tokDir = File("src/main/assets")
    private val fixDir = File("$home/models/bonsai-image-4b-tflite/device_fixtures")
    private val fix256 = File("$home/models/bonsai-image-256/fixtures256")
    private val meta256 = File("src/main/assets/pipeline_meta.json")

    private fun floats(name: String, dir: File = fixDir): FloatArray {
        val bytes = File(dir, name).readBytes()
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    private fun maxErr(a: FloatArray, b: FloatArray): Float {
        assertEquals("length", b.size, a.size)
        var m = 0f
        for (i in a.indices) m = maxOf(m, Math.abs(a[i] - b[i]))
        return m
    }

    @Test
    fun tokenizerMatchesPythonGolden() {
        assumeTrue(golden != null && File(tokDir, "vocab.json").exists())
        val g = JSONObject(golden!!.readText())
        val tok = QwenTokenizer(
            File(tokDir, "vocab.json").inputStream(),
            File(tokDir, "merges.txt").inputStream()
        )
        val cases = g.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val want = c.getJSONArray("body_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val got = tok.encode("user\n" + c.getString("prompt"))
            assertEquals("case $i: ${c.getString("prompt").take(40)}",
                want.toList(), got.toList())
        }

        // full padded encode on case 0
        val c0 = cases.getJSONObject(0)
        val enc = tok.encodePrompt(c0.getString("prompt"))
        val body = c0.getJSONArray("body_ids")
        val suffix = g.getJSONArray("suffix")
        val real = 1 + body.length() + suffix.length()
        assertEquals(256, enc.ids.size)
        assertEquals(g.getInt("im_start"), enc.ids[0])
        assertEquals(g.getInt("pad"), enc.ids[real])
        assertEquals(1, enc.mask[real - 1])
        assertEquals(0, enc.mask[real])
    }

    @Test
    fun sigmasMatchManifest() {
        assumeTrue(fixDir.exists())
        val manifest = JSONObject(File(fixDir, "manifest.json").readText())
        val want = manifest.getJSONArray("sigmas")
        val got = BonsaiMath.sigmas(manifest.getInt("steps"))
        assertEquals(want.length(), got.size)
        for (i in got.indices) {
            assertTrue("sigma[$i]", Math.abs(got[i] - want.getDouble(i)) < 2e-6)
        }
    }

    @Test
    fun positionIdsMatchFixtures() {
        assumeTrue(fixDir.exists())
        assertEquals(floats("img_ids_f32.bin").toList(), BonsaiMath.imgIds().toList())
        assertEquals(floats("txt_ids_f32.bin").toList(), BonsaiMath.txtIds().toList())
    }

    @Test
    fun eulerAndUnpatchifyMatchFixtures() {
        assumeTrue(fixDir.exists())
        val manifest = JSONObject(File(fixDir, "manifest.json").readText())
        val steps = manifest.getInt("steps")
        val sig = manifest.getJSONArray("sigmas")
        val lat = floats("lat0_f32.bin")
        for (k in 0 until steps) {
            val v = floats("dit_out_${k}_f32.bin")
            val ds = (sig.getDouble(k + 1) - sig.getDouble(k)).toFloat()
            for (i in lat.indices) lat[i] += ds * v[i]
        }
        val a = manifest.getJSONArray("affine_a")
        val b = manifest.getJSONArray("affine_b")
        val z = BonsaiMath.unpatchify(
            lat,
            FloatArray(a.length()) { a.getDouble(it).toFloat() },
            FloatArray(b.length()) { b.getDouble(it).toFloat() }
        )
        val zRef = floats("z_vae_f32.bin")
        assertEquals(zRef.size, z.size)
        var maxErr = 0f
        for (i in z.indices) maxErr = maxOf(maxErr, Math.abs(z[i] - zRef[i]))
        assertTrue("unpatchify max err $maxErr", maxErr < 1e-4f)
    }

    @Test
    fun noiseMatchesSwiftStream() {
        // The iOS app's SplitMix64+Box-Muller stream, seed 42: first 8 values
        // and the last, printed by the Mac Swift harness at %.9e —
        // cross-platform image reproducibility depends on this exact stream.
        val swiftFirst8 = floatArrayOf(
            4.147197604e-01f, 6.526812315e-01f, -8.918862343e-01f, 1.326833606e+00f,
            1.729593039e+00f, -1.883416772e+00f, 5.456204414e-01f, -1.656835794e+00f
        )
        val swiftLast = 2.426872700e-01f
        val n = BonsaiMath.noise(42)
        assertEquals(1024 * 128, n.size)
        for (i in swiftFirst8.indices) {
            assertEquals("noise[$i]", swiftFirst8[i], n[i], 1e-7f)
        }
        assertEquals("noise[last]", swiftLast, n[n.size - 1], 1e-7f)
        var mean = 0.0
        var sq = 0.0
        for (x in n) { mean += x; sq += x.toDouble() * x }
        mean /= n.size
        assertTrue("mean $mean", Math.abs(mean) < 0.02)
        assertTrue("var", Math.abs(sq / n.size - mean * mean - 1.0) < 0.03)
        assertEquals(n.toList(), BonsaiMath.noise(42).toList())
        assertTrue(n.toList() != BonsaiMath.noise(43).toList())
    }

    // ---- per-size host math vs generate.py (fixtures256) -------------------

    @Test
    fun sigmasMatchGenerateBothSizes() {
        assumeTrue(fix256.exists())
        val s256 = BonsaiMath.sigmas(4, 256)
        val s1024 = BonsaiMath.sigmas(4, 1024)
        val e256 = maxErr(s256, floats("sigmas_4_256_f32.bin", fix256))
        val e1024 = maxErr(s1024, floats("sigmas_4_1024_f32.bin", fix256))
        println("sigmas max err: 256 tokens $e256, 1024 tokens $e1024")
        assertTrue("sigmas(4, 256) max err $e256", e256 < 1e-5f)
        assertTrue("sigmas(4, 1024) max err $e1024", e1024 < 1e-5f)
        // the 512x512 default is the 1024-token schedule
        assertArrayEquals(s1024, BonsaiMath.sigmas(4), 0f)
        // mu depends on the token count: the two schedules differ
        assertTrue(s256[1] != s1024[1])
    }

    @Test
    fun positionIdsMatchGenerateBothSizes() {
        assumeTrue(fix256.exists())
        assertArrayEquals(floats("img_ids_g16_f32.bin", fix256), BonsaiMath.imgIds(16), 0f)
        assertArrayEquals(floats("img_ids_g32_f32.bin", fix256), BonsaiMath.imgIds(32), 0f)
        assertArrayEquals(floats("img_ids_g32_f32.bin", fix256), BonsaiMath.imgIds(), 0f)
        assertArrayEquals(floats("txt_ids_f32.bin", fix256), BonsaiMath.txtIds(), 0f)
        // at 256x256 img_ids and txt_ids share the shape (256, 4) but not the values
        assertEquals(BonsaiMath.imgIds(16).size, BonsaiMath.txtIds().size)
        assertTrue(!BonsaiMath.imgIds(16).contentEquals(BonsaiMath.txtIds()))
    }

    @Test
    fun unpatchifyMatchesGenerateBothSizes() {
        assumeTrue(fix256.exists())
        val scale = floats("bn_scale_f32.bin", fix256)
        val shift = floats("bn_shift_f32.bin", fix256)
        val z16 = BonsaiMath.unpatchify(floats("lat_g16_f32.bin", fix256), scale, shift, 16)
        val z32 = BonsaiMath.unpatchify(floats("lat_g32_f32.bin", fix256), scale, shift, 32)
        assertEquals(32 * 32 * 32, z16.size)
        assertEquals(32 * 64 * 64, z32.size)
        val e16 = maxErr(z16, floats("z_g16_f32.bin", fix256))
        val e32 = maxErr(z32, floats("z_g32_f32.bin", fix256))
        println("unpatchify max err: grid 16 $e16, grid 32 $e32")
        assertTrue("unpatchify grid 16 max err $e16", e16 < 1e-5f)
        assertTrue("unpatchify grid 32 max err $e32", e32 < 1e-5f)
        // the 512x512 default is grid 32
        assertArrayEquals(z32, BonsaiMath.unpatchify(floats("lat_g32_f32.bin", fix256), scale, shift), 0f)
    }

    @Test
    fun noise256MatchesAppNoisePy() {
        assumeTrue(fix256.exists())
        val manifest = JSONObject(File(fix256, "manifest.json").readText())
        val want = manifest.getJSONObject("noise").getJSONArray("first8")
        val n = BonsaiMath.noise(7, 256)
        assertEquals(256 * 128, n.size)
        for (i in 0 until want.length()) {
            assertEquals("noise(7, 256)[$i]", want.getDouble(i).toFloat(), n[i], 1e-7f)
        }
        val e = maxErr(n, floats("noise_s7_t256_f32.bin", fix256))
        println("noise(7, 256) vs app_noise.py: max err $e over ${n.size} values")
        assertTrue("noise(7, 256) max err $e", e < 1e-6f)
        // 256x256 takes the head of the 512x512 stream
        assertArrayEquals(BonsaiMath.noise(7).copyOf(256 * 128), n, 0f)
    }

    // ---- which graphs each size needs (pipeline_meta.json with variants) ----

    @Test
    fun graphFilesPerSize() {
        assumeTrue(meta256.exists())
        val meta = JSONObject(meta256.readText())
        assertEquals(listOf(256, 512), BonsaiPipeline.sizes(meta))
        assertEquals("dit_int4b32.tflite" to "vae_dec_fp32.tflite", BonsaiPipeline.graphFiles(meta, 512))
        assertEquals("dit_256_int4b32.tflite" to "vae_dec_256_fp32.tflite", BonsaiPipeline.graphFiles(meta, 256))

        val dir = Files.createTempDirectory("bonsai-models").toFile()
        try {
            val all = listOf("textenc_int4.tflite", "dit_256_int4b32.tflite", "vae_dec_256_fp32.tflite",
                "dit_int4b32.tflite", "vae_dec_fp32.tflite")
            assertEquals(all.toSet(), BonsaiPipeline.missingFiles(dir, meta).toSet())
            assertEquals(5, BonsaiPipeline.missingFiles(dir, meta).size)
            assertEquals(emptyList<Int>(), BonsaiPipeline.readySizes(dir, meta))
            // the 256 graphs without the shared text encoder: nothing runs
            File(dir, "dit_256_int4b32.tflite").writeText("")
            File(dir, "vae_dec_256_fp32.tflite").writeText("")
            assertEquals(emptyList<Int>(), BonsaiPipeline.readySizes(dir, meta))
            File(dir, "textenc_int4.tflite").writeText("")
            assertEquals(listOf(256), BonsaiPipeline.readySizes(dir, meta))
            assertEquals(setOf("dit_int4b32.tflite", "vae_dec_fp32.tflite"),
                BonsaiPipeline.missingFiles(dir, meta).toSet())
            // the zero-scale-patched sibling counts as the published DiT
            File(dir, "dit_int4b32_fixed.tflite").writeText("")
            File(dir, "vae_dec_fp32.tflite").writeText("")
            assertEquals(listOf(256, 512), BonsaiPipeline.readySizes(dir, meta))
            assertEquals(emptyList<String>(), BonsaiPipeline.missingFiles(dir, meta))
        } finally {
            dir.deleteRecursively()
        }
    }
}
