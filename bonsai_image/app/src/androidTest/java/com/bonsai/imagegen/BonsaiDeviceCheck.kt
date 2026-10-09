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

// On-device check of this sample, used the way a person uses it: the screen is
// driven through its views (type a prompt, press Generate, press Cancel, tap a
// size) and the outcome is read from what the app shows and writes (the pill,
// the line under the prompt box, the PNG in files/outputs, the run record in
// files/Documents). One logcat line per step under the tag bonsai-check,
//   RESULT step=<name> ok=<bool> key=value ...
// then RESULT ok=<every step ok> with the summary.
//
// Needs the three 256x256 graphs in the app's files folder (README). The
// 512x512 pair is optional; while it is absent, the 512 button is checked to
// say which files it needs. The first generation runs before anything else
// heats the phone, so its total is a cold-phone number.
//
//   adb shell am instrument -w -e class com.bonsai.imagegen.BonsaiDeviceCheck \
//     com.bonsai.imagegen.test/androidx.test.runner.AndroidJUnitRunner
//   adb logcat -d -s bonsai-check | grep RESULT

package com.bonsai.imagegen

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.TextView
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class BonsaiDeviceCheck {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val modelsDir: File get() = context.getExternalFilesDir(null)!!
    private val outputsDir: File get() = File(modelsDir, "outputs")

    /** (step, ok) in run order. */
    private val results = ArrayList<Pair<String, Boolean>>()
    private val failurePathsOk = ArrayList<String>()
    private var generateTotalS: Double? = null

    private lateinit var meta: JSONObject
    private lateinit var tokenizer: QwenTokenizer
    private var activity: Activity? = null

    @Test
    fun check() {
        step("assets") { checkAssets() }
        step("models") { checkModels() }
        step("missing-model", failurePath = true) { checkMissingModel() }
        val launched = step("launch") { launch() }
        val uiSteps = listOf<Triple<String, Boolean, () -> String>>(
            Triple("empty-prompt", true) { checkEmptyPrompt() },
            Triple("long-prompt", true) { checkLongPrompt() },
            Triple("size-unavailable", true) { checkSizeUnavailable() },
            Triple("generate", false) { checkGenerate() },
            Triple("cancel", true) { checkCancel() },
            Triple("regenerate", true) { checkRegenerate() },
        )
        for ((name, failurePath, body) in uiSteps) {
            if (launched) step(name, failurePath, body)
            else report(name, false, "skipped=launch_failed")
        }
        activity?.let { a -> instrumentation.runOnMainSync { a.finish() } }

        val ok = results.all { it.second }
        val stepOk = results.toMap()
        val linePass = listOf("assets", "models", "generate", "cancel", "regenerate").all { stepOk[it] == true } &&
            failurePathsOk.size >= 3
        val failed = results.filter { !it.second }.joinToString(",") { it.first }.ifEmpty { "none" }
        val summary = "ok=$ok steps=${results.size} failed=$failed failure_paths_ok=${failurePathsOk.size} " +
            "line_pass=$linePass generate_total_s=${generateTotalS?.let { f2(it) } ?: "none"}"
        Log.i(TAG, "RESULT $summary")
        assertTrue(summary, ok)
    }

    // ---- steps -------------------------------------------------------------------------------------------------

    /** The APK carries the tokenizer tables and the pipeline meta (prep_assets.sh). */
    private fun checkAssets(): String {
        val vocab = assetBytes("vocab.json")
        val merges = assetBytes("merges.txt")
        val metaBytes = assetBytes("pipeline_meta.json")
        meta = JSONObject(String(metaBytes, Charsets.UTF_8))
        tokenizer = context.assets.open("vocab.json").use { v ->
            context.assets.open("merges.txt").use { m -> QwenTokenizer(v, m) }
        }
        val sizes = BonsaiPipeline.sizes(meta)
        val tokens = tokenizer.encodePrompt(PROMPT).promptTokenCount
        check(256 in sizes) { "pipeline_meta.json has no 256x256 variant: $sizes" }
        check(tokens > 0) { "the prompt encoded to no tokens" }
        return "vocab_bytes=${vocab.size} merges_bytes=${merges.size} meta_bytes=${metaBytes.size} " +
            "sizes=${sizes.joinToString(",")} prompt_body_tokens=$tokens max_body_tokens=${PromptRules.MAX_BODY_TOKENS}"
    }

    /** The three 256x256 graphs are in the app's files folder, each at its published size. */
    private fun checkModels(): String {
        val names = graphs(256)
        val fields = names.map { name ->
            val bytes = File(modelsDir, name).takeIf { it.exists() }?.length() ?: -1L
            Triple(name, bytes, PUBLISHED_BYTES[name])
        }
        val size512 = graphs(512).all { BonsaiPipeline.resolveModel(it, modelsDir) != null }
        val text = fields.joinToString(" ") { (name, bytes, _) -> "$name=$bytes" } +
            " dir=${modelsDir.path} size512_present=$size512"
        val bad = fields.filter { (_, bytes, want) -> bytes < 0 || (want != null && bytes != want) }
        check(bad.isEmpty()) {
            "$text; missing or not the published size: " +
                bad.joinToString(",") { (name, bytes, want) -> "$name($bytes!=$want)" }
        }
        return text
    }

    /** A run pointed at a folder without the graphs fails with the list of files, before loading anything; so
     *  does a 512x512 run while the 512 pair is absent. */
    private fun checkMissingModel(): String {
        val empty = File(context.cacheDir, "bonsai-check-empty").apply { deleteRecursively(); mkdirs() }
        val t0 = SystemClock.elapsedRealtime()
        val err = runCatching { BonsaiPipeline(empty, meta).generate(tokenizer, PROMPT, SEED, STEPS, 256) }
            .exceptionOrNull()
        val ms = SystemClock.elapsedRealtime() - t0
        check(err is BonsaiPipeline.MissingModels) { "expected MissingModels, got $err" }
        check(err.files == graphs(256)) { "named ${err.files}, expected ${graphs(256)}" }
        val text = MainActivity.missingRunText(empty, err.files)
        check(err.files.all { it in text }) { "the screen text does not name the files: $text" }
        var size512 = "skipped_512_present"
        val absent512 = graphs(512).filter { BonsaiPipeline.resolveModel(it, modelsDir) == null }
        if (absent512.isNotEmpty()) {
            val err512 = runCatching { BonsaiPipeline(modelsDir, meta).generate(tokenizer, PROMPT, SEED, STEPS, 512) }
                .exceptionOrNull()
            check(err512 is BonsaiPipeline.MissingModels && err512.files == absent512) {
                "512x512 with $absent512 absent: expected MissingModels($absent512), got $err512"
            }
            size512 = err512.files.joinToString(",")
        }
        return "empty_dir_named=${err.files.joinToString(",")} returned_ms=$ms size512_named=$size512"
    }

    /** Starts the screen with a fixed seed and waits until the line under the prompt box counts tokens. */
    private fun launch(): String {
        val intent = Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            .putExtra("seed", SEED)
            .putExtra("steps", STEPS)
        val t0 = SystemClock.elapsedRealtime()
        activity = instrumentation.startActivitySync(intent)
        val counted = waitFor(LAUNCH_TIMEOUT_MS) { ui { status().matches(COUNTER) } }
        val ms = SystemClock.elapsedRealtime() - t0
        val state = ui { "pill=${kv(pill())} status=${kv(status())}" }
        check(counted) { "no token count after $ms ms: $state" }
        return "$state ready_ms=$ms"
    }

    /** Generate with only whitespace in the box says so on the line under it and starts nothing. */
    private fun checkEmptyPrompt(): String {
        val after = ui {
            prompt().setText("   ")
            generate().performClick()
            Triple(status(), pill(), label())
        }
        check(after.first == PromptRules.EMPTY_TEXT) { "status '${after.first}'" }
        check(after.second == "READY" && after.third == "Generate") { "a run started: ${after.second}" }
        return "status=${kv(after.first)} pill=${after.second}"
    }

    /** A prompt past the text encoder's window is counted in red as it is typed and refused at Generate. */
    private fun checkLongPrompt(): String {
        val long = List(LONG_PROMPT_WORDS) { "bonsai" }.joinToString(" ")
        val c = PromptRules.count(long, tokenizer)
        check(!c.fits) { "the long prompt fits: ${c.tokens}/${c.max}" }
        val typed = ui {
            prompt().setText(long)
            status()
        }
        val after = ui {
            generate().performClick()
            Triple(status(), pill(), label())
        }
        check(typed == PromptRules.tooLongText(c)) { "while typing: '$typed'" }
        check(after.first == PromptRules.refusal(long, tokenizer)) { "after Generate: '${after.first}'" }
        check(after.second == "READY" && after.third == "Generate") { "a run started: ${after.second}" }
        return "tokens=${c.tokens} max=${c.max} status=${kv(after.first)} pill=${after.second}"
    }

    /** With the 512x512 pair absent, its button names the missing files and 256x256 stays selected. */
    private fun checkSizeUnavailable(): String {
        val absent = graphs(512).filter { BonsaiPipeline.resolveModel(it, modelsDir) == null }
        if (absent.isEmpty()) return "skipped=512_present"
        val after = ui {
            sizeButton(512).performClick()
            Triple(status(), sizeButton(256).isSelected, sizeButton(512).isSelected)
        }
        check(absent.all { it in after.first }) { "status '${after.first}' does not name $absent" }
        check(after.second && !after.third) { "selection moved: 256=${after.second} 512=${after.third}" }
        return "status=${kv(after.first)} selected=256"
    }

    /** The fixed prompt at 256x256: a PNG in files/outputs, the stage times from the run record. */
    private fun checkGenerate(): String {
        val run = runToImage("generate")
        generateTotalS = run.totalS
        check(run.totalS <= GENERATE_LIMIT_S) { "${run.fields} over the ${GENERATE_LIMIT_S.toInt()} s line" }
        return run.fields + " limit_s=${GENERATE_LIMIT_S.toInt()}"
    }

    /** Cancel during the first DiT step: no crash, no PNG, READY again with the Cancelled text. */
    private fun checkCancel(): String {
        val before = pngNames()
        ui { prompt().setText(PROMPT) }
        startRunOrFail()
        val atStep = waitFor(STEP_TIMEOUT_MS) { ui { pill().startsWith("STEP 1/") } }
        check(atStep) { "never reached STEP 1: pill=${ui { pill() }}" }
        val t0 = SystemClock.elapsedRealtime()
        val pressedAt = ui {
            val before = label()
            generate().performClick()
            "${pill()}|$before"
        }
        val readyAgain = waitFor(CANCEL_TIMEOUT_MS) { ui { pill() == "READY" && label() == "Generate" } }
        val ms = SystemClock.elapsedRealtime() - t0
        val message = ui { image().message }
        check(readyAgain) { "not READY ${CANCEL_TIMEOUT_MS / 1000} s after Cancel: pill=${ui { pill() }}" }
        check(message == MainActivity.CANCELLED_TEXT) { "image area says '$message'" }
        check(pngNames() == before) { "a cancelled run wrote a PNG: ${pngNames() - before}" }
        return "pressed_at=${kv(pressedAt)} to_ready_ms=$ms pill=READY message=${kv(message ?: "")} png_added=false"
    }

    /** Generate right after the Cancel runs to an image again, the same pixels as the first run (same seed). */
    private fun checkRegenerate(): String {
        val run = runToImage("regenerate")
        return run.fields + " same_as_generate=${run.rgbSha256 == firstRgbSha256}"
    }

    // ---- one run to an image -----------------------------------------------------------------------------------

    private var firstRgbSha256: String? = null

    private class Run(val totalS: Double, val rgbSha256: String, val fields: String)

    /** Presses Generate with the fixed prompt and seed at 256x256 and checks the image the run leaves behind. */
    private fun runToImage(name: String): Run {
        val before = pngNames()
        ui {
            prompt().setText(PROMPT)
            sizeButton(256).performClick()
        }
        startRunOrFail()
        val pills = LinkedHashSet<String>()
        val done = waitFor(RUN_TIMEOUT_MS) {
            val now = ui { pill() }
            pills.add(now)
            now == "DONE" || now == "ERROR"
        }
        val end = ui { pill() }
        check(done && end == "DONE") {
            "pill $end after ${pills.joinToString(">")}; image area: ${ui { image().message }}"
        }
        var png: File? = null
        check(waitFor(RECORD_TIMEOUT_MS) { (pngNames() - before).isNotEmpty().also { if (it) png = newestPng(before) } }) {
            "no new PNG in ${outputsDir.path}"
        }
        val row = waitForRecord(png!!) ?: error("no run record names ${png!!.name}")
        val pixels = pixelStats(png!!)
        check(pixels.width == 256 && pixels.height == 256) { "PNG is ${pixels.width}x${pixels.height}" }
        check(pixels.std > MIN_PIXEL_STD) { "flat image: std ${f2(pixels.std)}" }
        if (name == "generate") firstRgbSha256 = pixels.rgbSha256
        val ms = row.getJSONObject("ms")
        val steps = ms.getJSONArray("steps").let { a -> (0 until a.length()).map { s(a.getDouble(it)) } }
        val totalS = ms.getDouble("total_wall") / 1000.0
        val fields = "total_s=${f2(totalS)} textenc_s=${s(ms.getDouble("textenc_run"))} " +
            "steps_s=${steps.joinToString(",")} vae_s=${s(ms.getDouble("vae_run"))} " +
            "model_load_s=${s(ms.getDouble("model_load"))} textenc_load_s=${s(ms.getDouble("textenc_load"))} " +
            "dit_load_s=${s(ms.getDouble("dit_load"))} vae_load_s=${s(ms.getDouble("vae_load"))} " +
            "png=${png!!.name} w=${pixels.width} h=${pixels.height} std=${f2(pixels.std)} mean=${f2(pixels.mean)} " +
            "rgb_sha256=${pixels.rgbSha256.take(16)} same_as_s26_reference=${pixels.rgbSha256 == REFERENCE_RGB_SHA256} " +
            "thermal_before=${thermal(row, "thermal_before")} thermal_after=${thermal(row, "thermal_after")} " +
            "vmhwm_kib=${row.opt("vmhwm_kib")} pills=${pills.joinToString(">") { it.replace(' ', '_') }}"
        return Run(totalS, pixels.rgbSha256, fields)
    }

    /** Presses Generate and fails at once, with the line under the prompt box, when no run starts. */
    private fun startRunOrFail() {
        val started = ui {
            generate().performClick()
            pill() != "READY" && label() == "Cancel"
        }
        check(started) { "Generate did not start a run: pill=${ui { pill() }} status=${ui { status() }}" }
    }

    /** The run record row of [png], from the newest bonsai-demo-*.json (written after the PNG). */
    private fun waitForRecord(png: File): JSONObject? {
        val docs = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: return null
        var found: JSONObject? = null
        waitFor(RECORD_TIMEOUT_MS) {
            val newest = docs.listFiles { f -> f.name.startsWith("bonsai-demo-") && f.name.endsWith(".json") }
                ?.maxByOrNull { it.lastModified() } ?: return@waitFor false
            val runs = runCatching { JSONObject(newest.readText()).getJSONArray("runs") }.getOrNull()
                ?: return@waitFor false
            found = (0 until runs.length()).map { runs.getJSONObject(it) }.lastOrNull { it.optString("png") == png.path }
            found != null
        }
        return found
    }

    private class Pixels(val width: Int, val height: Int, val mean: Double, val std: Double, val rgbSha256: String)

    /** Mean and standard deviation over the R, G and B values of every pixel, and the SHA-256 of those bytes. */
    private fun pixelStats(png: File): Pixels {
        val bmp = BitmapFactory.decodeFile(png.path) ?: error("${png.name} does not decode")
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val rgb = ByteArray(px.size * 3)
        var sum = 0.0
        var sq = 0.0
        for ((i, p) in px.withIndex()) {
            for ((c, v) in intArrayOf((p shr 16) and 0xFF, (p shr 8) and 0xFF, p and 0xFF).withIndex()) {
                rgb[i * 3 + c] = v.toByte()
                sum += v
                sq += v.toDouble() * v
            }
        }
        val n = rgb.size.toDouble()
        val mean = sum / n
        val sha = MessageDigest.getInstance("SHA-256").digest(rgb).joinToString("") { "%02x".format(it) }
        return Pixels(bmp.width, bmp.height, mean, sqrt(maxOf(0.0, sq / n - mean * mean)), sha)
    }

    // ---- screen access (main thread) ---------------------------------------------------------------------------

    private fun <T> ui(block: () -> T): T {
        var out: Result<T>? = null
        instrumentation.runOnMainSync { out = runCatching(block) }
        return out!!.getOrThrow()
    }

    private fun <V : View> view(tag: String): V =
        activity!!.window.decorView.findViewWithTag<V>(tag) ?: error("no view tagged $tag")

    private fun prompt(): EditText = view(MainActivity.VIEW_PROMPT)
    private fun generate(): TextView = view(MainActivity.VIEW_GENERATE)
    private fun label(): String = generate().text.toString()
    private fun pill(): String = view<TextView>(MainActivity.VIEW_PILL).text.toString()
    private fun status(): String = view<TextView>(MainActivity.VIEW_STATUS).text.toString()
    private fun image(): ImageArea = view(MainActivity.VIEW_IMAGE)
    private fun sizeButton(size: Int): TextView = view(MainActivity.viewSize(size))

    // ---- helpers -----------------------------------------------------------------------------------------------

    /** Runs one step, logs its RESULT line and records the outcome; an exception is a failed step. */
    private fun step(name: String, failurePath: Boolean = false, body: () -> String): Boolean {
        val outcome = runCatching(body)
        val ok = outcome.isSuccess
        val fields = outcome.getOrElse { e -> "error=${kv(e.message ?: e.toString())}" }
        report(name, ok, fields)
        if (ok && failurePath && !fields.startsWith("skipped")) failurePathsOk.add(name)
        return ok
    }

    private fun report(name: String, ok: Boolean, fields: String) {
        results.add(name to ok)
        Log.i(TAG, "RESULT step=$name ok=$ok $fields")
    }

    private fun waitFor(timeoutMs: Long, condition: () -> Boolean): Boolean {
        val end = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < end) {
            if (condition()) return true
            Thread.sleep(POLL_MS)
        }
        return condition()
    }

    private fun graphs(size: Int): List<String> =
        listOf(meta.getJSONObject("files").getString("textenc")) + BonsaiPipeline.graphFiles(meta, size).toList()

    private fun assetBytes(name: String): ByteArray = context.assets.open(name).use { it.readBytes() }

    private fun pngNames(): Set<String> =
        outputsDir.listFiles { f -> f.name.endsWith(".png") }?.map { it.name }?.toSet() ?: emptySet()

    private fun newestPng(before: Set<String>): File? =
        outputsDir.listFiles { f -> f.name.endsWith(".png") && f.name !in before }?.maxByOrNull { it.lastModified() }

    private fun thermal(row: JSONObject, key: String): String =
        row.optJSONObject(key)?.opt("status")?.toString() ?: "unknown"

    companion object {
        private const val TAG = "bonsai-check"

        /** The prompt and seed of the app's first Galaxy S26 runs (4 steps, 256x256). */
        private const val PROMPT = "a red fox sitting in fresh snow, soft morning light, shallow depth of field"
        private const val SEED = 7L
        private const val STEPS = 4

        /** SHA-256 of the RGB bytes of the image those inputs gave on a Galaxy S26 (2026-09-28, LiteRT 2.1.3,
         *  6 XNNPACK threads). Reported, not required: other CPUs' kernels may differ in the last bits. */
        private const val REFERENCE_RGB_SHA256 = "d7b860e2d681512bc238f15e2ee9990467dec4da823d800307ecc147fc0126cc"

        /** The graphs as published on litert-community/Bonsai-Image-ternary-4B (revision 8878f895). */
        private val PUBLISHED_BYTES = mapOf(
            "textenc_int4.tflite" to 1_798_100_240L,
            "dit_256_int4b32.tflite" to 2_267_356_304L,
            "vae_dec_256_fp32.tflite" to 198_831_180L,
        )

        /** The pre-registered line for the first, cold 256x256 run, from the press of Generate to the image. */
        private const val GENERATE_LIMIT_S = 60.0
        private const val MIN_PIXEL_STD = 10.0
        private const val LONG_PROMPT_WORDS = 300
        private val COUNTER = Regex("""\d+ / \d+ tokens""")

        private const val POLL_MS = 100L
        private const val LAUNCH_TIMEOUT_MS = 30_000L
        private const val STEP_TIMEOUT_MS = 120_000L
        private const val CANCEL_TIMEOUT_MS = 60_000L
        private const val RUN_TIMEOUT_MS = 300_000L
        private const val RECORD_TIMEOUT_MS = 30_000L

        private fun f2(v: Double) = String.format(Locale.US, "%.2f", v)
        private fun s(ms: Double) = String.format(Locale.US, "%.2f", ms / 1000.0)

        /** A value without spaces, so every RESULT field stays one key=value word. */
        private fun kv(text: String) = text.trim().replace(Regex("\\s+"), "_")
    }
}
