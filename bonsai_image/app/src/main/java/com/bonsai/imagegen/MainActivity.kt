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

// Bonsai — one-screen on-device text-to-image app around BonsaiPipeline:
// prompt, output size (256 | 512), steps, Generate; a status pill and a
// stopwatch that runs from the press of Generate until the image is on screen;
// under the image, the measured seconds of every stage. Every number on
// screen is measured by this app (System.nanoTime), never a constant.
//
// Under the prompt box a line counts the prompt's tokens against what the text
// encoder reads (PromptRules) and says why Generate did not start: an empty or
// over-long prompt, or a size whose graphs are not on the phone. Cancel stops
// the run at the next stage boundary and the app is READY again.
// BonsaiDeviceCheck (androidTest) drives this screen through the view tags
// below.
//
// Headless runs (adb shell am start -n com.bonsai.imagegen/.MainActivity ...):
//   --ez autorun true
//   --es prompts "p1|p2"   (or --es prompt "p")    one prompt per run
//   --es seeds "7,21"      (or --el seed 7)        one seed per run; none = random
//   --es sizes "256,512"                          one output size per run
//     parallel lists, run k takes position k; a list of one value applies to
//     every run
//   --ei steps 4  --ei threads 6
//   --ei delay_ms 2000     wait after the screen is up
//   --ei gap_ms 3000       wait after an image is shown, before the next run
//   --ez type_prompt true  type the prompt into the box (~30 ms per character)
//                          and show the button press before the run starts;
//                          the stopwatch starts with the real run either way
// Each launch writes getExternalFilesDir(DOCUMENTS)/bonsai-demo-<epoch s>.json
// (device, app, screen layout, every run's stage times, thermal status, VmHWM,
// PNG path) and logs under the tag BonsaiDemo: stage=<name> ms=<ms> per
// stage, then AUTORUN_DONE json=<path> or AUTORUN_FAILED reason=<reason>.
// PNGs land in getExternalFilesDir()/outputs (adb-pullable; Share exports
// via FileProvider).

package com.bonsai.imagegen

import android.app.Activity
import android.app.ActivityManager
import android.content.Intent
import android.database.ContentObserver
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.provider.Settings
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.Log
import android.util.TypedValue
import android.view.Choreographer
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.ViewTreeObserver
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Space
import android.widget.TextView
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.TensorFlowLite
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.random.Random

class MainActivity : Activity() {

    /** One autorun entry; [seed] null = a new random seed. */
    private class RunSpec(val prompt: String, val seed: Long?, val size: Int)

    private class Autorun(val runs: List<RunSpec>, val delayMs: Long, val gapMs: Long, val typePrompt: Boolean) {
        var next = 0
    }

    /** What the pill shows; READY and MISSING follow the model files, the others stay until the next run. */
    private enum class PillState { READY, MISSING, RUNNING, DONE, ERROR }

    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    private var meta: JSONObject? = null
    private var metaSha256: String? = null
    private var pipeline: BonsaiPipeline? = null
    @Volatile private var tokenizer: QwenTokenizer? = null
    @Volatile private var tokenizerProblem: String? = null
    private var sizes: List<Int> = listOf(256, 512)
    private var ready: List<Int> = emptyList()
    private var missing: List<String> = emptyList()

    // settings (UI thread)
    private var size = 256
    private var steps = 4
    private var threads = BonsaiPipeline.THREADS
    private var pinnedSeed: Long? = null
    private var gpuDit = false

    // what footer line 2 shows: the running / last run, or the selection
    private var shownSize = 256
    private var shownSteps = 4
    private var shownSeed: Long? = null

    // run state (UI thread)
    private var running = false
    private var pillState = PillState.READY
    private var runT0 = 0L
    private var runSteps = 4
    private var loadSumMs = 0.0
    private var stopwatchOn = false
    private var lastTenths = -1L
    private var lastPng: File? = null
    private var autorun: Autorun? = null

    // run record: touched on the worker thread only
    private val startEpochS = System.currentTimeMillis() / 1000
    private val report = JSONObject()
    private val runRows = JSONArray()
    private lateinit var reportFile: File

    private lateinit var root: LinearLayout
    private lateinit var shareButton: TextView
    private lateinit var promptEdit: EditText
    private lateinit var promptStatus: TextView
    private lateinit var sizeViews: Map<Int, TextView>
    private lateinit var stepsChip: TextView
    private lateinit var generateButton: TextView
    private lateinit var pill: TextView
    private lateinit var stopwatch: TextView
    private lateinit var imageArea: ImageArea
    private lateinit var table: LinearLayout
    private val cells = ArrayList<Pair<TextView, TextView>>()
    private lateinit var footer1: TextView
    private lateinit var footer2: TextView

    private val airplaneObserver = object : ContentObserver(Handler(Looper.getMainLooper())) {
        override fun onChange(selfChange: Boolean) = refreshFooter()
    }

    private val modelsDir: File
        get() = getExternalFilesDir(null)!!

    private val deviceName: String
        get() = if (Build.MODEL.startsWith("SM-S942")) "Galaxy S26" else Build.MODEL

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Show over the lock screen and keep the display on: a locked phone parks a hidden activity's process in
        // the background cpuset (little cores), which runs the same graphs about 10x slower.
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        reportFile = File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: filesDir, "bonsai-demo-$startEpochS.json")
        try {
            val bytes = assets.open("pipeline_meta.json").use { it.readBytes() }
            val m = JSONObject(String(bytes, Charsets.UTF_8))
            pipeline = BonsaiPipeline(modelsDir, m)
            sizes = BonsaiPipeline.sizes(m)
            metaSha256 = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            meta = m
        } catch (e: Exception) {
            Log.e(TAG, "pipeline_meta.json unreadable", e)
        }
        setContentView(buildUi())
        // The status bar stays (its airplane icon is part of the picture), the navigation bar goes.
        WindowCompat.getInsetsController(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.navigationBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = false
        }
        contentResolver.registerContentObserver(
            Settings.Global.getUriFor(Settings.Global.AIRPLANE_MODE_ON), false, airplaneObserver
        )
        refreshModels()
        val auto = if (wantsAutorun(intent)) prepareAutorun(intent) else null
        if (auto == null && !wantsAutorun(intent)) {
            // a plain launch may still set the seed, steps and threads for the Generate button
            pinnedSeed = runCatching { longExtraOrNull(intent, "seed") }.getOrNull()
            runCatching { intExtraOrNull(intent, "steps") }.getOrNull()?.takeIf { it in 1..50 }?.let {
                steps = it
                shownSteps = it
                rebuildTable(it)
            }
            runCatching { intExtraOrNull(intent, "threads") }.getOrNull()?.takeIf { it in 1..16 }?.let { threads = it }
            gpuDit = intent.getBooleanExtra("gpu", false)
        }
        worker.execute { prepare() }   // after the extras: the record's app block states the final settings
        refreshControls()
        refreshFooter()
        root.post {
            refreshFooter()   // with the laid-out width
            root.post { recordLayout("launch") }
            if (auto != null) main.postDelayed({ autorunStep() }, auto.delayMs)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (!wantsAutorun(intent)) return
        if (running || autorun != null) {
            Log.w(TAG, "autorun ignored: a run is in progress")
            return
        }
        prepareAutorun(intent)?.let { a -> main.postDelayed({ autorunStep() }, a.delayMs) }
    }

    override fun onResume() {
        super.onResume()
        if (!running) refreshModels()
        refreshFooter()
    }

    override fun onDestroy() {
        pipeline?.cancelled = true
        stopStopwatch(null)
        main.removeCallbacksAndMessages(null)
        contentResolver.unregisterContentObserver(airplaneObserver)
        worker.shutdown()
        super.onDestroy()
    }

    // ---- startup ----------------------------------------------------------------------------------------------

    /** Worker: tokenizer tables and the native runtime, so the first run's numbers are the pipeline's only. */
    private fun prepare() {
        val t0 = System.nanoTime()
        val runtime = runCatching { TensorFlowLite.runtimeVersion() }
        val nativeMs = msSince(t0)
        val t1 = System.nanoTime()
        val tok = runCatching {
            assets.open("vocab.json").use { v -> assets.open("merges.txt").use { m -> QwenTokenizer(v, m) } }
        }
        val tokMs = msSince(t1)
        tokenizer = tok.getOrNull()
        tok.exceptionOrNull()?.let { tokenizerProblem = "Tokenizer tables missing from the APK (run prep_assets.sh)." }
        ui { refreshPromptStatus() }
        val ram = ActivityManager.MemoryInfo().also {
            getSystemService(ActivityManager::class.java).getMemoryInfo(it)
        }
        val files = JSONObject()
        meta?.let { m ->
            val names = listOf(m.getJSONObject("files").getString("textenc")) +
                sizes.flatMap { BonsaiPipeline.graphFiles(m, it).toList() }
            for (n in names.distinct()) {
                files.put(n, BonsaiPipeline.resolveModel(n, modelsDir)?.length() ?: JSONObject.NULL)
            }
        }
        report.put("started_epoch_s", startEpochS)
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("device", Build.DEVICE)
                .put("shown_as", deviceName)
                .put("soc_manufacturer", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else JSONObject.NULL)
                .put("soc", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else JSONObject.NULL)
                .put("android", Build.VERSION.RELEASE).put("sdk_int", Build.VERSION.SDK_INT)
                .put("ram_total_bytes", ram.totalMem)
                .put("cpu_count", Runtime.getRuntime().availableProcessors()))
            .put("app", JSONObject()
                .put("package", packageName).put("version_name", BuildConfig.VERSION_NAME)
                .put("version_code", BuildConfig.VERSION_CODE).put("build_type", BuildConfig.BUILD_TYPE)
                .put("threads", threads).put("threads_default", BonsaiPipeline.THREADS)
                .put("litert", BuildConfig.LITERT_VERSION)
                .put("tflite_runtime_version", runtime.getOrElse { "unavailable: $it" })
                .put("native_init_ms", nativeMs)
                .put("tokenizer_load_ms", tokMs)
                .put("airplane_mode", airplaneMode())
                .put("gpu_dit", gpuDit)
                .put("pipeline_meta_sha256", metaSha256 ?: JSONObject.NULL)
                .put("models_dir", modelsDir.path)
                .put("model_files_bytes", files))
            .put("runs", runRows)
        writeReport()
        Log.i(TAG, "PREPARED tokenizer_ms=${f1(tokMs)} native_init_ms=${f1(nativeMs)} " +
            "runtime=${runtime.getOrNull()} json=${reportFile.path}")
        tok.exceptionOrNull()?.let { e ->
            Log.e(TAG, "tokenizer tables unreadable", e)
            ui { showProblem("Tokenizer tables missing from the APK assets (run prep_assets.sh): $e") }
        }
    }

    /** Which sizes have all three graphs; the pill and the placeholder say what is missing. */
    private fun refreshModels() {
        val m = meta
        if (m == null) {
            ready = emptyList()
            showProblem("pipeline_meta.json missing from the APK assets (run prep_assets.sh)")
            return
        }
        missing = BonsaiPipeline.missingFiles(modelsDir, m)
        ready = BonsaiPipeline.readySizes(modelsDir, m)
        if (!running && size !in ready && ready.isNotEmpty()) {
            size = ready.first()
            shownSize = size
        }
        if (!running && pillState in setOf(PillState.READY, PillState.MISSING)) {
            if (ready.isEmpty()) setPill("MODELS MISSING", C_ERR, PillState.MISSING)
            else setPill("READY", C_IDLE, PillState.READY)
            if (imageArea.bitmap == null) imageArea.message = missingMessage()
        }
        refreshControls()
        Log.i(TAG, "MODELS ready=$ready missing=$missing dir=${modelsDir.path}")
    }

    private fun missingMessage(): String? {
        if (missing.isEmpty()) return null
        val off = sizes.filter { it !in ready }.joinToString(", ") { "$it×$it" }
        return "Not available: $off\n\nMissing in ${modelsDir.path}:\n" + missing.joinToString("\n") +
            "\n\nadb push them from\nhuggingface.co/litert-community/Bonsai-Image-ternary-4B"
    }

    /** The line under the prompt box for a size whose graphs are not all on the phone. */
    private fun sizeMissingText(s: Int): String {
        val m = meta ?: return "No pipeline_meta.json in the APK assets (run prep_assets.sh)."
        val need = listOf(m.getJSONObject("files").getString("textenc")) + BonsaiPipeline.graphFiles(m, s).toList()
        return "$s×$s needs ${need.filter { it in missing }.joinToString(" and ")}, not on the phone."
    }

    private fun showProblem(text: String) {
        setPill("ERROR", C_ERR, PillState.ERROR)
        imageArea.bitmap = null
        imageArea.segments = 0
        imageArea.message = text
    }

    // ---- one run ----------------------------------------------------------------------------------------------

    /** UI thread: starts a run if possible; the stopwatch starts here. Returns null when the run started, else
     *  why not — also shown, in red, on the line under the prompt box. */
    private fun startRun(promptText: String, seedArg: Long?, runSize: Int, auto: Boolean): String? {
        if (running) return "a run is in progress"
        val m = meta
        val pipe = pipeline
        if (m == null || pipe == null) return refuse("No pipeline_meta.json in the APK assets (run prep_assets.sh).")
        val tok = tokenizer ?: return refuse(tokenizerProblem ?: "Loading the tokenizer…")
        PromptRules.refusal(promptText, tok)?.let { return refuse(it) }
        val prompt = promptText.trim()
        if (runSize !in ready) {
            refreshModels()   // the graphs may have been pushed while the app was open
            if (runSize !in ready) return refuse(sizeMissingText(runSize))
        }
        hideKeyboard()
        refreshPromptStatus()
        pipe.cancelled = false
        val seed = seedArg ?: pinnedSeed ?: Random.nextLong(0, 1_000_000)
        val stepsNow = steps
        val threadsNow = threads
        running = true
        runT0 = System.nanoTime()
        val epoch0 = System.currentTimeMillis()
        size = runSize
        runSteps = stepsNow
        loadSumMs = 0.0
        shownSize = runSize
        shownSteps = stepsNow
        shownSeed = seed
        refreshControls()
        refreshFooter()
        rebuildTable(stepsNow)
        imageArea.bitmap = null
        imageArea.message = null
        imageArea.segments = stepsNow
        imageArea.filled = 0
        imageArea.active = -1
        shareButton.visibility = View.INVISIBLE
        setPill("TEXT ENCODER", C_RUN, PillState.RUNNING)
        startStopwatch()
        setGenerateRunning(true)
        Log.i(TAG, "RUN_START size=$runSize seed=$seed steps=$stepsNow threads=$threadsNow gpu_dit=$gpuDit " +
            "autorun=$auto prompt=$prompt")
        worker.execute { generateOnWorker(pipe, m, prompt, seed, runSize, stepsNow, threadsNow, epoch0, auto) }
        return null
    }

    /** Shows why Generate did not start on the line under the prompt box; the next edit restores the count. */
    private fun refuse(reason: String): String {
        setPromptStatus(reason, error = true)
        Log.i(TAG, "RUN_REFUSED reason=$reason")
        return reason
    }

    private fun generateOnWorker(
        pipe: BonsaiPipeline, m: JSONObject, prompt: String, seed: Long, runSize: Int, runSteps: Int,
        runThreads: Int, epoch0: Long, auto: Boolean,
    ) {
        val row = JSONObject().put("size", runSize).put("prompt", prompt).put("seed", seed).put("steps", runSteps)
            .put("threads", runThreads).put("gpu_dit", gpuDit).put("autorun", auto)
            .put("started_epoch_ms", epoch0).put("airplane_mode", airplaneMode())
            .put("thermal_before", thermal())
            .put("vmrss_kib_before", procStatusKiB("VmRSS") ?: JSONObject.NULL)
            .put("vmhwm_reset", resetPeakRss())
        try {
            val tok = tokenizer ?: assets.open("vocab.json").use { v ->
                assets.open("merges.txt").use { mg -> QwenTokenizer(v, mg) }
            }.also { tokenizer = it }
            row.put("prompt_tokens", tok.encode(prompt).size)
            row.put("files_expected", JSONArray(BonsaiPipeline.graphFiles(m, runSize).toList()))
            val result = pipe.generate(
                tokenizer = tok, prompt = prompt, seed = seed, steps = runSteps, size = runSize,
                threads = runThreads, gpuDit = gpuDit,
                status = { Log.i(TAG, it) },
                progress = { onProgress(it) },
            )
            val tb = System.nanoTime()
            val bmp = toBitmap(result.rgb, runSize)
            val bitmapMs = msSince(tb)
            Log.i(TAG, "stage=bitmap ms=${f1(bitmapMs)}")
            val tPost = System.nanoTime()
            ui {
                showImage(bmp) { tShown, via ->
                    onShown(row, result, bmp, bitmapMs, tPost, tShown, via, epoch0, seed)
                }
            }
        } catch (c: BonsaiPipeline.Cancelled) {
            recordFailure(row, "cancelled", epoch0)
            ui { onRunCancelled() }
        } catch (t: Throwable) {
            Log.e(TAG, "run failed", t)
            recordFailure(row, t.toString(), epoch0)
            val text = if (t is BonsaiPipeline.MissingModels) missingRunText(modelsDir, t.files)
            else (t.message ?: t.toString())
            ui { onRunEnded("ERROR", C_ERR, PillState.ERROR, text) }
        }
    }

    /** Worker: log the finished stage, then move the screen. */
    private fun onProgress(ev: BonsaiPipeline.Progress) {
        when (ev) {
            is BonsaiPipeline.Progress.TextEncoderDone -> {
                Log.i(TAG, "stage=textenc_load ms=${f1(ev.loadMs)}")
                Log.i(TAG, "stage=textenc ms=${f1(ev.runMs)}")
            }
            is BonsaiPipeline.Progress.DitLoaded -> Log.i(TAG, "stage=dit_load ms=${f1(ev.loadMs)}")
            is BonsaiPipeline.Progress.StepDone -> Log.i(TAG, "stage=step k=${ev.k} n=${ev.n} ms=${f1(ev.ms)}")
            is BonsaiPipeline.Progress.DecodeDone -> {
                Log.i(TAG, "stage=vae_load ms=${f1(ev.loadMs)}")
                Log.i(TAG, "stage=vae ms=${f1(ev.runMs)}")
            }
            else -> {}
        }
        ui { applyProgress(ev) }
    }

    /** UI thread: pill, segments and table follow the stages. */
    private fun applyProgress(ev: BonsaiPipeline.Progress) {
        if (!running) return
        val n = runSteps
        when (ev) {
            BonsaiPipeline.Progress.TextEncoder -> setPill("TEXT ENCODER", C_RUN, PillState.RUNNING)
            is BonsaiPipeline.Progress.TextEncoderDone -> {
                loadSumMs += ev.loadMs
                setEntry(0, "text encoder", ev.runMs)
                setPill("LOADING DiT", C_RUN, PillState.RUNNING)
            }
            is BonsaiPipeline.Progress.DitLoaded -> loadSumMs += ev.loadMs
            is BonsaiPipeline.Progress.Step -> {
                setPill("STEP ${ev.k}/${ev.n}", C_RUN, PillState.RUNNING)
                imageArea.active = ev.k - 1
            }
            is BonsaiPipeline.Progress.StepDone -> {
                setEntry(ev.k, "step ${ev.k}", ev.ms)
                imageArea.filled = ev.k
                imageArea.active = -1
            }
            BonsaiPipeline.Progress.Decode -> setPill("DECODING", C_RUN, PillState.RUNNING)
            is BonsaiPipeline.Progress.DecodeDone -> {
                loadSumMs += ev.loadMs
                setEntry(n + 1, "VAE", ev.runMs)
                setEntry(n + 2, "model load", loadSumMs)
            }
        }
    }

    /** UI thread: set the image, then report the first frame that draws it (2 s fallback if nothing draws). */
    private fun showImage(bmp: Bitmap, onShown: (Long, String) -> Unit) {
        imageArea.bitmap = bmp
        var done = false
        val listener = object : ViewTreeObserver.OnDrawListener {
            override fun onDraw() {
                if (done) return
                done = true
                val t = System.nanoTime()
                val self = this
                // views may not change inside a draw pass
                main.post {
                    imageArea.viewTreeObserver.removeOnDrawListener(self)
                    onShown(t, "draw")
                }
            }
        }
        imageArea.viewTreeObserver.addOnDrawListener(listener)
        main.postDelayed({
            if (!done) {
                done = true
                imageArea.viewTreeObserver.removeOnDrawListener(listener)
                onShown(System.nanoTime(), "timeout")
            }
        }, 2000)
    }

    /** UI thread: the image is on screen — the run's wall time ends here. */
    private fun onShown(
        row: JSONObject, result: BonsaiPipeline.Result, bmp: Bitmap, bitmapMs: Double, tPost: Long, tShown: Long,
        via: String, epoch0: Long, seed: Long,
    ) {
        val totalMs = (tShown - runT0) / 1e6
        val n = runSteps
        stopStopwatch(totalMs)
        setPill("DONE", C_DONE, PillState.DONE)   // the total is on the stopwatch and in the table
        setEntry(n + 3, "total", totalMs, bold = true)
        running = false
        setGenerateRunning(false)
        refreshControls()
        recordLayout("done")   // the screen of this take, as drawn
        Log.i(TAG, "stage=total ms=${f1(totalMs)} shown_by=$via")
        val shown = JSONArray()
        for ((name, value) in cells) if (name.text.isNotEmpty()) {
            shown.put(JSONArray().put(name.text.toString()).put(value.text.toString()))
        }
        val a = autorun
        val last = a != null && a.next >= a.runs.size
        worker.execute {
            finishRun(row, result, bmp, bitmapMs, (tShown - tPost) / 1e6, totalMs, via, epoch0, seed, shown, last)
        }
        if (a != null) {
            if (last) autorun = null else main.postDelayed({ autorunStep() }, a.gapMs)
        }
    }

    /** Worker: PNG, then the run's row in the JSON. */
    private fun finishRun(
        row: JSONObject, result: BonsaiPipeline.Result, bmp: Bitmap, bitmapMs: Double, postToDrawMs: Double,
        totalMs: Double, via: String, epoch0: Long, seed: Long, shown: JSONArray, lastOfAutorun: Boolean,
    ) {
        val tp = System.nanoTime()
        val outDir = File(modelsDir, "outputs").apply { mkdirs() }
        val png = File(outDir, "bonsai_${result.size}_seed${seed}_${System.currentTimeMillis()}.png")
        png.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        val pngMs = msSince(tp)
        val tm = result.timings
        val ms = JSONObject()
            .put("tokenize", tm.tokenizeMs)
            .put("textenc_load", tm.textencLoadMs).put("textenc_run", tm.textencRunMs)
            .put("textenc_close", tm.textencCloseMs)
            .put("noise", tm.noiseMs)
            .put("dit_load", tm.ditLoadMs).put("steps", JSONArray(tm.stepMs)).put("dit_close", tm.ditCloseMs)
            .put("unpatchify", tm.unpatchifyMs)
            .put("vae_load", tm.vaeLoadMs).put("vae_run", tm.vaeRunMs).put("vae_close", tm.vaeCloseMs)
            .put("rgb", tm.rgbMs)
            .put("bitmap", bitmapMs)
            .put("post_to_draw", postToDrawMs)
            .put("model_load", tm.modelLoadMs)
            .put("total_wall", totalMs)
            .put("png", pngMs)
        row.put("ms", ms)
            .put("table_shown", shown)
            .put("sigmas", JSONArray(result.sigmas.map { it.toDouble() }))
            .put("attention_tokens", result.attentionTokens)
            .put("files", JSONArray(result.files))
            .put("shown_by", via)
            .put("finished_epoch_ms", epoch0 + Math.round(totalMs))
            .put("thermal_after", thermal())
            .put("vmhwm_kib", procStatusKiB("VmHWM") ?: JSONObject.NULL)
            .put("vmrss_kib_after", procStatusKiB("VmRSS") ?: JSONObject.NULL)
            .put("png", png.path)
        runRows.put(row)
        writeReport()
        Log.i(TAG, "stage=png ms=${f1(pngMs)}")
        Log.i(TAG, "RUN_DONE size=${result.size} seed=$seed total_ms=${f1(totalMs)} png=${png.path} json=${reportFile.path}")
        if (lastOfAutorun) {
            report.put("autorun_done_epoch_ms", System.currentTimeMillis())
            writeReport()
            Log.i(TAG, "AUTORUN_DONE json=${reportFile.path}")
        }
        ui {
            lastPng = png
            if (!running) shareButton.visibility = View.VISIBLE
        }
    }

    private fun recordFailure(row: JSONObject, error: String, epoch0: Long) {
        row.put("error", error).put("failed_epoch_ms", System.currentTimeMillis())
            .put("elapsed_ms", System.currentTimeMillis() - epoch0)
            .put("thermal_after", thermal())
            .put("vmhwm_kib", procStatusKiB("VmHWM") ?: JSONObject.NULL)
        runRows.put(row)
        writeReport()
        Log.w(TAG, "RUN_FAILED reason=$error json=${reportFile.path}")
    }

    /** UI thread: a run ended without an image. */
    private fun onRunEnded(label: String, color: Int, state: PillState, message: String) {
        stopStopwatch(null)
        setPill(label, color, state)
        imageArea.segments = 0
        imageArea.message = message
        running = false
        setGenerateRunning(false)
        refreshControls()
        if (autorun != null) autorunFailed("run: $message")
    }

    /** UI thread: a cancelled run has closed its graph; the app is READY for the next Generate. */
    private fun onRunCancelled() {
        stopStopwatch(null)
        running = false
        setGenerateRunning(false)
        Log.i(TAG, "RUN_CANCELLED")
        if (autorun != null) autorunFailed("cancelled")   // before the pill below, which it would replace
        imageArea.segments = 0
        imageArea.message = CANCELLED_TEXT
        if (ready.isEmpty()) setPill("MODELS MISSING", C_ERR, PillState.MISSING)
        else setPill("READY", C_IDLE, PillState.READY)
        refreshControls()
    }

    private fun cancel() {
        pipeline?.cancelled = true
        setPill("CANCELLING", C_IDLE, PillState.RUNNING)
    }

    // ---- autorun ----------------------------------------------------------------------------------------------

    private fun wantsAutorun(i: Intent?) = i?.getBooleanExtra("autorun", false) == true

    /** Reads the autorun extras and sets the screen to the first run (before it is drawn when launched). */
    private fun prepareAutorun(i: Intent): Autorun? {
        try {
            intExtraOrNull(i, "steps")?.let {
                require(it in 1..50) { "steps $it out of range" }
                steps = it
                shownSteps = it
            }
            intExtraOrNull(i, "threads")?.let {
                require(it in 1..16) { "threads $it out of range" }
                threads = it
            }
            gpuDit = i.getBooleanExtra("gpu", gpuDit)
            val runs = parseRuns(i)
            val delay = (intExtraOrNull(i, "delay_ms") ?: 2000).toLong()
            val gap = (intExtraOrNull(i, "gap_ms") ?: 3000).toLong()
            val type = i.getBooleanExtra("type_prompt", true)
            val a = Autorun(runs, delay, gap, type)
            autorun = a
            size = runs[0].size
            shownSize = size
            shownSeed = runs[0].seed
            if (type) promptEdit.setText("") else promptEdit.setText(runs[0].prompt)
            rebuildTable(steps)
            refreshControls()
            refreshFooter()
            val spec = JSONObject().put("runs", JSONArray().apply {
                runs.forEach { r -> put(JSONObject().put("prompt", r.prompt).put("seed", r.seed ?: JSONObject.NULL).put("size", r.size)) }
            }).put("steps", steps).put("threads", threads).put("delay_ms", delay).put("gap_ms", gap)
                .put("type_prompt", type).put("gpu_dit", gpuDit)
            worker.execute { report.put("autorun", spec); writeReport() }
            Log.i(TAG, "AUTORUN_START $spec")
            return a
        } catch (e: IllegalArgumentException) {
            autorunFailed(e.message ?: e.toString())
            return null
        }
    }

    /** Parallel lists (prompts | seeds | sizes); one value applies to every run. */
    private fun parseRuns(i: Intent): List<RunSpec> {
        val prompts = i.getStringExtra("prompts")?.split('|')?.map { it.trim() }
            ?: listOf(i.getStringExtra("prompt")?.trim() ?: DEFAULT_PROMPT)
        val seeds: List<Long?> = i.getStringExtra("seeds")?.split(',')?.map { s ->
            s.trim().toLongOrNull() ?: throw IllegalArgumentException("bad seed '$s' in seeds")
        } ?: listOf(longExtraOrNull(i, "seed"))
        val sizeList = i.getStringExtra("sizes")?.split(',')?.map { s ->
            s.trim().toIntOrNull() ?: throw IllegalArgumentException("bad size '$s' in sizes")
        } ?: listOf(size)
        val n = maxOf(prompts.size, seeds.size, sizeList.size)
        for ((name, count) in listOf("prompts" to prompts.size, "seeds" to seeds.size, "sizes" to sizeList.size)) {
            require(count == 1 || count == n) { "$name has $count values, expected 1 or $n" }
        }
        require(prompts.none { it.isEmpty() }) { "empty prompt" }
        val unknown = sizeList.distinct().filter { it !in sizes }
        require(unknown.isEmpty()) { "size $unknown not in pipeline_meta.json (has $sizes)" }
        val absent = sizeList.distinct().filter { it !in ready }
        require(absent.isEmpty()) { "size $absent not ready, missing: $missing" }
        fun <T> at(l: List<T>, k: Int) = if (l.size == 1) l[0] else l[k]
        return (0 until n).map { k -> RunSpec(at(prompts, k), at(seeds, k), at(sizeList, k)) }
    }

    /** One autorun entry as a person would do it: pick the size, type the prompt, press Generate. */
    private fun autorunStep() {
        val a = autorun ?: return
        if (running || a.next >= a.runs.size) return
        if (tokenizer == null && tokenizerProblem == null) {   // prepare() is still loading it
            main.postDelayed({ autorunStep() }, 200)
            return
        }
        val r = a.runs[a.next]
        a.next++
        val retype = a.typePrompt && promptEdit.text.toString() != r.prompt
        if (!a.typePrompt) promptEdit.setText(r.prompt)
        val afterSize = {
            if (retype) typePrompt(r.prompt) { main.postDelayed({ press(r) }, 300) } else press(r)
        }
        if (r.size != size) {
            size = r.size
            shownSize = r.size
            refreshControls()
            refreshFooter()
            main.postDelayed(afterSize, 400)
        } else afterSize()
    }

    private fun typePrompt(text: String, then: () -> Unit) {
        promptEdit.setText("")
        var k = 0
        main.post(object : Runnable {
            override fun run() {
                if (k >= text.length) {
                    then()
                    return
                }
                val next = text.offsetByCodePoints(k, 1)
                promptEdit.text.append(text, k, next)
                k = next
                main.postDelayed(this, TYPE_MS)
            }
        })
    }

    private fun press(r: RunSpec) {
        generateButton.isPressed = true
        main.postDelayed({
            generateButton.isPressed = false
            startRun(r.prompt, r.seed, r.size, auto = true)?.let { autorunFailed("run did not start: $it") }
        }, PRESS_MS)
    }

    private fun autorunFailed(reason: String) {
        Log.e(TAG, "AUTORUN_FAILED reason=$reason")
        autorun = null
        if (pillState != PillState.ERROR && !running) {
            setPill("ERROR", C_ERR, PillState.ERROR)
            imageArea.message = "Autorun: $reason"
        }
        worker.execute { report.put("autorun_failed", reason); writeReport() }
    }

    // ---- screen -----------------------------------------------------------------------------------------------

    private fun buildUi(): View {
        root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(C_BG)
            isFocusableInTouchMode = true   // keeps the prompt box (and the keyboard) unfocused at launch
        }
        val side = dp(16f)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            v.setPadding(side + bars.left, dp(4f) + bars.top, side + bars.right, dp(4f) + maxOf(bars.bottom, ime.bottom))
            WindowInsetsCompat.CONSUMED
        }

        val title = label(14f, C_TEXT, bold = true).apply { text = "Bonsai Image 4B · on-device" }
        shareButton = label(14f, C_ACCENT, bold = true).apply {
            text = "Share"
            visibility = View.INVISIBLE
            setPadding(dp(12f), 0, 0, 0)
            setOnClickListener { share() }
        }
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(title, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(shareButton)
        })

        promptEdit = EditText(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
            setTextColor(C_TEXT)
            setHintTextColor(C_SUB)
            hint = "Describe an image…"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            setLines(3)   // after inputType, which resets the line limits; fixed so typing never moves the screen
            gravity = Gravity.TOP or Gravity.START
            includeFontPadding = false
            background = rounded(C_FIELD, 14f, C_EDGE)
            setPadding(dp(12f), dp(6f), dp(12f), dp(6f))
            setText(DEFAULT_PROMPT)
            tag = VIEW_PROMPT
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
                override fun afterTextChanged(s: Editable?) = refreshPromptStatus()
            })
        }
        root.addView(promptEdit, lp(top = 6f))

        // the prompt's tokens against the text encoder's window, or why Generate did not start
        promptStatus = label(14f, C_SUB).apply {
            maxLines = 2
            includeFontPadding = false
            tag = VIEW_STATUS
        }
        root.addView(promptStatus, lp(top = 4f))

        val sizeBox = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = rounded(C_FIELD, 12f, C_EDGE)
            setPadding(dp(3f), dp(3f), dp(3f), dp(3f))
        }
        sizeViews = sizes.associateWith { s ->
            label(16f, C_TEXT, bold = true).apply {
                text = s.toString()
                gravity = Gravity.CENTER
                tag = viewSize(s)
                setOnClickListener {
                    if (running || autorun != null) return@setOnClickListener
                    if (s !in ready) refreshModels()   // the graphs may have been pushed while the app was open
                    if (s in ready) {
                        size = s
                        shownSize = s
                        refreshControls()
                        refreshFooter()
                        refreshPromptStatus()
                    } else {
                        setPromptStatus(sizeMissingText(s), error = true)
                    }
                }
            }.also { sizeBox.addView(it, LinearLayout.LayoutParams(dp(52f), MATCH_PARENT)) }
        }
        stepsChip = label(16f, C_TEXT, bold = true).apply {
            gravity = Gravity.CENTER
            background = rounded(C_FIELD, 12f, C_EDGE)
            setPadding(dp(12f), 0, dp(12f), 0)
            setOnClickListener {
                if (!running && autorun == null) {
                    steps = STEP_CHOICES.firstOrNull { it > steps } ?: STEP_CHOICES.first()
                    shownSteps = steps
                    rebuildTable(steps)
                    refreshControls()
                    refreshFooter()
                }
            }
        }
        generateButton = label(17f, C_BG, bold = true).apply {
            text = "Generate"
            gravity = Gravity.CENTER
            tag = VIEW_GENERATE
            setOnClickListener {
                if (running) cancel()
                else if (autorun == null) startRun(promptEdit.text.toString(), null, size, auto = false)
            }
        }
        setGenerateRunning(false)
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(sizeBox, LinearLayout.LayoutParams(WRAP_CONTENT, MATCH_PARENT))
            addView(stepsChip, LinearLayout.LayoutParams(WRAP_CONTENT, MATCH_PARENT).apply { marginStart = dp(8f) })
            addView(generateButton, LinearLayout.LayoutParams(0, MATCH_PARENT, 1f).apply { marginStart = dp(8f) })
        }, LinearLayout.LayoutParams(MATCH_PARENT, dp(44f)).apply { topMargin = dp(6f) })

        pill = label(20f, Color.WHITE, bold = true).apply {
            gravity = Gravity.CENTER
            tag = VIEW_PILL
            includeFontPadding = false
            maxLines = 1
            letterSpacing = 0.03f
            setPadding(dp(14f), 0, dp(14f), 0)
            background = GradientDrawable().apply { cornerRadius = dp(22f).toFloat(); setColor(C_IDLE) }
        }
        stopwatch = label(32f, C_SUB, bold = true).apply {
            gravity = Gravity.END or Gravity.CENTER_VERTICAL
            includeFontPadding = false
            maxLines = 1
            minWidth = TabularSpan().getSize(paint, "88.8 s", 0, 6, null) + 1   // a steady box for the video tools
            text = TabularSpan.of(fmtS(0.0))
        }
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(pill, LinearLayout.LayoutParams(WRAP_CONTENT, dp(44f)))
            addView(Space(context), LinearLayout.LayoutParams(0, 0, 1f))
            addView(stopwatch, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginStart = dp(8f) })
        }, lp(top = 6f))

        // takes the height the rest leaves, up to a full-width square
        imageArea = ImageArea(this).apply { tag = VIEW_IMAGE }
        root.addView(imageArea, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f).apply { topMargin = dp(6f) })

        table = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        root.addView(table, lp(top = 8f))
        rebuildTable(steps)

        footer1 = label(14f, C_SUB).apply { includeFontPadding = false }
        footer2 = label(14f, C_SUB).apply { includeFontPadding = false }
        root.addView(footer1, lp(top = 6f))
        root.addView(footer2, lp(top = 2f))
        return root
    }

    /** Two columns of (stage, seconds): text encoder, step 1..n, VAE, model load, total — filled as stages end. */
    private fun rebuildTable(n: Int) {
        val entries = n + 4
        val rows = (entries + 1) / 2
        if (cells.size == rows * 2) {
            for ((name, value) in cells) {
                name.text = ""
                value.text = ""
            }
            return
        }
        table.removeAllViews()
        cells.clear()
        val cols = List(2) { c ->
            LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }.also {
                table.addView(it, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply {
                    if (c == 1) marginStart = dp(20f)
                })
            }
        }
        for (i in 0 until rows * 2) {
            val name = label(14f, C_SUB).apply { maxLines = 1; minLines = 1; includeFontPadding = false }
            val value = label(14f, C_TEXT).apply {
                maxLines = 1
                minLines = 1
                includeFontPadding = false
                gravity = Gravity.END
            }
            cols[i / rows].addView(LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(name, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
                addView(value, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
            }, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { if (i % rows > 0) topMargin = dp(2f) })
            cells.add(name to value)
        }
    }

    private fun setEntry(i: Int, name: String, ms: Double, bold: Boolean = false) {
        val (n, v) = cells.getOrNull(i) ?: return
        n.text = name
        v.text = TabularSpan.of(fmtS(ms))
        v.typeface = if (bold) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        n.setTextColor(if (bold) C_TEXT else C_SUB)
    }

    private fun setPill(text: String, color: Int, state: PillState, big: Boolean = false) {
        pill.text = text
        pill.setTextSize(TypedValue.COMPLEX_UNIT_SP, if (big) 26f else 20f)
        (pill.background as GradientDrawable).setColor(color)
        pillState = state
    }

    private fun setGenerateRunning(on: Boolean) {
        generateButton.text = if (on) "Cancel" else "Generate"
        generateButton.setTextColor(if (on) C_TEXT else C_BG)
        generateButton.background = if (on) rounded(C_FIELD, 12f, C_EDGE) else StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_pressed), rounded(C_BUTTON_PRESSED, 12f))
            addState(intArrayOf(), rounded(C_BUTTON, 12f))
        }
    }

    private fun refreshControls() {
        for ((s, v) in sizeViews) {
            val selected = s == size
            v.isSelected = selected
            v.background = if (selected) rounded(C_SELECTED, 9f) else null
            v.setTextColor(if (selected) C_TEXT else C_SUB)
            v.alpha = when {
                s !in ready -> 0.3f
                running && !selected -> 0.5f
                else -> 1f
            }
        }
        stepsChip.text = if (steps == 1) "1 step" else "$steps steps"
        stepsChip.alpha = if (running) 0.5f else 1f
        // dimmed, not disabled: a press re-reads the models folder, then runs or says what is missing
        generateButton.alpha = if (running || size in ready) 1f else 0.4f
    }

    /** UI thread: the line under the prompt box, back to the token count (or the tokenizer's state). */
    private fun refreshPromptStatus() {
        if (!::promptStatus.isInitialized) return
        val tok = tokenizer
        if (tok == null) {
            setPromptStatus(tokenizerProblem ?: "", error = tokenizerProblem != null)
            return
        }
        val c = PromptRules.count(promptEdit.text.toString(), tok)
        setPromptStatus(PromptRules.counterText(c), error = !c.fits)
    }

    private fun setPromptStatus(text: String, error: Boolean) {
        promptStatus.text = text
        promptStatus.setTextColor(if (error) C_ERR else C_SUB)
    }

    private fun hideKeyboard() {
        getSystemService(InputMethodManager::class.java)?.hideSoftInputFromWindow(promptEdit.windowToken, 0)
        promptEdit.clearFocus()
    }

    /** Footer: device and runtime, then model and run settings — each broken at a separator when too wide. */
    private fun refreshFooter() {
        if (!::footer1.isInitialized) return
        val backend = if (gpuDit) "CPU + GPU DiT" else "CPU"
        setParts(footer1, listOf(
            deviceName,
            "$backend, XNNPACK $threads threads",
            "LiteRT ${BuildConfig.LITERT_VERSION}",
            "airplane mode ${if (airplaneMode()) "on" else "off"}",
        ))
        setParts(footer2, listOf(
            MODEL_LABEL,
            "$shownSize×$shownSize",
            if (shownSteps == 1) "1 step" else "$shownSteps steps",
            shownSeed?.let { "seed $it" } ?: pinnedSeed?.let { "seed $it" } ?: "random seed",
        ))
    }

    private fun setParts(tv: TextView, parts: List<String>) {
        val avail = (if (root.width > 0) root.width - root.paddingLeft - root.paddingRight
        else resources.displayMetrics.widthPixels - 2 * dp(16f)).toFloat()
        val whole = parts.joinToString(SEP)
        if (parts.size < 2 || tv.paint.measureText(whole) <= avail) {
            tv.text = whole
            return
        }
        // the break that balances the two lines best; never inside a part
        val k = (1 until parts.size).minByOrNull { k ->
            maxOf(tv.paint.measureText(parts.subList(0, k).joinToString(SEP)),
                tv.paint.measureText(parts.subList(k, parts.size).joinToString(SEP)))
        }!!
        tv.text = parts.subList(0, k).joinToString(SEP) + "\n" + parts.subList(k, parts.size).joinToString(SEP)
    }

    // ---- stopwatch --------------------------------------------------------------------------------------------

    private val tick = object : Choreographer.FrameCallback {
        override fun doFrame(frameTimeNanos: Long) {
            if (!stopwatchOn) return
            val tenths = (System.nanoTime() - runT0) / 100_000_000L
            if (tenths != lastTenths) {
                lastTenths = tenths
                stopwatch.text = TabularSpan.of(String.format(Locale.US, "%d.%d s", tenths / 10, tenths % 10))
            }
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    private fun startStopwatch() {
        stopwatchOn = true
        lastTenths = -1
        stopwatch.setTextColor(C_TEXT)
        stopwatch.text = TabularSpan.of(fmtS(0.0))
        Choreographer.getInstance().postFrameCallback(tick)
    }

    /** Stops the stopwatch; [finalMs] (the run's wall time) replaces the last tick when given. */
    private fun stopStopwatch(finalMs: Double?) {
        stopwatchOn = false
        Choreographer.getInstance().removeFrameCallback(tick)
        finalMs?.let { stopwatch.text = TabularSpan.of(fmtS(it)) }
    }

    // ---- records ----------------------------------------------------------------------------------------------

    /** Screen pixels of the pill, the stopwatch and the image, for the video tools (the pill colour marks the state). */
    private fun recordLayout(reason: String) {
        fun rect(v: View): JSONObject {
            val p = IntArray(2)
            v.getLocationOnScreen(p)
            return JSONObject().put("left", p[0]).put("top", p[1]).put("width", v.width).put("height", v.height)
        }
        val pillRect = rect(pill).put("pad_left", pill.paddingLeft)
        val image = rect(imageArea).put("side", imageArea.side)
        image.put("square_left", image.getInt("left") + (imageArea.width - imageArea.side) / 2)
        val bottomUsed = IntArray(2).also { footer2.getLocationOnScreen(it) }[1] + footer2.height
        val layout = JSONObject().put("reason", reason)
            .put("window_px", JSONArray().put(window.decorView.width).put(window.decorView.height))
            .put("density", resources.displayMetrics.density.toDouble())
            .put("font_scale", resources.configuration.fontScale.toDouble())
            .put("pill_px", pillRect)
            .put("pill_sample_px", JSONArray().put(pillRect.getInt("left") + pill.paddingLeft / 2)
                .put(pillRect.getInt("top") + pill.height / 2))
            .put("pill_colors", JSONObject().put("idle", hex(C_IDLE)).put("running", hex(C_RUN))
                .put("done", hex(C_DONE)).put("error", hex(C_ERR)))
            .put("stopwatch_px", rect(stopwatch))
            .put("image_px", image)
            .put("image_full_width", imageArea.side == imageArea.width)
            .put("table_px", rect(table))
            .put("prompt_px", rect(promptEdit))
            .put("generate_px", rect(generateButton))
            .put("footer_bottom_px", bottomUsed)
        Log.i(TAG, "LAYOUT $layout")
        worker.execute { report.put("layout", layout); writeReport() }
    }

    private fun writeReport() {
        try {
            val tmp = File(reportFile.path + ".tmp")
            tmp.writeText(report.toString(1))
            check(tmp.renameTo(reportFile)) { "could not rename ${tmp.path}" }
        } catch (e: Exception) {
            Log.e(TAG, "could not write ${reportFile.path}", e)
        }
    }

    private fun thermal(): JSONObject {
        val o = JSONObject()
        if (Build.VERSION.SDK_INT >= 29) {
            val pm = getSystemService(PowerManager::class.java)
            o.put("status", pm.currentThermalStatus)
            if (Build.VERSION.SDK_INT >= 30) {
                val h = pm.getThermalHeadroom(10)
                o.put("headroom_10s", if (h.isNaN()) JSONObject.NULL else h.toDouble())
            }
        }
        return o
    }

    private fun procStatusKiB(key: String): Long? = runCatching {
        File("/proc/self/status").readLines().first { it.startsWith("$key:") }
            .trim().split(Regex("\\s+"))[1].toLong()
    }.getOrNull()

    /** Resets VmHWM to the current RSS (Linux clear_refs "5"), so each run's VmHWM is its own peak; false if the
     *  platform refuses, and VmHWM is then the process peak so far. */
    private fun resetPeakRss(): Boolean = runCatching { File("/proc/self/clear_refs").writeText("5") }.isSuccess

    private fun airplaneMode(): Boolean =
        Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) != 0

    // ---- helpers ----------------------------------------------------------------------------------------------

    private fun share() {
        val png = lastPng ?: return
        val uri: Uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", png)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Share image"))
    }

    private fun toBitmap(rgb: ByteArray, side: Int): Bitmap {
        val px = IntArray(side * side)
        for (p in px.indices) {
            px[p] = Color.rgb(
                rgb[p * 3].toInt() and 0xFF,
                rgb[p * 3 + 1].toInt() and 0xFF,
                rgb[p * 3 + 2].toInt() and 0xFF
            )
        }
        return Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888).apply {
            setPixels(px, 0, side, 0, 0, side, side)
        }
    }

    /** An int extra given as --ei, --el or --es. */
    private fun intExtraOrNull(i: Intent, key: String): Int? = longExtraOrNull(i, key)?.let {
        require(it in Int.MIN_VALUE..Int.MAX_VALUE) { "$key $it out of range" }
        it.toInt()
    }

    /** A long extra given as --el, --ei or --es. */
    private fun longExtraOrNull(i: Intent, key: String): Long? {
        @Suppress("DEPRECATION")
        return when (val v = i.extras?.get(key)) {
            null -> null
            is Number -> v.toLong()
            is String -> v.trim().toLongOrNull() ?: throw IllegalArgumentException("bad $key '$v'")
            else -> throw IllegalArgumentException("bad $key: ${v.javaClass.simpleName}")
        }
    }

    private fun ui(block: () -> Unit) {
        main.post(block)
    }

    private fun dp(v: Float): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    private fun lp(top: Float) = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(top) }

    private fun label(sizeSp: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
        setTextColor(color)
        if (bold) typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private fun rounded(color: Int, radiusDp: Float, stroke: Int? = null) = GradientDrawable().apply {
        cornerRadius = dp(radiusDp).toFloat()
        setColor(color)
        stroke?.let { setStroke(dp(1f).coerceAtLeast(1), it) }
    }

    companion object {
        private const val TAG = "BonsaiDemo"
        private const val DEFAULT_PROMPT = "a small bonsai tree in a blue ceramic pot"
        private const val MODEL_LABEL = "Bonsai Image 4B (ternary weights, int4)"
        private const val SEP = " · "
        private const val TYPE_MS = 30L
        private const val PRESS_MS = 180L
        private val STEP_CHOICES = listOf(2, 4, 6, 8)

        /** The image area's text after a Cancel. */
        internal const val CANCELLED_TEXT = "Cancelled. Press Generate to run again."

        // View tags: BonsaiDeviceCheck finds the views a person would touch and read by these.
        internal const val VIEW_PROMPT = "prompt"
        internal const val VIEW_STATUS = "prompt_status"
        internal const val VIEW_GENERATE = "generate"
        internal const val VIEW_PILL = "pill"
        internal const val VIEW_IMAGE = "image"
        internal fun viewSize(size: Int) = "size_$size"

        /** The image area's text when a run finds graphs missing (BonsaiPipeline.MissingModels). */
        internal fun missingRunText(dir: File, files: List<String>) =
            "Missing in ${dir.path}:\n" + files.joinToString("\n")

        private const val C_BG = 0xFF0E1116.toInt()
        private const val C_TEXT = 0xFFE6E8EB.toInt()
        private const val C_SUB = 0xFF8A919C.toInt()
        private const val C_FIELD = 0xFF1A1F27.toInt()
        private const val C_EDGE = 0xFF2C333D.toInt()
        private const val C_SELECTED = 0xFF343C48.toInt()
        private const val C_BUTTON = 0xFFE6E8EB.toInt()
        private const val C_BUTTON_PRESSED = 0xFFA9B0BA.toInt()
        private const val C_ACCENT = 0xFF8AB4F8.toInt()
        private const val C_IDLE = 0xFF5F6368.toInt()
        private const val C_RUN = 0xFF1565C0.toInt()
        private const val C_DONE = 0xFF2E7D32.toInt()
        private const val C_ERR = 0xFFE53935.toInt()

        private fun msSince(t0: Long) = (System.nanoTime() - t0) / 1e6
        private fun f1(ms: Double) = String.format(Locale.US, "%.1f", ms)
        private fun fmtS(ms: Double) = String.format(Locale.US, "%.1f s", ms / 1000)
        private fun hex(c: Int) = String.format("#%06X", c and 0xFFFFFF)
    }
}
