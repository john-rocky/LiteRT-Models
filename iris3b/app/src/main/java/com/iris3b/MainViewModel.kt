package com.iris3b

import android.app.Application
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Owns the model session and makes one image at a time. Every engine call runs on one thread
 * ([engine]); the screen only reads [uiState].
 *
 * Each image also leaves `files/last_run.json` (stage times, memory, status) and `IRIS3B_APP` lines in
 * logcat; the verify entry adds `files/verify_<key>.npy` (the image as float32 [1, 3, 768, 768]).
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application
    private val dir: File = application.filesDir
    private val engineThread = Executors.newSingleThreadExecutor { r -> Thread(r, "iris3b-engine") }
    private val engine = engineThread.asCoroutineDispatcher()

    private val _uiState = MutableStateFlow(UiState(prompt = Example.APPLE.text))
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    @Volatile
    private var session: ImageSession? = null

    @Volatile
    private var stopRequested = false
    private var checking: Job? = null
    private var counting: Job? = null

    /** A scripted request that came before the model was ready; runs once it is. */
    private var pending: (() -> Unit)? = null
    private var guardKb = MemoryWatch.DEFAULT_GUARD_KB

    init {
        checkModel()
    }

    /** Looks for the installed model and opens it. */
    fun checkModel() {
        if (checking?.isActive == true || _uiState.value.running) return
        _uiState.update { it.copy(model = ModelState.Checking, message = null) }
        checking = viewModelScope.launch {
            val outcome = withContext(engine) {
                runCatching {
                    val missing = ImageSession.missingFiles(dir)
                    if (missing.isEmpty() && session == null) session = ImageSession.open(app, dir)
                    missing
                }
            }
            outcome.onSuccess { missing ->
                if (missing.isNotEmpty()) {
                    Log.w(TAG, "MISSING ${missing.size} first=${missing.first()}")
                    _uiState.update { it.copy(model = ModelState.Missing(missing)) }
                    return@onSuccess
                }
                val s = requireNotNull(session)
                Log.i(TAG, "READY residency=${s.residency} max_caption_tokens=${s.maxCaptionTokens}")
                _uiState.update { it.copy(model = ModelState.Ready, maxPromptTokens = s.maxCaptionTokens) }
                countTokens(_uiState.value.prompt)
                pending?.let { request ->
                    pending = null
                    request()
                }
            }.onFailure { failure ->
                Log.e(TAG, "MODEL_FAILED", failure)
                _uiState.update { it.copy(model = ModelState.Failed(failure.toString())) }
            }
        }
    }

    fun setPrompt(text: String) {
        _uiState.update { it.copy(prompt = text) }
        countTokens(text)
    }

    fun selectExample(example: Example) = setPrompt(example.text)

    fun setSeed(text: String) {
        _uiState.update { it.copy(seedText = text.filter { c -> c in '0'..'9' }.take(MAX_SEED_DIGITS)) }
    }

    fun randomSeed() {
        _uiState.update { it.copy(seedText = Random.nextInt(0, 1_000_000).toString()) }
    }

    /** The Generate button. A second press while an image is in progress is ignored. */
    fun generate() {
        val state = _uiState.value
        if (state.running) {
            Log.i(TAG, "IGNORED generate: an image is in progress")
            return
        }
        if (!state.canGenerate) return
        start(Request(state.prompt, state.seedText.toLongOrNull() ?: 0L, verifyKey = null))
    }

    /** Ends the image before its next graph. */
    fun stop() {
        if (!_uiState.value.running) return
        stopRequested = true
        _uiState.update { it.copy(stage = app.getString(R.string.stage_stopping)) }
    }

    fun save() {
        val bitmap = _uiState.value.image ?: return
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) { runCatching { ImageStore.savePng(app, bitmap) } }
            _uiState.update { s ->
                saved.fold(
                    { s.copy(message = app.getString(R.string.saved, it), messageIsError = false) },
                    { s.copy(message = app.getString(R.string.save_failed, it.toString()), messageIsError = true) },
                )
            }
        }
    }

    /** Scripted (debug): [example] from the reference noise; writes files/verify_<key>.npy and .json. */
    fun verify(example: Example) {
        if (!whenReady { verify(example) }) return
        if (_uiState.value.running) {
            Log.i(TAG, "IGNORED verify ${example.key}: an image is in progress")
            return
        }
        setPrompt(example.text)
        start(Request(example.text, seed = null, verifyKey = example.key))
    }

    /** Scripted (debug): presses Generate once the model is ready. */
    fun autoplay() {
        if (whenReady { autoplay() }) generate()
    }

    /** Scripted (debug): the free-memory floor below which an image stops. */
    fun setGuardKb(kb: Long) {
        guardKb = kb
        Log.i(TAG, "GUARD_KB $kb")
    }

    /** True when the model is ready; otherwise keeps [request] for when it is. */
    private fun whenReady(request: () -> Unit): Boolean {
        if (_uiState.value.ready) return true
        pending = request
        return false
    }

    private class Request(val prompt: String, val seed: Long?, val verifyKey: String?)

    private fun start(request: Request) {
        val s = session ?: return
        stopRequested = false
        val t0 = SystemClock.elapsedRealtime()
        _uiState.update {
            it.copy(
                running = true,
                stage = app.getString(R.string.stage_preparing),
                progress = 0f,
                elapsedSeconds = 0,
                message = null,
                messageIsError = false,
            )
        }
        viewModelScope.launch {
            val ticker = launch {
                while (true) {
                    delay(1000)
                    _uiState.update { it.copy(elapsedSeconds = ((SystemClock.elapsedRealtime() - t0) / 1000).toInt()) }
                }
            }
            val finish = withContext(engine) { runOnce(s, request) }
            ticker.cancel()
            _uiState.update(finish)
        }
    }

    /** Makes the image on the engine thread; returns how the screen changes. */
    private fun runOnce(s: ImageSession, r: Request): (UiState) -> UiState {
        val key = r.verifyKey ?: "user"
        val promptSha = sha256(r.prompt)
        val noise = if (r.verifyKey != null) s.referenceNoise() else ImageSession.seedNoise(r.seed ?: 0L)
        val watch = MemoryWatch(guardKb).start()
        var runs = 0
        val log = StageLog(System::nanoTime, { MemoryWatch.snapshot() }) { e ->
            Log.i(TAG, "EVT $key ${e.stage} ${e.graph ?: "-"} s${e.step} ${e.ms} ${e.memory}")
            if (e.stage == "run") runs++
            // After Stop the line keeps saying so until the pipeline reaches its next graph.
            val text = if (stopRequested) null else stageText(s, e)
            _uiState.update { it.copy(stage = text ?: it.stage, progress = runs.toFloat() / s.totalRuns) }
        }
        Log.i(TAG, "START key=$key seed=${r.seed ?: "reference"} prompt_sha256=$promptSha chars=${r.prompt.length} guard_kb=$guardKb")
        val start = System.nanoTime()
        fun wallMs() = (System.nanoTime() - start) / 1e6
        try {
            val result = s.generate(r.prompt, noise, stop = { stopRequested || watch.tripped }, log = log)
            val wall = wallMs()
            val report = report(s, r, promptSha, "DONE", log, wall, result, watch, null)
            r.verifyKey?.let { NpyFile.write(File(dir, "verify_$it.npy"), result.image, intArrayOf(1, 3, HostOps.IMAGE_SIZE, HostOps.IMAGE_SIZE)) }
            writeReport(report, r.verifyKey)
            val stages = RunReport.stageMs(log.events)
            Log.i(
                TAG,
                "DONE key=$key wall_ms=${f1(wall)} " +
                    stages.entries.joinToString(" ") { (k, v) -> "${k}_ms=${f1(v)}" } +
                    " real_len=${result.realLen} caption_tokens=${result.captionTokens} nonfinite=${HostOps.nonFinite(result.image)}" +
                    " mem_min_kb=${watch.minKb} vmhwm_kb=${RunReport.peak(log.events, "VmHWM_kb")}",
            )
            val bitmap = ImageSession.toBitmap(result.image)
            val noiseText = r.seed?.let { app.getString(R.string.seed_value, it) } ?: app.getString(R.string.reference_noise)
            var info = app.getString(R.string.result_info, wall / 1000, noiseText)
            if (result.captionTokens > s.maxCaptionTokens) {
                info += "\n" + app.getString(R.string.result_cut, s.maxCaptionTokens, result.captionTokens)
            }
            return { it.copy(running = false, stage = "", progress = 1f, image = bitmap, imageInfo = info) }
        } catch (stop: StopRequested) {
            val lowKb = watch.trippedKb
            val status = if (lowKb != null) "STOPPED_LOW_MEMORY" else "STOPPED"
            writeReport(report(s, r, promptSha, status, log, wallMs(), null, watch, null), r.verifyKey)
            Log.w(TAG, "$status key=$key wall_ms=${f1(wallMs())} tripped_kb=$lowKb guard_kb=${watch.guardKb} mem_min_kb=${watch.minKb}")
            val message = if (lowKb != null) {
                app.getString(R.string.stopped_low_memory, lowKb / 1024, watch.guardKb / 1024)
            } else {
                app.getString(R.string.stopped_by_user)
            }
            return { it.copy(running = false, stage = "", message = message, messageIsError = lowKb != null) }
        } catch (failure: Throwable) {
            writeReport(report(s, r, promptSha, "FAILED", log, wallMs(), null, watch, failure), r.verifyKey)
            Log.e(TAG, "FAILED key=$key", failure)
            return { it.copy(running = false, stage = "", message = app.getString(R.string.failed, failure.toString()), messageIsError = true) }
        } finally {
            watch.close()
        }
    }

    /** The status line for what comes after timeline event [e]. */
    private fun stageText(s: ImageSession, e: StageLog.Event): String? {
        fun encoder(i: Int) = app.getString(R.string.stage_text_encoder, i + 1, s.teGraphs)
        fun denoise(step: Int) = app.getString(R.string.stage_denoise, step + 1, s.steps)
        val graph = e.graph
        return when {
            graph == null -> when (e.stage) {
                "tokenize", "embed", "te_mask" -> encoder(0)
                "adapter_bias" -> app.getString(R.string.stage_adapter)
                "sampler" -> if (e.step + 1 < s.steps) denoise(e.step + 1) else app.getString(R.string.stage_finishing)
                else -> null
            }
            graph.startsWith("te") -> {
                val i = graph.removePrefix("te").toIntOrNull() ?: return null
                val next = if (e.stage == "close") i + 1 else i
                if (next < s.teGraphs) encoder(next) else app.getString(R.string.stage_adapter)
            }
            graph == "dit_adapter" -> if (e.stage == "close") denoise(0) else app.getString(R.string.stage_adapter)
            else -> denoise(e.step)
        }
    }

    private fun report(
        s: ImageSession,
        r: Request,
        promptSha: String,
        status: String,
        log: StageLog,
        wallMs: Double,
        result: PipelineResult?,
        watch: MemoryWatch,
        failure: Throwable?,
    ): Map<String, Any?> = linkedMapOf(
        "app" to "com.iris3b sample",
        "status" to status,
        "key" to (r.verifyKey ?: "user"),
        "prompt" to r.prompt,
        "prompt_sha256" to promptSha,
        "noise" to (r.seed?.let { "seed $it" } ?: "reference ${ImageSession.REFERENCE_NOISE}"),
        "device" to Build.MODEL,
        "android" to Build.VERSION.SDK_INT,
        "litert" to LITERT_VERSION,
        "config" to ImageSession.CONFIG_FILE,
        "residency" to s.residency,
        "real_len" to result?.realLen,
        "caption_tokens" to result?.captionTokens,
        "max_caption_tokens" to s.maxCaptionTokens,
        "wall_ms" to wallMs,
        "stage_ms" to RunReport.stageMs(log.events),
        "memory" to linkedMapOf(
            "VmHWM_kb_max" to RunReport.peak(log.events, "VmHWM_kb"),
            "VmRSS_kb_max" to RunReport.peak(log.events, "VmRSS_kb"),
            "VmSwap_kb_max" to RunReport.peak(log.events, "VmSwap_kb"),
            "MemAvailable_kb_min" to watch.minKb.takeIf { it != Long.MAX_VALUE },
            "guard_kb" to watch.guardKb,
            "guard_tripped_kb" to watch.trippedKb,
        ),
        "nonfinite" to result?.let { HostOps.nonFinite(it.image) },
        "error" to failure?.toString(),
        "events" to RunReport.events(log.events),
    )

    /** files/last_run.json, and files/verify_<key>.json for the verify entry (written after its .npy). */
    private fun writeReport(report: Map<String, Any?>, verifyKey: String?) {
        val text = Json.write(report)
        val names = listOfNotNull(LAST_RUN, verifyKey?.let { "verify_$it.json" })
        for (name in names) {
            val partial = File(dir, "$name.partial")
            partial.writeText(text)
            partial.renameTo(File(dir, name))
        }
    }

    private fun countTokens(text: String) {
        counting?.cancel()
        val s = session
        if (s == null) {
            _uiState.update { it.copy(promptTokens = null) }
            return
        }
        counting = viewModelScope.launch {
            delay(COUNT_DELAY_MS)
            val n = withContext(Dispatchers.Default) { s.countTokens(text) }
            _uiState.update { if (it.prompt == text) it.copy(promptTokens = n) else it }
        }
    }

    override fun onCleared() {
        stopRequested = true
        // After the image in progress (one engine thread): close the session, then the thread.
        engineThread.execute {
            session?.close()
            session = null
        }
        engineThread.shutdown()
    }

    private companion object {
        const val TAG = "IRIS3B_APP"
        const val LITERT_VERSION = "2.2.0"
        const val LAST_RUN = "last_run.json"
        const val MAX_SEED_DIGITS = 9
        const val COUNT_DELAY_MS = 150L

        fun sha256(text: String): String =
            MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }

        /** One decimal with a '.' whatever the phone's locale (log lines are parsed on a computer). */
        fun f1(v: Double): String = String.format(Locale.US, "%.1f", v)
    }
}
