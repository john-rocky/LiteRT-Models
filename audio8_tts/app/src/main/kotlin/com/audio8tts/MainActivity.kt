package com.audio8tts

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.LocaleList
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.provider.Settings
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.Choreographer
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.log10

/**
 * Audio8-TTS-Preview-0.6b on the phone: type a sentence, pick a voice, tap Speak, and the speech plays from the
 * speaker. Every graph runs on LiteRT from Kotlin ([Audio8Tts] around [Audio8Engine]): slow + fast AR on the CPU
 * (4 threads), codec decoder on the GPU when its output passes a check against the CPU int8 decoder (else CPU int8),
 * codec encoder created for a voice recording only.
 *
 * A normal launch shows the input screen: a voice picker (the model repository's two voices, and "My voice", recorded
 * with "Record my voice (10 s)" into <external files dir>/voices/my_voice/), a text box, Speak and Cancel, and a footer
 * with the device, the runtime, the backends, the load time and the last `generate / audio = RTF`.
 *
 * Model files live in <external files dir>/ (adb push). Launch extras (the recording and check tools of the demo lane;
 * no taps needed; the S26 shows an anti-touch screen when tapped in the dark):
 *   --es text "..."   (normal launch) puts the text in the text box
 *   --ez autorun true --es ref_mode mic|file|bundled --es voice ja_funasr_example --es text_ja "..." --es text_en "..."
 *       --ei seed 42 --ez greedy false --ei delay_ms 3000 --ei gap_ms 2500 [--ei rest_ms <gap_ms>] [--es ref_wav ref_ja_44k.wav]
 *       rest_ms = the pause between the end of the registration and the first GENERATING (default gap_ms; 0 = none),
 *       so the S26's CPU frequency cap can lift before the timed segments; the pill shows REGISTERED (green) meanwhile
 *   --ez kv_probe true [--ei probe_steps 200]            dummy decode / fast steps, ms per step
 *   --ez parity true --es text "..." --es voice ja_funasr_example [--es tag ja] [--es teacher greedy_ja.json]
 *       greedy generation with the bundled voice codes; with a teacher file the Python greedy sequence is forced and
 *       the app records its own pick + logits at every position (teacher-forced comparison)
 *   --ez codec_probe true [--es teacher greedy_ja.json]   GPU codec variants vs the CPU int8 codec
 *   engine variants (any mode): --es kv_owner output|input, --ez weight_cache true, --ei threads 4,
 *       --es codec auto|gpu|cpu (auto = GPU fp16 only if its output matches the CPU int8 codec, corr >= 0.99; without
 *       the extra a launch uses auto, or cpu once the check has failed on this OS build and LiteRT version)
 * Every number on screen is measured by this app (SystemClock.elapsedRealtimeNanos). Each launch writes
 * <external files dir>/Documents/audio8-demo-<epoch>.json; logs use the tag Audio8Demo.
 */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    @Volatile private var tts: Audio8Tts? = null
    @Volatile private var engine: Audio8Engine? = null
    @Volatile private var recording = false
    private var track: AudioTrack? = null
    private var loadMs = 0.0
    private var mode = "interactive"

    // Run record: touched on the worker thread only (UI-thread fields are handed over through worker.execute).
    private val startEpochS = System.currentTimeMillis() / 1000
    private val report = JSONObject()
    private val segments = JSONArray()
    private lateinit var reportFile: File
    private lateinit var docs: File
    private lateinit var modelDir: File
    private lateinit var prefs: SharedPreferences

    private lateinit var pill: TextView
    private lateinit var body: TextView
    private lateinit var bar: PlayBar
    private lateinit var progress: TextView
    private lateinit var stats: TextView
    private lateinit var footer: TextView

    // Input screen (normal launch). UI thread only, except where marked @Volatile.
    private lateinit var inputPanel: LinearLayout
    private lateinit var voiceGroup: RadioGroup
    private val voiceButtons = LinkedHashMap<String, RadioButton>()
    private lateinit var recordButton: Button
    private lateinit var textInput: EditText
    private lateinit var tokenLine: TextView
    private lateinit var message: TextView
    private lateinit var buttonRow: LinearLayout
    private lateinit var speakButton: Button
    private lateinit var cancelButton: Button

    /** What the input screen is doing: null (idle), "speak", "record", "register" or "play". */
    private var phase: String? = null

    /** The cancel flag of the running Speak; Cancel sets it, the engine reads it once per frame. */
    private var job = AtomicBoolean(false)
    @Volatile private var keepRecording = true
    @Volatile private var genFrames = 0
    private var spoken = 0
    private var lastSpeech: Audio8Tts.Speech? = null

    private var tickerStartNs = 0L
    private var tickerText: ((Double) -> String)? = null
    private val ticker = object : Runnable {
        override fun run() {
            val f = tickerText ?: return
            progress.text = f((SystemClock.elapsedRealtimeNanos() - tickerStartNs) / 1e9)
            main.postDelayed(this, 100)
        }
    }

    private val deviceName: String
        get() = if (Build.MODEL.startsWith("SM-S942")) "Galaxy S26" else Build.MODEL

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Show over the lock screen and keep the display on: a locked phone parks a hidden activity's process in the
        // background cpuset (little cores), which runs the same model about 10x slower.
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        mode = modeOf(intent)
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        // Creates <external files dir>/ on the first launch: the folder adb push writes the model files into.
        modelDir = getExternalFilesDir(null)!!
        // The voice folders adb push fills are made here, so they belong to the app: a folder adb creates belongs to
        // the shell user, and the app may not be allowed to open it.
        for (id in Audio8Tts.BUNDLED_VOICES) Audio8Tts.voiceDir(modelDir, id).mkdirs()
        docs = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)!!
        reportFile = File(docs, "audio8-demo-$startEpochS.json")
        setContentView(buildUi())
        // The status bar stays visible (its airplane icon is the offline evidence); light icons on the dark background.
        if (Build.VERSION.SDK_INT >= 30) {
            window.insetsController?.setSystemBarsAppearance(0,
                WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS)
        }
        setPill("LOADING", C_IDLE)
        footer.text = deviceName
        worker.execute { prepare() }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        Log.i(TAG, "onNewIntent ignored (one run per launch): ${intent.extras?.keySet()}")
    }

    override fun onDestroy() {
        recording = false
        keepRecording = false
        job.set(true)
        track?.release()
        track = null
        worker.execute {
            tts?.close()
            tts = null
            engine = null
        }
        worker.shutdown()
        super.onDestroy()
    }

    private fun modeOf(i: Intent): String = when {
        i.getBooleanExtra("kv_probe", false) -> "kv_probe"
        i.getBooleanExtra("codec_probe", false) -> "codec_probe"
        i.getBooleanExtra("parity", false) -> "parity"
        i.getBooleanExtra("autorun", false) -> "autorun"
        else -> "interactive"
    }

    // ---- load ----------------------------------------------------------------------------------------------

    private fun prepare() {
        val threads = intent.getIntExtra("threads", 4)
        val kvOwner = intent.getStringExtra("kv_owner") ?: "output"
        val weightCache = if (intent.getBooleanExtra("weight_cache", false)) File(cacheDir, "xnnpack_slow_ar_int8.cache").path else null
        val codecMode = intent.getStringExtra("codec") ?: Audio8Tts.rememberedCodecMode(this, modelDir)
        report.put("app", packageName).put("started_epoch_s", startEpochS).put("mode", mode)
            .put("extras", JSONObject().apply { intent.extras?.let { b -> for (k in b.keySet()) put(k, b.get(k).toString()) } })
            .put("device", JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
                .put("device", Build.DEVICE).put("android", Build.VERSION.RELEASE).put("sdk_int", Build.VERSION.SDK_INT)
                .put("soc", if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else JSONObject.NULL)
                .put("shown_as", deviceName).put("airplane_mode_on", airplaneOn()))
            .put("runtime", JSONObject().put("litert", BuildConfig.LITERT_VERSION).put("api", "CompiledModel (Kotlin)")
                .put("ar_backend", "CPU").put("threads", threads).put("kv_owner", kvOwner)
                .put("slow_weight_cache", weightCache ?: JSONObject.NULL).put("gpu_cache_dir", cacheDir.path)
                .put("codec_mode", codecMode)
                .put("gpu_codec_failure_remembered", Audio8Tts.rememberedGpuFailure(this, modelDir) ?: JSONObject.NULL))
            .put("files", JSONObject().apply {
                for (n in Audio8Tts.REQUIRED_FILES + Audio8Tts.OPTIONAL_FILES) {
                    val f = File(modelDir, n)
                    put(n, if (f.isFile) f.length() else JSONObject.NULL)
                }
            })
            .put("segments", segments)
        report.put("thermal_at_load", thermal())
        writeReport()
        val missing = Audio8Tts.missingFiles(modelDir)
        if (missing.isNotEmpty()) {
            Log.e(TAG, "MODEL_FILES_MISSING dir=${modelDir.path} missing=$missing")
            report.put("error", "model files missing: $missing")
            writeReport()
            ui { showMissingFiles(missing) }
            return
        }
        try {
            val constants = PromptConstants(assets.open("prompt_constants.json").bufferedReader().readText())
            val t = Audio8Tts.load(modelDir, cacheDir, constants, threads, codecMode, kvOwner, weightCache)
            Audio8Tts.rememberCodecCheck(this, t)
            val e = t.engine
            tts = t
            engine = e
            loadMs = t.loadSeconds * 1000
            val selfTest = t.tokenizerSelfTest()
            report.put("load", JSONObject().put("total_ms", loadMs).put("tokenizer_load_ms", e.tokenizerLoadMs)
                .put("create_ms", JSONObject(e.createMs as Map<*, *>)).put("warmup_ms", JSONObject(e.warmupMs as Map<*, *>))
                .put("codec_backend", e.codecBackend).put("gpu_error", e.gpuError ?: JSONObject.NULL)
                .put("gpu_codec_check", e.gpuCheck?.let { JSONObject(it as Map<*, *>) } ?: JSONObject.NULL)
                .put("vm", vm()).put("thermal_after", thermal()))
            report.put("tokenizer_self_test", selfTest)
            report.getJSONObject("runtime").put("codec_backend", e.codecBackend)
            writeReport()
            Log.i(TAG, "READY load_ms=$loadMs create_ms=${e.createMs} warmup_ms=${e.warmupMs} codec=${e.codecBackend} " +
                "codec_mode=${e.codecMode} self_test=${selfTest.getBoolean("pass")} vm=${vm()}")
            if (!selfTest.getBoolean("pass")) {
                ui { setPill("ERROR", C_LIVE); showError("tokenizer self-test failed"); progress.text = selfTest.toString() }
                return
            }
            ui {
                footer.text = if (mode == "interactive") interactiveFooter(t) else footerText(e)
                setPill("READY", C_IDLE)
                pill.post { recordLayout() }
                if (mode == "interactive") onReady()
            }
            when (mode) {
                "kv_probe" -> runKvProbe(e)
                "codec_probe" -> runCodecProbe(e)
                "parity" -> runParity(e)
                "autorun" -> ui { startAutorun() }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "load failed", t)
            report.put("error", t.toString())
            writeReport()
            ui { setPill("ERROR", C_LIVE); showError(t.message ?: t.toString()) }
        }
    }

    /** Two explicit lines (the break never falls inside a phrase): device, network state, backends / runtime, load time. */
    private fun footerText(e: Audio8Engine): String = String.format(Locale.US, "%s · airplane mode %s · %s\nLiteRT %s · models loaded in %.1f s",
        deviceName, if (airplaneOn()) "on" else "off",
        if (e.codecBackend == "CPU") "CPU only, ${e.threads} threads" else "CPU ${e.threads} threads + ${e.codecBackend} codec",
        BuildConfig.LITERT_VERSION, loadMs / 1000)

    /** The input screen's footer: device and runtime, the backend of each graph, the last Speak's times. */
    private fun interactiveFooter(t: Audio8Tts): String {
        val last = lastSpeech
        return String.format(Locale.US, "%s · LiteRT %s · models loaded in %.1f s\nslow + fast AR: CPU %d threads · codec: %s\n%s",
            deviceName, BuildConfig.LITERT_VERSION, t.loadSeconds, t.threads,
            if (t.codecBackend == "GPU") "GPU fp16" else "CPU int8",
            if (last == null) "generate – / audio – = RTF –"
            else String.format(Locale.US, "generate %.1f s / audio %.1f s = RTF %.2f", last.generateSeconds, last.audioSeconds, last.rtf))
    }

    /** In the probe and autorun modes the message goes into the large text line, as before. */
    private fun showError(text: String) {
        if (mode == "interactive") showMessage(text, C_ERROR) else body.text = text
    }

    // ---- kv probe ------------------------------------------------------------------------------------------

    private fun runKvProbe(e: Audio8Engine) {
        val steps = intent.getIntExtra("probe_steps", 200)
        ui { setPill("KV PROBE", C_WORK); progress.text = "$steps decode + $steps fast steps" }
        val th0 = thermal()
        val r = e.kvProbe(steps)
        val o = JSONObject().put("steps", steps).put("thermal_before", th0).put("thermal_after", thermal())
            .put("kv_owner", e.kvOwner).put("slow_weight_cache", e.slowWeightCache ?: JSONObject.NULL).put("threads", e.threads)
            .put("slow_decode_total", dist(r.decodeTotal)).put("slow_decode_run_only", dist(r.decodeRun))
            .put("fast_step_total", dist(r.fastTotal)).put("fast_step_run_only", dist(r.fastRun))
            .put("reference_rows", JSONObject().put("slow_decode_ms", 12.3).put("fast_step_ms", 0.97)
                .put("source", "benchmark_model S26 CPU 4t (NOTES.md clean legs)"))
            .put("vm", vm())
        report.put("kv_probe", o)
        writeReport()
        val d = o.getJSONObject("slow_decode_total")
        val f = o.getJSONObject("fast_step_total")
        Log.i(TAG, "KV_PROBE_DONE decode_median_ms=${d.getDouble("median")} decode_run_median_ms=${o.getJSONObject("slow_decode_run_only").getDouble("median")} " +
            "fast_median_ms=${f.getDouble("median")} fast_run_median_ms=${o.getJSONObject("fast_step_run_only").getDouble("median")} json=${reportFile.path}")
        ui {
            setPill("DONE", C_DONE)
            stats.text = String.format(Locale.US, "slow decode %.2f ms/step · fast step %.2f ms/step (median of %d)",
                d.getDouble("median"), f.getDouble("median"), steps)
        }
    }

    // ---- codec probe ---------------------------------------------------------------------------------------

    /** The GPU codec variants against the CPU int8 codec on the frames of a greedy reference file (<= 128 frames). */
    private fun runCodecProbe(e: Audio8Engine) {
        val t = loadTeacher(File(modelDir, intent.getStringExtra("teacher") ?: "greedy_ja.json"))
        ui { setPill("CODEC PROBE", C_WORK) }
        val rows = e.codecProbe(t.codes)
        val arr = JSONArray()
        for (r in rows) arr.put(JSONObject().put("variant", r.variant).put("create_ms", r.createMs)
            .put("run_ms", JSONArray(r.runMs)).put("corr_vs_cpu_int8", r.corrVsCpu.takeIf { !it.isNaN() } ?: JSONObject.NULL)
            .put("corr_ja_vs_zero_input", r.corrJaVsZeros.takeIf { !it.isNaN() } ?: JSONObject.NULL)
            .put("rms", r.rms.takeIf { !it.isNaN() } ?: JSONObject.NULL).put("input_buffer", r.inTypes).put("output_buffer", r.outTypes)
            .put("error", r.error ?: JSONObject.NULL))
        report.put("codec_probe", JSONObject().put("frames", t.codes[0].size).put("rows", arr).put("thermal", thermal()).put("vm", vm()))
        writeReport()
        Log.i(TAG, "CODEC_PROBE_DONE json=${reportFile.path}")
        ui { setPill("DONE", C_DONE) }
    }

    // ---- parity --------------------------------------------------------------------------------------------

    private fun runParity(e: Audio8Engine) {
        val text = intent.getStringExtra("text") ?: SCRIPT_JA
        val tag = intent.getStringExtra("tag") ?: if (isJapanese(text)) "ja" else "en"
        val voice = intent.getStringExtra("voice") ?: "ja_funasr_example"
        val (refText, refCodes) = loadVoice(voice)
        val teacher = intent.getStringExtra("teacher")?.let { loadTeacher(File(modelDir, it)) }
        ui {
            setPill("GENERATING", C_WORK)
            body.textLocale = if (tag == "ja") Locale.JAPANESE else Locale.US
            body.text = text
        }
        val th0 = thermal()
        val g = e.generate(text, refText, refCodes, Sampler(greedy = true, seed = 0), record = true, teacher = teacher,
            dumpLogits = teacher != null) { n, s -> ui { progress.text = String.format(Locale.US, "frame %d · %.1f s", n, s / 1000) } }
        val info = LinkedHashMap<String, Any>()
        val calls = ArrayList<Audio8Engine.CodecCall>()
        val wav = e.decodeAudio(g.frames, info, calls)
        val wavFile = File(docs, "audio8-demo-$startEpochS-parity-$tag${if (teacher != null) "-teacher" else ""}.wav")
        Wav.writeMono16(wavFile, Wav.floatToPcm16(wav), Audio8Engine.SR)
        val p = g.prompt
        val md = MessageDigest.getInstance("SHA-256")
        val bb = ByteBuffer.allocate(8 * (p.size - 1) * p[0].size).order(ByteOrder.LITTLE_ENDIAN)
        for (r in 1 until p.size) for (v in p[r]) bb.putLong(v.toLong())
        val o = JSONObject().put("text", text).put("tag", tag).put("voice", voice).put("sampler", "greedy (all uniform draws = 0.5)")
            .put("teacher", intent.getStringExtra("teacher") ?: JSONObject.NULL)
            .put("prompt_ids", JSONArray(p[0].toList())).put("prompt_shape", JSONArray(listOf(p.size, p[0].size)))
            .put("prompt_rows_1_10_sha", md.digest(bb.array()).joinToString("") { "%02x".format(it) })
            .put("frames", g.frames.size).put("stopped_by", g.stoppedBy)
            .put("semantic", JSONArray(g.semantic.toList()))
            .put("codes", JSONArray((0 until Audio8Engine.NUM_CB).map { q -> JSONArray(g.frames.map { it[q] }) }))
            .put("sem_margin", JSONArray(g.semMargin!!.map { it.toDouble() }))
            .put("sem_top2_ids", JSONArray(g.semTop2!!.map { JSONArray(it.toList()) }))
            .put("code_margin", JSONArray(g.codeMargin!!.map { r -> JSONArray(r.map { it.toDouble() }) }))
            .put("stats", JSONObject(g.stats as Map<*, *>))
            .put("slow_decode_ms_per_frame", dist(g.decodeMs)).put("fast_10_steps_ms_per_frame", dist(g.fastMs))
            .put("codec", codecJson(info, calls)).put("wav", wavFile.path)
            .put("thermal_before", th0).put("thermal_after", thermal()).put("vm", vm())
        if (teacher != null) {
            o.put("own_semantic", JSONArray(g.ownSemantic!!.toList()))
            o.put("own_codes", JSONArray((0 until Audio8Engine.NUM_CB).map { q -> JSONArray(g.ownCodes!!.map { it[q] }) }))
            val semFile = File(docs, "audio8-demo-$startEpochS-parity-$tag-sem.f32")
            val fastFile = File(docs, "audio8-demo-$startEpochS-parity-$tag-fast.f32")
            writeFloats(semFile, g.semLogitsDump!!)
            writeFloats(fastFile, g.fastLogitsDump!!)
            o.put("sem_logits_file", semFile.path).put("fast_logits_file", fastFile.path)
                .put("sem_logits_rows", g.semLogitsDump.size).put("fast_logits_rows", g.fastLogitsDump.size)
        }
        report.put("parity", o)
        writeReport()
        Log.i(TAG, "PARITY_DONE tag=$tag prompt_len=${p[0].size} frames=${g.frames.size} stopped_by=${g.stoppedBy} " +
            "teacher=${teacher != null} json=${reportFile.path}")
        ui {
            setPill("DONE", C_DONE)
            stats.text = String.format(Locale.US, "greedy · prompt %d ids · %d frames", p[0].size, g.frames.size)
        }
    }

    private fun loadTeacher(f: File): Audio8Engine.Teacher {
        val o = JSONObject(f.readText())
        val sem = o.getJSONArray("semantic").let { a -> IntArray(a.length()) { a.getInt(it) } }
        val c = o.getJSONArray("codes")
        val codes = Array(c.length()) { q -> c.getJSONArray(q).let { a -> IntArray(a.length()) { a.getInt(it) } } }
        return Audio8Engine.Teacher(sem, codes)
    }

    private fun writeFloats(f: File, rows: List<FloatArray>) {
        f.outputStream().buffered(1 shl 20).use { out ->
            for (r in rows) {
                val bb = ByteBuffer.allocate(r.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                bb.asFloatBuffer().put(r)
                out.write(bb.array())
            }
        }
    }

    // ---- autorun -------------------------------------------------------------------------------------------

    private class Sentence(val id: String, val text: String, val locale: Locale)

    private fun startAutorun() {
        val delay = intent.getIntExtra("delay_ms", 3000).toLong()
        val gap = intent.getIntExtra("gap_ms", 2500).toLong()
        val rest = intent.getIntExtra("rest_ms", gap.toInt()).toLong()
        val refMode = intent.getStringExtra("ref_mode") ?: "file"
        val voice = intent.getStringExtra("voice") ?: "ja_funasr_example"
        val sentences = listOf(
            Sentence("ja", intent.getStringExtra("text_ja") ?: SCRIPT_JA, Locale.JAPANESE),
            Sentence("en", intent.getStringExtra("text_en") ?: SCRIPT_EN, Locale.US),
        )
        Log.i(TAG, "AUTORUN_START ref_mode=$refMode voice=$voice delay_ms=$delay gap_ms=$gap rest_ms=$rest")
        worker.execute {
            report.put("autorun", JSONObject().put("ref_mode", refMode).put("voice", voice).put("delay_ms", delay)
                .put("gap_ms", gap).put("rest_ms", rest).put("seed", intent.getIntExtra("seed", 42))
                .put("greedy", intent.getBooleanExtra("greedy", false)))
            writeReport()
        }
        main.postDelayed({
            register(refMode, voice) { refText, refCodes ->
                if (rest <= 0) {
                    speak(sentences, 0, refText, refCodes, gap)
                } else {
                    // The registration is over: the pill must not keep saying REGISTERING while the phone rests.
                    setPill("REGISTERED", C_DONE)
                    val restEpoch = System.currentTimeMillis()
                    worker.execute {
                        report.getJSONObject("autorun").put("rest_started_epoch_ms", restEpoch).put("rest_thermal", thermal())
                        writeReport()
                    }
                    main.postDelayed({ speak(sentences, 0, refText, refCodes, gap) }, rest)
                }
            }
        }, delay)
    }

    /** Reference voice -> (transcript, codes [10][n]); the transcript is the voice's meta.json reference_text. */
    private fun register(refMode: String, voice: String, onDone: (String, Array<IntArray>) -> Unit) {
        val e = engine ?: return
        val voiceDir = File(modelDir, "voices/$voice")
        val refText = JSONObject(File(voiceDir, "meta.json").readText()).getString("reference_text")
        val refWav = File(modelDir, intent.getStringExtra("ref_wav") ?: "ref_ja_44k.wav")
        when (refMode) {
            "bundled" -> worker.execute {
                val (t, c) = loadVoice(voice)
                segments.put(JSONObject().put("id", "register").put("ref_mode", "bundled").put("voice", voice)
                    .put("frames", c[0].size))
                writeReport()
                ui { onDone(t, c) }
            }
            "mic" -> {
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    Log.e(TAG, "MIC needs RECORD_AUDIO (adb shell pm grant $packageName android.permission.RECORD_AUDIO); using file")
                    worker.execute { report.put("mic_fallback", "RECORD_AUDIO not granted"); writeReport() }
                    encodeFile(e, refWav, refText, "file (mic fallback)", null, onDone)
                } else {
                    recordReference(refWav) { rec, row -> encodeFile(e, refWav, refText, "mic", rec to row, onDone) }
                }
            }
            else -> encodeFile(e, refWav, refText, "file", null, onDone)
        }
    }

    private fun encodeFile(e: Audio8Engine, refWav: File, refText: String, mode: String, recorded: Pair<FloatArray, JSONObject>?,
                           onDone: (String, Array<IntArray>) -> Unit) {
        val audio = recorded?.first ?: Wav.monoFloat(refWav).also { require(it.second == Audio8Engine.SR) { "reference wav must be 44.1 kHz" } }.first
        val frames = kotlin.math.ceil(minOf(audio.size, Audio8Engine.ENC_SAMPLES) / Audio8Engine.FRAME.toDouble()).toInt()
        setPill("REGISTERING", C_WORK)
        body.text = ""
        stats.text = ""
        startTicker { s -> String.format(Locale.US, "codec encoder · %d frames · %.1f s", frames, s) }
        worker.execute {
            val row = (recorded?.second ?: JSONObject()).put("id", "register").put("ref_mode", mode)
                .put("ref_wav", refWav.path).put("screen_epoch_ms", System.currentTimeMillis()).put("thermal_before", thermal())
            try {
                val r = e.encodeReference(audio)
                row.put("samples", r.samples).put("frames", r.codes[0].size).put("encoder_create_ms", r.createMs)
                    .put("encode_ms", r.encodeMs).put("encoder_close_ms", r.closeMs).put("thermal_after", thermal()).put("vm", vm())
                segments.put(row)
                writeReport()
                Log.i(TAG, "REGISTERED mode=$mode frames=${r.codes[0].size} create_ms=${r.createMs} encode_ms=${r.encodeMs} " +
                    "close_ms=${r.closeMs} vm=${vm()}")
                ui {
                    stopTicker()
                    progress.text = String.format(Locale.US, "codec encoder · %d frames · %.1f s", r.codes[0].size,
                        (r.createMs + r.encodeMs + r.closeMs) / 1000)
                    onDone(refText, r.codes)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "registration failed", t)
                segments.put(row.put("error", t.toString()))
                writeReport()
                ui { stopTicker(); setPill("ERROR", C_LIVE); body.text = t.message ?: t.toString() }
            }
        }
    }

    private fun speak(list: List<Sentence>, i: Int, refText: String, refCodes: Array<IntArray>, gap: Long) {
        val e = engine ?: return
        val sen = list[i]
        val seed = intent.getIntExtra("seed", 42).toLong()
        val greedy = intent.getBooleanExtra("greedy", false)
        setPill("GENERATING", C_WORK)
        body.textLocale = sen.locale
        body.text = sen.text
        bar.visibility = View.INVISIBLE
        progress.text = ""
        stats.text = ""
        val genStartNs = SystemClock.elapsedRealtimeNanos()
        worker.execute {
            val row = JSONObject().put("id", sen.id).put("text", sen.text).put("seed", seed).put("greedy", greedy)
                .put("gen_screen_epoch_ms", System.currentTimeMillis()).put("thermal_before", thermal())
            try {
                val g = e.generate(sen.text, refText, refCodes, Sampler(greedy, seed)) { n, _ ->
                    val el = (SystemClock.elapsedRealtimeNanos() - genStartNs) / 1e9
                    ui { progress.text = String.format(Locale.US, "frame %d · %.1f s of audio · %.1f s", n, n * FRAME_S, el) }
                }
                val thGen = thermal()
                val decStartNs = SystemClock.elapsedRealtimeNanos()
                ui {
                    setPill("DECODING", C_WORK)
                    startTicker { s -> String.format(Locale.US, "codec · %s · %.1f s", e.codecBackend, s) }
                }
                row.put("decoding_screen_epoch_ms", System.currentTimeMillis())
                val info = LinkedHashMap<String, Any>()
                val calls = ArrayList<Audio8Engine.CodecCall>()
                val wav = e.decodeAudio(g.frames, info, calls)
                val codecMs = (SystemClock.elapsedRealtimeNanos() - decStartNs) / 1e6
                val wavFile = File(docs, "audio8-demo-$startEpochS-${sen.id}.wav")
                val pcm = Wav.floatToPcm16(wav)
                Wav.writeMono16(wavFile, pcm, Audio8Engine.SR)
                val audioS = wav.size / Audio8Engine.SR.toDouble()
                val genMs = g.stats["generate_ms"] as Double
                val prefillMs = g.stats["prefill_ms"] as Double
                val loopMs = g.stats["loop_ms"] as Double
                val rtf = (genMs + codecMs) / 1000 / audioS
                row.put("prompt_len", g.prompt[0].size).put("frames", g.frames.size).put("stopped_by", g.stoppedBy)
                    .put("audio_s", audioS).put("stats", JSONObject(g.stats as Map<*, *>))
                    .put("slow_decode_ms_per_frame", dist(g.decodeMs)).put("fast_10_steps_ms_per_frame", dist(g.fastMs))
                    .put("codec", codecJson(info, calls)).put("codec_ms", codecMs).put("rtf", rtf).put("wav", wavFile.path)
                    .put("thermal_after_generate", thGen).put("thermal_after_codec", thermal()).put("vm", vm())
                Log.i(TAG, "SEGMENT id=${sen.id} prompt_len=${g.prompt[0].size} frames=${g.frames.size} prefill_ms=$prefillMs " +
                    "loop_ms=$loopMs decode_median=${median(g.decodeMs)} fast_median=${median(g.fastMs)} codec_ms=$codecMs " +
                    "backend=${info["codec_backend"]} audio_s=$audioS rtf=$rtf")
                val statsText = String.format(Locale.US, "prefill %.2f s · %d frames in %.1f s · codec %.1f s %s · %.1f s audio · RTF %.2f",
                    prefillMs / 1000, g.frames.size, loopMs / 1000, codecMs / 1000, info["codec_backend"], audioS, rtf)
                ui {
                    stopTicker()
                    progress.text = String.format(Locale.US, "codec · %s · %.1f s", info["codec_backend"], codecMs / 1000)
                    play(sen, pcm, audioS, row, statsText) {
                        if (i + 1 < list.size) {
                            main.postDelayed({ speak(list, i + 1, refText, refCodes, gap) }, gap)
                        } else {
                            worker.execute {
                                report.put("autorun_done_epoch_ms", System.currentTimeMillis()).put("thermal_at_done", thermal()).put("vm_at_done", vm())
                                writeReport()
                                Log.i(TAG, "AUTORUN_DONE json=${reportFile.path}")
                            }
                        }
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "synthesis failed for ${sen.id}", t)
                segments.put(row.put("error", t.toString()))
                writeReport()
                ui { stopTicker(); setPill("ERROR", C_LIVE); body.text = t.message ?: t.toString() }
            }
        }
    }

    // ---- input screen (normal launch) -------------------------------------------------------------------------

    private fun onReady() {
        intent.getStringExtra("text")?.let { if (textInput.text.isEmpty()) textInput.setText(it) }
        setPhase(null)
        refreshVoices()
        updateTokenLine()
    }

    private fun showMissingFiles(missing: List<String>) {
        setPill("ERROR", C_LIVE)
        showMessage("Model files missing in ${modelDir.path}:\n" + missing.joinToString("\n") { "  $it" } +
            "\nPush them with adb (README, step 3), then open the app again.", C_ERROR)
        setPhase(null)
    }

    /** Enables what can be used in [p] (null = idle): Cancel stops a Speak, a recording or a playback. */
    private fun setPhase(p: String?) {
        phase = p
        val ready = tts != null
        speakButton.isEnabled = ready && p == null
        cancelButton.isEnabled = p == "speak" || p == "record" || p == "play"
        recordButton.isEnabled = ready && (p == null || p == "record")
        recordButton.text = if (p == "record") "Stop and save" else RECORD_LABEL
        textInput.isEnabled = p == null
        for ((id, b) in voiceButtons) b.isEnabled = p == null && Audio8Tts.hasVoice(modelDir, id)
        for (v in listOf(speakButton, cancelButton, recordButton)) v.alpha = if (v.isEnabled) 1f else 0.35f
    }

    /** Marks each voice installed or not and checks the remembered one (else the first installed one). */
    private fun refreshVoices() {
        for ((id, b) in voiceButtons) {
            val installed = Audio8Tts.hasVoice(modelDir, id)
            b.text = voiceLabel(id) + when {
                id == Audio8Tts.MY_VOICE -> if (installed) " (recorded on this phone)" else " (not recorded yet)"
                installed -> ""
                else -> " (not installed)"
            }
        }
        val wanted = prefs.getString(PREF_VOICE, null) ?: if (Locale.getDefault().language == "ja") JA_VOICE else EN_VOICE
        val pick = (listOf(wanted) + voiceButtons.keys).firstOrNull { Audio8Tts.hasVoice(modelDir, it) }
        if (pick != null) voiceButtons.getValue(pick).isChecked = true
    }

    private fun selectedVoice(): String? =
        voiceButtons.entries.firstOrNull { it.value.isChecked && Audio8Tts.hasVoice(modelDir, it.key) }?.key

    private fun updateTokenLine() {
        val t = tts ?: return
        when (val c = t.checkText(textInput.text.toString())) {
            is Audio8Tts.TextCheck.Ok -> {
                tokenLine.setTextColor(C_SUB)
                tokenLine.text = "${c.tokens} / ${TextRules.MAX_TEXT_TOKENS} tokens"
            }
            is Audio8Tts.TextCheck.TooLong -> {
                tokenLine.setTextColor(C_ERROR)
                tokenLine.text = "${if (c.truncated) "more than " else ""}${c.tokens} / ${TextRules.MAX_TEXT_TOKENS} tokens: too long"
            }
            else -> {
                tokenLine.setTextColor(C_SUB)
                tokenLine.text = "0 / ${TextRules.MAX_TEXT_TOKENS} tokens"
            }
        }
    }

    private fun onSpeak() {
        val t = tts ?: return
        if (phase != null) return
        hideKeyboard()
        val text = textInput.text.toString()
        val textCheck = t.checkText(text)
        if (textCheck !is Audio8Tts.TextCheck.Ok) {
            showMessage(textCheck.message ?: "", C_ERROR)
            return
        }
        val voiceId = selectedVoice() ?: run {
            showMessage("No voice installed: push voices/ (README) or record your voice.", C_ERROR)
            return
        }
        val seed = intent.getIntExtra("seed", Audio8Tts.DEFAULT_SEED.toInt()).toLong()
        val greedy = intent.getBooleanExtra("greedy", false)
        val cancel = AtomicBoolean(false)
        job = cancel
        spoken++
        val id = "speak-$spoken"
        val wavFile = File(docs, "audio8-$startEpochS-$spoken.wav")
        setPhase("speak")
        showMessage("")
        setPill("GENERATING", C_WORK)
        bar.visibility = View.INVISIBLE
        stats.text = ""
        genFrames = 0
        startTicker { s -> String.format(Locale.US, "frame %d · %.1f s of audio · %.1f s", genFrames, genFrames * FRAME_S, s) }
        reveal(progress)
        worker.execute {
            val row = JSONObject().put("id", id).put("text", text).put("voice", voiceId).put("seed", seed)
                .put("greedy", greedy).put("tokens", textCheck.tokens).put("gen_screen_epoch_ms", System.currentTimeMillis())
                .put("thermal_before", thermal())
            try {
                val voice = Audio8Tts.loadVoice(modelDir, voiceId)
                val sp = t.speak(text, voice, seed, greedy, cancel, onFrame = { n, _ -> genFrames = n }, onDecoding = {
                    row.put("decoding_screen_epoch_ms", System.currentTimeMillis())
                    ui {
                        if (job === cancel && !cancel.get()) {
                            setPill("DECODING", C_WORK)
                            startTicker { s -> String.format(Locale.US, "codec · %s · %.1f s", t.codecBackend, s) }
                        }
                    }
                })
                wavFile.writeBytes(sp.wavBytes())
                val g = sp.gen
                val prefillMs = g.stats["prefill_ms"] as Double
                val loopMs = g.stats["loop_ms"] as Double
                row.put("prompt_len", g.prompt[0].size).put("frames", g.frames.size).put("stopped_by", g.stoppedBy)
                    .put("audio_s", sp.audioSeconds).put("generate_s", sp.generateSeconds).put("codec_s", sp.codecSeconds)
                    .put("rtf", sp.rtf).put("rms", sp.rms).put("stats", JSONObject(g.stats as Map<*, *>))
                    .put("slow_decode_ms_per_frame", dist(g.decodeMs)).put("fast_10_steps_ms_per_frame", dist(g.fastMs))
                    .put("codec", codecJson(sp.codecInfo, sp.codecCalls)).put("wav", wavFile.path)
                    .put("thermal_after", thermal()).put("vm", vm())
                Log.i(TAG, "SPEAK id=$id voice=$voiceId tokens=${textCheck.tokens} prompt_len=${g.prompt[0].size} " +
                    "frames=${g.frames.size} stopped_by=${g.stoppedBy} prefill_ms=$prefillMs loop_ms=$loopMs " +
                    "codec_ms=${sp.codecSeconds * 1000} backend=${sp.codecInfo["codec_backend"]} generate_s=${sp.generateSeconds} " +
                    "audio_s=${sp.audioSeconds} rtf=${sp.rtf} wav=${wavFile.path}")
                val statsText = String.format(Locale.US, "prefill %.2f s · %d frames in %.1f s · codec %.1f s %s · seed %d",
                    prefillMs / 1000, g.frames.size, loopMs / 1000, sp.codecSeconds, sp.codecInfo["codec_backend"], seed)
                ui {
                    stopTicker()
                    if (cancel.get()) {
                        speakCancelled(row, "cancelled before playback")
                        return@ui
                    }
                    lastSpeech = sp
                    footer.text = interactiveFooter(t)
                    val rtfLine = String.format(Locale.US, "generate %.1f s / audio %.1f s = RTF %.2f",
                        sp.generateSeconds, sp.audioSeconds, sp.rtf)
                    progress.text = rtfLine
                    showMessage(if (g.stoppedBy == "limit") "The speech reached the 512-frame cap (23.8 s) and stops " +
                        "there; split the text." else "Saved Android/${wavFile.path.substringAfter("/Android/")}")
                    setPhase("play")
                    // Playback shows "<s> of audio" while it plays; once it is over the times come back.
                    play(Sentence(id, text, if (isJapanese(text)) Locale.JAPANESE else Locale.US), sp.pcm,
                        sp.audioSeconds, row, statsText) {
                        progress.text = rtfLine
                        setPhase(null)
                        reveal(stats)
                    }
                }
            } catch (c: CancellationException) {
                Log.i(TAG, "SPEAK_CANCELLED id=$id ${c.message}")
                ui { stopTicker(); speakCancelled(row, c.message ?: "cancelled") }
            } catch (e: Throwable) {
                Log.e(TAG, "speak failed for $id", e)
                segments.put(row.put("error", e.toString()))
                writeReport()
                ui {
                    stopTicker()
                    setPill("READY", C_IDLE)
                    showMessage("Failed: ${e.message ?: e}", C_ERROR)
                    setPhase(null)
                }
            }
        }
    }

    private fun speakCancelled(row: JSONObject, why: String) {
        worker.execute { segments.put(row.put("cancelled", why)); writeReport() }
        setPill("READY", C_IDLE)
        progress.text = String.format(Locale.US, "%s · %d frames", why, genFrames)
        showMessage("Cancelled.")
        setPhase(null)
    }

    private fun onCancel() {
        when (phase) {
            "speak" -> {
                job.set(true)
                showMessage("Cancelling…")
            }
            "record" -> {
                keepRecording = false
                recording = false
            }
            "play" -> {
                val t = track ?: return
                track = null
                runCatching { t.stop() }
                t.release()
                bar.set(0f, C_IDLE)
                setPill("READY", C_IDLE)
                showMessage("Playback stopped.")
                setPhase(null)
            }
        }
    }

    private fun onRecord() {
        if (phase == "record") {
            recording = false   // "Stop and save": the take so far is registered
            return
        }
        if (phase != null || tts == null) return
        hideKeyboard()
        when (val problem = Audio8Tts.recordingProblem(this, modelDir)) {
            null -> recordMyVoice()
            Audio8Tts.MIC_DENIED -> requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)
            else -> showMessage(problem, C_ERROR)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQ_MIC) return
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            recordMyVoice()
        } else {
            Log.i(TAG, "RECORD_AUDIO denied: recording is off, the bundled voices stay available")
            showMessage(Audio8Tts.MIC_DENIED, C_ERROR)
        }
    }

    /**
     * Records up to 10.03 s (the encoder's input) from the microphone (VOICE_RECOGNITION, 44.1 kHz mono) while the
     * screen shows the sentence to read; "Stop and save" ends the take early, Cancel drops it. The sentence becomes
     * the voice's transcript, so it must be read as written.
     */
    private fun recordMyVoice() {
        val t = tts ?: return
        val script = if (isJapanese(textInput.text.toString())) MY_VOICE_SCRIPT_JA else MY_VOICE_SCRIPT_EN
        recording = true
        keepRecording = true
        setPhase("record")
        setPill("● RECORDING", C_LIVE)
        showMessage("Read aloud, as written:\n$script", C_TEXT)
        stats.text = ""
        bar.set(0f, C_LIVE)
        bar.visibility = View.VISIBLE
        val maxS = Audio8Engine.ENC_SAMPLES / Audio8Engine.SR.toDouble()
        startTicker { s -> String.format(Locale.US, "recording · %.1f s of %.0f s", minOf(s, maxS), maxS) }
        Thread {
            val pcm = ByteArrayOutputStream()
            var failure: Throwable? = null
            try {
                val minBuf = AudioRecord.getMinBufferSize(Audio8Engine.SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
                val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, Audio8Engine.SR,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, Audio8Engine.SR))
                try {
                    check(rec.state == AudioRecord.STATE_INITIALIZED) { "the microphone did not open" }
                    val buf = ByteArray(Audio8Engine.SR / 10 * 2)
                    rec.startRecording()
                    while (recording && pcm.size() < Audio8Engine.ENC_SAMPLES * 2) {
                        val n = rec.read(buf, 0, minOf(buf.size, Audio8Engine.ENC_SAMPLES * 2 - pcm.size()))
                        if (n < 0) break
                        pcm.write(buf, 0, n)
                        // Level bar: the block's (100 ms) peak in dBFS mapped linearly, -50 dBFS -> empty, -5 dBFS -> full.
                        var pk = 0.0
                        val sb = ByteBuffer.wrap(buf, 0, n).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        for (k in 0 until sb.limit()) pk = maxOf(pk, abs(sb.get(k) / 32768.0))
                        val level = ((20 * log10(pk) + 50) / 45).coerceIn(0.0, 1.0).toFloat()
                        ui { bar.fraction = level }
                    }
                } finally {
                    runCatching { rec.stop() }
                    rec.release()
                }
            } catch (e: Throwable) {
                Log.e(TAG, "recording failed", e)
                failure = e
            }
            recording = false
            val sb = ByteBuffer.wrap(pcm.toByteArray()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            val x = FloatArray(sb.limit()) { sb.get(it) / 32768f }
            Log.i(TAG, "MY_VOICE_RECORDED samples=${x.size} keep=$keepRecording error=$failure")
            ui {
                stopTicker()
                bar.visibility = View.INVISIBLE
                when {
                    failure != null -> recordingEnded("Recording failed: ${failure.message ?: failure}", C_ERROR)
                    !keepRecording -> recordingEnded("Recording cancelled.", C_SUB)
                    else -> registerMyVoice(t, x, script)
                }
            }
        }.start()
    }

    private fun recordingEnded(text: String, color: Int) {
        setPill("READY", C_IDLE)
        showMessage(text, color)
        setPhase(null)
    }

    /** Trims and normalizes the take ([ReferenceAudio]), encodes it and saves it as voices/my_voice, then selects it. */
    private fun registerMyVoice(t: Audio8Tts, raw: FloatArray, script: String) {
        val audio = ReferenceAudio.normalize(ReferenceAudio.trimSilence(raw, Audio8Engine.SR))
        if (audio.size < Audio8Engine.SR) {
            recordingEnded("Too short: read the whole sentence, then tap Stop and save.", C_ERROR)
            return
        }
        setPhase("register")
        setPill("REGISTERING", C_WORK)
        val frames = kotlin.math.ceil(minOf(audio.size, Audio8Engine.ENC_SAMPLES) / Audio8Engine.FRAME.toDouble()).toInt()
        startTicker { s -> String.format(Locale.US, "codec encoder · %d frames · %.1f s", frames, s) }
        worker.execute {
            val row = JSONObject().put("id", "register-my-voice").put("transcript", script)
                .put("recorded_s", raw.size / Audio8Engine.SR.toDouble()).put("kept_s", audio.size / Audio8Engine.SR.toDouble())
                .put("thermal_before", thermal())
            try {
                val (voice, r) = t.registerVoice(audio, script)
                row.put("frames", voice.frames).put("encoder_create_ms", r.createMs).put("encode_ms", r.encodeMs)
                    .put("encoder_close_ms", r.closeMs).put("thermal_after", thermal()).put("vm", vm())
                segments.put(row)
                writeReport()
                Log.i(TAG, "REGISTERED my_voice frames=${voice.frames} create_ms=${r.createMs} encode_ms=${r.encodeMs} " +
                    "close_ms=${r.closeMs} vm=${vm()}")
                ui {
                    stopTicker()
                    progress.text = String.format(Locale.US, "codec encoder · %d frames · %.1f s", voice.frames,
                        (r.createMs + r.encodeMs + r.closeMs) / 1000)
                    setPill("REGISTERED", C_DONE)
                    prefs.edit().putString(PREF_VOICE, Audio8Tts.MY_VOICE).apply()
                    setPhase(null)
                    refreshVoices()
                    showMessage(String.format(Locale.US, "Your voice is saved (%.1f s) in voices/my_voice and selected.",
                        voice.seconds))
                }
            } catch (e: Throwable) {
                Log.e(TAG, "registration failed", e)
                segments.put(row.put("error", e.toString()))
                writeReport()
                ui { stopTicker(); recordingEnded("Registration failed: ${e.message ?: e}", C_ERROR) }
            }
        }
    }

    private fun showMessage(text: String, color: Int = C_SUB) {
        message.setTextColor(color)
        message.text = text
        message.visibility = if (text.isEmpty()) View.GONE else View.VISIBLE
        if (text.isNotEmpty()) reveal(message)
    }

    private fun hideKeyboard() {
        getSystemService(InputMethodManager::class.java)?.hideSoftInputFromWindow(textInput.windowToken, 0)
        textInput.clearFocus()
    }

    private fun voiceLabel(id: String): String = when (id) {
        JA_VOICE -> "Japanese sample voice"
        EN_VOICE -> "English sample voice"
        else -> "My voice"
    }

    /** Scrolls the input screen until [v] is in view: with the keyboard up the content is taller than the screen. */
    private fun reveal(v: View) {
        v.post { v.requestRectangleOnScreen(Rect(0, 0, v.width, v.height), false) }
    }

    // ---- playback (the Fun-ASR demo's PLAYING-at-first-output logic) --------------------------------------------

    private fun play(sen: Sentence, pcm: ByteArray, audioS: Double, row: JSONObject, statsText: String, onDone: () -> Unit) {
        try {
            val t = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(Audio8Engine.SR).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STATIC)
                .setBufferSizeInBytes(pcm.size)
                .build()
            track = t
            check(t.write(pcm, 0, pcm.size) == pcm.size) { "AudioTrack took a partial buffer" }
            Playback(sen, t, pcm.size / 2, Audio8Engine.SR, audioS, row, statsText, onDone).start()
        } catch (t: Throwable) {
            Log.e(TAG, "playback failed for ${sen.id}", t)
            track?.release()
            track = null
            worker.execute { segments.put(row.put("play_error", t.toString())); writeReport() }
            setPill("ERROR", C_LIVE)
            showError(t.message ?: t.toString())
            if (mode == "interactive") setPhase(null)
        }
    }

    /**
     * Follows one playback every display frame. The AudioTrack timestamp gives the time frame 0 left the speaker, so
     * the PLAYING screen goes up in the first frame after the sound has started, the bar follows the output position,
     * and the playback ends once the last sample has left the speaker. Fallbacks, recorded in the row: no timestamp
     * within 0.5 s (screen up anyway; end = playback head at the last frame + 200 ms), or the clip length + 3 s.
     */
    private inner class Playback(
        val sen: Sentence, val t: AudioTrack, val frames: Int, val rate: Int, val audioS: Double, val row: JSONObject,
        val statsText: String, val onDone: () -> Unit,
    ) : Choreographer.FrameCallback {
        private val ts = AudioTimestamp()
        private val durNs = frames * 1_000_000_000L / rate
        private var playNs = 0L
        private var firstOutNs = 0L
        private var tsFrame = 0L
        private var screenNs = 0L
        private var headEndNs = 0L

        fun start() {
            row.put("play_started_epoch_ms", System.currentTimeMillis())
            playNs = System.nanoTime()
            t.play()
            Choreographer.getInstance().postFrameCallback(this)
        }

        override fun doFrame(frameTimeNanos: Long) {
            if (track !== t) return
            val now = System.nanoTime()
            if (t.getTimestamp(ts) && ts.framePosition > tsFrame && ts.framePosition <= frames) {
                tsFrame = ts.framePosition
                firstOutNs = ts.nanoTime - ts.framePosition * 1_000_000_000L / rate
            }
            if (screenNs == 0L && (firstOutNs > 0 || now - playNs >= 500_000_000L)) {
                screenNs = now
                bar.set(0f, C_LIVE)
                bar.visibility = View.VISIBLE
                progress.text = String.format(Locale.US, "%.1f s of audio", audioS)
                setPill("● PLAYING", C_LIVE)
            }
            val head = t.playbackHeadPosition
            if (head >= frames - rate / 50 && headEndNs == 0L) headEndNs = now
            if (screenNs > 0) {
                val played = if (firstOutNs > 0) (now - firstOutNs).toDouble() / durNs else head.toDouble() / frames
                bar.fraction = played.coerceIn(0.0, 1.0).toFloat()
            }
            val end = when {
                screenNs == 0L || now < screenNs + durNs -> null
                firstOutNs > 0 && headEndNs > 0 && now >= firstOutNs + durNs -> "output_timestamp"
                firstOutNs == 0L && headEndNs > 0 && now >= headEndNs + 200_000_000L -> "head_plus_200ms"
                now - playNs >= durNs + 3_000_000_000L -> "timeout"
                else -> null
            }
            if (end == null) {
                Choreographer.getInstance().postFrameCallback(this)
                return
            }
            t.release()
            track = null
            fun ms(ns: Long): Any = if (ns > 0) (ns - playNs) / 1e6 else JSONObject.NULL
            row.put("play_ms", (now - playNs) / 1e6).put("play_end", end).put("play_frames", frames)
                .put("play_head_final", head).put("play_rate_hz", rate).put("play_output_start_ms", ms(firstOutNs))
                .put("play_output_end_ms", if (firstOutNs > 0) ms(firstOutNs + durNs) else JSONObject.NULL)
                .put("play_screen_ms", ms(screenNs)).put("play_head_end_ms", ms(headEndNs)).put("play_ts_last_frame", tsFrame)
                .put("done_screen_epoch_ms", System.currentTimeMillis())
            Log.i(TAG, "PLAYED id=${sen.id} play_ms=${(now - playNs) / 1e6} end=$end output_start_ms=${ms(firstOutNs)} " +
                "screen_ms=${ms(screenNs)} head_end_ms=${ms(headEndNs)} head=$head frames=$frames rate=$rate")
            bar.set(1f, C_IDLE)
            setPill("DONE", C_DONE)
            stats.text = statsText
            worker.execute { segments.put(row); writeReport() }
            onDone()
        }
    }

    // ---- mic reference ---------------------------------------------------------------------------------------

    /**
     * Plays the reference clip through the speaker while AudioRecord (VOICE_RECOGNITION, 44.1 kHz mono) captures it,
     * then peak-normalises the recording to -3 dBFS and saves it as Documents/audio8-demo-<epoch>-ref_recorded.wav.
     * The pill says MIC ON (blue) while the mic runs before the clip is heard, and turns to ● RECORDING (red) in the
     * first display frame after the clip's AudioTrack timestamp says its first sample has left the speaker ([ClipWatch],
     * the rule of [Playback]), so make_media.sh can lay the clip at the first red frame of this segment.
     */
    private fun recordReference(clip: File, onDone: (FloatArray, JSONObject) -> Unit) {
        val clipPcm = Wav.pcm16(clip)
        val clipFrames = clipPcm.data.size / (2 * clipPcm.channels)
        val clipS = clipFrames / clipPcm.sampleRate.toDouble()
        val row = JSONObject().put("clip", clip.path).put("clip_s", clipS)
        recording = true
        setPill("MIC ON", C_WORK)
        val micOnEpoch = System.currentTimeMillis()
        body.text = ""
        bar.set(0f, C_LIVE)
        bar.visibility = View.VISIBLE
        startTicker { s -> String.format(Locale.US, "reference voice · mic · %.1f s", s) }
        val recordMs = (clipS * 1000).toLong() + 800
        Thread {
            val pcm = ByteArrayOutputStream()
            var player: AudioTrack? = null
            var watch: ClipWatch? = null
            row.put("mic_on_screen_epoch_ms", micOnEpoch)
            try {
                val minBuf = AudioRecord.getMinBufferSize(Audio8Engine.SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
                val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, Audio8Engine.SR,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, Audio8Engine.SR))
                val buf = ByteArray(Audio8Engine.SR / 10 * 2)
                rec.startRecording()
                row.put("record_started_epoch_ms", System.currentTimeMillis())
                val t0 = SystemClock.elapsedRealtime()
                try {
                    while (recording && SystemClock.elapsedRealtime() - t0 < recordMs) {
                        if (player == null && SystemClock.elapsedRealtime() - t0 >= 300) {
                            val p = AudioTrack.Builder()
                                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                                .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                    .setSampleRate(clipPcm.sampleRate)
                                    .setChannelMask(if (clipPcm.channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO).build())
                                .setTransferMode(AudioTrack.MODE_STATIC).setBufferSizeInBytes(clipPcm.data.size).build()
                            player = p
                            p.write(clipPcm.data, 0, clipPcm.data.size)
                            val w = ClipWatch(p, clipFrames, clipPcm.sampleRate)
                            watch = w
                            row.put("clip_play_epoch_ms", System.currentTimeMillis())
                            w.playNs = System.nanoTime()
                            p.play()
                            ui { w.start() }
                        }
                        val n = rec.read(buf, 0, buf.size)
                        if (n < 0) break
                        pcm.write(buf, 0, n)
                        // Level bar: the block's (100 ms) peak in dBFS mapped linearly, -50 dBFS -> empty, -5 dBFS -> full.
                        var pk = 0.0
                        val sb = ByteBuffer.wrap(buf, 0, n).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        for (k in 0 until sb.limit()) pk = maxOf(pk, abs(sb.get(k) / 32768.0))
                        val level = ((20 * log10(pk) + 50) / 45).coerceIn(0.0, 1.0).toFloat()
                        ui { bar.fraction = level }
                    }
                } finally {
                    rec.stop()
                    rec.release()
                    row.put("record_end_epoch_ms", System.currentTimeMillis())
                    synchronized(clipLock) {
                        watch?.let { w -> w.released = true; w.putInto(row) }
                        player?.release()
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "recording failed", t)
                row.put("record_error", t.toString())
            }
            recording = false
            val bytes = pcm.toByteArray()
            val sb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            val x = FloatArray(sb.limit()) { sb.get(it) / 32768f }
            var peak = 0f
            for (v in x) peak = maxOf(peak, abs(v))
            val gain = if (peak > 0f) 0.70794576f / peak else 1f   // -3 dBFS
            for (k in x.indices) x[k] *= gain
            val out = File(docs, "audio8-demo-$startEpochS-ref_recorded.wav")
            Wav.writeMono16(out, Wav.floatToPcm16(x), Audio8Engine.SR)
            row.put("recorded_s", x.size / Audio8Engine.SR.toDouble()).put("peak_before", peak.toDouble()).put("gain", gain.toDouble())
                .put("recorded_wav", out.path)
            Log.i(TAG, "MIC_RECORDED samples=${x.size} peak=$peak gain=$gain clip_output_start_ms=${row.opt("clip_output_start_ms")} " +
                "clip_screen_ms=${row.opt("clip_screen_ms")} clip_screen_by=${row.opt("clip_screen_by")} wav=${out.path}")
            ui { stopTicker(); bar.visibility = View.INVISIBLE; onDone(x, row) }
        }.start()
    }

    private val clipLock = Any()

    /**
     * Follows the reference clip's AudioTrack every display frame while the mic records. The pill turns red in the
     * first frame after the timestamp gives the time the clip's frame 0 left the speaker (only timestamps whose
     * position advanced are used: a static track keeps returning its last position with a new time); fallback when no
     * timestamp arrives within 0.5 s of play(): red anyway, recorded as clip_screen_by = timeout_500ms. All state is
     * read and written under [clipLock]; the recording thread sets [released] (under the lock) before it releases the
     * track, and the watch stops there.
     */
    private inner class ClipWatch(val t: AudioTrack, val frames: Int, val rate: Int) : Choreographer.FrameCallback {
        private val ts = AudioTimestamp()
        @Volatile var playNs = 0L
        var released = false
        private var firstOutNs = 0L
        private var tsFrame = 0L
        private var screenNs = 0L
        private var screenBy: String? = null

        fun start() = Choreographer.getInstance().postFrameCallback(this)

        override fun doFrame(frameTimeNanos: Long) {
            synchronized(clipLock) {
                if (released) return
                val now = System.nanoTime()
                if (t.getTimestamp(ts) && ts.framePosition > tsFrame && ts.framePosition <= frames) {
                    tsFrame = ts.framePosition
                    firstOutNs = ts.nanoTime - ts.framePosition * 1_000_000_000L / rate
                }
                if (screenNs == 0L && (firstOutNs > 0 || now - playNs >= 500_000_000L)) {
                    screenNs = now
                    screenBy = if (firstOutNs > 0) "output_timestamp" else "timeout_500ms"
                    setPill("● RECORDING", C_LIVE)
                }
            }
            Choreographer.getInstance().postFrameCallback(this)
        }

        /** Called on the recording thread with [clipLock] held. Times are ms after play(). */
        fun putInto(row: JSONObject) {
            fun ms(ns: Long): Any = if (ns > 0) (ns - playNs) / 1e6 else JSONObject.NULL
            val durNs = frames * 1_000_000_000L / rate
            row.put("clip_output_start_ms", ms(firstOutNs))
                .put("clip_output_end_ms", if (firstOutNs > 0) ms(firstOutNs + durNs) else JSONObject.NULL)
                .put("clip_screen_ms", ms(screenNs)).put("clip_screen_by", screenBy ?: JSONObject.NULL)
                .put("clip_ts_last_frame", tsFrame).put("clip_frames", frames).put("clip_rate_hz", rate)
                .put("record_end_after_play_ms", (System.nanoTime() - playNs) / 1e6)
        }
    }

    // ---- helpers ---------------------------------------------------------------------------------------------

    private fun loadVoice(voice: String): Pair<String, Array<IntArray>> =
        Audio8Tts.loadVoice(modelDir, voice).let { it.transcript to it.codes }

    private fun codecJson(info: Map<String, Any>, calls: List<Audio8Engine.CodecCall>) = JSONObject(info as Map<*, *>)
        .put("calls", JSONArray(calls.map { JSONObject().put("T", it.t).put("start", it.start).put("n", it.n).put("ms", it.ms) }))

    private fun median(x: DoubleArray): Double = if (x.isEmpty()) Double.NaN else x.sorted()[x.size / 2]

    private fun dist(x: DoubleArray): JSONObject {
        if (x.isEmpty()) return JSONObject().put("n", 0)
        val s = x.sorted()
        fun q(f: Double) = s[((s.size - 1) * f).toInt()]
        val steady = if (x.size > 10) x.copyOfRange(10, x.size) else x
        return JSONObject().put("n", x.size).put("mean", x.average()).put("median", q(0.5)).put("p10", q(0.1)).put("p90", q(0.9))
            .put("min", s.first()).put("max", s.last()).put("first", x[0]).put("mean_after_10", steady.average())
    }

    private fun isJapanese(text: String) = text.any { Character.UnicodeScript.of(it.code).let { s ->
        s == Character.UnicodeScript.HIRAGANA || s == Character.UnicodeScript.KATAKANA || s == Character.UnicodeScript.HAN } }

    private fun airplaneOn(): Boolean = Settings.Global.getInt(contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) != 0

    private fun recordLayout() {
        val p = IntArray(2).also { pill.getLocationOnScreen(it) }
        val b = IntArray(2).also { bar.getLocationOnScreen(it) }
        val t = IntArray(2).also { body.getLocationOnScreen(it) }
        val (w, h) = if (Build.VERSION.SDK_INT >= 30) windowManager.currentWindowMetrics.bounds.let { it.width() to it.height() }
            else resources.displayMetrics.let { it.widthPixels to it.heightPixels }
        val layout = JSONObject().put("screen_px", JSONArray().put(w).put(h))
            .put("density", resources.displayMetrics.density.toDouble())
            .put("pill_px", JSONObject().put("left", p[0]).put("top", p[1]).put("height", pill.height).put("width", pill.width)
                .put("pad_left", pill.paddingLeft))
            .put("bar_px", JSONObject().put("left", b[0]).put("top", b[1]).put("width", bar.width).put("height", bar.height))
            .put("text_px", JSONObject().put("left", t[0]).put("top", t[1]).put("width", body.width))
        Log.i(TAG, "LAYOUT $layout")
        worker.execute { report.put("layout", layout); writeReport() }
    }

    private fun thermal(): JSONObject {
        val o = JSONObject()
        if (Build.VERSION.SDK_INT >= 29) {
            val pm = getSystemService(PowerManager::class.java)
            o.put("status", pm.currentThermalStatus)
            if (Build.VERSION.SDK_INT >= 30) {
                val headroom = pm.getThermalHeadroom(10)
                o.put("headroom_10s", if (headroom.isNaN()) JSONObject.NULL else headroom.toDouble())
            }
        }
        // The S26 caps scaling_max_freq after heavy legs even at thermal status 0 (round 1: policy6 1.98 of 4.74 GHz);
        // a speed row is clean only when every policy's scaling_max_freq equals cpuinfo_max_freq.
        runCatching {
            val caps = JSONObject()
            File("/sys/devices/system/cpu/cpufreq").listFiles { f -> f.name.startsWith("policy") }?.sortedBy { it.name }?.forEach { d ->
                val cur = File(d, "scaling_max_freq").readText().trim()
                val max = File(d, "cpuinfo_max_freq").readText().trim()
                caps.put(d.name, "$cur/$max")
            }
            o.put("cpufreq_max_khz", caps)
        }.onFailure { o.put("cpufreq_max_khz", "unreadable: $it") }
        return o.put("epoch_ms", System.currentTimeMillis())
    }

    /** VmHWM (peak RSS) and VmRSS of this process, KiB. */
    private fun vm(): JSONObject {
        val o = JSONObject()
        runCatching {
            for (line in File("/proc/self/status").readLines()) {
                if (line.startsWith("VmHWM:") || line.startsWith("VmRSS:")) {
                    o.put(line.substringBefore(':').lowercase() + "_kib", line.substringAfter(':').trim().split(Regex("\\s+"))[0].toLong())
                }
            }
        }
        return o
    }

    private fun writeReport() {
        val tmp = File(reportFile.path + ".tmp")
        tmp.writeText(report.toString(1))
        check(tmp.renameTo(reportFile)) { "could not write ${reportFile.path}" }
    }

    private fun ui(block: () -> Unit) {
        main.post(block)
    }

    private fun startTicker(text: (Double) -> String) {
        tickerStartNs = SystemClock.elapsedRealtimeNanos()
        tickerText = text
        main.removeCallbacks(ticker)
        ticker.run()
    }

    private fun stopTicker() {
        tickerText = null
        main.removeCallbacks(ticker)
    }

    private fun setPill(label: String, color: Int) {
        pill.text = label
        pill.background = GradientDrawable().apply { cornerRadius = dp(999f); setColor(color) }
    }

    private fun dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)

    private fun label(sizeSp: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
        setTextColor(color)
        if (bold) typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private fun button(text: String, color: Int, viewId: Int) = Button(this).apply {
        id = viewId
        this.text = text
        isAllCaps = false
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
        setTextColor(C_TEXT)
        background = GradientDrawable().apply { cornerRadius = dp(12f); setColor(color) }
        stateListAnimator = null
        minHeight = dp(48f).toInt()
        setPadding(dp(18f).toInt(), dp(10f).toInt(), dp(18f).toInt(), dp(10f).toInt())
    }

    private fun buildUi(): LinearLayout {
        val side = dp(22f).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(C_BG)
            setOnApplyWindowInsetsListener { v, insets ->
                if (Build.VERSION.SDK_INT >= 30) {
                    val i = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                    // The keyboard: the bottom edge moves up with it, so the Speak row stays above the keys.
                    val ime = insets.getInsets(WindowInsets.Type.ime())
                    v.setPadding(side + i.left, dp(20f).toInt() + i.top, side + i.right,
                        dp(20f).toInt() + maxOf(i.bottom, ime.bottom))
                } else {
                    @Suppress("DEPRECATION")
                    v.setPadding(side + insets.systemWindowInsetLeft, dp(20f).toInt() + insets.systemWindowInsetTop,
                        side + insets.systemWindowInsetRight, dp(20f).toInt() + insets.systemWindowInsetBottom)
                }
                insets
            }
        }
        val title = label(14f, C_TEXT, bold = true).apply { text = "Audio8-TTS-Preview-0.6b · LiteRT · on-device" }
        pill = label(13f, C_TEXT, bold = true).apply {
            letterSpacing = 0.08f
            setPadding(dp(12f).toInt(), dp(5f).toInt(), dp(12f).toInt(), dp(5f).toInt())
            gravity = Gravity.CENTER_VERTICAL
        }
        // A label with "●" lays out taller (the glyph comes from a fallback font): measure that one and fix the pill's
        // height to it, so a pill change never moves the text and bar below.
        pill.text = "● RECORDING"
        pill.measure(View.MeasureSpec.UNSPECIFIED, View.MeasureSpec.UNSPECIFIED)
        val pillHeight = pill.measuredHeight
        body = label(30f, C_TEXT).apply { setLineSpacing(0f, 1.12f) }
        bar = PlayBar(this, C_BUTTON_EDGE).apply { visibility = View.INVISIBLE }
        progress = label(19f, C_SUB)
        stats = label(16f, C_SUB)
        footer = label(13f, C_SUB)

        voiceGroup = RadioGroup(this).apply { orientation = LinearLayout.VERTICAL }
        for ((id, viewId) in listOf(JA_VOICE to R.id.voice_ja, EN_VOICE to R.id.voice_en, Audio8Tts.MY_VOICE to R.id.voice_mine)) {
            val b = RadioButton(this).apply {
                this.id = viewId
                text = voiceLabel(id)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
                setTextColor(C_TEXT)
                buttonTintList = ColorStateList.valueOf(C_ACCENT)
                isEnabled = false
            }
            voiceGroup.addView(b)
            voiceButtons[id] = b
        }
        voiceGroup.setOnCheckedChangeListener { _, checkedId ->
            val id = voiceButtons.entries.firstOrNull { it.value.id == checkedId }?.key
            if (id != null && phase == null) prefs.edit().putString(PREF_VOICE, id).apply()
        }
        recordButton = button(RECORD_LABEL, C_BUTTON_EDGE, R.id.record_button).apply {
            isEnabled = false
            alpha = 0.35f
            setOnClickListener { onRecord() }
        }
        textInput = EditText(this).apply {
            id = R.id.text_input
            hint = TEXT_HINT
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            setTextColor(C_TEXT)
            setHintTextColor(C_HINT)
            background = GradientDrawable().apply {
                cornerRadius = dp(12f)
                setColor(C_FIELD)
                setStroke(dp(1f).toInt(), C_BUTTON_EDGE)
            }
            setPadding(dp(14f).toInt(), dp(12f).toInt(), dp(14f).toInt(), dp(12f).toInt())
            minLines = 3
            maxLines = 6
            gravity = Gravity.TOP or Gravity.START
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            // Han characters in Japanese glyphs on a phone set to another language.
            textLocales = LocaleList(Locale.JAPANESE, Locale.US)
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                override fun afterTextChanged(s: Editable?) = updateTokenLine()
            })
        }
        tokenLine = label(13f, C_SUB)
        message = label(15f, C_SUB).apply {
            id = R.id.message
            visibility = View.GONE
            setLineSpacing(0f, 1.1f)
        }
        speakButton = button("Speak", C_ACCENT, R.id.speak_button).apply {
            isEnabled = false
            alpha = 0.35f
            setOnClickListener { onSpeak() }
        }
        cancelButton = button("Cancel", C_BUTTON_EDGE, R.id.cancel_button).apply {
            isEnabled = false
            alpha = 0.35f
            setOnClickListener { onCancel() }
        }

        inputPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(label(13f, C_SUB, bold = true).apply { text = "VOICE" }, lp(top = 18f))
            addView(voiceGroup, lp(top = 2f))
            addView(recordButton, lp(top = 6f, match = false))
            addView(label(13f, C_SUB, bold = true).apply { text = "TEXT" }, lp(top = 20f))
            addView(textInput, lp(top = 8f))
            addView(tokenLine, lp(top = 6f))
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(inputPanel)
            addView(body, lp(top = 36f))
            addView(message, lp(top = 12f))
            addView(bar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(6f).toInt()).apply {
                topMargin = dp(20f).toInt()
            })
            addView(progress, lp(top = 16f))
            addView(stats, lp(top = 12f))
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(content)
        }
        buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(speakButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(cancelButton, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                leftMargin = dp(12f).toInt()
            })
        }
        val interactive = mode == "interactive"
        inputPanel.visibility = if (interactive) View.VISIBLE else View.GONE
        buttonRow.visibility = if (interactive) View.VISIBLE else View.GONE
        body.visibility = if (interactive) View.GONE else View.VISIBLE
        root.addView(title)
        root.addView(pill, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, pillHeight).apply { topMargin = dp(16f).toInt() })
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        root.addView(buttonRow, lp(top = 12f))
        root.addView(footer, lp(top = 14f))
        return root
    }

    private fun lp(top: Float, match: Boolean = true) = LinearLayout.LayoutParams(
        if (match) ViewGroup.LayoutParams.MATCH_PARENT else ViewGroup.LayoutParams.WRAP_CONTENT,
        ViewGroup.LayoutParams.WRAP_CONTENT,
    ).apply { topMargin = dp(top).toInt() }

    companion object {
        private const val TAG = "Audio8Demo"
        const val SCRIPT_JA = "録音した声を登録して、その声で文章を読み上げます。"
        const val SCRIPT_EN = "Everything you hear was made on this phone, with no network at all."

        /** What "Record my voice" asks the user to read; the text becomes the recorded voice's transcript. */
        const val MY_VOICE_SCRIPT_EN = "I am recording this short sample so that the app can learn the sound of my voice."
        const val MY_VOICE_SCRIPT_JA = "この短い録音で、アプリに私の声の特徴を覚えてもらいます。"
        private const val RECORD_LABEL = "Record my voice (10 s)"
        private const val TEXT_HINT = "Type a sentence: English, 日本語, 中文, 한국어, Deutsch, Français, Español, Italiano, " +
            "Nederlands, Polski"
        private const val JA_VOICE = "ja_funasr_example"
        private const val EN_VOICE = "en_librispeech_1272"
        private const val PREFS = "audio8"
        private const val PREF_VOICE = "voice"
        private const val REQ_MIC = 1
        private val FRAME_S = Audio8Engine.FRAME / Audio8Engine.SR.toDouble()
        private const val C_BG = 0xFF0E1116.toInt()
        private const val C_IDLE = 0xFF5F6368.toInt()
        private const val C_LIVE = 0xFFE53935.toInt()      // sound out: RECORDING, PLAYING (make_media.sh looks for it)
        private const val C_WORK = 0xFF1565C0.toInt()      // compute: REGISTERING, GENERATING, DECODING
        private const val C_DONE = 0xFF2E7D32.toInt()
        private const val C_TEXT = 0xFFE6E8EB.toInt()
        private const val C_SUB = 0xFF8A919C.toInt()
        private const val C_BUTTON_EDGE = 0xFF2C333D.toInt()
        private const val C_ACCENT = 0xFF1565C0.toInt()
        private const val C_FIELD = 0xFF161B22.toInt()
        private const val C_HINT = 0xFF5F6670.toInt()
        private const val C_ERROR = 0xFFFF8A80.toInt()
    }
}
