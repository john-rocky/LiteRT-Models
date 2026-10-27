package com.ming3tts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors

/**
 * Ming TTS: type text, pick a language and a built-in voice, tap Speak. Speech is generated on
 * the phone (LiteRT CompiledModel, no network) and streamed to the speaker as it is decoded.
 *
 * Graphs are compiled and the frames run on one thread; the decoder runs on the engine's own
 * thread next to them. On a phone whose NPU LiteRT can use (the Qualcomm vendor libraries are in
 * the APK), the placement can put graphs on the NPU; elsewhere those graphs run on the GPU and the
 * status line says so. There the screen also offers the faster placement with every graph on the
 * NPU (remembered across launches). Intent extras drive the device checks of the conversion (see
 * SelfTest) and hands-free runs:
 *   adb shell am start -n com.ming3tts/.MainActivity --es text "'Good morning.'" --es lang en --es voice US_Emma --ez autoplay true
 * (the inner single quotes keep the text one argument in the phone's shell)
 */
class MainActivity : Activity() {

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "ming3tts-model") }
    private var engine: MingEngine? = null
    @Volatile private var cancel = false
    @Volatile private var busy = false
    private var player: StreamingPlayer? = null

    private lateinit var input: EditText
    private lateinit var langs: Spinner
    private lateinit var voices: Spinner
    private lateinit var button: Button
    private lateinit var status: TextView
    private lateinit var fast: Switch
    private lateinit var fastHint: TextView

    // `modeldir` names another folder under files/ (the device gates use it for the missing-file path).
    private val modelDir by lazy { File(filesDir, intent.getStringExtra("modeldir") ?: "models") }

    /** The faster all-NPU placement, as chosen on the switch (only read where the NPU is usable). */
    private val prefs by lazy { getSharedPreferences("ming3tts", MODE_PRIVATE) }

    // load settings of this launch, kept for a placement switch
    private var share = true
    private var decoderThread = true
    private var decoderGate = false
    private var accel: Accelerators? = null
    private var settingSwitch = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        val extra = intent.getStringExtra("placement")
        // Without a placement extra: the default, or the faster all-NPU placement when it was chosen on the switch.
        val fromSwitch = extra == null && prefs.getBoolean(PREF_FAST, false)
        val requested = try {
            Placement.parse(extra ?: if (fromSwitch) Placement.FAST else null)
        } catch (e: Exception) {
            status.text = "Bad placement: ${e.message}"
            return
        }
        // Two signatures of one file (prefill + step / frame) share their weights on the GPU;
        // without it the FP32 LM weights are uploaded twice (7.4 GB peak on a Galaxy S26).
        share = intent.getBooleanExtra("share", true)
        decoderThread = intent.getBooleanExtra("dec_thread", true)
        decoderGate = intent.getBooleanExtra("dec_gate", false)
        val selftest = intent.getStringExtra("selftest")
        status.text = "Loading…"
        worker.execute {
            if (selftest == "snippet") {
                // MingGraphs (the model card's Kotlin block) makes its own LiteRT Environment: the app's
                // one is not created in this process.
                runSnippet()
                return@execute
            }
            val acc = try {
                accelerators(this)
            } catch (t: Throwable) {
                Log.e(Graph.TAG, "accelerators failed", t)
                runOnUiThread { status.text = "Could not start LiteRT: ${t.message}" }
                return@execute
            }
            accel = acc
            var placement = requested
            var note = ""
            if (placement.usesNpu && !acc.npu) {
                if (fromSwitch) {
                    // chosen on a build or phone with the NPU; here the default placement takes over
                    placement = Placement.parse(null).withoutNpu()
                    note = "The faster NPU mode is not available here (${acc.npuNote}), so the model runs on the GPU.\n"
                } else {
                    placement = placement.withoutNpu()
                    note = "The NPU is not available here (${acc.npuNote}), so the model runs on the GPU.\n"
                }
                Log.w(Graph.TAG, "FAILPATH NPU unavailable (${acc.npuNote}): ${requested.describe()} -> ${placement.describe()}")
            }
            // The switch is offered only where the NPU is usable and the launch did not fix the placement.
            if (acc.npu && extra == null && selftest == null) runOnUiThread { showFastSwitch(fromSwitch) }
            load(placement, note, selftest)
        }
    }

    /**
     * On the worker thread: the engine on [placement], then Ready (or the self-test; the launch's extras
     * unless [switched]). A placement with several NPU graphs first compiles them one at a time
     * ([MingEngine.prepareNpu]).
     */
    private fun load(placement: Placement, note: String, selftest: String?, switched: Boolean = false) {
        val acc = accel!!
        val missing = MingEngine.missingFiles(modelDir, placement)
        if (missing.isNotEmpty()) {
            runOnUiThread {
                status.text = note + "Model files not found in ${modelDir.absolutePath}:\n${missing.joinToString(", ")}\n\n" +
                    "Download them and copy them to the phone, on a computer:\n" +
                    "hf download litert-community/Ming-3.0-tts-0.5B --local-dir Ming-3.0-tts-0.5B\n" +
                    "bash Ming-3.0-tts-0.5B/scripts/install_to_device.sh Ming-3.0-tts-0.5B"
            }
            Log.w(Graph.TAG, "FAILPATH missing model files: $missing")
            return
        }
        if (selftest == "graphs" || selftest == "host" || selftest == "jit") {
            runOnUiThread { runSelfTest(selftest, placement, share, null, acc) }
            return
        }
        runOnUiThread {
            status.text = note + if (placement.usesNpu) {
                "Preparing the NPU: the first launch compiles the model for this phone (about a minute)…"
            } else {
                "Loading model (${placement.summary()})…"
            }
        }
        val e = try {
            if (placement.npuRoles.size > 1) {
                MingEngine.prepareNpu(acc, modelDir, placement) { i, n ->
                    runOnUiThread { status.text = note + "Preparing the NPU, graph $i of $n: the first time, each graph is compiled for this phone…" }
                }
            }
            MingEngine(modelDir, placement, share, acc, decoderThread, decoderGate)
        } catch (t: Throwable) {
            Log.e(Graph.TAG, "load failed", t)
            if (selftest != null) SelfTest(this, modelDir, tagOf(selftest, placement))
                .status("FAILED", org.json.JSONObject().put("error", t.toString()).put("stage", "load"))
            runOnUiThread {
                status.text = "Could not load the model: ${t.message}"
                fast.isEnabled = true     // the switch can still go back to the other placement
            }
            return
        }
        engine = e
        readyNote = note
        Log.i(Graph.TAG, "ready ${placement.describe()} compile ms ${e.compileMs}, decoder thread $decoderThread, gate $decoderGate")
        runOnUiThread {
            status.text = note + "Ready · ${placement.summary()}"
            button.isEnabled = true
            fast.isEnabled = true
            if (selftest != null) runSelfTest(selftest, placement, share, e, acc) else if (!switched) handleExtras(intent)
        }
    }

    private fun showFastSwitch(on: Boolean) {
        setFast(on)
        fast.visibility = View.VISIBLE
        fastHint.visibility = View.VISIBLE
    }

    private fun setFast(on: Boolean) {
        settingSwitch = true
        fast.isChecked = on
        settingSwitch = false
    }

    /** The switch: the faster all-NPU placement on or off. The current engine closes first (one model in memory). */
    private fun onFastSwitch(on: Boolean) {
        val target = Placement.parse(if (on) Placement.FAST else null)
        val missing = MingEngine.missingFiles(modelDir, target)
        if (missing.isNotEmpty()) {
            Log.w(Graph.TAG, "FAILPATH faster mode files missing: $missing")
            setFast(!on)
            status.text = "This mode needs model files that are not on the phone: ${missing.joinToString(", ")}\n\n" +
                "Copy them to the phone, on a computer:\n" +
                "bash Ming-3.0-tts-0.5B/scripts/install_to_device.sh Ming-3.0-tts-0.5B"
            return
        }
        prefs.edit().putBoolean(PREF_FAST, on).apply()
        val old = engine
        engine = null
        button.isEnabled = false
        fast.isEnabled = false
        readyNote = ""
        status.text = "Loading…"
        Log.i(Graph.TAG, "placement switch: faster mode ${if (on) "on" else "off"}, ${target.describe()}")
        worker.execute {
            old?.close()
            load(target, "", null, switched = true)
        }
    }

    /** Shown above the result lines, e.g. that the NPU graphs moved to the GPU. */
    private var readyNote = ""

    private fun tagOf(selftest: String, p: Placement) = intent.getStringExtra("tag") ?: "${selftest}_${p.name}"

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true
            setPadding(48, 48, 48, 48)
        }
        input = EditText(this).apply {
            hint = "Type something to say"
            setText(EXAMPLES.getValue("en"))
            minLines = 3
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        }
        langs = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, LANGS.map { it.second })
        }
        langs.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                // Swap in the example of the new language unless the user typed their own text.
                if (input.text.toString() in EXAMPLES.values) input.setText(EXAMPLES.getValue(LANGS[pos].first))
            }

            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
        voices = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, VOICES.map { it.second })
        }
        button = Button(this).apply {
            text = "Speak"
            isEnabled = false
            setOnClickListener { onButton() }
        }
        // Shown once the NPU turns out to be usable (showFastSwitch).
        fast = Switch(this).apply {
            text = "Faster: run everything on the NPU"
            textSize = 16f
            isEnabled = false
            visibility = View.GONE
            setOnCheckedChangeListener { _, on -> if (!settingSwitch) onFastSwitch(on) }
        }
        fastHint = TextView(this).apply {
            text = "The NPU computes in 16-bit floats: faster, but less exact than the default."
            textSize = 12f
            visibility = View.GONE
        }
        status = TextView(this).apply {
            text = ""
            textSize = 14f
            setTextIsSelectable(true)
        }
        val caption = TextView(this).apply {
            text = "Type any text: the phone speaks it in a 44.1 kHz voice, generated on the device, no network."
            textSize = 14f
        }
        for ((i, v) in listOf(caption, input, langs, voices, fast, fastHint, button, status).withIndex()) {
            val lp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            lp.topMargin = if (i == 0) 0 else if (v === fastHint) 4 else 24
            root.addView(v, lp)
        }
        setContentView(root)
    }

    private fun onButton() {
        if (busy) {
            // Second tap while speaking = stop, safely: the model thread ends at the next frame.
            Log.i(Graph.TAG, "FAILPATH second press while generating: stop requested")
            cancel = true
            player?.stop()
            status.text = "Stopping…"
            return
        }
        val text = input.text.toString()
        val lang = LANGS[langs.selectedItemPosition].first
        val voice = VOICES[voices.selectedItemPosition].first
        val e = engine ?: return
        val fragments = e.frontend.prepare(text, lang)
        if (fragments.isEmpty()) {
            Log.i(Graph.TAG, "FAILPATH empty input: nothing generated")
            status.text = "Type some text first."
            return
        }
        speak(e, fragments, lang, voice)
    }

    private fun speak(e: MingEngine, fragments: List<String>, lang: String, voice: String) {
        busy = true
        cancel = false
        fast.isEnabled = false
        button.text = "Stop"
        status.text = "Speaking… (${fragments.size} part${if (fragments.size > 1) "s" else ""})"
        val p = StreamingPlayer()
        player = p
        worker.execute {
            val t0 = System.nanoTime()
            var lastPcm = t0
            var firstPcm = -1L
            var audioSamples = 0L
            var untrimmed = 0L
            val frameMs = ArrayList<Double>()
            val decMs = ArrayList<Double>()
            var outcome = "done"
            try {
                for ((i, frag) in fragments.withIndex()) {
                    val trim = SilenceTrimmer(HostMath.SAMPLE_RATE) { pcm ->
                        if (firstPcm < 0) firstPcm = System.nanoTime()
                        audioSamples += pcm.size
                        lastPcm = System.nanoTime()
                        p.offer(pcm)
                    }
                    runOnUiThread { if (busy) status.text = "Speaking… part ${i + 1} of ${fragments.size}" }
                    val st = e.generate(Request(frag, lang, voice), MingEngine.gaussianNoise(System.nanoTime()),
                        { trim.push(it) }, { cancel })
                    trim.finish()
                    untrimmed += st.samples
                    frameMs.addAll(st.frameMs)
                    decMs.addAll(st.decMs)
                    Log.i(Graph.TAG, "part ${i + 1}/${fragments.size}: ${st.steps} steps, ${st.samples} samples \"$frag\"")
                }
                p.finish()
            } catch (c: CancellationException) {
                outcome = "stopped"
                p.stop()
            } catch (t: Throwable) {
                outcome = "error: ${t.message}"
                Log.e(Graph.TAG, "generation failed", t)
                p.stop()
            }
            val audio = audioSamples / HostMath.SAMPLE_RATE.toDouble()
            val elapsed = (lastPcm - t0) / 1e9
            val line = if (outcome == "done") {
                readyNote + "Spoke %.1f s in %.1f s · RTFx %.2f · first audio %.2f s · frame %.0f ms · decoder %.0f ms/chunk\n%s"
                    .format(audio, elapsed, audio / elapsed, (firstPcm - t0) / 1e9, MingEngine.median(frameMs),
                        MingEngine.median(decMs), e.placement.summary())
            } else {
                readyNote + "Stopped${if (outcome != "stopped") " ($outcome)" else ""} · ${e.placement.summary()}"
            }
            Log.i(Graph.TAG, "SPEAK $outcome: ${fragments.size} parts, audio ${"%.2f".format(audio)} s (untrimmed " +
                "${"%.2f".format(untrimmed / 44100.0)} s), to last sample ${"%.2f".format(elapsed)} s, RTFx " +
                "${"%.3f".format(audio / elapsed)}, TTFA ${"%.0f".format((firstPcm - t0) / 1e6)} ms")
            runOnUiThread {
                busy = false
                button.text = "Speak"
                fast.isEnabled = true
                if (!isFinishing && !isDestroyed) status.text = line
            }
        }
    }

    private fun runSelfTest(mode: String, placement: Placement, share: Boolean, e: MingEngine?, acc: Accelerators) {
        val tag = tagOf(mode, placement)
        status.text = "Self-test $mode ($tag)…"
        worker.execute {
            val t = SelfTest(this, modelDir, tag)
            t.status("RUNNING", org.json.JSONObject().put("placement", placement.describe()).put("npu", acc.npu)
                .put("npu_note", acc.npuNote))
            try {
                when (mode) {
                    "graphs" -> t.graphs(acc, placement, share, intent.getIntExtra("reps", 5))
                    "jit" -> t.jit(acc, placement)
                    "host" -> t.host()
                    "chain" -> t.chain(e!!, intent.getStringExtra("fixtures")?.split(',')?.toSet() ?: emptySet())
                    "bench" -> t.bench(e!!, intent.getStringExtra("fixture") ?: "E5", intent.getIntExtra("warmup", 1),
                        intent.getIntExtra("reps", 3), intent.getLongExtra("rest_ms", 0L), intent.getBooleanExtra("play", true))
                    else -> error("unknown selftest $mode")
                }
                t.status("DONE")
                runOnUiThread { status.text = "Self-test $mode done ($tag)" }
            } catch (x: Throwable) {
                Log.e(Graph.TAG, "selftest $mode failed", x)
                t.status("FAILED", org.json.JSONObject().put("error", x.toString())
                    .put("stack", Log.getStackTraceString(x).take(4000)))
                runOnUiThread { status.text = "Self-test $mode failed: ${x.message}" }
            }
        }
    }

    /** The snippet self-test, on the worker thread. */
    private fun runSnippet() {
        val tag = intent.getStringExtra("tag") ?: "snippet"
        runOnUiThread { status.text = "Self-test snippet ($tag)…" }
        val t = SelfTest(this, modelDir, tag)
        t.status("RUNNING")
        try {
            t.snippet(intent.getStringExtra("fixtures")?.split(',')?.toSet() ?: emptySet())
            t.status("DONE")
            runOnUiThread { status.text = "Self-test snippet done ($tag)" }
        } catch (x: Throwable) {
            Log.e(Graph.TAG, "selftest snippet failed", x)
            t.status("FAILED", org.json.JSONObject().put("error", x.toString())
                .put("stack", Log.getStackTraceString(x).take(4000)))
            runOnUiThread { status.text = "Self-test snippet failed: ${x.message}" }
        }
    }

    /** Hands-free run: text / lang / voice extras, then Speak when autoplay is set. */
    private fun handleExtras(i: Intent?) {
        if (i == null || i.getStringExtra("selftest") != null) return
        i.getStringExtra("lang")?.let { l ->
            val k = LANGS.indexOfFirst { it.first == l }
            if (k >= 0) langs.setSelection(k, false)
        }
        i.getStringExtra("voice")?.let { v ->
            val k = VOICES.indexOfFirst { it.first == v }
            if (k >= 0) voices.setSelection(k, false)
        }
        i.getStringExtra("text")?.let { input.setText(it) }
        if (i.getBooleanExtra("autoplay", false) && button.isEnabled) button.performClick()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (engine != null && !busy) handleExtras(intent)
    }

    override fun onStop() {
        super.onStop()
        if (busy) {
            // Leaving the app while speaking stops the speech; nothing keeps running unseen.
            Log.i(Graph.TAG, "FAILPATH app left the foreground while generating: stopped")
            cancel = true
            player?.stop()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cancel = true
        player?.stop()
        val e = engine
        engine = null
        worker.execute { e?.close() }
        worker.shutdown()
    }

    companion object {
        // One LiteRT Environment per process: the NPU's dispatch options are fixed by the first
        // model a process loads, and the Environment holds the GPU and NPU state for all graphs.
        @Volatile private var acc: Accelerators? = null

        @Synchronized
        fun accelerators(ctx: android.content.Context): Accelerators =
            acc ?: Accelerators.create(ctx.applicationContext).also { acc = it }

        private const val PREF_FAST = "fast_all_npu"

        val LANGS = listOf(
            "en" to "English", "zh" to "中文 (Chinese)", "ja" to "日本語 (Japanese)", "ko" to "한국어 (Korean)",
            "de" to "Deutsch (German)", "fr" to "Français (French)", "es" to "Español (Spanish)",
        )

        // Built-in voices of the checkpoint (no voice file needed): cookbooks/instruction_en.md
        val VOICES = listOf(
            "US_Emma" to "Emma · US English, female", "US_James" to "James · US English, male",
            "US_Stella" to "Stella · US English, female", "UK_George" to "George · British English, male",
            "UK_Charlotte" to "Charlotte · British English, female", "灵小萌" to "灵小萌 · Mandarin, female",
            "灵岩" to "灵岩 · Mandarin, male", "灵小雅" to "灵小雅 · Mandarin, female",
        )

        // Example sentences with invented names and places.
        val EXAMPLES = mapOf(
            "en" to "Good morning! This is Marisol Fenwick from Larkspur Lane Bakery. Your rosemary loaf is ready, and it is still warm.",
            "zh" to "早上好！我是云杉书店的林若溪。您预订的那本旅行画册已经到了，欢迎随时来取。",
            "ja" to "おはようございます。こもれび通りベーカリーの春野ミナです。ご注文のパンが焼き上がりました。",
            "ko" to "안녕하세요. 별빛골 베이커리의 한소율입니다. 주문하신 빵이 방금 나왔어요.",
            "de" to "Guten Morgen! Hier ist Lina Brandauer von der Bäckerei Lerchenweg. Ihr Rosmarinbrot ist fertig und noch warm.",
            "fr" to "Bonjour ! Ici Margot Delaunay, de la boulangerie Les Trois Tilleuls. Votre pain au romarin est prêt et encore tiède.",
            "es" to "¡Buenos días! Le habla Inés Valcárcel, de la panadería El Romero Azul. Su pan de romero ya está listo y todavía caliente.",
        )
    }
}
