package com.ming3tts

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * The conversion's device checks, started by intent extras and written to <external files>/r2out/<tag>/
 * for the scorers on the computer. They never judge; they record. Their inputs (files/selftest/) belong to
 * the conversion's verification and are not part of the model download.
 *  - graphs: every signature of the placement's graphs on fixed inputs (outputs, delegation)
 *  - jit:    each NPU graph of the placement compiled alone and closed (fills the JIT cache one graph at
 *            a time, so the compile's memory peak is one graph's; the memory line goes to jit.json)
 *  - chain:  the 10 test sentences through the app's host loop with the reference noise
 *  - bench:  fixture E5 with streaming playback, warm-up + reps (RTFx, TTFA, frame / decoder ms)
 *  - host:   the host-code vectors on the device (tokenizer, prompt contract, frontend, ISTFT)
 *  - snippet: the model card's Kotlin block (MingGraphs) under the app's host loop, as chain
 */
class SelfTest(private val ctx: Context, private val models: File, private val tag: String) {
    private val selftest = File(ctx.filesDir, "selftest")

    // External files dir: adb can pull it from a release (non-debuggable) build as well.
    val outDir = File(ctx.getExternalFilesDir(null), "r2out/$tag").also { it.mkdirs() }

    private fun log(msg: String) = Log.i(Graph.TAG, "SELFTEST $tag $msg")

    /** JSON cannot hold NaN (an empty list's median): null instead. */
    private fun num(x: Double): Any = if (x.isNaN() || x.isInfinite()) JSONObject.NULL else x

    fun status(state: String, extra: JSONObject = JSONObject()) {
        extra.put("status", state).put("tag", tag).put("device", Build.MODEL).put("sdk", Build.VERSION.SDK_INT)
        File(outDir, "status.json").writeText(extra.toString(1))
        log("status $state")
    }

    fun conditions(): JSONObject {
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        val o = JSONObject()
        if (Build.VERSION.SDK_INT >= 29) o.put("thermal_status", pm.currentThermalStatus)
        if (Build.VERSION.SDK_INT >= 30) o.put("thermal_headroom_10s", pm.getThermalHeadroom(10).toDouble())
        val caps = JSONObject()
        File("/sys/devices/system/cpu/cpufreq").listFiles()?.filter { it.name.startsWith("policy") }?.sortedBy { it.name }?.forEach {
            caps.put(it.name, runCatching { File(it, "scaling_max_freq").readText().trim() }.getOrDefault("?") + "/" +
                runCatching { File(it, "cpuinfo_max_freq").readText().trim() }.getOrDefault("?"))
        }
        o.put("cpu_max_freq", caps)
        runCatching { o.put("kgsl_max_clock_mhz", File("/sys/class/kgsl/kgsl-3d0/max_clock_mhz").readText().trim()) }
        o.put("time_ms", System.currentTimeMillis())
        return o
    }

    // ------------------------------------------------------------------ graphs (per-graph outputs)

    private val groupsOf = mapOf(
        "frame" to listOf("lm2_prefill" to "prefill", "frame2" to "frame"),
        "lm" to listOf("lm2_prefill" to "prefill", "lm2_step" to "step"),
        "prefill" to listOf("lm2_prefill" to "prefill"),
        "step" to listOf("lm2_step" to "step"),
        "flow" to listOf("flow" to "flow"),
        "agg" to listOf("agg" to "agg"),
        "dec" to listOf("dec_c8" to "dec"),
    )

    /** This process's VmHWM / VmRSS / VmSwap and the phone's available memory (MB). */
    fun memory(): JSONObject {
        val status = File("/proc/self/status").readLines().associate { line ->
            val p = line.split(":", limit = 2)
            p[0] to (p.getOrNull(1)?.trim() ?: "")
        }
        fun mb(k: String) = (status[k]?.removeSuffix("kB")?.trim()?.toLongOrNull() ?: -1L) / 1024
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        return JSONObject().put("vm_hwm_mb", mb("VmHWM")).put("vm_rss_mb", mb("VmRSS")).put("vm_swap_mb", mb("VmSwap"))
            .put("device_avail_mb", mi.availMem shr 20)
    }

    /** Files under cacheDir (the NPU JIT cache) and their bytes. */
    fun cacheBytes(): Long = ctx.cacheDir.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    fun jit(acc: Accelerators, placement: Placement) {
        val res = JSONObject().put("placement", placement.describe()).put("npu", acc.npu).put("conditions_start", conditions())
        for ((role, r) in placement.roles) {
            if (r.backend != Backend.NPU) continue
            log("jit $role ${r.file}")
            val before = cacheBytes()
            val g = Graph(acc, File(models, r.file), r.backend, Graph.signaturesOf(role))
            val mem = memory()
            res.put(role, JSONObject().put("file", r.file).put("compile_ms", g.compileMs).put("memory", mem)
                .put("cache_bytes_before", before).put("cache_bytes_after", cacheBytes()))
            g.close()
            log("jit $role ${r.file} ${"%.0f".format(g.compileMs)} ms, $mem")
        }
        res.put("conditions_end", conditions())
        File(outDir, "jit.json").writeText(res.toString(1))
    }

    fun graphs(acc: Accelerators, placement: Placement, share: Boolean, reps: Int) {
        val dev = File(selftest, "devfix")
        val shapes = JSONObject(File(dev, "shapes.json").readText()).getJSONObject("groups")
        val res = JSONObject().put("placement", placement.describe()).put("share_constants", share)
            .put("npu", acc.npu).put("conditions_start", conditions())
        run {
            // NPU graphs first, as in the engine
            val order = placement.roles.entries.sortedBy { if (it.value.backend == Backend.NPU) 0 else 1 }
            for ((role, spec) in order) {
                val be = spec.backend
                log("compile $role ${be.label} ${spec.file}")
                val g = Graph(acc, File(models, spec.file), be, Graph.signaturesOf(role), share)
                val rr = JSONObject().put("file", g.file.name).put("backend", be.label).put("compile_ms", g.compileMs)
                    .put("memory_after_compile", memory())
                // the decoder's device fixture follows its chunk (dec_c8, dec_c16)
                val groups = if (role == "dec") listOf("dec_c${spec.decChunk}" to "dec") else groupsOf.getValue(role)
                for ((group, sig) in groups) {
                    val items = shapes.getJSONArray(group)
                    val ins = ArrayList<FloatArray>()
                    val outs = ArrayList<String>()
                    for (i in 0 until items.length()) {
                        val it = items.getJSONObject(i)
                        when (it.getString("role")) {
                            "input" -> ins.add(HostMath.readF32(File(dev, it.getString("file"))))
                            "output" -> outs.add(it.getString("name"))
                        }
                    }
                    val t0 = System.nanoTime()
                    var o = g.call(sig, ins.toTypedArray())
                    val first = (System.nanoTime() - t0) / 1e6
                    val times = ArrayList<Double>()
                    for (r in 0 until reps) {
                        val t = System.nanoTime()
                        o = g.call(sig, ins.toTypedArray())
                        times.add((System.nanoTime() - t) / 1e6)
                    }
                    val gd = File(outDir, group).also { it.mkdirs() }
                    for ((k, name) in outs.withIndex()) HostMath.writeF32(File(gd, "out_%02d_%s.f32".format(k, name)), o[k]!!)
                    rr.put(group, JSONObject().put("signature", sig).put("first_call_ms", first)
                        .put("call_ms", JSONArray(times)).put("call_ms_median", MingEngine.median(times))
                        .put("write_ms_last", g.lastWriteMs).put("run_ms_last", g.lastRunMs).put("read_ms_last", g.lastReadMs))
                    log("$role/$group first ${"%.1f".format(first)} ms, median ${"%.2f".format(MingEngine.median(times))} ms")
                }
                g.close()
                res.put(role, rr)
            }
        }
        res.put("conditions_end", conditions())
        File(outDir, "graphs.json").writeText(res.toString(1))
    }

    // ------------------------------------------------------------------ fixtures

    class Fixture(val id: String, val languageId: String, val ip: String?, val voice: String?, val text: String)

    fun fixtures(only: Set<String>): List<Fixture> =
        File(selftest, "fixtures.tsv").readLines(Charsets.UTF_8).filter { it.isNotEmpty() }.map {
            val c = it.split('\t')
            Fixture(c[0], c[1], c[2].ifEmpty { null }, c[3].ifEmpty { null }, c[4])
        }.filter { only.isEmpty() || it.id in only }

    private val noiseRows: FloatArray by lazy { HostMath.readF32(File(selftest, "chain_noise.bin")) }

    fun referenceNoise(): (Int) -> FloatArray = { s -> noiseRows.copyOfRange(s * 64, s * 64 + 64) }

    fun request(f: Fixture): Request {
        val v = f.voice?.let { VoiceFile.load(File(selftest, "voice_$it.bin"), File(selftest, "voice_$it.txt")) }
        return Request(f.text, f.languageId, f.ip, v)
    }

    // ------------------------------------------------------------------ chain (whole sentences)

    fun chain(engine: MingEngine, only: Set<String>) {
        val summary = JSONObject().put("placement", engine.placement.describe())
            .put("decoder_thread", engine.decoderThread).put("decoder_gate", engine.decoderGate)
            .put("compile_ms", JSONObject(engine.compileMs)).put("memory", memory()).put("conditions_start", conditions())
        val rows = JSONArray()
        for (f in fixtures(only)) {
            val pcm = ArrayList<FloatArray>()
            val st = engine.generate(request(f), referenceNoise(), { pcm.add(it) }, keepLatents = true)
            val wav = FloatArray(pcm.sumOf { it.size })
            var o = 0
            for (p in pcm) {
                p.copyInto(wav, o)
                o += p.size
            }
            HostMath.writeF32(File(outDir, "${f.id}.wav.f32"), wav)
            val lat = st.latents!!
            val la = FloatArray(lat.size * 64)
            for (i in lat.indices) lat[i].copyInto(la, i * 64)
            HostMath.writeF32(File(outDir, "${f.id}_latent.f32"), la)
            val j = statsJson(st).put("id", f.id).put("seconds", wav.size / HostMath.SAMPLE_RATE.toDouble())
                .put("ids", JSONArray(st.ids.toList()))
            File(outDir, "${f.id}.json").writeText(j.toString(1))
            rows.put(j.put("ids", JSONArray()))
            log("chain ${f.id}: ${st.steps} steps, ${"%.2f".format(wav.size / 44100.0)} s audio in " +
                "${"%.0f".format(st.totalMs)} ms (frame median ${"%.1f".format(MingEngine.median(st.frameMs))} ms)")
        }
        summary.put("fixtures", rows).put("conditions_end", conditions())
        File(outDir, "chain.json").writeText(summary.toString(1))
    }

    fun statsJson(st: FragmentStats): JSONObject = JSONObject()
        .put("prompt_tokens", st.promptTokens).put("steps", st.steps).put("stopped", st.stopped)
        .put("samples", st.samples).put("prefill_ms", st.prefillMs).put("total_ms", st.totalMs)
        .put("first_pcm_ms", st.firstPcmMs)
        .put("frame_ms_median", MingEngine.median(st.frameMs)).put("frame_ms", JSONArray(st.frameMs))
        .put("frame_ms_mean", if (st.frameMs.isEmpty()) JSONObject.NULL else st.frameMs.average())
        .put("frame_write_ms_median", MingEngine.median(st.frameWriteMs))
        .put("lm_ms_median", num(MingEngine.median(st.lmMs))).put("flow_ms_median", num(MingEngine.median(st.flowMs)))
        .put("agg_ms_median", num(MingEngine.median(st.aggMs))).put("frames_end_ms", st.framesEndMs)
        .put("dec_ms_median", MingEngine.median(st.decMs)).put("dec_ms", JSONArray(st.decMs))
        .put("istft_ms_median", MingEngine.median(st.istftMs))
        .put("stop_p1_last", st.stopP1.lastOrNull()?.toDouble() ?: Double.NaN)
        .put("near_stop", JSONArray(st.stopP1.withIndex().filter { kotlin.math.abs(it.value - 0.5f) < 0.05f }
            .map { JSONArray().put(it.index).put(it.value.toDouble()) }))

    // ------------------------------------------------------------------ snippet (the model card's Kotlin block)

    /**
     * MingGraphs (the model card's Kotlin block, MingGraphs.kt) under the app's host loop: it loads the four
     * graphs on its own placement, and every graph call of the fixtures goes through MingGraphs.run. Writes
     * what chain() writes, plus snippet.json (placement and load time, or the error of a missing file).
     */
    fun snippet(only: Set<String>) {
        val t0 = System.nanoTime()
        val g = try {
            MingGraphs(ctx, models)
        } catch (e: IllegalArgumentException) {
            log("FAILPATH snippet: ${e.message}")
            File(outDir, "snippet.json").writeText(JSONObject().put("error", e.message).toString(1))
            return
        }
        val loadMs = (System.nanoTime() - t0) / 1e6
        val placement = Placement.parse("Q2D16").let { if (g.npu) it else it.withoutNpu() }
        log("snippet: npu=${g.npu}, ${placement.describe()}, graphs loaded in ${"%.0f".format(loadMs)} ms")
        File(outDir, "snippet.json").writeText(JSONObject().put("npu", g.npu).put("placement", placement.describe())
            .put("load_ms", loadMs).put("memory", memory()).toString(1))
        val byRole = mapOf("lm" to g.lm, "flow" to g.flow, "agg" to g.agg, "dec" to g.dec)
        val engine = MingEngine(models, placement, graphFactory = { role, _, _ ->
            object : GraphCall {
                override val compileMs = 0.0
                override val lastWriteMs = 0.0

                override fun call(sig: String, inputs: Array<FloatArray>, read: IntArray?): Array<FloatArray?> {
                    val r = g.run(byRole.getValue(role), sig, *inputs)
                    return arrayOfNulls<FloatArray>(r.size).also { a -> r.forEachIndexed { i, x -> a[i] = x } }
                }

                override fun close() {}
            }
        })
        try {
            chain(engine, only)
        } finally {
            engine.close()
            g.close()
        }
    }

    // ------------------------------------------------------------------ bench (speed)

    fun bench(engine: MingEngine, fixtureId: String, warmup: Int, reps: Int, restMs: Long, play: Boolean) {
        val f = fixtures(setOf(fixtureId)).single()
        val req = request(f)
        val res = JSONObject().put("placement", engine.placement.describe()).put("fixture", fixtureId)
            .put("decoder_thread", engine.decoderThread).put("decoder_gate", engine.decoderGate)
            .put("compile_ms", JSONObject(engine.compileMs)).put("memory", memory()).put("play", play)
        val runs = JSONArray()
        for (r in 0 until warmup + reps) {
            if (r > 0 && restMs > 0) Thread.sleep(restMs)
            val cond = conditions()
            val player = if (play) StreamingPlayer() else null
            val t0 = System.nanoTime()
            var lastPcm = 0L
            val st = engine.generate(req, referenceNoise(), { pcm ->
                player?.offer(pcm)
                lastPcm = System.nanoTime()
            })
            player?.finish()
            val elapsed = (lastPcm - t0) / 1e9
            val audio = st.samples / HostMath.SAMPLE_RATE.toDouble()
            player?.awaitEnd(audio.toLong() * 1000 + 5000)
            val firstWrite = player?.firstWriteNanos?.let { if (it > 0) (it - t0) / 1e6 else -1.0 } ?: -1.0
            val j = statsJson(st).put("rep", r).put("warmup", r < warmup).put("audio_s", audio)
                .put("to_last_sample_s", elapsed).put("rtfx", audio / elapsed)
                .put("ttfa_first_pcm_ms", st.firstPcmMs).put("ttfa_first_write_ms", firstWrite)
                .put("underruns", player?.underruns ?: -1).put("conditions", cond)
            j.remove("frame_ms")
            j.remove("dec_ms")
            runs.put(j)
            log("bench rep $r${if (r < warmup) " (warm-up)" else ""}: ${"%.2f".format(audio)} s audio, " +
                "RTFx ${"%.3f".format(audio / elapsed)}, TTFA ${"%.0f".format(st.firstPcmMs)} ms " +
                "(first write ${"%.0f".format(firstWrite)} ms), frame ${"%.1f".format(MingEngine.median(st.frameMs))} ms " +
                "(lm ${"%.1f".format(MingEngine.median(st.lmMs))}, flow ${"%.1f".format(MingEngine.median(st.flowMs))}, " +
                "agg ${"%.1f".format(MingEngine.median(st.aggMs))}), dec ${"%.1f".format(MingEngine.median(st.decMs))} ms/chunk, " +
                "thermal ${cond.optInt("thermal_status")}")
        }
        val warm = (0 until runs.length()).map { runs.getJSONObject(it) }.filter { !it.getBoolean("warmup") }
        res.put("runs", runs)
            .put("rtfx_median", MingEngine.median(warm.map { it.getDouble("rtfx") }))
            .put("ttfa_ms_median", MingEngine.median(warm.map { it.getDouble("ttfa_first_pcm_ms") }))
            .put("frame_ms_median", MingEngine.median(warm.map { it.getDouble("frame_ms_median") }))
            .put("lm_ms_median", num(MingEngine.median(warm.map { it.optDouble("lm_ms_median") })))
            .put("flow_ms_median", num(MingEngine.median(warm.map { it.optDouble("flow_ms_median") })))
            .put("agg_ms_median", num(MingEngine.median(warm.map { it.optDouble("agg_ms_median") })))
            .put("dec_ms_median", MingEngine.median(warm.map { it.getDouble("dec_ms_median") }))
        File(outDir, "bench.json").writeText(res.toString(1))
        log("BENCH ${engine.placement.describe()} E5 warm x${warm.size}: RTFx median ${"%.3f".format(res.getDouble("rtfx_median"))}, " +
            "TTFA ${"%.0f".format(res.getDouble("ttfa_ms_median"))} ms, frame ${"%.1f".format(res.getDouble("frame_ms_median"))} ms, " +
            "dec ${"%.1f".format(res.getDouble("dec_ms_median"))} ms/chunk")
    }

    // ------------------------------------------------------------------ host (host code on the device)

    fun host() {
        val dh = File(selftest, "dhost")
        val cc = CharClasses.load(File(models, "char_classes.bin"))
        val t0 = System.nanoTime()
        val tok = QwenTokenizer(File(models, "tokenizer_vocab.txt"), File(models, "tokenizer_merges.txt"),
            File(models, "tokenizer_added.txt"), cc)
        val loadMs = (System.nanoTime() - t0) / 1e6
        fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
        var tokEq = 0
        var tokN = 0
        val bad = JSONArray()
        val t1 = System.nanoTime()
        File(dh, "tok_cases.jsonl").forEachLine(Charsets.UTF_8) { line ->
            if (line.isBlank()) return@forEachLine
            val r = JSONObject(line)
            tokN++
            if (tok.encode(r.getString("text")).contentEquals(ints(r.getJSONArray("ids_mirror")))) tokEq++
            else if (bad.length() < 10) bad.put(r.getString("text"))
        }
        val tokMs = (System.nanoTime() - t1) / 1e6
        var prEq = 0
        var prN = 0
        File(dh, "prompt_cases.jsonl").forEachLine(Charsets.UTF_8) { line ->
            if (line.isBlank()) return@forEachLine
            val r = JSONObject(line)
            prN++
            val ip = if (r.isNull("ip")) null else r.getString("ip")
            val ids = PromptContract.buildInputIds(tok, PromptContract.ttsPrompt(r.getString("language_id")),
                r.getString("text"), ip?.let { PromptContract.ipInstruction(it) },
                if (r.isNull("prompt_text")) null else r.getString("prompt_text"), r.getInt("latent_steps"))
            if (ids.ids.contentEquals(ints(r.getJSONArray("ids")))) prEq++
        }
        val fe = TextFrontend(cc)
        var feEq = 0
        var feN = 0
        File(dh, "frontend_cases.jsonl").forEachLine(Charsets.UTF_8) { line ->
            if (line.isBlank()) return@forEachLine
            val r = JSONObject(line)
            feN++
            val text = r.getString("text")
            val ok = if (r.isNull("lang")) {
                fe.addPeriod(text) == r.getString("add_period") && fe.stripMarkdownCode(text) == r.getString("preprocess") &&
                    fe.semanticLength(text) == r.getInt("semantic_length")
            } else {
                val want = r.getJSONArray("fragments").let { a -> List(a.length()) { a.getString(it) } }
                fe.prepare(text, r.getString("lang")) == want
            }
            if (ok) feEq++
        }
        val ist = Istft(HostMath.readF32(File(models, "istft_window.bin")))
        val head = HostMath.readF32(File(dh, "istft_E1_head.bin"))
        val ref = HostMath.readF32(File(dh, "istft_E1_wav.bin"))
        val frame = FloatArray(ist.n)
        val wav = ArrayList<FloatArray>()
        val s = ist.Stream()
        val t2 = System.nanoTime()
        val rows = head.size / HostMath.HEAD_WIDTH
        for (r in 0 until rows) {
            ist.frame(head, r * HostMath.HEAD_WIDTH, frame)
            s.push(frame)
            if (r % 32 == 31) s.flush { wav.add(it) }
        }
        s.flush { wav.add(it) }
        s.finish { wav.add(it) }
        val istMs = (System.nanoTime() - t2) / 1e6
        val got = FloatArray(wav.sumOf { it.size })
        var o = 0
        for (a in wav) {
            a.copyInto(got, o)
            o += a.size
        }
        var mx = 0.0
        var amax = 0.0
        for (i in got.indices) {
            mx = maxOf(mx, kotlin.math.abs(got[i] - ref[i]).toDouble())
            amax = maxOf(amax, kotlin.math.abs(ref[i]).toDouble())
        }
        val res = JSONObject().put("tokenizer_load_ms", loadMs)
            .put("tokenizer", JSONObject().put("cases", tokN).put("equal", tokEq).put("ms", tokMs).put("mismatches", bad))
            .put("prompt_contract", JSONObject().put("cases", prN).put("equal", prEq))
            .put("text_frontend", JSONObject().put("cases", feN).put("equal", feEq))
            .put("istft", JSONObject().put("rows", rows).put("samples", got.size).put("same_length", got.size == ref.size)
                .put("max_abs_diff", mx).put("absmax", amax).put("ratio", mx / amax).put("ms", istMs))
        File(outDir, "host.json").writeText(res.toString(1))
        log("host tokenizer $tokEq/$tokN, prompt $prEq/$prN, frontend $feEq/$feN, ISTFT max|d|/absmax ${mx / amax} " +
            "(${"%.0f".format(istMs)} ms for $rows rows), tokenizer load ${"%.0f".format(loadMs)} ms")
    }
}
