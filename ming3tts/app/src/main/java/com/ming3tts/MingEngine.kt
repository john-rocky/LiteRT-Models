package com.ming3tts

import android.util.Log
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Random
import java.util.concurrent.CancellationException
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.Semaphore
import kotlin.math.min

/** One graph of a placement: where it runs and which file it loads. */
class RoleSpec(val backend: Backend, val file: String) {
    /** Latents per decoder call: dec_c<N>_*.tflite decodes N (dec_c8 and dec_c16 were exported). */
    val decChunk: Int get() = Regex("dec_c(\\d+)").find(file)?.groupValues?.get(1)?.toInt() ?: HostMath.DEC_CHUNK
}

/**
 * Where each graph runs. Fused = `frame2` (LM step + stop head + 10-step flow + aggregator in one
 * call, plus the LM prefill signature); split = the LM (`lm2` with prefill + step, or the two
 * single-signature files `prefill` + `step` for the NPU) + `flow` + `agg`, with the stop head on
 * the host. The VAE decoder `dec_c8` decodes 8 latents per call either way.
 */
class Placement(val name: String, val roles: Map<String, RoleSpec>) {
    val fused = "frame" in roles
    val usesNpu = roles.values.any { it.backend == Backend.NPU }

    /** Where each graph runs, in plain words for the screen (logs and self-test records use [describe]). */
    fun summary(): String {
        val parts = ArrayList<String>()
        for ((role, r) in roles) {
            // the two single-signature LM files on one backend read as one language model
            val pair = if (role == "prefill") "step" else if (role == "step") "prefill" else null
            if (pair != null && roles[pair]?.backend == r.backend) {
                if (role == "prefill") parts.add("${ROLE_NAMES.getValue("lm")} ${r.backend.label}")
            } else {
                parts.add("${ROLE_NAMES.getValue(role)} ${r.backend.label}")
            }
        }
        return parts.joinToString(" · ")
    }

    /** NPU graphs of this placement; with more than one, [MingEngine.prepareNpu] compiles them one at a time. */
    val npuRoles: List<String> get() = roles.filterValues { it.backend == Backend.NPU }.keys.toList()

    fun describe() = name + ": " + roles.entries.joinToString(", ") { (role, r) ->
        "$role ${r.backend.label}" + if (r.file != defaultFile(role)) " (${r.file})" else ""
    }

    /** The same placement with every NPU graph moved to the GPU (phones or APKs without the NPU). */
    fun withoutNpu(): Placement {
        if (!usesNpu) return this
        val out = LinkedHashMap<String, RoleSpec>()
        for ((role, r) in roles) {
            if (r.backend != Backend.NPU) {
                out[role] = r
                continue
            }
            when (role) {
                // The NPU LM is two single-signature files; the GPU runs the two-signature lm2.
                "prefill", "step" -> out["lm"] = RoleSpec(Backend.GPU32, defaultFile("lm"))
                // the same file on the GPU (the decoder keeps its chunk: dec_c16 stays dec_c16)
                "flow", "dec" -> out[role] = RoleSpec(Backend.GPU, r.file)
                else -> out[role] = RoleSpec(Backend.GPU32, r.file)
            }
        }
        return Placement("$name-gpu", out)
    }

    fun requiredFiles(): List<String> = roles.values.map { it.file }.distinct() + HOST_FILES

    companion object {
        /**
         * Q2D16 = the fastest placement that passed every device check on the Galaxy S26 (
         * LM + aggregator on the GPU at FP32, the flow on the NPU, the decoder on the GPU on its own
         * thread, 16 latents per decoder call). Without a usable NPU it runs as Q2D16-gpu (the flow
         * on the GPU at its default precision).
         */
        const val DEFAULT = "Q2D16"

        /**
         * Q4D16 = every graph on the NPU, the faster placement the screen offers where the NPU is usable
         * (the LM as the two single-signature epsdiv files, the decoder at 16 latents per call). The NPU
         * computes in fp16: whole sentences stay within the audio checks, single graphs miss their
         * per-graph limits (numbers on the model card).
         */
        const val FAST = "Q4D16"

        val FILES = mapOf(
            "frame" to "frame2_wfp16.tflite",
            "lm" to "lm2_wfp16.tflite",
            "prefill" to "lm2_epsdiv_prefill_wfp16.tflite",
            "step" to "lm2_epsdiv_step_wfp16.tflite",
            "flow" to "flow_r4b_s10_wfp16.tflite",
            "agg" to "agg_r4b_wfp16.tflite",
            "dec" to "dec_c8_wfp16.tflite",
        )

        fun defaultFile(role: String) = FILES.getValue(role)

        val ROLE_NAMES = mapOf(
            "frame" to "LM step + flow", "lm" to "language model", "prefill" to "LM prefill", "step" to "LM step",
            "flow" to "flow", "agg" to "aggregator", "dec" to "decoder",
        )

        /**
         * Placements measured during the conversion (P*: the first exports, Q*: the final files) and custom strings
         * like "lm:gpu32,flow:npu,agg:gpu32,dec:gpu" or "flow:gpu@flow_wfp16.tflite". The P
         * presets keep the first flow and aggregator files they were measured with.
         */
        val PRESETS = mapOf(
            "P1" to "frame:gpu32,dec:gpu",
            "P2" to "lm:gpu32,flow:gpu32@flow_wfp16.tflite,agg:gpu32@agg_wfp16.tflite,dec:gpu",
            "P3" to "lm:cpu,flow:gpu32@flow_wfp16.tflite,agg:gpu32@agg_wfp16.tflite,dec:gpu",
            "P4" to "frame:gpu16a32,dec:gpu",
            "P5" to "lm:gpu32,flow:gpu@flow_wfp16.tflite,agg:gpu32@agg_wfp16.tflite,dec:gpu",
            "Q1" to "lm:gpu32,flow:gpu,agg:gpu32,dec:gpu",
            "Q1C" to "lm:gpu32,flow:gpu,agg:gpu32,dec:cpu",
            "Q2" to "lm:gpu32,flow:npu,agg:gpu32,dec:gpu",
            "Q2C" to "lm:gpu32,flow:npu,agg:gpu32,dec:cpu",
            "Q2D16" to "lm:gpu32,flow:npu,agg:gpu32,dec:gpu@dec_c16_wfp16.tflite",
            "Q4" to "prefill:npu,step:npu,flow:npu,agg:npu,dec:npu",
            "Q4D16" to "prefill:npu,step:npu,flow:npu,agg:npu,dec:npu@dec_c16_wfp16.tflite",
            "Q5" to "lm:cpu,flow:cpu,agg:cpu,dec:cpu",
        )

        fun parse(s: String?): Placement {
            val key = (s ?: DEFAULT).uppercase()
            val spec = PRESETS[key] ?: s!!
            val roles = LinkedHashMap<String, RoleSpec>()
            for (part in spec.split(',').map { it.trim() }.filter { it.isNotEmpty() }) {
                val (role, rest) = part.split(':', limit = 2)
                require(role in FILES) { "unknown graph role '$role'" }
                val at = rest.indexOf('@')
                val be = Backend.parse(if (at < 0) rest else rest.substring(0, at))
                roles[role] = RoleSpec(be, if (at < 0) defaultFile(role) else rest.substring(at + 1))
            }
            val lm = "lm" in roles || ("prefill" in roles && "step" in roles)
            require(("frame" in roles) != (lm && "flow" in roles && "agg" in roles)) {
                "a placement needs either frame, or lm (or prefill + step) + flow + agg"
            }
            require(!("lm" in roles && ("prefill" in roles || "step" in roles))) { "lm and prefill / step exclude each other" }
            require("dec" in roles) { "a placement needs dec" }
            return Placement(if (key in PRESETS) key else "custom", roles)
        }

        val HOST_FILES = listOf(
            "embed_tokens_f16.bin", "temb_epss10.bin", "istft_window.bin", "stop_head.bin",
            "char_classes.bin", "tokenizer_vocab.txt", "tokenizer_merges.txt", "tokenizer_added.txt",
        )
    }
}

/** A voice file made on the Mac (prompt latents + projected rows + speaker row); self-test only. */
class VoiceFile(val promptText: String, val steps: Int, val promptLatent: FloatArray,
                val latentRows: FloatArray, val speakerRow: FloatArray) {
    companion object {
        fun load(bin: File, txt: File): VoiceFile {
            val meta = txt.readLines(Charsets.UTF_8).filter { it.isNotEmpty() }.associate {
                val t = it.indexOf('\t')
                it.substring(0, t) to it.substring(t + 1)
            }
            val steps = meta.getValue("prompt_latent").split(',')[0].toInt()
            val all = HostMath.readF32(bin)
            val a = steps * HostMath.LATENT
            val b = a + steps * HostMath.HIDDEN
            return VoiceFile(meta.getValue("prompt_text"), steps, all.copyOfRange(0, a),
                all.copyOfRange(a, b), all.copyOfRange(b, b + HostMath.HIDDEN))
        }
    }
}

/** One synthesis request: a built-in (IP) voice, or a voice file (self-test fixtures). */
class Request(val text: String, val languageId: String, val ip: String?, val voice: VoiceFile? = null)

/** Per-fragment measurements (ms are wall times; the decoder's are on its own thread when it has one). */
class FragmentStats {
    var promptTokens = 0
    var steps = 0
    var stopped = false
    var samples = 0
    var prefillMs = 0.0
    var frameMs = ArrayList<Double>()      // graph time per frame (incl. host stop head, agg)
    var frameWriteMs = ArrayList<Double>()
    var lmMs = ArrayList<Double>()         // split placements: LM step, flow and aggregator calls
    var flowMs = ArrayList<Double>()
    var aggMs = ArrayList<Double>()
    var decMs = ArrayList<Double>()        // decoder call per chunk
    var istftMs = ArrayList<Double>()      // host ISTFT per chunk
    var firstPcmMs = -1.0                  // from generate() start
    var framesEndMs = 0.0                  // last frame done (the decoder may still be working)
    var totalMs = 0.0
    val stopP1 = ArrayList<Float>()
    var ids: IntArray = IntArray(0)
    var latents: ArrayList<FloatArray>? = null
}

/**
 * Ming-3.0-tts-0.5B on LiteRT CompiledModel. The host loop is contract.json's
 * `host_loop` (scripts/host_mirror.py on the Mac): prompt ids -> embedding rows -> prefill in 64-token
 * chunks -> one frame per 80 ms latent -> the decoder every 8 latents once the next latent
 * exists -> streaming ISTFT -> 44.1 kHz float PCM. Graph calls run on the caller's thread; with
 * [decoderThread] the decoder and the ISTFT run on the engine's own thread, next to the frames,
 * and [generate]'s PCM callback is called from that thread. A GPU runs one graph at a time, so a
 * GPU decoder chunk next to GPU frames delays the frame it overlaps; with [decoderGate] each chunk
 * starts when the next flow call starts (the GPU's idle window while the flow runs on the NPU).
 */
class MingEngine(
    private val dir: File,
    val placement: Placement,
    shareConstants: Boolean = false,
    acc: Accelerators? = null,
    val decoderThread: Boolean = true,
    val decoderGate: Boolean = false,
    graphFactory: ((role: String, file: File, backend: Backend) -> GraphCall)? = null,
) : Closeable {
    val cc = CharClasses.load(File(dir, "char_classes.bin"))
    val tokenizer = QwenTokenizer(
        File(dir, "tokenizer_vocab.txt"), File(dir, "tokenizer_merges.txt"),
        File(dir, "tokenizer_added.txt"), cc,
    )
    val frontend = TextFrontend(cc)
    private val embedChannel = RandomAccessFile(File(dir, "embed_tokens_f16.bin"), "r").channel
    private val embed: MappedByteBuffer =
        embedChannel.map(FileChannel.MapMode.READ_ONLY, 0, embedChannel.size()).also { it.order(ByteOrder.LITTLE_ENDIAN) }
    private val temb = HostMath.readF32(File(dir, "temb_epss10.bin"))
    val istft = Istft(HostMath.readF32(File(dir, "istft_window.bin")))
    private val stopHead = StopHead(File(dir, "stop_head.bin"))
    private val decExec: ExecutorService? =
        if (decoderThread) Executors.newSingleThreadExecutor { Thread(it, "ming3tts-decoder") } else null

    val graphs = LinkedHashMap<String, GraphCall>()

    /** Released when a flow call starts; a gated decoder chunk waits for it. */
    private val flowWindow = Semaphore(0)

    init {
        require(temb.size == HostMath.FLOW_STEPS * HostMath.TEMB)
        require(graphFactory != null || acc != null) { "a LiteRT engine needs the process's accelerators" }
        // NPU graphs first: the first load of each is a JIT compile with a high memory peak, best
        // taken before the GPU graphs hold their weights.
        val order = placement.roles.entries.sortedBy { if (it.value.backend == Backend.NPU) 0 else 1 }
        for ((role, r) in order) {
            val file = File(dir, r.file)
            graphs[role] = graphFactory?.invoke(role, file, r.backend)
                ?: Graph(acc!!, file, r.backend, Graph.signaturesOf(role), shareConstants)
        }
    }

    val compileMs: Map<String, Double> get() = graphs.mapValues { it.value.compileMs }

    /** Latents per decoder call of this placement's decoder file. */
    val decChunk = placement.roles.getValue("dec").decChunk

    // ---- host state, reused across fragments ------------------------------
    private val pk = FloatArray(HostMath.KV * HostMath.LM_PAST * HostMath.HEAD_DIM)
    private val pv = FloatArray(pk.size)
    private val cos64 = FloatArray(64 * 64)
    private val sin64 = FloatArray(64 * 64)
    private val cos1 = FloatArray(64)
    private val sin1 = FloatArray(64)
    private val mask1 = FloatArray(HostMath.LM_PAST + 1)

    fun embedRow(id: Int, out: FloatArray, off: Int) {
        var b = id * HostMath.HIDDEN * 2
        for (j in 0 until HostMath.HIDDEN) {
            out[off + j] = HostMath.halfToFloat(embed.getShort(b).toInt() and 0xffff)
            b += 2
        }
    }

    /**
     * The VAE decoder of one fragment: chunks of 8 latents (plus one on each side for the x4
     * upsample) with a 64-position KV cache per layer, then the streaming ISTFT. Owns all of its
     * state, so it can run on the decoder thread while the frames go on.
     */
    private inner class Decoder(private val st: FragmentStats, private val sink: (FloatArray) -> Unit) {
        val lats = ArrayList<FloatArray>()
        private val chunk = decChunk
        private val positions = HostMath.POS_PER_LATENT * chunk
        private var a = 0
        private var dpk = FloatArray(HostMath.KV * HostMath.DEC_PAST * HostMath.HEAD_DIM)
        private var dpv = FloatArray(dpk.size)
        private val stream = istft.Stream()
        private val cosC = FloatArray(positions * 64)
        private val sinC = FloatArray(positions * 64)
        private val window = FloatArray((chunk + 2) * HostMath.LATENT)
        private val frameBuf = FloatArray(3528)
        private val dec = graphs["dec"]!!

        /** Decode every chunk whose next latent exists; with [last] also the rest. */
        fun drain(last: Boolean) {
            while (lats.size >= a + chunk + 1 || (last && a < lats.size)) {
                if (decoderGate && !last) {
                    // wait for the next flow call (the frames release it; the end of the fragment too)
                    flowWindow.drainPermits()
                    flowWindow.acquire()
                }
                val td = System.nanoTime()
                HostMath.decLatentWindow(lats, a, chunk, window)
                HostMath.rope(4 * a, positions, cosC, sinC)
                val o = dec.call("dec", arrayOf(window, cosC, sinC, HostMath.decMask(4 * a, positions), dpk, dpv))
                dpk = o[1]!!
                dpv = o[2]!!
                val head = o[0]!!
                val ti = System.nanoTime()
                st.decMs.add((ti - td) / 1e6)
                val real = if (last) min(chunk, lats.size - a) else chunk
                for (r in 0 until 4 * real) {
                    istft.frame(head, r * HostMath.HEAD_WIDTH, frameBuf)
                    stream.push(frameBuf)
                }
                stream.flush(sink)
                st.istftMs.add((System.nanoTime() - ti) / 1e6)
                a += chunk
            }
        }

        fun finish() = stream.finish(sink)
    }

    /** Feeds a [Decoder] on the decoder thread: latents in order, then the end (or an abort). */
    private class DecoderFeed(private val d: MingEngine.Decoder, private val window: Semaphore) {
        private val queue = LinkedBlockingQueue<FloatArray>()
        @Volatile private var abort = false
        lateinit var future: Future<*>

        fun start(exec: ExecutorService) {
            future = exec.submit(Runnable {
                while (true) {
                    val x = queue.take()
                    if (abort) break
                    if (x === END) {
                        d.drain(true)
                        d.finish()
                        break
                    }
                    d.lats.add(x)
                    d.drain(false)
                }
            })
        }

        fun add(latent: FloatArray) {
            if (future.isDone) await()     // the decoder failed: surface its error now
            queue.put(latent)
        }

        fun end() {
            queue.put(END)
            window.release()       // a gated chunk waiting for a flow call that will not come
        }

        fun abort() {
            abort = true
            queue.put(END)
            window.release()
            runCatching { future.get() }
        }

        fun await() {
            try {
                future.get()
            } catch (e: ExecutionException) {
                throw e.cause ?: e
            }
        }

        companion object {
            private val END = FloatArray(0)
        }
    }

    /**
     * Synthesize one fragment. [noise] gives the 64 Gaussian values of frame `step`; [onPcm]
     * receives final 44.1 kHz samples as they become available (on the decoder thread when the
     * engine has one); [cancelled] is polled per frame.
     */
    fun generate(
        req: Request,
        noise: (Int) -> FloatArray,
        onPcm: (FloatArray) -> Unit,
        cancelled: () -> Boolean = { false },
        keepLatents: Boolean = false,
        maxDecodeSteps: Int = 1000,
    ): FragmentStats {
        val st = FragmentStats()
        val t0 = System.nanoTime()
        fun ms(a: Long, b: Long) = (b - a) / 1e6
        val v = req.voice
        val built = PromptContract.buildInputIds(
            tokenizer, PromptContract.ttsPrompt(req.languageId), req.text,
            req.ip?.let { PromptContract.ipInstruction(it) }, v?.promptText, v?.steps ?: 0,
        )
        val ids = built.ids
        val L = ids.size
        st.ids = ids
        st.promptTokens = L
        require(L < HostMath.MAX_CACHE_LEN - 1) { "prompt of $L tokens leaves no room to speak" }
        val H = HostMath.HIDDEN
        val rows = FloatArray(L * H)
        for (i in 0 until L) embedRow(ids[i], rows, i * H)
        for (p in built.speakerPositions) {
            if (v == null) rows.fill(0f, p * H, (p + 1) * H) else System.arraycopy(v.speakerRow, 0, rows, p * H, H)
        }
        if (v != null) System.arraycopy(v.latentRows, 0, rows, built.latentStart * H, v.steps * H)

        // ---- prefill rows[0 : L-1] in 64-token chunks
        pk.fill(0f)
        pv.fill(0f)
        val pre = graphs["frame"] ?: graphs["lm"] ?: graphs["prefill"]!!
        val x64 = FloatArray(64 * H)
        val slot = HostMath.LM_PAST * HostMath.HEAD_DIM
        var start = 0
        while (start < L - 1) {
            val n = min(64, L - 1 - start)
            x64.fill(0f)
            System.arraycopy(rows, start * H, x64, 0, n * H)
            HostMath.rope(start, 64, cos64, sin64)
            val out = pre.call("prefill", arrayOf(x64, cos64, sin64, HostMath.lmMask(start, 64), pk, pv), intArrayOf(1, 2))
            val k = out[1]!!
            val vv = out[2]!!
            for (g in 0 until HostMath.KV) {
                System.arraycopy(k, g * 64 * 64, pk, g * slot + start * 64, n * 64)
                System.arraycopy(vv, g * 64 * 64, pv, g * slot + start * 64, n * 64)
            }
            start += 64
            if (cancelled()) throw CancellationException("cancelled")
        }
        val tPre = System.nanoTime()
        st.prefillMs = ms(t0, tPre)

        // ---- frames
        var hist = if (v != null) v.promptLatent.copyOfRange((v.steps - 1) * 64, v.steps * 64) else FloatArray(64)
        var x = rows.copyOfRange((L - 1) * H, L * H)
        val lats = ArrayList<FloatArray>()
        val budget = HostMath.MAX_CACHE_LEN - L
        val maxSteps = min(maxDecodeSteps, budget)
        val sink: (FloatArray) -> Unit = { pcm ->
            if (st.firstPcmMs < 0) st.firstPcmMs = ms(t0, System.nanoTime())
            st.samples += pcm.size
            onPcm(pcm)
        }
        val decoder = Decoder(st, sink)
        flowWindow.drainPermits()
        val feed = decExec?.let { DecoderFeed(decoder, flowWindow).also { f -> f.start(it) } }
        val stepGraph = graphs["lm"] ?: graphs["step"]
        try {
            for (step in 0 until maxSteps) {
                if (cancelled()) throw CancellationException("cancelled")
                val pos = L - 1 + step
                HostMath.rope(pos, 1, cos1, sin1)
                HostMath.lmMask(pos, 1, mask1)
                val nz = noise(step)
                val tf = System.nanoTime()
                val latent: FloatArray
                var eNext: FloatArray? = null
                val logits: FloatArray
                var writeMs: Double
                if (placement.fused) {
                    val g = graphs["frame"]!!
                    val out = g.call("frame", arrayOf(x, hist, nz, temb, cos1, sin1, mask1, pk, pv))[0]!!
                    writeMs = g.lastWriteMs
                    latent = out.copyOfRange(HostMath.OUT_LATENT, HostMath.OUT_LATENT + 64)
                    eNext = out.copyOfRange(HostMath.OUT_ENEXT, HostMath.OUT_ENEXT + H)
                    logits = out.copyOfRange(HostMath.OUT_STOP, HostMath.OUT_STOP + 2)
                    for (gi in 0 until HostMath.KV) {
                        System.arraycopy(out, HostMath.OUT_K + gi * 64, pk, gi * slot + pos * 64, 64)
                        System.arraycopy(out, HostMath.OUT_V + gi * 64, pv, gi * slot + pos * 64, 64)
                    }
                } else {
                    val lm = stepGraph!!
                    val o = lm.call("step", arrayOf(x, cos1, sin1, mask1, pk, pv))
                    writeMs = lm.lastWriteMs
                    val z = o[0]!!
                    val k = o[1]!!
                    val vv = o[2]!!
                    for (gi in 0 until HostMath.KV) {
                        System.arraycopy(k, gi * 64, pk, gi * slot + pos * 64, 64)
                        System.arraycopy(vv, gi * 64, pv, gi * slot + pos * 64, 64)
                    }
                    logits = stopHead.logits(z)
                    val tl = System.nanoTime()
                    st.lmMs.add(ms(tf, tl))
                    val fl = graphs["flow"]!!
                    if (decoderGate) flowWindow.release()
                    latent = fl.call("flow", arrayOf(z, hist, nz, temb))[0]!!
                    st.flowMs.add(ms(tl, System.nanoTime()))
                    writeMs += fl.lastWriteMs
                }
                var frameMs = ms(tf, System.nanoTime())
                lats.add(latent)
                val (stop, p1) = HostMath.stopDecision(logits[0], logits[1], step)
                st.stopP1.add(p1)
                val last = stop || step == maxSteps - 1
                if (feed != null) {
                    feed.add(latent)
                    if (last) feed.end()
                } else {
                    decoder.lats.add(latent)
                    decoder.drain(last)
                }
                if (stop) {
                    st.stopped = true
                    st.frameMs.add(frameMs)
                    st.frameWriteMs.add(writeMs)
                    break
                }
                if (!last) {
                    if (eNext == null) {
                        val ta = System.nanoTime()
                        val ag = graphs["agg"]!!
                        eNext = ag.call("agg", arrayOf(latent))[0]!!
                        val aggMs = ms(ta, System.nanoTime())
                        st.aggMs.add(aggMs)
                        frameMs += aggMs
                        writeMs += ag.lastWriteMs
                    }
                    x = eNext
                    hist = latent
                }
                st.frameMs.add(frameMs)
                st.frameWriteMs.add(writeMs)
            }
            st.framesEndMs = ms(t0, System.nanoTime())
            if (feed != null) feed.await() else decoder.finish()
        } catch (e: Throwable) {
            feed?.abort()
            throw e
        }
        st.steps = lats.size
        if (keepLatents) st.latents = lats
        st.totalMs = ms(t0, System.nanoTime())
        Log.i(Graph.TAG, "fragment: $L prompt tokens, ${st.steps} steps (stop=${st.stopped}), " +
            "${st.samples} samples, prefill ${"%.0f".format(st.prefillMs)} ms, frame median " +
            "${"%.1f".format(median(st.frameMs))} ms (lm ${"%.1f".format(median(st.lmMs))}, flow " +
            "${"%.1f".format(median(st.flowMs))}, agg ${"%.1f".format(median(st.aggMs))}), dec median " +
            "${"%.1f".format(median(st.decMs))} ms${if (feed != null) " on its own thread" else ""}" +
            "${if (decoderGate) " (gated to the flow calls)" else ""}, frames end " +
            "${"%.0f".format(st.framesEndMs)} ms, total ${"%.0f".format(st.totalMs)} ms")
        return st
    }

    override fun close() {
        decExec?.shutdown()
        graphs.values.forEach { it.close() }
        graphs.clear()
        embedChannel.close()
    }

    companion object {
        fun median(a: List<Double>): Double {
            if (a.isEmpty()) return Double.NaN
            val s = a.sorted()
            return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
        }

        /**
         * Compiles each NPU graph of [placement] alone and closes it, so that the engine then loads them all
         * from the NPU's compiled cache. The first load of an NPU graph compiles it on the phone with a memory
         * peak of several GB; one at a time, the peak is one graph's instead of piling up on the graphs already
         * loaded. Once compiled, each load takes about a second. [progress] gets (graph, of).
         */
        fun prepareNpu(acc: Accelerators, dir: File, placement: Placement, progress: (Int, Int) -> Unit = { _, _ -> }) {
            val roles = placement.npuRoles
            for ((i, role) in roles.withIndex()) {
                progress(i + 1, roles.size)
                Graph(acc, File(dir, placement.roles.getValue(role).file), Backend.NPU, Graph.signaturesOf(role)).close()
            }
        }

        /** Files of [placement] that are not in [dir] (listing only: no stat of each file). */
        fun missingFiles(dir: File, placement: Placement): List<String> {
            val have = dir.list()?.toSet() ?: emptySet()
            return placement.requiredFiles().filter { it !in have }
        }

        /** Frame noise for normal synthesis: N(0, 1), one 64-vector per frame. */
        fun gaussianNoise(seed: Long): (Int) -> FloatArray {
            val r = Random(seed)
            return { FloatArray(64) { r.nextGaussian().toFloat() } }
        }
    }
}
