package com.audio8tts

import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.SplittableRandom
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Audio8-TTS-Preview-0.6b on LiteRT: a line-by-line Kotlin port of the reference host loop
 * `audio8_tts_work/audio8_tts_litert.py` (class Audio8LiteRT) on the CompiledModel API.
 *
 * Graphs (FINDINGS §3): slow AR `slow_ar_int8.tflite` (prefill_256 + decode, KV cache 2048 as 48 f32 tensors
 * [1,2,2048,64]), fast AR `fast_ar_int8.tflite` (step, 10 calls per frame, KV [4,1,2,10,64]), codec decoder
 * (decode: codes i32 [1,10,T] -> wav f32 [1,1,T*2048]) and the codec encoder (encode: audio [1,1,442368] ->
 * codes [1,10,216]). Every buffer is addressed by tensor NAME through run(Map, Map, signature): the positional
 * lists follow the subgraph's input order, not the signature's (NOTES.md trap).
 *
 * KV cache (SUPERVISOR design decision 1): two persistent sets of 48 TensorBuffers; each slow call reads one set and
 * writes the other (ping-pong), so no cache bytes cross the JNI boundary per step. The fast AR does the same with its
 * two small sets plus a write-once zero set for the first call of every frame.
 */
class Audio8Engine(
    private val dir: File,
    private val gpuCacheDir: File,
    val constants: PromptConstants,
    val threads: Int = 4,
    /** "output": KV sets from createOutputBuffer (as the Qwen3-TTS sample); "input": from createInputBuffer. */
    val kvOwner: String = "output",
    /** XNNPACK weight cache file for the slow graph (null = none). */
    val slowWeightCache: String? = null,
    /** "auto": GPU fp16 codec if it passes the output check, else CPU int8; "gpu": GPU without the check; "cpu". */
    val codecMode: String = "auto",
    /** Codes [10][n] (n <= 128) for the GPU codec output check (the bundled voice's codes). */
    private val checkCodes: Array<IntArray>? = null,
) : AutoCloseable {

    companion object {
        const val TAG = "Audio8Demo"
        const val SEM_BEGIN = 151678
        const val SEM_END = 155773
        const val EOS = 151645
        const val NUM_CB = 10
        const val ROWS = NUM_CB + 1
        const val CB_SIZE = 4096
        const val SEM_LOGITS = 4097          // lm_head rows: semantic range (4096) + EOS
        const val DIM = 896
        const val CACHE = 2048
        const val MAX_SEQ = 2048
        const val KV_FLOATS = 2 * CACHE * 64 // [1,2,2048,64]
        const val FAST_KV_FLOATS = 4 * 1 * 2 * NUM_CB * 64
        const val PREFILL_T = 256
        const val SR = 44100
        const val FRAME = 2048
        const val NEG = -1e9f
        const val ENC_SAMPLES = 442368
        const val ENC_FRAMES = 216
        const val MAX_NEW = 512
        const val CODEC_CTX = 128
        const val SLOW = "slow_ar_int8.tflite"
        const val FAST = "fast_ar_int8.tflite"
        const val CODEC_GPU_128 = "codec_decoder_fp16_T128.tflite"
        const val CODEC_GPU_192 = "codec_decoder_fp16_T192.tflite"
        const val CODEC_CPU_128 = "codec_decoder_int8_T128.tflite"
        const val ENCODER = "codec_encoder_fp16_10s.tflite"
        val KV_NAMES: List<String> = (0 until 24).flatMap { listOf("k_$it", "v_$it") }

        fun nowNs(): Long = SystemClock.elapsedRealtimeNanos()
        fun ms(ns: Long): Double = ns / 1e6

        /** Python's str.isspace() closely enough for clean(): Unicode whitespace incl. U+3000 and U+00A0. */
        private fun isPySpace(c: Char) = Character.isWhitespace(c) || Character.isSpaceChar(c) || c == '\u0085'

        /** clean(t) = " ".join(str(t).strip().split()) */
        fun clean(t: String): String {
            val words = ArrayList<String>()
            val sb = StringBuilder()
            for (c in t) {
                if (isPySpace(c)) {
                    if (sb.isNotEmpty()) { words.add(sb.toString()); sb.setLength(0) }
                } else sb.append(c)
            }
            if (sb.isNotEmpty()) words.add(sb.toString())
            return words.joinToString(" ")
        }

        fun top2(x: FloatArray): Triple<Int, Int, Float> {
            var i1 = 0
            for (i in 1 until x.size) if (x[i] > x[i1]) i1 = i
            var i2 = if (i1 == 0) 1 else 0
            for (i in 0 until x.size) if (i != i1 && x[i] > x[i2]) i2 = i
            return Triple(i1, i2, x[i1] - x[i2])
        }

        fun allowed(idx: Int): Int = if (idx < CB_SIZE) SEM_BEGIN + idx else EOS
    }

    val tokenizer: QwenBpeTokenizer
    val tokenizerLoadMs: Double
    val createMs = LinkedHashMap<String, Double>()
    val warmupMs = LinkedHashMap<String, Double>()
    var codecBackend = "none"
        private set
    var gpuError: String? = null
        private set
    /** GPU codec output check: correlation with the CPU int8 codec on the same codes, and GPU(codes) vs GPU(zeros). */
    var gpuCheck: Map<String, Any>? = null
        private set

    // ---------------- model creation ----------------

    private fun file(name: String): File = File(dir, name).also { check(it.isFile) { "model missing: ${it.path}" } }

    private fun cpuModel(name: String, weightCache: String? = null): CompiledModel {
        val o = CompiledModel.Options(Accelerator.CPU)
        o.cpuOptions = CompiledModel.CpuOptions(threads, null, weightCache)
        return CompiledModel.create(file(name).absolutePath, o, null)
    }

    /**
     * GPU (OpenCL) with the CPU accelerator as the fallback for the ops the GPU delegate rejects (the codec's fp16
     * DEQUANTIZE + EMBEDDING_LOOKUP, 133 of 1069 nodes). Options(Accelerator.GPU) alone fails at create on the S26
     * ("Failed to compile model" after the delegate has taken 936 nodes), measured in round 1.
     */
    private fun gpuModel(
        name: String,
        precision: CompiledModel.GpuOptions.Precision? = null,
        storage: CompiledModel.GpuOptions.BufferStorageType? = null,
        cacheKey: String = name,
    ): CompiledModel {
        val o = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
        o.cpuOptions = CompiledModel.CpuOptions(threads, null, null)
        o.gpuOptions = CompiledModel.GpuOptions(precision = precision, bufferStorageType = storage,
            serializationDir = gpuCacheDir.absolutePath, modelCacheKey = cacheKey)
        return CompiledModel.create(file(name).absolutePath, o, null)
    }

    // ---------------- codec probe (round 1 diagnosis of the GPU codec output) ----------------

    class ProbeRow(val variant: String, val createMs: Double, val runMs: List<Double>, val corrVsCpu: Double,
                   val corrJaVsZeros: Double, val rms: Double, val inTypes: String, val outTypes: String, val error: String?)

    private fun pearson(a: FloatArray, b: FloatArray): Double {
        val n = min(a.size, b.size)
        var ma = 0.0; var mb = 0.0
        for (i in 0 until n) { ma += a[i]; mb += b[i] }
        ma /= n; mb /= n
        var sab = 0.0; var saa = 0.0; var sbb = 0.0
        for (i in 0 until n) { val x = a[i] - ma; val y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y }
        return sab / kotlin.math.sqrt(saa * sbb)
    }

    /**
     * Runs the T128 codec on [codes] ([10][n], n <= 128) on the CPU int8 graph (reference) and on GPU variants, and
     * reports each variant's correlation with the reference and whether its output depends on the input at all
     * (JA codes vs all-zero codes).
     */
    fun codecProbe(codes: Array<IntArray>): List<ProbeRow> {
        val n = codes[0].size
        val buf = IntArray(NUM_CB * 128)
        for (q in 0 until NUM_CB) System.arraycopy(codes[q], 0, buf, q * 128, n)
        val zeros = IntArray(NUM_CB * 128)
        val rows = ArrayList<ProbeRow>()
        val ref = codec128!!.let { c -> if (c.backend == "CPU") c.run(buf) else null } ?: run {
            val m = cpuModel(CODEC_CPU_128)
            Codec(CODEC_CPU_128, m, 128, "CPU").use { it.run(buf) }
        }
        fun types(m: CompiledModel, input: Boolean): String = runCatching {
            val r = if (input) m.getInputBufferRequirements("codes", "decode") else m.getOutputBufferRequirements("wav", "decode")
            "${r.supportedTypes} size=${r.bufferSize}"
        }.getOrElse { "error: $it" }
        data class V(val name: String, val precision: CompiledModel.GpuOptions.Precision?,
                     val storage: CompiledModel.GpuOptions.BufferStorageType?, val warmZeros: Boolean, val freshInput: Boolean)
        val variants = listOf(
            V("gpu_default_warm_zeros", null, null, true, false),
            V("gpu_default_cold", null, null, false, false),
            V("gpu_default_fresh_input_per_call", null, null, false, true),
            V("gpu_fp32", CompiledModel.GpuOptions.Precision.FP32, null, false, false),
            V("gpu_buffer_storage", null, CompiledModel.GpuOptions.BufferStorageType.BUFFER, false, false),
        )
        for (v in variants) {
            var m: CompiledModel? = null
            try {
                val t0 = nowNs()
                m = gpuModel(CODEC_GPU_128, v.precision, v.storage, "${CODEC_GPU_128}_${v.name}")
                val createMs = ms(nowNs() - t0)
                val inT = types(m, true)
                val outT = types(m, false)
                val times = ArrayList<Double>()
                fun runOnce(x: IntArray): FloatArray {
                    val inB = m.createInputBuffer("codes", "decode")
                    val outB = m.createOutputBuffer("wav", "decode")
                    inB.writeInt(x)
                    val r0 = nowNs()
                    m.run(mapOf("codes" to inB), mapOf("wav" to outB), "decode")
                    times.add(ms(nowNs() - r0))
                    val o = outB.readFloat()
                    inB.close(); outB.close()
                    return o
                }
                val outJa: FloatArray
                val outZero: FloatArray
                if (v.freshInput) {
                    outJa = runOnce(buf)
                    outZero = runOnce(zeros)
                } else {
                    val inB = m.createInputBuffer("codes", "decode")
                    val outB = m.createOutputBuffer("wav", "decode")
                    fun go(x: IntArray): FloatArray {
                        inB.writeInt(x)
                        val r0 = nowNs()
                        m.run(mapOf("codes" to inB), mapOf("wav" to outB), "decode")
                        times.add(ms(nowNs() - r0))
                        return outB.readFloat()
                    }
                    if (v.warmZeros) go(zeros)
                    outJa = go(buf)
                    outZero = go(zeros)
                    inB.close(); outB.close()
                }
                var sq = 0.0
                for (i in 0 until n * FRAME) sq += outJa[i] * outJa[i]
                val row = ProbeRow(v.name, createMs, times, pearson(outJa.copyOf(n * FRAME), ref.copyOf(n * FRAME)),
                    pearson(outJa, outZero), kotlin.math.sqrt(sq / (n * FRAME)), inT, outT, null)
                rows.add(row)
                Log.i(TAG, "CODEC_PROBE ${v.name} create_ms=$createMs run_ms=$times corr_vs_cpu=${row.corrVsCpu} " +
                    "corr_ja_vs_zeros=${row.corrJaVsZeros} rms=${row.rms} in=$inT out=$outT")
            } catch (e: Exception) {
                Log.e(TAG, "CODEC_PROBE ${v.name} failed: $e", e)
                rows.add(ProbeRow(v.name, 0.0, emptyList(), Double.NaN, Double.NaN, Double.NaN, "", "", e.toString()))
            } finally {
                m?.close()
            }
        }
        return rows
    }

    private inline fun <T> timed(key: String, map: MutableMap<String, Double>, block: () -> T): T {
        val t0 = nowNs()
        val r = block()
        map[key] = ms(nowNs() - t0)
        return r
    }

    // ---------------- slow AR ----------------
    private val slow: CompiledModel
    private val kvA: Map<String, TensorBuffer>
    private val kvB: Map<String, TensorBuffer>
    private var cur: Map<String, TensorBuffer>
    private val decCodes: TensorBuffer
    private val decPos: TensorBuffer
    private val decMaskBuf: TensorBuffer
    private val decLogits: TensorBuffer
    private val decHidden: TensorBuffer
    private val preCodes: TensorBuffer
    private val prePos: TensorBuffer
    private val preMaskBuf: TensorBuffer
    private val preLogits: TensorBuffer
    private val preHidden: TensorBuffer
    private val decMask = FloatArray(CACHE)
    private val preMask = FloatArray(PREFILL_T * CACHE)
    private val zeroKv = FloatArray(KV_FLOATS)
    private val decInA: Map<String, TensorBuffer>   // inputs reading set A
    private val decOutB: Map<String, TensorBuffer>  // outputs writing set B
    private val decInB: Map<String, TensorBuffer>
    private val decOutA: Map<String, TensorBuffer>

    // ---------------- fast AR ----------------
    private val fast: CompiledModel
    private val fHidden: TensorBuffer
    private val fToken: TensorBuffer
    private val fLogits: TensorBuffer
    private val fIns: Array<Map<String, TensorBuffer>>
    private val fOuts: Array<Map<String, TensorBuffer>>
    private val fOwned = ArrayList<TensorBuffer>()
    private val tokenArr = IntArray(1)

    // ---------------- codec ----------------
    inner class Codec(val fileName: String, val model: CompiledModel, val t: Int, val backend: String) : AutoCloseable {
        val input: TensorBuffer = model.createInputBuffer("codes", "decode")
        val output: TensorBuffer = model.createOutputBuffer("wav", "decode")
        private val ins = mapOf("codes" to input)
        private val outs = mapOf("wav" to output)
        fun run(codes: IntArray): FloatArray {
            input.writeInt(codes)
            model.run(ins, outs, "decode")
            return output.readFloat()
        }
        override fun close() {
            input.close(); output.close(); model.close()
        }
    }

    private var codec128: Codec? = null
    private var codec192: Codec? = null
    private var codec192Failed = false

    private fun cpuCodec(): Codec {
        val m = timed("codec_int8_T128_cpu", createMs) { cpuModel(CODEC_CPU_128) }
        val c = Codec(CODEC_CPU_128, m, 128, "CPU")
        timed("codec_int8_T128_cpu", warmupMs) { c.run(IntArray(NUM_CB * 128)) }
        codecBackend = "CPU"
        return c
    }

    private fun selectCodec(): Codec {
        if (codecMode == "cpu") return cpuCodec()
        var gpu: Codec? = null
        try {
            val m = timed("codec_fp16_T128_gpu", createMs) { gpuModel(CODEC_GPU_128) }
            gpu = Codec(CODEC_GPU_128, m, 128, "GPU")
            timed("codec_fp16_T128_gpu", warmupMs) { gpu.run(IntArray(NUM_CB * 128)) }
        } catch (e: Exception) {
            gpu?.close()
            gpuError = e.toString()
            Log.e(TAG, "GPU codec failed, falling back to CPU int8 T128: $e", e)
            return cpuCodec()
        }
        val g = gpu!!
        if (codecMode == "gpu" || checkCodes == null) {
            codecBackend = "GPU"
            return g
        }
        val n = min(checkCodes[0].size, 128)
        val buf = IntArray(NUM_CB * 128)
        for (q in 0 until NUM_CB) System.arraycopy(checkCodes[q], 0, buf, q * 128, n)
        val t0 = nowNs()
        val outG = g.run(buf).copyOf(n * FRAME)
        val outGz = g.run(IntArray(NUM_CB * 128)).copyOf(n * FRAME)
        val t1 = nowNs()
        val cpu = cpuCodec()
        val t2 = nowNs()
        val outC = cpu.run(buf).copyOf(n * FRAME)
        val t3 = nowNs()
        val corr = pearson(outG, outC)
        val dep = pearson(outG, outGz)
        val pass = corr >= 0.99
        gpuCheck = linkedMapOf("frames" to n, "corr_gpu_vs_cpu_int8" to corr, "corr_gpu_codes_vs_gpu_zeros" to dep,
            "pass" to pass, "threshold" to 0.99, "gpu_runs_ms" to ms(t1 - t0), "cpu_create_warmup_ms" to ms(t2 - t1),
            "cpu_run_ms" to ms(t3 - t2))
        Log.i(TAG, "GPU_CODEC_CHECK $gpuCheck")
        return if (pass) {
            cpu.close()
            createMs.remove("codec_int8_T128_cpu"); warmupMs.remove("codec_int8_T128_cpu")
            codecBackend = "GPU"
            g
        } else {
            g.close()
            gpuError = "output check failed: corr vs CPU int8 = $corr, corr vs zero input = $dep"
            codecBackend = "CPU"
            cpu
        }
    }

    init {
        val t0 = nowNs()
        tokenizer = QwenBpeTokenizer(file("tokenizer.json"))
        tokenizerLoadMs = ms(nowNs() - t0)
        Log.i(TAG, "TOKENIZER vocab=${tokenizer.vocabSize} merges=${tokenizer.mergeCount} added=${tokenizer.addedCount} load_ms=$tokenizerLoadMs")

        slow = timed("slow_ar_int8", createMs) { cpuModel(SLOW, slowWeightCache) }
        fun kvSet(): Map<String, TensorBuffer> = KV_NAMES.associateWith {
            if (kvOwner == "input") slow.createInputBuffer(it, "decode") else slow.createOutputBuffer(it, "decode")
        }
        kvA = kvSet()
        kvB = kvSet()
        cur = kvB
        decCodes = slow.createInputBuffer("codes", "decode")
        decPos = slow.createInputBuffer("input_pos", "decode")
        decMaskBuf = slow.createInputBuffer("mask", "decode")
        decLogits = slow.createOutputBuffer("logits", "decode")
        decHidden = slow.createOutputBuffer("hidden", "decode")
        preCodes = slow.createInputBuffer("codes", "prefill_256")
        prePos = slow.createInputBuffer("input_pos", "prefill_256")
        preMaskBuf = slow.createInputBuffer("mask", "prefill_256")
        preLogits = slow.createOutputBuffer("logits", "prefill_256")
        preHidden = slow.createOutputBuffer("hidden", "prefill_256")
        fun decIn(set: Map<String, TensorBuffer>) = HashMap<String, TensorBuffer>(64).apply {
            put("codes", decCodes); put("input_pos", decPos); put("mask", decMaskBuf); putAll(set)
        }
        fun decOut(set: Map<String, TensorBuffer>) = HashMap<String, TensorBuffer>(64).apply {
            put("logits", decLogits); put("hidden", decHidden); putAll(set)
        }
        decInA = decIn(kvA); decOutB = decOut(kvB); decInB = decIn(kvB); decOutA = decOut(kvA)

        fast = timed("fast_ar_int8", createMs) { cpuModel(FAST) }
        fHidden = fast.createInputBuffer("hidden", "step")
        fToken = fast.createInputBuffer("token", "step")
        fLogits = fast.createOutputBuffer("logits", "step")
        val use1 = fast.createInputBuffer("use_hidden", "step").also { it.writeFloat(floatArrayOf(1f)) }
        val use0 = fast.createInputBuffer("use_hidden", "step").also { it.writeFloat(floatArrayOf(0f)) }
        // fast_masks[p] = 0 where j <= p else -1e9, shape [1,1,1,10]; pos p; both constant per call index.
        val posBufs = Array(NUM_CB) { p -> fast.createInputBuffer("pos", "step").also { it.writeInt(intArrayOf(p)) } }
        val maskBufs = Array(NUM_CB) { p ->
            fast.createInputBuffer("mask", "step").also { b -> b.writeFloat(FloatArray(NUM_CB) { j -> if (j <= p) 0f else NEG }) }
        }
        val zero = FloatArray(FAST_KV_FLOATS)
        val kZ = fast.createInputBuffer("k_all", "step").also { it.writeFloat(zero) }
        val vZ = fast.createInputBuffer("v_all", "step").also { it.writeFloat(zero) }
        val kA = fast.createOutputBuffer("k_all", "step")
        val vA = fast.createOutputBuffer("v_all", "step")
        val kB = fast.createOutputBuffer("k_all", "step")
        val vB = fast.createOutputBuffer("v_all", "step")
        fOwned.addAll(listOf(use1, use0, kZ, vZ, kA, vA, kB, vB) + posBufs + maskBufs)
        // Call 0 reads the zero set and writes A; call p >= 1 reads A (p odd) or B (p even) and writes the other.
        fIns = Array(NUM_CB) { p ->
            val (k, v) = when {
                p == 0 -> kZ to vZ
                p % 2 == 1 -> kA to vA
                else -> kB to vB
            }
            mapOf("hidden" to fHidden, "token" to fToken, "use_hidden" to (if (p == 0) use1 else use0),
                "pos" to posBufs[p], "mask" to maskBufs[p], "k_all" to k, "v_all" to v)
        }
        fOuts = Array(NUM_CB) { p ->
            val (k, v) = if (p % 2 == 0) kA to vA else kB to vB
            mapOf("logits" to fLogits, "k_all" to k, "v_all" to v)
        }

        // Codec: fp16 T128 on the GPU when it both builds and passes an output check against the CPU int8 graph on the
        // bundled voice's codes; otherwise (or with codecMode "cpu") CPU int8 T128. Round 1 measured the GPU graph
        // building and running without an exception while returning the same wav for every input (corr -0.01 vs CPU),
        // so an exception alone is not a usable fallback signal.
        codec128 = selectCodec()

        // First invokes of the AR graphs (one decode, one fast step), timed apart from generation.
        timed("slow_decode", warmupMs) {
            zeroCache()
            decode(IntArray(ROWS), 0)
        }
        timed("fast_step", warmupMs) {
            fHidden.writeFloat(FloatArray(DIM))
            fToken.writeInt(intArrayOf(0))
            fast.run(fIns[0], fOuts[0], "step")
        }
        Log.i(TAG, "ENGINE create_ms=$createMs warmup_ms=$warmupMs codec_backend=$codecBackend gpu_error=$gpuError")
    }

    // ---------------- prompt (build_prompt) ----------------

    fun buildPrompt(text: String, refText: String?, refCodes: Array<IntArray>?): Array<IntArray> {
        val f = constants.fragments
        val target = clean(text)
        if (refCodes == null) {
            val row0 = f.getValue("sys_open") + f.getValue("sys_noref") + f.getValue("im_end_nl") +
                f.getValue("user_open") + tokenizer.encode(target) + f.getValue("im_end_nl") + f.getValue("asst_voice")
            return Array(ROWS) { r -> if (r == 0) row0 else IntArray(row0.size) }
        }
        var rt = clean(refText ?: "")
        if (!rt.contains("<|speaker:")) rt = "<|speaker:0|>$rt"
        val prefix = f.getValue("sys_open") + f.getValue("sys_ref") + tokenizer.encode(rt) + f.getValue("speech")
        val suffix = f.getValue("im_end_nl") + f.getValue("user_open") + tokenizer.encode(target) +
            f.getValue("im_end_nl") + f.getValue("asst_voice")
        val l = refCodes[0].size
        val row0 = prefix + IntArray(l) { refCodes[0][it] + SEM_BEGIN } + suffix
        val prompt = Array(ROWS) { IntArray(row0.size) }
        prompt[0] = row0
        for (r in 1 until ROWS) System.arraycopy(refCodes[r - 1], 0, prompt[r], prefix.size, l)
        return prompt
    }

    // ---------------- slow AR (_prefill / _decode) ----------------

    private fun zeroCache() {
        for (n in KV_NAMES) kvB.getValue(n).writeFloat(zeroKv)
        cur = kvB
    }

    class Step(val logits: FloatArray, val hidden: FloatArray)

    /** Prefill prompt[:-1] in right-padded chunks of 256; the last prompt token goes through decode. */
    fun prefill(prompt: Array<IntArray>, stats: MutableMap<String, Any>? = null): Step {
        val p = prompt[0].size
        val t0 = nowNs()
        zeroCache()
        val t1 = nowNs()
        val codes = IntArray(ROWS * PREFILL_T)
        val pos = IntArray(PREFILL_T)
        var s = 0
        var chunks = 0
        while (s < p - 1) {
            val rem = p - 1 - s
            val n = min(rem, PREFILL_T)
            codes.fill(0)
            for (r in 0 until ROWS) System.arraycopy(prompt[r], s, codes, r * PREFILL_T, n)
            for (i in 0 until PREFILL_T) pos[i] = s + i
            preMask.fill(NEG)
            for (i in 0 until PREFILL_T) {
                val allowed = min(s + i + 1, CACHE)
                java.util.Arrays.fill(preMask, i * CACHE, i * CACHE + allowed, 0f)
            }
            preCodes.writeInt(codes)
            prePos.writeInt(pos)
            preMaskBuf.writeFloat(preMask)
            val next = if (cur === kvA) kvB else kvA
            val ins = HashMap<String, TensorBuffer>(64).apply {
                put("codes", preCodes); put("input_pos", prePos); put("mask", preMaskBuf); putAll(cur)
            }
            val outs = HashMap<String, TensorBuffer>(64).apply {
                put("logits", preLogits); put("hidden", preHidden); putAll(next)
            }
            slow.run(ins, outs, "prefill_256")
            cur = next
            s += n
            chunks++
        }
        val t2 = nowNs()
        val step = decode(IntArray(ROWS) { prompt[it][p - 1] }, p - 1)
        val t3 = nowNs()
        stats?.apply {
            put("kv_zero_ms", ms(t1 - t0)); put("prefill_chunks_ms", ms(t2 - t1)); put("prefill_chunks", chunks)
            put("prefill_last_token_decode_ms", ms(t3 - t2))
        }
        return step
    }

    /** One slow decode step: reads the current KV set, writes the other one. */
    fun decode(column: IntArray, pos: Int, runNs: LongArray? = null): Step {
        decCodes.writeInt(column)
        decPos.writeInt(intArrayOf(pos))
        decMask.fill(NEG)
        java.util.Arrays.fill(decMask, 0, min(pos + 1, CACHE), 0f)
        decMaskBuf.writeFloat(decMask)
        val fromA = cur === kvA
        val r0 = nowNs()
        slow.run(if (fromA) decInA else decInB, if (fromA) decOutB else decOutA, "decode")
        if (runNs != null) runNs[0] += nowNs() - r0
        cur = if (fromA) kvB else kvA
        return Step(decLogits.readFloat(), decHidden.readFloat())
    }

    // ---------------- fast AR (_fast_frame) ----------------

    /** 10 step calls: pos 0 = hidden (use_hidden 1, token 0), pos p = previous code; codes[0] = semantic index. */
    fun fastFrame(hidden: FloatArray, semantic: Int, pick: (FloatArray, Int) -> Int, runNs: LongArray? = null): IntArray {
        fHidden.writeFloat(hidden)
        tokenArr[0] = 0
        fToken.writeInt(tokenArr)
        var r0 = nowNs()
        fast.run(fIns[0], fOuts[0], "step")
        if (runNs != null) runNs[0] += nowNs() - r0
        var c = (semantic - SEM_BEGIN).coerceIn(0, CB_SIZE - 1)
        val codes = IntArray(NUM_CB)
        codes[0] = c
        for (p in 1 until NUM_CB) {
            tokenArr[0] = c
            fToken.writeInt(tokenArr)
            r0 = nowNs()
            fast.run(fIns[p], fOuts[p], "step")
            if (runNs != null) runNs[0] += nowNs() - r0
            c = pick(fLogits.readFloat(), p)
            codes[p] = c
        }
        return codes
    }

    // ---------------- generation (generate_codes) ----------------

    /** Forced sequence for the teacher-forced parity check: the Python greedy semantic ids + codes [10][N]. */
    class Teacher(val semantic: IntArray, val codes: Array<IntArray>) {
        val frames: Int get() = semantic.size
    }

    class Gen(
        val prompt: Array<IntArray>,
        val semantic: IntArray,
        val frames: List<IntArray>,
        val stoppedBy: String,
        val stats: MutableMap<String, Any>,
        val decodeMs: DoubleArray,      // per frame: the slow decode after the frame
        val fastMs: DoubleArray,        // per frame: the 10 fast steps incl. sampling
        val semMargin: FloatArray?,
        val semTop2: Array<IntArray>?,
        val codeMargin: Array<FloatArray>?,
        val ownSemantic: IntArray?,     // teacher mode: the app's own greedy pick at each position
        val ownCodes: Array<IntArray>?,
        val semLogitsDump: ArrayList<FloatArray>?,
        val fastLogitsDump: ArrayList<FloatArray>?,
    )

    /**
     * [cancel] is read once per frame: when it is set the loop stops before the next frame and the result carries
     * stoppedBy = "cancelled" (the next call starts from a zeroed KV cache, so nothing has to be reset).
     */
    fun generate(
        text: String, refText: String?, refCodes: Array<IntArray>?, sampler: Sampler,
        record: Boolean = false, teacher: Teacher? = null, dumpLogits: Boolean = false,
        cancel: AtomicBoolean? = null,
        onFrame: ((Int, Double) -> Unit)? = null,
    ): Gen {
        val stats = LinkedHashMap<String, Any>()
        val tStart = nowNs()
        val prompt = buildPrompt(text, refText, refCodes)
        val p = prompt[0].size
        require(p < MAX_SEQ) { "prompt length $p must be smaller than $MAX_SEQ" }
        val tPrompt = nowNs()
        var step = prefill(prompt, stats)
        val tPrefill = nowNs()
        stats["prompt_len"] = p
        stats["prompt_build_ms"] = ms(tPrompt - tStart)
        stats["prefill_ms"] = ms(tPrefill - tPrompt)

        val previous = ArrayDeque<Int>()
        val frames = ArrayList<IntArray>()
        val semantic = ArrayList<Int>()
        val decodeMs = ArrayList<Double>()
        val fastMs = ArrayList<Double>()
        val semMargin = if (record) ArrayList<Float>() else null
        val semTop2 = if (record) ArrayList<IntArray>() else null
        val codeMargin = if (record) ArrayList<FloatArray>() else null
        val ownSem = if (teacher != null) ArrayList<Int>() else null
        val ownCodes = if (teacher != null) ArrayList<IntArray>() else null
        val semDump = if (dumpLogits) ArrayList<FloatArray>() else null
        val fastDump = if (dumpLogits) ArrayList<FloatArray>() else null
        var stoppedBy = "limit"
        val decRun = LongArray(1)
        val fastRun = LongArray(1)
        var samplerNs = 0L
        val limit = min(MAX_NEW, MAX_SEQ - p)
        for (i in 0 until limit) {
            if (cancel?.get() == true) {
                stoppedBy = "cancelled"
                break
            }
            if (record) {
                val (i1, i2, m) = top2(step.logits)
                semMargin!!.add(m); semTop2!!.add(intArrayOf(allowed(i1), allowed(i2)))
            }
            semDump?.add(step.logits)
            val s0 = nowNs()
            var sem = sampler.semantic(step.logits, previous)
            samplerNs += nowNs() - s0
            if (teacher != null) {
                ownSem!!.add(sem)
                sem = if (i < teacher.frames) teacher.semantic[i] else EOS
            }
            if (sem == EOS) {
                stoppedBy = "eos"
                break
            }
            val margins = if (record) FloatArray(NUM_CB - 1) else null
            val own = if (teacher != null) IntArray(NUM_CB) else null
            val f0 = nowNs()
            val cb = fastFrame(step.hidden, sem, { logits, pIdx ->
                if (margins != null) margins[pIdx - 1] = top2(logits).third
                fastDump?.add(logits)
                val q0 = nowNs()
                val c = sampler.code(logits)
                samplerNs += nowNs() - q0
                if (own != null) {
                    own[pIdx] = c
                    teacher!!.codes[pIdx][i]
                } else c
            }, fastRun)
            val f1 = nowNs()
            if (own != null) { own[0] = cb[0]; ownCodes!!.add(own) }
            if (margins != null) codeMargin!!.add(margins)
            frames.add(cb)
            semantic.add(sem)
            previous.addLast(sem)
            if (previous.size > 10) previous.removeFirst()
            onFrame?.invoke(frames.size, ms(nowNs() - tStart))
            val column = IntArray(ROWS)
            column[0] = sem
            System.arraycopy(cb, 0, column, 1, NUM_CB)
            val d0 = nowNs()
            step = decode(column, p + i, decRun)
            val d1 = nowNs()
            fastMs.add(ms(f1 - f0))
            decodeMs.add(ms(d1 - d0))
        }
        val tEnd = nowNs()
        stats["frames"] = frames.size
        stats["loop_ms"] = ms(tEnd - tPrefill)
        stats["generate_ms"] = ms(tEnd - tStart)
        stats["slow_decode_run_only_ms_total"] = ms(decRun[0])
        stats["fast_run_only_ms_total"] = ms(fastRun[0])
        stats["sampler_ms_total"] = ms(samplerNs)
        return Gen(prompt, semantic.toIntArray(), frames, stoppedBy, stats, decodeMs.toDoubleArray(), fastMs.toDoubleArray(),
            semMargin?.toFloatArray(), semTop2?.toTypedArray(), codeMargin?.toTypedArray(),
            ownSem?.toIntArray(), ownCodes?.toTypedArray(), semDump, fastDump)
    }

    // ---------------- codec (decode_audio) ----------------

    class CodecCall(val t: Int, val start: Int, val n: Int, val ms: Double)

    private fun codec192OrNull(calls: MutableMap<String, Any>): Codec? {
        if (codec192 != null || codec192Failed) return codec192
        if (codecBackend != "GPU") return null
        codec192 = try {
            val m = timed("codec_fp16_T192_gpu", createMs) { gpuModel(CODEC_GPU_192) }
            Codec(CODEC_GPU_192, m, 192, "GPU")
        } catch (e: Exception) {
            Log.e(TAG, "GPU codec T192 failed: $e", e)
            calls["t192_error"] = e.toString()
            codec192Failed = true
            null
        }
        return codec192
    }

    /**
     * Causal codec: T128 if N <= 128, T192 if N <= 192, else T192 windows with 128 frames of left context, each window
     * right-padded. Without a T192 graph (CPU fallback) longer inputs use T128 windows with 64 frames of context.
     * [cancel] is read before every window; when it is set the decode stops with a [CancellationException].
     */
    fun decodeAudio(
        frames: List<IntArray>, info: MutableMap<String, Any>, calls: MutableList<CodecCall>,
        cancel: AtomicBoolean? = null,
    ): FloatArray {
        val n = frames.size
        if (n == 0) return FloatArray(0)
        val c = if (n <= 128) codec128!! else (codec192OrNull(info) ?: codec128!!)
        val t = c.t
        val ctx = if (t > CODEC_CTX) CODEC_CTX else t / 2
        info["codec_file"] = c.fileName
        info["codec_backend"] = c.backend
        info["codec_T"] = t
        info["codec_ctx"] = ctx
        val wav = FloatArray(n * FRAME)
        var done = 0
        val buf = IntArray(NUM_CB * t)
        while (done < n) {
            if (cancel?.get() == true) throw CancellationException("cancelled during the codec decode")
            val s = if (done == 0) 0 else max(0, done - ctx)
            val k = min(t, n - s)
            buf.fill(0)
            for (q in 0 until NUM_CB) for (j in 0 until k) buf[q * t + j] = frames[s + j][q]
            val t0 = nowNs()
            val out = c.run(buf)
            calls.add(CodecCall(t, s, k, ms(nowNs() - t0)))
            System.arraycopy(out, (done - s) * FRAME, wav, done * FRAME, (s + k - done) * FRAME)
            done = s + k
        }
        return wav
    }

    // ---------------- registration (encode_reference) ----------------

    class Registration(val codes: Array<IntArray>, val createMs: Double, val encodeMs: Double, val closeMs: Double,
                       val samples: Int)

    /** Codec encoder created for this call and closed right after (2 GB footprint on the S26). */
    fun encodeReference(wav44k: FloatArray): Registration {
        val t0 = nowNs()
        val m = cpuModel(ENCODER)
        val t1 = nowNs()
        val inB = m.createInputBuffer("audio", "encode")
        val outB = m.createOutputBuffer("codes", "encode")
        val nIn = min(wav44k.size, ENC_SAMPLES)
        val a = FloatArray(ENC_SAMPLES)
        System.arraycopy(wav44k, 0, a, 0, nIn)
        inB.writeFloat(a)
        m.run(mapOf("audio" to inB), mapOf("codes" to outB), "encode")
        val raw = outB.readInt()
        val t2 = nowNs()
        inB.close(); outB.close(); m.close()
        val t3 = nowNs()
        val keep = ceil(nIn / FRAME.toDouble()).toInt().coerceAtMost(ENC_FRAMES)
        val codes = Array(NUM_CB) { q -> IntArray(keep) { raw[q * ENC_FRAMES + it] } }
        return Registration(codes, ms(t1 - t0), ms(t2 - t1), ms(t3 - t2), nIn)
    }

    // ---------------- kv_probe ----------------

    class ProbeResult(val decodeTotal: DoubleArray, val decodeRun: DoubleArray, val fastTotal: DoubleArray, val fastRun: DoubleArray)

    /** [steps] dummy slow decode steps (ping-pong, pos 0..steps-1) and [steps] fast step calls (cycling p = i % 10). */
    fun kvProbe(steps: Int): ProbeResult {
        zeroCache()
        val dt = DoubleArray(steps)
        val dr = DoubleArray(steps)
        val col = IntArray(ROWS)
        for (i in 0 until steps) {
            col[0] = SEM_BEGIN + (i * 37) % CB_SIZE
            for (r in 1 until ROWS) col[r] = (i * 11 + r) % CB_SIZE
            val run = LongArray(1)
            val t0 = nowNs()
            decode(col, i, run)
            dt[i] = ms(nowNs() - t0)
            dr[i] = ms(run[0])
        }
        val ft = DoubleArray(steps)
        val fr = DoubleArray(steps)
        fHidden.writeFloat(FloatArray(DIM) { ((it % 7) - 3) * 0.1f })
        for (i in 0 until steps) {
            val p = i % NUM_CB
            val t0 = nowNs()
            tokenArr[0] = (i * 13) % CB_SIZE
            fToken.writeInt(tokenArr)
            val r0 = nowNs()
            fast.run(fIns[p], fOuts[p], "step")
            val r1 = nowNs()
            fLogits.readFloat()
            ft[i] = ms(nowNs() - t0)
            fr[i] = ms(r1 - r0)
        }
        return ProbeResult(dt, dr, ft, fr)
    }

    private var closed = false

    /** Releases every buffer and graph once; a second call does nothing (a TensorBuffer must not be closed twice). */
    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        codec128?.close(); codec192?.close()
        codec128 = null; codec192 = null
        for (b in kvA.values) b.close()
        for (b in kvB.values) b.close()
        for (b in listOf(decCodes, decPos, decMaskBuf, decLogits, decHidden, preCodes, prePos, preMaskBuf, preLogits, preHidden)) b.close()
        slow.close()
        for (b in fOwned) b.close()
        fHidden.close(); fToken.close(); fLogits.close()
        fast.close()
    }
}

/**
 * The vendor sampler (audio8_tts_litert.py _processed/_sample/_sample_semantic): top-k 50 / top-p 0.9 /
 * temperature 0.7, RAS re-draw at temperature 1.0 / top-p 0.9 when the semantic pick repeats one of the last 10,
 * exponential-race sampling argmax(p / -ln u). The RNG is SplittableRandom(seed); with [greedy] every u is 0.5, which
 * reduces each draw to argmax (the Python twin greedy_ref.py does the same).
 */
class Sampler(val greedy: Boolean, val seed: Long, val temperature: Double = 0.7, val topP: Double = 0.9, val topK: Int = 50) {
    private val rng = SplittableRandom(seed)
    private val idx = IntArray(64)
    private val kept = IntArray(64)
    private val vals = DoubleArray(64)

    private fun u(): Double = if (greedy) 0.5 else 1.0 - rng.nextDouble()   // (0, 1]

    /**
     * _sample(_processed(scores, k, p, t), u): keep the top-k by score (stable order), drop those whose cumulative
     * softmax exceeds top_p (the first always stays), scale by 1/t, draw argmax(p_i / -ln u_i) over the kept ones in
     * ascending index order (numpy argmax takes the first maximum).
     */
    fun draw(scores: FloatArray, k: Int, pTop: Double, t: Double): Int {
        val kk = min(k, scores.size)
        // top-k indices, descending score, ties -> lower index first (np.argsort(-scores, kind="stable"))
        var cnt = 0
        for (i in scores.indices) {
            val v = scores[i]
            if (cnt == kk && v <= scores[idx[cnt - 1]]) continue
            var j = if (cnt < kk) cnt++ else kk - 1
            while (j > 0 && scores[idx[j - 1]] < v) { idx[j] = idx[j - 1]; j-- }
            idx[j] = i
        }
        val mx = scores[idx[0]].toDouble()
        var sum = 0.0
        for (v in scores) sum += exp(v - mx)
        var cum = 0.0
        var nk = 0
        for (j in 0 until cnt) {
            cum += exp(scores[idx[j]] - mx) / sum
            if (j == 0 || cum <= pTop) kept[nk++] = idx[j]
        }
        java.util.Arrays.sort(kept, 0, nk)
        val tt = max(t, 1e-5)
        var m2 = Double.NEGATIVE_INFINITY
        for (j in 0 until nk) { vals[j] = scores[kept[j]] / tt; if (vals[j] > m2) m2 = vals[j] }
        var s2 = 0.0
        for (j in 0 until nk) { vals[j] = exp(vals[j] - m2); s2 += vals[j] }
        var best = -1
        var bestV = Double.NEGATIVE_INFINITY
        for (j in 0 until nk) {
            val r = (vals[j] / s2) / (-ln(u()))
            if (r > bestV) { bestV = r; best = kept[j] }
        }
        return best
    }

    /** _sample_semantic over the 4097 slow logits (semantic range + EOS). */
    fun semantic(logits: FloatArray, previous: Collection<Int>): Int {
        val normal = draw(logits, topK, topP, temperature)
        val high = draw(logits, topK, 0.9, 1.0)
        val normalId = Audio8Engine.allowed(normal)
        val highId = Audio8Engine.allowed(high)
        return if (previous.isNotEmpty() && normalId in previous &&
            normalId >= Audio8Engine.SEM_BEGIN && normalId <= Audio8Engine.SEM_END) highId else normalId
    }

    /** One fast codebook draw over 4096 logits. */
    fun code(logits: FloatArray): Int = draw(logits, topK, topP, temperature)
}
