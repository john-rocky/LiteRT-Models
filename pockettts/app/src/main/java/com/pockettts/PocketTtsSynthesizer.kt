package com.pockettts

import android.content.Context
import android.util.Half
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Random
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pocket TTS (Kyutai, 100M) on LiteRT CompiledModel.
 *
 * Pocket TTS is a flow-matching LM over continuous 32-dim Mimi latents: per
 * 12.5 Hz frame a 6-layer/1024-wide causal transformer conditions a 6-block
 * AdaLN MLP flow head that turns one Gaussian draw into the next latent
 * (Lagrangian Self Distillation, 1 step — no iterative sampling loop); a 20M
 * tiny Mimi (x16 ConvTranspose upsample + 2-layer transformer + SEANet)
 * decodes latents to 24 kHz audio. The voice is a precomputed prompt KV cache
 * (`pt_voice_*.bin`, repacked from Kyutai's published per-voice states).
 *
 * Four graphs, all stateless with host-side state (the dia2/vibevoice
 * packed-KV pattern):
 *  * `pt_flowlm_step`  — one AR step; packed KV `[1,96,512,64]` in/out.
 *  * `pt_flow_head`    — cond + noise -> latent (LSD time embeds baked in).
 *  * `pt_mimi_dec_tx`  — 64-latent-frame block of the Mimi decoder
 *    transformer; blocks overlap 32 frames because the 2-layer sliding-window
 *    (250) attention has a stacked receptive field of 498 positions.
 *  * `pt_mimi_deconly` — SEANet decoder, one-shot 256-frame window (causal,
 *    so real frames are exact regardless of the zero tail).
 *
 * Text chunking, EOS handling and the noise schedule mirror the reference
 * pocket_tts Python package; the sentencepiece unigram tokenizer is ported in
 * [SpTokenizer]. Host-vs-reference parity of every graph and of the full
 * pipeline is checked in scripts/build_pockettts.py.
 */
class PocketTtsSynthesizer(context: Context) : Closeable {

    companion object {
        const val H = 1024               // flow-LM width
        const val HD = 64                // head dim
        const val NH = 16                // heads
        const val LAYERS = 6
        const val G = LAYERS * NH        // packed KV groups
        const val PMAX = 512             // KV capacity: voice + text + audio frames
        const val LDIM = 32              // Mimi latent dim
        const val THETA = 10000.0

        const val UPS = 16               // 12.5 Hz -> 200 Hz
        const val MIMI_D = 512
        const val F_BLK = 64             // dec_tx block payload frames
        const val F_HOP = 32             // dec_tx block hop
        const val S_BLK = F_BLK * UPS
        const val DEC_FRAMES = 256       // deconly window frames
        const val S_DEC = DEC_FRAMES * UPS
        const val SPF = 1920             // samples per 12.5 Hz frame
        const val SAMPLE_RATE = 24000

        // Generation defaults from the english config / pocket_tts defaults.
        const val TEMP = 0.3f
        const val EOS_THRESHOLD = -4.0f
        const val MAX_TOKENS_PER_CHUNK = 50
        const val TOKENS_PER_SECOND = 3.0
        const val GEN_SECONDS_PADDING = 2.0
        const val FRAME_RATE = 12.5
        const val MASK_NEG = -1e4f

        // step + flow head fused into one graph with one output tensor: on
        // Mali the per-frame cost is dispatch/sync-bound, and two invocations
        // plus four readbacks per frame cost more than the math itself.
        const val LM = "pt_flowlm_fused_fp16.tflite"
        const val DEC_TX = "pt_mimi_dec_tx_fp16.tflite"
        const val DECONLY = "pt_mimi_deconly_fp16.tflite"
        const val EMBED = "pt_embed_f16.bin"
        const val INPUT_LINEAR = "pt_input_linear_f32.bin"
        const val BOS = "pt_bos_input_f32.bin"
        const val NEUTRAL = "pt_neutral_latent_f32.bin"
        const val TOKENIZER = "pt_tokenizer.tsv"

        // CC-BY-4.0 (alba-mackenna, VCTK) and CC0 (voice-donations, voice-zero)
        // voices only; the CC-BY-NC ones (expresso, ears) are not bundled.
        val VOICES = listOf("alba", "marius", "javert", "charles", "mary", "eve")
    }

    private val modelDir =
        requireNotNull(context.getExternalFilesDir(null)) { "External storage unavailable" }

    private fun path(name: String): File {
        val f = File(modelDir, name)
        check(f.exists()) { "Missing $name — push files first: scripts/install_to_device.sh" }
        return f
    }

    // Debug override: a `force_cpu.txt` in the files dir listing keys (lm, head,
    // dectx, dec — comma/newline separated) pins those graphs to CPU without a
    // rebuild. Placement experiments on new devices only; absent in normal use.
    private val forceCpu: Set<String> =
        File(modelDir, "force_cpu.txt").takeIf { it.exists() }
            ?.readText()?.split(',', '\n')?.map { it.trim() }?.filter { it.isNotEmpty() }
            ?.toSet() ?: emptySet()

    /** Compile on GPU; fall back to CPU (fp16 weights dequantize to fp32 there). */
    private fun load(name: String, key: String): Pair<CompiledModel, String> {
        val p = path(name).absolutePath
        if (key in forceCpu) {
            return CompiledModel.create(p, CompiledModel.Options(Accelerator.CPU), null) to "CPU*"
        }
        return try {
            CompiledModel.create(p, CompiledModel.Options(Accelerator.GPU), null) to "GPU"
        } catch (e: Throwable) {
            CompiledModel.create(p, CompiledModel.Options(Accelerator.CPU), null) to "CPU"
        }
    }

    private val lmP = load(LM, "lm")
    private val dectxP = load(DEC_TX, "dectx")
    private val deconlyP = load(DECONLY, "dec")
    private val lm = lmP.first
    private val dectx = dectxP.first
    private val deconly = deconlyP.first

    /** e.g. "lm:GPU dectx:GPU dec:GPU" — shown in the UI status line. */
    val placements =
        "lm:${lmP.second} dectx:${dectxP.second} dec:${deconlyP.second}"

    private val lmIn = lm.createInputBuffers()
    private val lmOut = lm.createOutputBuffers()
    private val dectxIn = dectx.createInputBuffers()
    private val dectxOut = dectx.createOutputBuffers()
    private val deconlyIn = deconly.createInputBuffers()
    private val deconlyOut = deconly.createOutputBuffers()

    // ---- host assets ------------------------------------------------------
    private val embChannel = RandomAccessFile(path(EMBED), "r").channel
    private val embMap = embChannel
        .map(FileChannel.MapMode.READ_ONLY, 0, embChannel.size()).order(ByteOrder.LITTLE_ENDIAN)
    private val inputLinear = readF32(path(INPUT_LINEAR))      // [1024, 32] row-major
    private val bosInput = readF32(path(BOS))                  // [1024]
    private val neutral = readF32(path(NEUTRAL))               // [32]
    val tokenizer = SpTokenizer(path(TOKENIZER))

    private val endTokens: Set<Int>
    private val fallbackTokens: Set<Int>

    init {
        endTokens = tokenizer.encode(".!...?").drop(1).toSet()
        fallbackTokens = tokenizer.encode(",;:").drop(1).toSet()
    }

    // ---- host state -------------------------------------------------------
    private val pk = FloatArray(G * PMAX * HD)
    private val pv = FloatArray(G * PMAX * HD)
    private val mask = FloatArray(NH * (PMAX + 1))
    private var pos = 0

    private var voiceName = ""
    private var voiceK = FloatArray(0)
    private var voiceV = FloatArray(0)
    private var voiceLen = 0

    private val cosArr = FloatArray(HD)
    private val sinArr = FloatArray(HD)
    private val invFreq = DoubleArray(HD / 2) { 1.0 / Math.pow(THETA, it / 32.0) }
    private val rnd = Random()

    data class Result(val audio: FloatArray, val frames: Int, val ms: Long)

    /** Load a repacked voice state: int32 T, then k and v as fp16 `[96][T][64]`. */
    fun loadVoice(name: String) {
        if (name == voiceName) return
        val bb = ByteBuffer.wrap(path("pt_voice_$name.bin").readBytes())
            .order(ByteOrder.LITTLE_ENDIAN)
        val t = bb.int
        check(t <= PMAX) { "voice state longer than KV capacity: $t > $PMAX" }
        val n = G * t * HD
        val k = FloatArray(n) { Half.toFloat(bb.short) }
        val v = FloatArray(n) { Half.toFloat(bb.short) }
        voiceK = k; voiceV = v; voiceLen = t; voiceName = name
    }

    private fun resetToVoice() {
        pk.fill(0f); pv.fill(0f)
        for (g in 0 until G) {
            System.arraycopy(voiceK, g * voiceLen * HD, pk, g * PMAX * HD, voiceLen * HD)
            System.arraycopy(voiceV, g * voiceLen * HD, pv, g * PMAX * HD, voiceLen * HD)
        }
        mask.fill(MASK_NEG)
        for (h in 0 until NH) {
            val base = h * (PMAX + 1)
            for (p in 0 until voiceLen) mask[base + p] = 0f
            mask[base + PMAX] = 0f                        // current token, concatenated at tail
        }
        pos = voiceLen
    }

    // ---- small host math --------------------------------------------------
    private fun embRow(id: Int): FloatArray {
        val out = FloatArray(H)
        var b = id * H * 2
        for (j in 0 until H) { out[j] = Half.toFloat(embMap.getShort(b)); b += 2 }
        return out
    }

    private fun projectLatent(lat: FloatArray): FloatArray {
        val out = FloatArray(H)
        for (o in 0 until H) {
            var acc = 0f
            val row = o * LDIM
            for (i in 0 until LDIM) acc += inputLinear[row + i] * lat[i]
            out[o] = acc
        }
        return out
    }

    private fun ropeFill(p: Int) {
        for (j in 0 until HD / 2) {
            val ang = p * invFreq[j]
            val c = cos(ang).toFloat(); val s = sin(ang).toFloat()
            cosArr[j] = c; cosArr[j + HD / 2] = c
            sinArr[j] = s; sinArr[j + HD / 2] = s
        }
    }

    private val zeroNoise = FloatArray(LDIM)

    /**
     * One fused frame: flow-LM step + flow head in a single invocation.
     * Output layout: eos(1) | latent(32) | new-k(96*64) | new-v(96*64).
     * Returns (latent, eosLogit) and appends this step's K/V at [pos].
     * Text prompting passes zero noise and ignores the latent.
     */
    private fun step(emb: FloatArray, noise: FloatArray): Pair<FloatArray, Float> {
        check(pos < PMAX) { "KV cache overflow at $pos" }
        ropeFill(pos)
        lmIn[0].writeFloat(emb)
        lmIn[1].writeFloat(cosArr)
        lmIn[2].writeFloat(sinArr)
        lmIn[3].writeFloat(mask)
        lmIn[4].writeFloat(pk)
        lmIn[5].writeFloat(pv)
        lmIn[6].writeFloat(noise)
        lm.run(lmIn, lmOut)
        val out = lmOut[0].readFloat()
        val eos = out[0]
        val latent = out.copyOfRange(1, 1 + LDIM)
        val kvBase = 1 + LDIM
        for (g in 0 until G) {
            System.arraycopy(out, kvBase + g * HD, pk, g * PMAX * HD + pos * HD, HD)
            System.arraycopy(out, kvBase + G * HD + g * HD, pv, g * PMAX * HD + pos * HD, HD)
        }
        for (h in 0 until NH) mask[h * (PMAX + 1) + pos] = 0f
        pos++
        return latent to eos
    }

    /** Generate speech for `text` with the currently loaded voice. */
    fun synthesize(text: String, voice: String): Result {
        val t0 = System.nanoTime()
        loadVoice(voice)
        val audio = ArrayList<FloatArray>()
        var frames = 0
        val chunks = splitIntoBestSentences(text)
        for (chunk in chunks) {
            val (prepared, eosGuess) = prepareTextPrompt(chunk)
            val ids = tokenizer.encode(prepared)
            val latents = generateChunk(ids, framesAfterEos = eosGuess + 2)
            android.util.Log.i(
                "PocketTTS",
                "chunk: ${ids.size} tokens -> ${latents.size} frames",
            )
            frames += latents.size
            if (latents.isNotEmpty()) audio.add(decode(latents))
        }
        val total = audio.sumOf { it.size }
        val out = FloatArray(total)
        var o = 0
        for (a in audio) { System.arraycopy(a, 0, out, o, a.size); o += a.size }
        return Result(out, frames, (System.nanoTime() - t0) / 1_000_000)
    }

    /** The reference autoregressive loop for one <=50-token chunk. */
    private fun generateChunk(ids: IntArray, framesAfterEos: Int): List<FloatArray> {
        resetToVoice()
        for (id in ids) step(embRow(id), zeroNoise)
        val estimate = ceil((ids.size / TOKENS_PER_SECOND + GEN_SECONDS_PADDING) * FRAME_RATE)
        val maxGen = minOf(estimate.toInt(), PMAX - pos - 1)
        val latents = ArrayList<FloatArray>(maxGen)
        var emb = bosInput
        var eosStep = -1
        for (g in 0 until maxGen) {
            val noise = FloatArray(LDIM) {
                (rnd.nextGaussian() * sqrt(TEMP.toDouble())).toFloat()
            }
            val (lat, eosLogit) = step(emb, noise)
            if (eosLogit > EOS_THRESHOLD && eosStep < 0) eosStep = g
            if (eosStep >= 0 && g >= eosStep + framesAfterEos) break
            latents.add(lat)
            emb = projectLatent(lat)
        }
        return latents
    }

    /** Mimi decode: overlapped dec_tx blocks -> one-shot SEANet window. */
    private fun decode(latents: List<FloatArray>): FloatArray {
        val t = minOf(latents.size, DEC_FRAMES)
        val feat = FloatArray(MIMI_D * S_DEC)
        val blk = FloatArray((1 + F_BLK) * LDIM)

        fun runBlock(prev: FloatArray, start: Int): FloatArray {
            System.arraycopy(prev, 0, blk, 0, LDIM)
            for (f in 0 until F_BLK) {
                val src = if (start + f < t) latents[start + f] else neutral
                System.arraycopy(src, 0, blk, (1 + f) * LDIM, LDIM)
            }
            dectxIn[0].writeFloat(blk)
            dectx.run(dectxIn, dectxOut)
            return dectxOut[0].readFloat()               // [512 * 1024]
        }

        var out = runBlock(neutral, 0)
        val n0 = minOf(F_BLK, t)
        for (c in 0 until MIMI_D)
            System.arraycopy(out, c * S_BLK, feat, c * S_DEC, n0 * UPS)
        var kept = F_BLK
        while (kept < t) {
            val start = kept - F_HOP
            out = runBlock(latents[start - 1], start)
            val n = minOf(F_BLK, t - start)
            val keepN = (n - F_HOP) * UPS
            for (c in 0 until MIMI_D)
                System.arraycopy(out, c * S_BLK + F_HOP * UPS, feat, c * S_DEC + kept * UPS, keepN)
            kept += n - F_HOP
        }

        deconlyIn[0].writeFloat(feat)
        deconly.run(deconlyIn, deconlyOut)
        val wav = deconlyOut[0].readFloat()
        return FloatArray(t * SPF) { wav[it].coerceIn(-1f, 1f) }
    }

    // ---- text preparation (ports of pocket_tts.models.tts_model) ----------

    /** prepare_text_prompt: normalize whitespace/case/punctuation; guess EOS tail. */
    internal fun prepareTextPrompt(raw: String): Pair<String, Int> {
        var text = raw.trim()
        require(text.isNotEmpty()) { "Text prompt cannot be empty" }
        text = text.replace('\n', ' ').replace('\r', ' ').replace("  ", " ")
        val words = text.trim().split(Regex("\\s+")).size
        val guess = if (words <= 4) 3 else 1
        if (!text[0].isUpperCase()) text = text[0].uppercaseChar() + text.substring(1)
        if (text.last().isLetterOrDigit()) text += "."
        return text to guess
    }

    /** split_into_best_sentences: sentence segments greedily packed <=50 tokens. */
    internal fun splitIntoBestSentences(raw: String): List<String> {
        val (prepared, _) = prepareTextPrompt(raw)
        val tokens = tokenizer.encode(prepared.trim()).toList()

        fun boundaries(list: List<Int>, marks: Set<Int>): List<Int> {
            val idx = ArrayList<Int>()
            idx.add(0)
            var prevWasBoundary = false
            for ((i, tok) in list.withIndex()) {
                if (tok in marks) prevWasBoundary = true
                else {
                    if (prevWasBoundary) idx.add(i)
                    prevWasBoundary = false
                }
            }
            idx.add(list.size)
            return idx
        }

        fun segments(list: List<Int>, idx: List<Int>): List<Pair<Int, String>> =
            (0 until idx.size - 1).map { i ->
                val part = list.subList(idx[i], idx[i + 1])
                part.size to tokenizer.decode(part)
            }

        val sentences = segments(tokens, boundaries(tokens, endTokens))
        val refined = ArrayList<Pair<Int, String>>()
        for ((n, textSeg) in sentences) {
            if (n <= MAX_TOKENS_PER_CHUNK) { refined.add(n to textSeg); continue }
            val sub = tokenizer.encode(textSeg.trim()).toList()
            val subSegs = segments(sub, boundaries(sub, fallbackTokens))
            if (subSegs.size > 1) refined.addAll(subSegs) else refined.add(n to textSeg)
        }

        val chunks = ArrayList<String>()
        var current = ""
        var count = 0
        for ((n, sentence) in refined) {
            when {
                current.isEmpty() -> { current = sentence; count = n }
                count + n > MAX_TOKENS_PER_CHUNK -> {
                    chunks.add(current.trim()); current = sentence; count = n
                }
                else -> { current += " $sentence"; count += n }
            }
        }
        if (current.isNotEmpty()) chunks.add(current.trim())
        return chunks
    }

    override fun close() {
        listOf(lmIn, lmOut, dectxIn, dectxOut, deconlyIn, deconlyOut)
            .forEach { l -> l.forEach { it.close() } }
        lm.close(); dectx.close(); deconly.close(); embChannel.close()
    }

    private fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }
}
