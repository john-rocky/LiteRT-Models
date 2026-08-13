package com.jazipformer

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * japanese-zipformer-base (reazon-research, ReazonSpeech) speech recognition, fully on the
 * LiteRT CompiledModel GPU (ML Drift / LITERT_CL) on a Pixel 8a — with ZERO FFT anywhere:
 * the raw 16 kHz waveform goes straight into a wav2vec2-style 1D-conv frontend (no mel/fbank
 * even on the host), then a Zipformer encoder (6 multi-rate stacks) and a CTC head, all in
 * one GPU graph.
 *
 * Fixed 16 s window: waveform [1,256000] with a 0.5 s zero lead pad (upstream model-card
 * convention) plus 4 additive attention-bias masks (0 = real frame, -1000 = padding) at the
 * internal frame rates [1,799], [1,400], [1,200], [1,100]. Output: raw CTC logits
 * [1,799,3004] at 50 Hz. Blank id 0 (icefall convention), BPE vocab 3004.
 *
 * Greedy CTC without an LM transcribes phonetically exactly; occasional kanji homophone
 * swaps (e.g. 選挙 -> 占拠) are the expected no-LM behavior.
 *
 * Device-verified (Pixel 8a): per-frame argmax agreement 98.5-100% vs the desktop float
 * reference, transcripts identical; GPU compile 2.4 s, ~621 ms run+readback per 16 s window.
 */
class JaZipformerAsr(private val ctx: Context) : Closeable {

    companion object {
        const val MODEL = "ja_zipformer_ctc_fp16.tflite"
        const val N_IN = 256000           // 16 s @ 16 kHz
        const val LEAD = 8000             // 0.5 s zero pad before the audio
        const val T50 = 799               // 50 Hz frames for the full window
        const val NCLASS = 3004           // BPE pieces; CTC blank id = 0
        const val BLANK = 0
        const val MAX_SAMPLES = N_IN - 2 * LEAD
        private val BIAS_LENS = intArrayOf(799, 400, 200, 100)   // ds 1, 2, 4, 8
    }

    private fun loadModel(): CompiledModel {
        val f = File(ctx.filesDir, MODEL)
        check(f.exists()) { "Model not found: $MODEL. Push it first: scripts/install_to_device.sh" }
        return CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }

    private val model = loadModel()
    private val inBufs = model.createInputBuffers()
    private val outBufs = model.createOutputBuffers()
    // resolve slots by float capacity (robust to converter ordering)
    private val waveSlot = inBufs.indexOfFirst { it.readFloat().size == N_IN }
    private val biasSlots = BIAS_LENS.map { len -> inBufs.indexOfFirst { it.readFloat().size == len } }
    private val logitsSlot = outBufs.indexOfFirst { it.readFloat().size == T50 * NCLASS }

    private val tokens: List<String> =
        ctx.assets.open("tokens.txt").bufferedReader().readLines()

    private val waveInput = FloatArray(N_IN)

    data class Result(val text: String, val gpuMs: Long)

    /** 50 Hz frame count for n input samples (the conv stack's length arithmetic). */
    private fun frames(n: Int): Int {
        var len = n
        for ((k, s) in listOf(10 to 5, 3 to 2, 3 to 2, 3 to 2, 3 to 2, 2 to 2, 2 to 2)) {
            len = (len - k) / s + 1
        }
        return len
    }

    /** @param audio mono PCM in [-1,1] at 16 kHz (clipped to the 15 s usable window). */
    fun transcribe(audio: FloatArray): Result {
        val n = minOf(audio.size, MAX_SAMPLES)
        java.util.Arrays.fill(waveInput, 0f)
        System.arraycopy(audio, 0, waveInput, LEAD, n)
        val valid50 = minOf(frames(n + 2 * LEAD), T50)

        val t0 = System.nanoTime()
        inBufs[waveSlot].writeFloat(waveInput)
        for ((r, len) in BIAS_LENS.withIndex()) {
            val ds = (T50 + len - 1) / len                       // 1, 2, 4, 8
            val bias = FloatArray(len) { i -> if (i * ds < valid50) 0f else -1000f }
            inBufs[biasSlots[r]].writeFloat(bias)
        }
        model.run(inBufs, outBufs)
        val logits = outBufs[logitsSlot].readFloat()             // readback syncs the GPU
        val t1 = System.nanoTime()

        return Result(decode(logits, valid50), (t1 - t0) / 1_000_000)
    }

    /** Greedy CTC over the real frames: argmax per frame, drop blanks and repeats, detok. */
    private fun decode(logits: FloatArray, valid: Int): String {
        val sb = StringBuilder()
        var prev = -1
        for (t in 0 until valid) {
            var best = -Float.MAX_VALUE
            var arg = 0
            val base = t * NCLASS
            for (c in 0 until NCLASS) { val v = logits[base + c]; if (v > best) { best = v; arg = c } }
            if (arg != BLANK && arg != prev && arg < tokens.size) sb.append(tokens[arg])
            prev = arg
        }
        return sb.toString().replace('▁', ' ').trim()
    }

    override fun close() {
        inBufs.forEach { it.close() }; outBufs.forEach { it.close() }; model.close()
    }
}
