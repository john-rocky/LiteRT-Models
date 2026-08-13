package com.zipformer

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * Zipformer medium CR-CTC (icefall, LibriSpeech) speech recognition, fully on the LiteRT
 * CompiledModel GPU (ML Drift / LITERT_CL) on a Pixel 8a.
 *
 * Pipeline:  host kaldi-fbank  ->  [GPU] Conv2dSubsampling + Zipformer2 (6 stacks) + CTC linear
 * ->  host greedy-CTC + BPE detokenize. Single GPU graph with a FIXED 16 s window: inputs are
 * the log(1e-10)-padded fbank [1,1600,80] plus 4 additive attention-bias masks (0 = real frame,
 * -1000 = padding), one per internal frame rate ([1,796], [1,398], [1,199], [1,100]). The
 * output is raw CTC logits [1,398,500] at 25 Hz (log-softmax lives host-side; greedy argmax is
 * unaffected). Blank id 0, BPE vocab 500.
 *
 * Device-verified (Pixel 8a): valid-region logits corr 0.9993 vs PyTorch, transcripts identical
 * on the 5-wav sweep, GPU compile 1.8 s, ~156 ms run+readback per 16 s window (RTF ~0.01).
 */
class ZipformerAsr(private val ctx: Context) : Closeable {

    companion object {
        const val MODEL = "zipformer_ctc_fp16.tflite"
        const val FBANK_LEN = 1600        // 16 s of 10 ms-hop fbank frames
        const val T50 = (FBANK_LEN - 7) / 2               // 796 post-subsampling frames
        const val T_OUT = 398             // 25 Hz output frames
        const val NCLASS = 500            // BPE pieces; CTC blank id = 0
        const val BLANK = 0
        const val MAX_SAMPLES = FBANK_LEN * ZipformerFbank.HOP - ZipformerFbank.HOP / 2  // 16 s cap
        private val BIAS_LENS = intArrayOf(796, 398, 199, 100)   // ds 1, 2, 4, 8
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
    private val fbankSlot = inBufs.indexOfFirst { it.readFloat().size == FBANK_LEN * ZipformerFbank.NMEL }
    private val biasSlots = BIAS_LENS.map { len -> inBufs.indexOfFirst { it.readFloat().size == len } }
    private val logitsSlot = outBufs.indexOfFirst { it.readFloat().size == T_OUT * NCLASS }

    private val fbank = ZipformerFbank(ctx)
    private val pieces: Map<Int, String> =
        ctx.assets.open("tokens.txt").bufferedReader().readLines().associate { line ->
            val cut = line.lastIndexOf(' ')
            line.substring(cut + 1).trim().toInt() to line.substring(0, cut)
        }

    private val fbankInput = FloatArray(FBANK_LEN * ZipformerFbank.NMEL)

    data class Result(val text: String, val fbankMs: Long, val gpuMs: Long)

    /** @param audio mono PCM in [-1,1] at 16 kHz (clipped to the 16 s window). */
    fun transcribe(audio: FloatArray): Result {
        val clip = if (audio.size > MAX_SAMPLES) audio.copyOf(MAX_SAMPLES) else audio

        val t0 = System.nanoTime()
        val mel = fbank.compute(clip)
        val nfr = minOf(mel.size, FBANK_LEN)
        java.util.Arrays.fill(fbankInput, ZipformerFbank.LOG_PAD)
        for (t in 0 until nfr) {
            System.arraycopy(mel[t], 0, fbankInput, t * ZipformerFbank.NMEL, ZipformerFbank.NMEL)
        }
        val valid50 = (nfr - 7) / 2
        val t1 = System.nanoTime()

        inBufs[fbankSlot].writeFloat(fbankInput)
        for ((r, len) in BIAS_LENS.withIndex()) {
            val ds = T50 / len + if (T50 % len != 0) 1 else 0    // 1, 2, 4, 8
            val bias = FloatArray(len) { i -> if (i * ds < valid50) 0f else -1000f }
            inBufs[biasSlots[r]].writeFloat(bias)
        }
        model.run(inBufs, outBufs)
        val logits = outBufs[logitsSlot].readFloat()     // [T_OUT * NCLASS] (readback syncs GPU)
        val t2 = System.nanoTime()

        val validOut = minOf((valid50 + 1) / 2, T_OUT)
        return Result(decode(logits, validOut), (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000)
    }

    /** Greedy CTC over the real frames: argmax per frame, drop blanks and repeats, detok. */
    private fun decode(logits: FloatArray, validOut: Int): String {
        val sb = StringBuilder()
        var prev = -1
        for (t in 0 until validOut) {
            var best = -Float.MAX_VALUE
            var arg = 0
            val base = t * NCLASS
            for (c in 0 until NCLASS) { val v = logits[base + c]; if (v > best) { best = v; arg = c } }
            if (arg != BLANK && arg != prev) sb.append(pieces[arg] ?: "")
            prev = arg
        }
        return sb.toString().replace('▁', ' ').trim()
    }

    override fun close() {
        inBufs.forEach { it.close() }; outBufs.forEach { it.close() }; model.close()
    }
}
