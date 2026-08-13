package com.w2vasr

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File

/**
 * wav2vec2-base-960h character-CTC speech recognition, fully on the LiteRT CompiledModel GPU
 * (ML Drift / LITERT_CL) on a Pixel 8a — with ZERO FFT anywhere: the raw 16 kHz waveform goes
 * straight into the 1D-conv feature extractor (no mel/fbank even on the host).
 *
 * Two GPU graphs (the fused graph exceeds the Mali whole-graph shader-compile limit; each half
 * compiles fully delegated):
 *   frontend: waveform [1,256000] -> features [1,799,768]
 *   head:     features -> CTC logits [1,799,32]  (12-layer transformer + lm_head)
 * Host side: zero-pad to the fixed 16 s window, then greedy char-CTC over the valid frames
 * (blank id 0, '|' = word delimiter). Greedy without an LM has the model's known spelling
 * quirks on hard words.
 *
 * Device-verified (Pixel 8a): valid-region logits corr 0.9928 vs PyTorch, transcript matches
 * the desktop reference; frontend 448 ms + head 391 ms per 16 s window (RTF ~0.05).
 */
class Wav2Vec2Asr(private val ctx: Context) : Closeable {

    companion object {
        const val FRONTEND = "w2v2_asr_frontend_fp16.tflite"
        const val HEAD = "w2v2_asr_head_fp16.tflite"
        const val MAX_SAMPLES = 256000    // 16 s @ 16 kHz
        const val T_PRIME = 799           // 50 Hz frames for the full window
        const val NCLASS = 32             // char vocab; CTC blank id = 0 (<pad>)
        const val BLANK = 0
    }

    private fun load(name: String): CompiledModel {
        val f = File(ctx.filesDir, name)
        check(f.exists()) { "Model not found: $name. Push it first: scripts/install_to_device.sh" }
        return CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }

    private val frontend = load(FRONTEND)
    private val head = load(HEAD)
    private val fIn = frontend.createInputBuffers()
    private val fOut = frontend.createOutputBuffers()
    private val hIn = head.createInputBuffers()
    private val hOut = head.createOutputBuffers()

    private val tokens: List<String> =
        ctx.assets.open("tokens.txt").bufferedReader().readLines()

    private val waveInput = FloatArray(MAX_SAMPLES)

    data class Result(val text: String, val gpuMs: Long)

    /** 50 Hz frame count for n input samples (the conv stack's length arithmetic). */
    private fun frames(n: Int): Int {
        var len = n
        for ((k, s) in listOf(10 to 5, 3 to 2, 3 to 2, 3 to 2, 3 to 2, 2 to 2, 2 to 2)) {
            len = (len - k) / s + 1
        }
        return len
    }

    /** @param audio mono PCM in [-1,1] at 16 kHz (clipped to the 16 s window). */
    fun transcribe(audio: FloatArray): Result {
        val n = minOf(audio.size, MAX_SAMPLES)
        java.util.Arrays.fill(waveInput, 0f)
        System.arraycopy(audio, 0, waveInput, 0, n)
        val valid = minOf(frames(n), T_PRIME)

        val t0 = System.nanoTime()
        fIn[0].writeFloat(waveInput)
        frontend.run(fIn, fOut)
        hIn[0].writeFloat(fOut[0].readFloat())
        head.run(hIn, hOut)
        val logits = hOut[0].readFloat()          // [T_PRIME * NCLASS] (readback syncs GPU)
        val t1 = System.nanoTime()

        return Result(decode(logits, valid), (t1 - t0) / 1_000_000)
    }

    /** Greedy char-CTC over the valid frames: argmax, drop blanks + repeats, '|' -> space. */
    private fun decode(logits: FloatArray, valid: Int): String {
        val sb = StringBuilder()
        var prev = -1
        for (t in 0 until valid) {
            var best = -Float.MAX_VALUE
            var arg = 0
            val base = t * NCLASS
            for (c in 0 until NCLASS) { val v = logits[base + c]; if (v > best) { best = v; arg = c } }
            if (arg != BLANK && arg != prev && arg < tokens.size) {
                val tok = tokens[arg]
                if (tok == "|") sb.append(' ') else if (!tok.startsWith("<")) sb.append(tok)
            }
            prev = arg
        }
        return sb.toString().trim()
    }

    override fun close() {
        fIn.forEach { it.close() }; fOut.forEach { it.close() }
        hIn.forEach { it.close() }; hOut.forEach { it.close() }
        frontend.close(); head.close()
    }
}
