package com.asr

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import kotlin.math.sqrt

/**
 * wav2vec2-base-960h CTC speech recognition on the LiteRT CompiledModel GPU — fully GPU, single forward pass.
 *   waveform[1,160000] (16 kHz, normalized) -> logits[1,499,32]
 * Decoding (CTC greedy: argmax per frame, collapse repeats, drop blanks) runs on the host. No autoregressive
 * decoder, so the whole model is one GPU graph (~22 ms / 10 s on a Pixel 8a).
 */
class Wav2Vec2CTC(ctx: Context, accelerator: Accelerator = Accelerator.GPU) : Closeable {

    companion object {
        const val SR = 16000
        const val SECONDS = 10
        const val SAMPLES = SR * SECONDS               // 160000
        // wav2vec2-base-960h CTC vocab (id -> token); id 0 = blank/pad, "|" = word boundary.
        val VOCAB = arrayOf(
            "<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D", "L",
            "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"
        )
    }

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "w2v2_ctc_fp16.tflite")
        check(f.exists()) { "Model not found: w2v2_ctc_fp16.tflite. Push first: scripts/install_to_device.sh" }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(accelerator), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()
    private val vocab = VOCAB.size                     // 32

    /** pcm: mono float [-1,1], any length (truncated/zero-padded to 10 s). Returns the transcription. */
    fun transcribe(pcm: FloatArray): String {
        val x = FloatArray(SAMPLES)
        val n = minOf(pcm.size, SAMPLES)
        // per-utterance zero-mean / unit-variance normalization (the Wav2Vec2 feature extractor), over the
        // actual audio only; the zero padding stays zero.
        var mean = 0f
        for (i in 0 until n) mean += pcm[i]
        mean /= n.coerceAtLeast(1)
        var varSum = 0f
        for (i in 0 until n) { val d = pcm[i] - mean; varSum += d * d }
        val std = sqrt(varSum / n.coerceAtLeast(1) + 1e-7f)
        for (i in 0 until n) x[i] = (pcm[i] - mean) / std

        inBuf[0].writeFloat(x)
        model.run(inBuf, outBuf)
        val logits = outBuf[0].readFloat()             // [frames * vocab]
        val frames = logits.size / vocab
        return greedyDecode(logits, frames)
    }

    private fun greedyDecode(logits: FloatArray, frames: Int): String {
        val sb = StringBuilder()
        var prev = -1
        for (f in 0 until frames) {
            var best = 0; var bv = logits[f * vocab]
            for (v in 1 until vocab) { val x = logits[f * vocab + v]; if (x > bv) { bv = x; best = v } }
            if (best != prev) {                        // collapse consecutive duplicates
                when {
                    best <= 3 -> {}                    // blank/pad, <s>, </s>, <unk> -> drop
                    best == 4 -> sb.append(' ')        // "|" -> word boundary
                    else -> sb.append(VOCAB[best])
                }
            }
            prev = best
        }
        return sb.toString().trim().replace(Regex(" +"), " ")
    }

    override fun close() { inBuf.forEach { it.close() }; outBuf.forEach { it.close() }; model.close() }
}
