package com.kokoro

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * Neural grapheme-to-phoneme fallback for out-of-vocabulary English words — running on
 * **LiteRT** (not ONNX). DeepPhonemizer (MIT) `en_us_cmudict_forward`, a non-autoregressive
 * forward Transformer (char → stress-less ARPABET), converted to tflite via litert-torch.
 *
 * Free-text input works despite tflite's static-shape requirement: the model is a single
 * fixed-length graph (`[1, 96]`) with the padding mask computed **in-graph** from `id == 0`,
 * so any word ≤ 31 chars is right-padded to 96 and decoded back to its real length. (The
 * "natural" variable-length export does not convert — a known litert-torch dynamic-shape gap —
 * so this fixed-max + in-graph-mask form is the workaround.) CPU only: the graph uses
 * EQUAL / SELECT_V2 / 5D attention tensors, which the GPU delegate rejects but the CPU
 * accelerator runs fine.
 *
 * Input  `[1, 96]` float32 = `[LANG] + each lowercased char id ×3 + [END]`, 0.0-padded.
 * Output `[1, 96, 42]` float32 logits → argmax per position → CTC collapse → ARPABET.
 */
class NeuralG2p private constructor(
    private val model: CompiledModel,
    private val inputBuffers: List<TensorBuffer>,
) {

    /** ARPABET symbols (no stress) for [word], or empty if it has no usable letters. */
    fun predict(word: String): List<String> {
        val ids = tokenize(word)
        if (ids.size <= 2) return emptyList()             // only LANG + END
        val len = minOf(ids.size, MAXT)
        val input = FloatArray(MAXT)                      // pad with 0.0
        for (i in 0 until len) input[i] = ids[i].toFloat()
        return try {
            inputBuffers[0].writeFloat(input)
            val out = model.run(inputBuffers)
            ctcDecode(out[0].readFloat(), len)            // [96 * 42] row-major
        } catch (e: Exception) {
            Log.w(TAG, "predict('$word') failed: ${e.message}")
            emptyList()
        }
    }

    private fun tokenize(word: String): IntArray {
        val out = ArrayList<Int>(word.length * REP + 2)
        out.add(LANG)
        for (c in word.lowercase()) {
            val id = CHAR2IDX[c] ?: continue
            repeat(REP) { out.add(id) }
        }
        out.add(END)
        return out.toIntArray()
    }

    /** CTC greedy decode over the first [len] positions: argmax → drop consecutive dups, pad & specials. */
    private fun ctcDecode(logits: FloatArray, len: Int): List<String> {
        val res = ArrayList<String>()
        var prev = -1
        for (t in 0 until len) {
            val base = t * NPHON
            var best = 0
            var bestV = logits[base]
            for (k in 1 until NPHON) {
                val v = logits[base + k]
                if (v > bestV) { bestV = v; best = k }
            }
            if (best == prev) continue
            prev = best
            if (best == PAD || best == LANG || best == END) continue
            IDX2ARPA[best]?.let { res.add(it) }
        }
        return res
    }

    fun close() {
        try {
            inputBuffers.forEach { it.close() }
            model.close()
        } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "NeuralG2p"
        private const val ASSET = "dp_g2p_litert.tflite"
        private const val MAXT = 96
        private const val NPHON = 42
        private const val PAD = 0
        private const val LANG = 1
        private const val END = 2
        private const val REP = 3

        // Text vocab (en_us_cmudict_forward): '_'=0 <en_us>=1 <end>=2 "'"=3 a..z=4..29
        private val CHAR2IDX: Map<Char, Int> = buildMap {
            put('\'', 3)
            for (i in 0..25) put('a' + i, 4 + i)
        }

        // Phoneme idx -> bare ARPABET (no stress).
        private val IDX2ARPA: Map<Int, String> = mapOf(
            3 to "AA", 4 to "AE", 5 to "AH", 6 to "AO", 7 to "AW", 8 to "AY", 9 to "B",
            10 to "CH", 11 to "DH", 12 to "D", 13 to "EH", 14 to "ER", 15 to "EY",
            16 to "F", 17 to "G", 18 to "HH", 19 to "IH", 20 to "IY", 21 to "JH",
            22 to "K", 23 to "L", 24 to "M", 25 to "NG", 26 to "N", 27 to "OW",
            28 to "OY", 29 to "P", 30 to "R", 31 to "SH", 32 to "S", 33 to "TH",
            34 to "T", 35 to "UH", 36 to "UW", 37 to "V", 38 to "W", 39 to "Y",
            40 to "ZH", 41 to "Z",
        )

        /** Load on the LiteRT CPU accelerator; returns null if the model is absent. */
        fun load(context: Context): NeuralG2p? = try {
            val model = CompiledModel.create(
                context.assets, ASSET, CompiledModel.Options(Accelerator.CPU), null
            )
            val inputs = model.createInputBuffers()
            Log.i(TAG, "neural G2P loaded (LiteRT CPU, $ASSET)")
            NeuralG2p(model, inputs)
        } catch (e: Exception) {
            Log.w(TAG, "neural G2P unavailable: ${e.message}")
            null
        }
    }
}
