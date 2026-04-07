package com.voiceassistant

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Text-only language model wrapping the SmolVLM decoder (which is built on
 * SmolLM2-135M-Instruct). Vision tokens are skipped — we feed only text
 * embeddings, so this becomes a small chat completion LM.
 *
 * Quality is limited (135M parameters, fine-tuned for vision-language not
 * pure chat) but produces coherent short responses to simple questions.
 *
 * Pipeline:
 *   user_text → BPE-ish tokenize → [<|im_start|>User: ... <end_of_utterance>\nAssistant:]
 *   → embedding lookup → autoregressive ONNX decode → response text
 */
class LanguageModel(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "LanguageModel"
        private const val DECODER_MODEL = "smolvlm_decoder.onnx"
        private const val EMBED_FILE = "embed_tokens.bin"
        private const val MAX_NEW_TOKENS = 80
        private const val REPETITION_PENALTY = 1.2f
    }

    private val embedDim: Int
    private val vocabSize: Int
    private val eosTokenId: Int
    private val promptPrefix: IntArray   // <|im_start|>User:
    private val promptSuffix: IntArray   // <end_of_utterance>\nAssistant:

    private val idToToken: Map<Int, String>
    private val tokenToId: Map<String, Int>
    private val embedTokens: FloatArray

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val decoderSession: OrtSession

    var lastGenerateMs = 0L; private set
    var lastTokenCount = 0; private set

    init {
        val config = JSONObject(context.assets.open("llm_config.json").bufferedReader().readText())
        vocabSize = config.getInt("vocab_size")
        embedDim = config.getInt("embed_dim")
        eosTokenId = config.getInt("eos_token_id")
        val prefixArr = config.getJSONArray("prompt_prefix")
        promptPrefix = IntArray(prefixArr.length()) { prefixArr.getInt(it) }
        val suffixArr = config.getJSONArray("prompt_suffix")
        promptSuffix = IntArray(suffixArr.length()) { suffixArr.getInt(it) }

        val vocabJson = JSONObject(context.assets.open("llm_vocab.json").bufferedReader().readText())
        tokenToId = mutableMapOf<String, Int>().apply {
            vocabJson.keys().forEach { put(it, vocabJson.getInt(it)) }
        }
        idToToken = tokenToId.entries.associate { (k, v) -> v to k }
        Log.i(TAG, "Loaded LM vocab: ${tokenToId.size} tokens")

        val embedFile = resolveLargeFile(context, EMBED_FILE)
        val embedBytes = embedFile.readBytes()
        val embedBuf = ByteBuffer.wrap(embedBytes).order(ByteOrder.LITTLE_ENDIAN)
        embedTokens = FloatArray(embedBytes.size / 4)
        embedBuf.asFloatBuffer().get(embedTokens)

        val decoderFile = resolveLargeFile(context, DECODER_MODEL)
        decoderSession = ortEnv.createSession(decoderFile.absolutePath)
        Log.i(TAG, "LM decoder ready")
    }

    /**
     * Generate a short chat response to user input. Greedy decoding with
     * repetition penalty. Stops at EOS or max tokens.
     */
    fun generate(userText: String, onToken: ((String) -> Unit)? = null): String {
        val t0 = System.nanoTime()

        // Tokenize user text greedy longest-match against BPE vocab.
        // Encode word-level: split on whitespace, encode each word + leading space.
        val userTokens = encodeForChat(userText)

        // Full sequence: prefix + user tokens + suffix
        val promptTokens = ArrayList<Int>(promptPrefix.size + userTokens.size + promptSuffix.size)
        promptTokens.addAll(promptPrefix.toList())
        promptTokens.addAll(userTokens.toList())
        promptTokens.addAll(promptSuffix.toList())
        Log.i(TAG, "Prompt: ${promptTokens.size} tokens")

        // Build initial embedding sequence
        val embedsList = ArrayList<FloatArray>(promptTokens.size + MAX_NEW_TOKENS)
        for (id in promptTokens) {
            val e = FloatArray(embedDim)
            System.arraycopy(embedTokens, id * embedDim, e, 0, embedDim)
            embedsList.add(e)
        }

        val generatedTokens = mutableListOf<Int>()
        val sb = StringBuilder()

        for (step in 0 until MAX_NEW_TOKENS) {
            val seqLen = embedsList.size
            val flat = FloatArray(seqLen * embedDim)
            for (i in 0 until seqLen) {
                System.arraycopy(embedsList[i], 0, flat, i * embedDim, embedDim)
            }
            val embedsTensor = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(flat),
                longArrayOf(1, seqLen.toLong(), embedDim.toLong())
            )
            val results = decoderSession.run(mapOf("inputs_embeds" to embedsTensor))
            val logitsTensor = results.get("logits").get() as OnnxTensor
            val logitsBuf = logitsTensor.floatBuffer
            val totalFloats = seqLen * vocabSize
            val logitsArray = FloatArray(totalFloats)
            logitsBuf.rewind()
            logitsBuf.get(logitsArray)

            val lastOffset = (seqLen - 1) * vocabSize
            // Repetition penalty
            for (tid in generatedTokens) {
                val idx = lastOffset + tid
                if (logitsArray[idx] > 0) logitsArray[idx] /= REPETITION_PENALTY
                else logitsArray[idx] *= REPETITION_PENALTY
            }

            var maxIdx = 0
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until vocabSize) {
                val v = logitsArray[lastOffset + i]
                if (v > maxVal) { maxVal = v; maxIdx = i }
            }

            embedsTensor.close()
            results.close()

            if (maxIdx == eosTokenId || maxIdx == 0 || maxIdx == 1) break

            generatedTokens.add(maxIdx)
            val tokenText = decodeGpt2Token(idToToken[maxIdx] ?: "")
            sb.append(tokenText)
            onToken?.invoke(sb.toString())

            val newEmbed = FloatArray(embedDim)
            System.arraycopy(embedTokens, maxIdx * embedDim, newEmbed, 0, embedDim)
            embedsList.add(newEmbed)
        }

        lastGenerateMs = (System.nanoTime() - t0) / 1_000_000
        lastTokenCount = generatedTokens.size
        return sb.toString().trim()
    }

    /** Greedy longest-match BPE-style tokenization for user text. */
    private fun encodeForChat(text: String): IntArray {
        val tokens = mutableListOf<Int>()
        // SmolLM2 BPE uses Ġ (U+0120) for space prefix on tokens. Replicate that.
        val normalized = text.replace(' ', 'Ġ')
        // Prepend space prefix (matches "User: <text>" pattern after the colon)
        val withLeading = "Ġ$normalized"
        var i = 0
        while (i < withLeading.length) {
            var bestLen = 0
            var bestId = -1
            val maxLen = minOf(20, withLeading.length - i)
            for (len in maxLen downTo 1) {
                val sub = withLeading.substring(i, i + len)
                val id = tokenToId[sub]
                if (id != null) { bestLen = len; bestId = id; break }
            }
            if (bestId >= 0) { tokens.add(bestId); i += bestLen }
            else i++
        }
        return tokens.toIntArray()
    }

    private fun decodeGpt2Token(token: String): String {
        val sb = StringBuilder()
        for (ch in token) {
            sb.append(when (ch) {
                'Ġ' -> ' '
                'Ċ' -> '\n'
                'ĉ' -> '\t'
                else -> ch
            })
        }
        return sb.toString()
    }

    private fun resolveLargeFile(context: Context, name: String): File {
        val file = File(context.filesDir, name)
        if (!file.exists()) {
            throw IllegalStateException("$name not found in files dir, push via adb")
        }
        return file
    }

    override fun close() {
        decoderSession.close()
    }
}
