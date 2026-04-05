package com.smolvlm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * SmolVLM-256M: Vision encoder (TFLite GPU) + LM decoder (ONNX CPU).
 *
 * Describe images, answer questions about them.
 *
 * Push models via adb:
 *   adb push smolvlm_vision.tflite /data/local/tmp/
 *   adb shell run-as com.smolvlm cp /data/local/tmp/smolvlm_vision.tflite /data/data/com.smolvlm/files/
 *   adb push smolvlm_decoder.onnx /data/local/tmp/
 *   adb shell run-as com.smolvlm cp /data/local/tmp/smolvlm_decoder.onnx /data/data/com.smolvlm/files/
 *   adb push embed_tokens.bin /data/local/tmp/
 *   adb shell run-as com.smolvlm cp /data/local/tmp/embed_tokens.bin /data/data/com.smolvlm/files/
 */
class VLMInference(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "SmolVLM"
        private const val VISION_MODEL = "smolvlm_vision.tflite"
        private const val DECODER_MODEL = "smolvlm_decoder.onnx"
        private const val EMBED_FILE = "embed_tokens.bin"
        private const val IMAGE_SIZE = 512
        private const val N_VISUAL_TOKENS = 64
        private const val MAX_NEW_TOKENS = 128
        private const val REPETITION_PENALTY = 1.2f
    }

    private val embedDim: Int
    private val vocabSize: Int
    private val imageTokenId: Int
    private val eosTokenId: Int

    // Prompt template tokens (pre-tokenized)
    private val promptPrefix: IntArray   // <|im_start|>User:
    private val promptSuffix: IntArray   // <end_of_utterance>\nAssistant:
    private val commonPrompts: Map<String, IntArray>  // pre-tokenized prompts

    // Vocab for decoding (id -> token string)
    private val idToToken: Map<Int, String>
    // Vocab for encoding (token string -> id)
    private val tokenToId: Map<String, Int>

    // Token embeddings [vocab_size, embed_dim]
    private val embedTokens: FloatArray

    // Vision encoder
    private val compiledModel: CompiledModel
    private val encoderInputBuffers: List<TensorBuffer>

    // LM decoder
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val decoderSession: OrtSession

    // Pre-allocated
    private val inputFloats = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
    private val resizedBitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
    private val inputPixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
    private val scaleMatrix = Matrix()
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    var lastEncodeMs = 0L; private set
    var lastDecodeMs = 0L; private set
    var acceleratorName = ""; private set

    init {
        // Load config
        val config = JSONObject(context.assets.open("config.json").bufferedReader().readText())
        vocabSize = config.getInt("vocab_size")
        embedDim = config.getInt("embed_dim")
        imageTokenId = config.getInt("image_token_id")
        eosTokenId = config.getInt("eos_token_id")

        // Load pre-tokenized prompt template
        val prefixArr = config.getJSONArray("prompt_prefix")
        promptPrefix = IntArray(prefixArr.length()) { prefixArr.getInt(it) }
        val suffixArr = config.getJSONArray("prompt_suffix")
        promptSuffix = IntArray(suffixArr.length()) { suffixArr.getInt(it) }

        // Load pre-tokenized common prompts
        val promptsObj = config.getJSONObject("common_prompts")
        commonPrompts = mutableMapOf<String, IntArray>().apply {
            promptsObj.keys().forEach { key ->
                val arr = promptsObj.getJSONArray(key)
                put(key, IntArray(arr.length()) { arr.getInt(it) })
            }
        }
        Log.i(TAG, "Loaded ${commonPrompts.size} pre-tokenized prompts")

        // Load vocab (token_string -> id for encoding, id -> token_string for decoding)
        val vocabJson = JSONObject(context.assets.open("vocab.json").bufferedReader().readText())
        tokenToId = mutableMapOf<String, Int>().apply {
            vocabJson.keys().forEach { put(it, vocabJson.getInt(it)) }
        }
        idToToken = tokenToId.entries.associate { (k, v) -> v to k }
        Log.i(TAG, "Loaded vocab: ${tokenToId.size} tokens")

        // Load token embeddings from filesDir
        val embedFile = resolveFile(context, EMBED_FILE)
        val embedBytes = embedFile.readBytes()
        val embedBuf = ByteBuffer.wrap(embedBytes).order(ByteOrder.LITTLE_ENDIAN)
        embedTokens = FloatArray(embedBytes.size / 4)
        embedBuf.asFloatBuffer().get(embedTokens)
        Log.i(TAG, "Embed tokens: ${vocabSize} x ${embedDim}")

        // Load vision encoder
        val visionFile = resolveFile(context, VISION_MODEL)
        Log.i(TAG, "Loading vision: ${visionFile.absolutePath}")
        compiledModel = try {
            val gpuOpts = CompiledModel.Options(Accelerator.GPU)
            try {
                gpuOpts.gpuOptions = CompiledModel.GpuOptions(
                    null, null, null,
                    CompiledModel.GpuOptions.Precision.FP32,
                    null, null, null, null, null, null, null, null, null, null, null
                )
            } catch (_: Exception) {}
            val m = CompiledModel.create(visionFile.absolutePath, gpuOpts, null)
            acceleratorName = "GPU"
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, CPU fallback")
            val m = CompiledModel.create(visionFile.absolutePath, CompiledModel.Options(Accelerator.CPU), null)
            acceleratorName = "CPU"
            m
        }
        encoderInputBuffers = compiledModel.createInputBuffers()
        Log.i(TAG, "Vision encoder ready ($acceleratorName)")

        // Load decoder
        val decoderFile = resolveFile(context, DECODER_MODEL)
        Log.i(TAG, "Loading decoder: ${decoderFile.absolutePath}")
        decoderSession = ortEnv.createSession(decoderFile.absolutePath)
        Log.i(TAG, "Decoder ready")
    }

    /**
     * Generate text description for an image with an optional prompt.
     * @param bitmap Input image
     * @param prompt User prompt (e.g., "Describe this image", "What is this?")
     * @param onToken Called for each generated token (streaming)
     * @return Full generated text
     */
    fun generate(bitmap: Bitmap, prompt: String = "Describe this image.", onToken: ((String) -> Unit)? = null): String {
        // 1. Encode image → visual tokens [1, 64, 576]
        val t1 = System.nanoTime()
        val visualTokens = encodeImage(bitmap)
        lastEncodeMs = (System.nanoTime() - t1) / 1_000_000
        Log.i(TAG, "Vision encode: ${lastEncodeMs}ms")

        // 2. Build prompt token sequence
        // ChatML format: <|im_start|>user\n<image>...<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        val promptTokenIds = buildPromptTokens(prompt)

        // 3. Build initial embeddings: visual tokens replace <image> placeholders
        val t2 = System.nanoTime()
        val result = decodeAutoregressive(visualTokens, promptTokenIds, onToken)
        lastDecodeMs = (System.nanoTime() - t2) / 1_000_000
        Log.i(TAG, "Decode: ${lastDecodeMs}ms")

        return result
    }

    private fun encodeImage(bitmap: Bitmap): FloatArray {
        // Center crop + resize to 512x512
        val canvas = Canvas(resizedBitmap)
        val srcSize = minOf(bitmap.width, bitmap.height)
        val srcX = (bitmap.width - srcSize) / 2
        val srcY = (bitmap.height - srcSize) / 2
        scaleMatrix.setRectToRect(
            android.graphics.RectF(srcX.toFloat(), srcY.toFloat(),
                (srcX + srcSize).toFloat(), (srcY + srcSize).toFloat()),
            android.graphics.RectF(0f, 0f, IMAGE_SIZE.toFloat(), IMAGE_SIZE.toFloat()),
            Matrix.ScaleToFit.FILL
        )
        canvas.drawBitmap(bitmap, scaleMatrix, scalePaint)
        resizedBitmap.getPixels(inputPixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // Normalize to [-1, 1]: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
        val planeSize = IMAGE_SIZE * IMAGE_SIZE
        for (i in inputPixels.indices) {
            val pixel = inputPixels[i]
            inputFloats[i] = ((pixel shr 16) and 0xFF).toFloat() / 127.5f - 1f
            inputFloats[planeSize + i] = ((pixel shr 8) and 0xFF).toFloat() / 127.5f - 1f
            inputFloats[2 * planeSize + i] = (pixel and 0xFF).toFloat() / 127.5f - 1f
        }
        encoderInputBuffers[0].writeFloat(inputFloats)

        val resultBuffers = compiledModel.run(encoderInputBuffers)
        return resultBuffers[0].readFloat()  // [1, 64, 576]
    }

    private fun buildPromptTokens(prompt: String): List<Int> {
        // Format: <|im_start|>User:<image>...<image>{prompt}<end_of_utterance>\nAssistant:
        val tokens = mutableListOf<Int>()

        // Prefix: <|im_start|>User:
        tokens.addAll(promptPrefix.toList())

        // 64 image placeholder tokens
        repeat(N_VISUAL_TOKENS) { tokens.add(imageTokenId) }

        // Prompt text (use pre-tokenized if available, otherwise word-level fallback)
        val promptTokens = commonPrompts[prompt] ?: tokenizeText(prompt)
        tokens.addAll(promptTokens.toList())

        // Suffix: <end_of_utterance>\nAssistant:
        tokens.addAll(promptSuffix.toList())

        Log.i(TAG, "Prompt: ${tokens.size} tokens (${promptPrefix.size} prefix + $N_VISUAL_TOKENS visual + ${promptTokens.size} text + ${promptSuffix.size} suffix)")
        return tokens
    }

    private fun tokenizeText(text: String): IntArray {
        // Greedy longest-match tokenization against the BPE vocab
        // Not perfect BPE but works reasonably for most English text
        val tokens = mutableListOf<Int>()
        var i = 0
        while (i < text.length) {
            var bestLen = 0
            var bestId = -1
            // Try longest match first (up to 20 chars)
            val maxLen = minOf(20, text.length - i)
            for (len in maxLen downTo 1) {
                val sub = text.substring(i, i + len)
                val id = tokenToId[sub]
                if (id != null) {
                    bestLen = len
                    bestId = id
                    break
                }
            }
            if (bestId >= 0) {
                tokens.add(bestId)
                i += bestLen
            } else {
                // Fallback: skip unknown character
                i++
            }
        }
        return tokens.toIntArray()
    }

    private fun decodeAutoregressive(
        visualTokens: FloatArray,
        promptTokenIds: List<Int>,
        onToken: ((String) -> Unit)?
    ): String {
        // Build initial embedding sequence
        // Replace image token positions with visual tokens
        val embedsList = mutableListOf<FloatArray>()

        var visualIdx = 0
        for (tokenId in promptTokenIds) {
            if (tokenId == imageTokenId && visualIdx < N_VISUAL_TOKENS) {
                // Insert visual token embedding
                val offset = visualIdx * embedDim
                val ve = FloatArray(embedDim)
                System.arraycopy(visualTokens, offset, ve, 0, embedDim)
                embedsList.add(ve)
                visualIdx++
            } else {
                // Text token embedding
                val offset = tokenId * embedDim
                val te = FloatArray(embedDim)
                System.arraycopy(embedTokens, offset, te, 0, embedDim)
                embedsList.add(te)
            }
        }

        val generatedTokens = mutableListOf<Int>()
        val sb = StringBuilder()

        for (step in 0 until MAX_NEW_TOKENS) {
            val seqLen = embedsList.size
            val embedsArray = FloatArray(seqLen * embedDim)
            for (i in 0 until seqLen) {
                System.arraycopy(embedsList[i], 0, embedsArray, i * embedDim, embedDim)
            }

            val embedsTensor = OnnxTensor.createTensor(
                ortEnv, FloatBuffer.wrap(embedsArray),
                longArrayOf(1, seqLen.toLong(), embedDim.toLong())
            )

            val results = decoderSession.run(mapOf("inputs_embeds" to embedsTensor))
            val logitsTensor = results.get("logits").get() as OnnxTensor

            // Get ALL logits as flat array then index the last position
            val allLogits = logitsTensor.floatBuffer
            val totalFloats = seqLen * vocabSize
            val logitsArray = FloatArray(totalFloats)
            allLogits.rewind()
            allLogits.get(logitsArray)

            // Apply repetition penalty to last position logits
            val lastOffset = (seqLen - 1) * vocabSize
            for (tid in generatedTokens) {
                val idx = lastOffset + tid
                if (logitsArray[idx] > 0) {
                    logitsArray[idx] /= REPETITION_PENALTY
                } else {
                    logitsArray[idx] *= REPETITION_PENALTY
                }
            }

            // Find argmax at last position
            var maxIdx = 0
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until vocabSize) {
                val v = logitsArray[lastOffset + i]
                if (v > maxVal) { maxVal = v; maxIdx = i }
            }

            embedsTensor.close()
            results.close()

            if (step < 5) {
                Log.i(TAG, "Step $step: seqLen=$seqLen maxIdx=$maxIdx token=${idToToken[maxIdx]}")
            }

            // Stop on EOS, BOS (start of next turn), or newline after sufficient output
            if (maxIdx == eosTokenId || maxIdx == 0 || maxIdx == 1) break

            generatedTokens.add(maxIdx)

            // Decode GPT-2 byte-encoded token to readable text
            val tokenText = decodeGpt2Token(idToToken[maxIdx] ?: "")
            sb.append(tokenText)
            onToken?.invoke(sb.toString())

            // Add new token embedding for next step
            val newEmbed = FloatArray(embedDim)
            System.arraycopy(embedTokens, maxIdx * embedDim, newEmbed, 0, embedDim)
            embedsList.add(newEmbed)
        }

        return sb.toString().trim()
    }

    /**
     * Decode GPT-2 byte-encoded token string to readable UTF-8.
     * GPT-2 BPE uses a byte-to-unicode mapping where e.g. Ġ=space, Ċ=newline.
     */
    private fun decodeGpt2Token(token: String): String {
        val sb = StringBuilder()
        for (ch in token) {
            sb.append(
                when (ch) {
                    'Ġ' -> ' '   // U+0120 → space
                    'Ċ' -> '\n'  // U+010A → newline
                    'ĉ' -> '\t'  // U+0109 → tab
                    'Ī' -> '\r'  // placeholder
                    else -> ch
                }
            )
        }
        return sb.toString()
    }

    private fun resolveFile(context: Context, name: String): java.io.File {
        val file = java.io.File(context.filesDir, name)
        if (!file.exists()) {
            try {
                context.assets.open(name).use { input ->
                    file.outputStream().use { output -> input.copyTo(output) }
                }
            } catch (_: Exception) {
                throw IllegalStateException(
                    "$name not found. Push via adb:\n" +
                    "adb push $name /data/local/tmp/ && " +
                    "adb shell run-as com.smolvlm cp /data/local/tmp/$name /data/data/com.smolvlm/files/"
                )
            }
        }
        return file
    }

    override fun close() {
        encoderInputBuffers.forEach { it.close() }
        compiledModel.close()
        decoderSession.close()
        ortEnv.close()
        resizedBitmap.recycle()
    }
}
