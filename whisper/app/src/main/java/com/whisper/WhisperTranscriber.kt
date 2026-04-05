package com.whisper

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Whisper speech-to-text: Encoder (TFLite GPU) + Decoder (ONNX Runtime CPU).
 *
 * Usage:
 *   1. Call transcribe(audioSamples) — returns transcribed text
 *
 * Push models via adb:
 *   adb push whisper_encoder.tflite /data/local/tmp/
 *   adb shell run-as com.whisper cp /data/local/tmp/whisper_encoder.tflite /data/data/com.whisper/files/
 *   adb push whisper_decoder.onnx /data/local/tmp/
 *   adb shell run-as com.whisper cp /data/local/tmp/whisper_decoder.onnx /data/data/com.whisper/files/
 */
class WhisperTranscriber(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "Whisper"
        private const val ENCODER_MODEL = "whisper_encoder.tflite"
        private const val DECODER_MODEL = "whisper_decoder.onnx"
        private const val MAX_TOKENS = 224  // max generation length
    }

    // Config
    private val nAudioState: Int
    private val sotToken: Int
    private val eotToken: Int
    private val transcribeToken: Int
    private val noTimestampsToken: Int
    private val langTokens: Map<String, Int>

    // Vocab for decoding token IDs → text
    private val vocab: Map<Int, String>

    // Mel spectrogram processor
    private val melProcessor: MelSpectrogram

    // Encoder: CompiledModel GPU
    private val compiledModel: CompiledModel
    private val encoderInputBuffers: List<TensorBuffer>

    // Decoder: ONNX Runtime CPU
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val decoderSession: OrtSession

    var lastEncodeMs = 0L; private set
    var lastDecodeMs = 0L; private set
    var acceleratorName = ""; private set

    init {
        // Load config
        val configJson = JSONObject(context.assets.open("config.json").bufferedReader().readText())
        nAudioState = configJson.getInt("n_audio_state")
        sotToken = configJson.getInt("sot_token")
        eotToken = configJson.getInt("eot_token")
        transcribeToken = configJson.getInt("transcribe_token")
        noTimestampsToken = configJson.getInt("no_timestamps_token")

        val langObj = configJson.getJSONObject("language_tokens")
        langTokens = mutableMapOf<String, Int>().apply {
            langObj.keys().forEach { key -> put(key, langObj.getInt(key)) }
        }

        // Load vocab
        val vocabJson = JSONObject(context.assets.open("vocab.json").bufferedReader().readText())
        vocab = mutableMapOf<Int, String>().apply {
            vocabJson.keys().forEach { key -> put(key.toInt(), vocabJson.getString(key)) }
        }
        Log.i(TAG, "Loaded vocab: ${vocab.size} tokens")

        // Mel processor
        melProcessor = MelSpectrogram(context)

        // Load encoder from files dir
        val encoderFile = java.io.File(context.filesDir, ENCODER_MODEL)
        if (!encoderFile.exists()) {
            try {
                context.assets.open(ENCODER_MODEL).use { input ->
                    encoderFile.outputStream().use { output -> input.copyTo(output) }
                }
            } catch (_: Exception) {
                throw IllegalStateException(
                    "Encoder not found. Push via adb:\n" +
                    "adb push whisper_encoder.tflite /data/local/tmp/ && " +
                    "adb shell run-as com.whisper cp /data/local/tmp/$ENCODER_MODEL " +
                    "/data/data/com.whisper/files/"
                )
            }
        }

        Log.i(TAG, "Loading encoder: ${encoderFile.absolutePath}")
        compiledModel = try {
            val gpuOpts = CompiledModel.Options(Accelerator.GPU)
            try {
                gpuOpts.gpuOptions = CompiledModel.GpuOptions(
                    null, null, null,
                    CompiledModel.GpuOptions.Precision.FP32,
                    null, null, null, null, null, null, null, null, null, null, null
                )
            } catch (_: Exception) {}
            val m = CompiledModel.create(encoderFile.absolutePath, gpuOpts, null)
            acceleratorName = "GPU"
            Log.i(TAG, "Encoder GPU ready")
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, falling back to CPU")
            val cpuOpts = CompiledModel.Options(Accelerator.CPU)
            val m = CompiledModel.create(encoderFile.absolutePath, cpuOpts, null)
            acceleratorName = "CPU"
            m
        }
        encoderInputBuffers = compiledModel.createInputBuffers()

        // Load decoder from files dir
        val decoderFile = java.io.File(context.filesDir, DECODER_MODEL)
        if (!decoderFile.exists()) {
            try {
                context.assets.open(DECODER_MODEL).use { input ->
                    decoderFile.outputStream().use { output -> input.copyTo(output) }
                }
            } catch (_: Exception) {
                throw IllegalStateException(
                    "Decoder not found. Push via adb:\n" +
                    "adb push whisper_decoder.onnx /data/local/tmp/ && " +
                    "adb shell run-as com.whisper cp /data/local/tmp/$DECODER_MODEL " +
                    "/data/data/com.whisper/files/"
                )
            }
        }

        Log.i(TAG, "Loading decoder: ${decoderFile.absolutePath}")
        decoderSession = ortEnv.createSession(decoderFile.absolutePath)
        Log.i(TAG, "Decoder ready")
    }

    /**
     * Transcribe audio samples to text.
     * @param samples PCM float samples at 16kHz, mono
     * @param language Language code ("en", "ja", etc.)
     * @return Transcribed text
     */
    fun transcribe(samples: FloatArray, language: String = "en"): String {
        // Step 1: Mel spectrogram
        val mel = melProcessor.compute(samples)

        // Step 2: Encode
        val t1 = System.nanoTime()
        encoderInputBuffers[0].writeFloat(mel)
        val encoderOutput = compiledModel.run(encoderInputBuffers)
        val audioFeatures = encoderOutput[0].readFloat()  // [1, 1500, n_audio_state]
        lastEncodeMs = (System.nanoTime() - t1) / 1_000_000
        Log.i(TAG, "Encode: ${lastEncodeMs}ms")

        // Step 3: Decode (autoregressive)
        val t2 = System.nanoTime()
        val langToken = langTokens[language] ?: langTokens["en"]!!

        // Initial prompt: SOT + language + transcribe + no_timestamps
        val tokens = mutableListOf(sotToken, langToken, transcribeToken, noTimestampsToken)

        val audioTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(audioFeatures),
            longArrayOf(1, 1500, nAudioState.toLong())
        )

        for (step in 0 until MAX_TOKENS) {
            val tokenArray = LongArray(tokens.size) { tokens[it].toLong() }
            val tokenTensor = OnnxTensor.createTensor(
                ortEnv, LongBuffer.wrap(tokenArray),
                longArrayOf(1, tokenArray.size.toLong())
            )

            val results = decoderSession.run(mapOf(
                "tokens" to tokenTensor,
                "audio_features" to audioTensor,
            ))

            // Get logits for the last token position
            val logitsTensor = results.get("logits").get() as OnnxTensor
            val logitsBuffer = logitsTensor.floatBuffer
            val vocabSize = logitsTensor.info.shape[2].toInt()
            val seqLen = logitsTensor.info.shape[1].toInt()

            // Skip to last position
            val lastOffset = (seqLen - 1) * vocabSize
            var maxIdx = 0
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until vocabSize) {
                val v = logitsBuffer.get(lastOffset + i)
                if (v > maxVal) {
                    maxVal = v
                    maxIdx = i
                }
            }

            tokenTensor.close()
            results.close()

            // Check for end of transcript
            if (maxIdx == eotToken) break

            tokens.add(maxIdx)
        }

        audioTensor.close()
        lastDecodeMs = (System.nanoTime() - t2) / 1_000_000
        Log.i(TAG, "Decode: ${lastDecodeMs}ms, ${tokens.size - 4} tokens generated")

        // Step 4: Decode tokens to text (skip prompt tokens)
        val textTokens = tokens.subList(4, tokens.size)
        return decodeTokens(textTokens)
    }

    private fun decodeTokens(tokenIds: List<Int>): String {
        val sb = StringBuilder()
        for (id in tokenIds) {
            val text = vocab[id] ?: ""
            // Skip special tokens (>= 50257)
            if (id >= 50257) continue
            sb.append(text)
        }
        return sb.toString().trim()
    }

    override fun close() {
        encoderInputBuffers.forEach { it.close() }
        compiledModel.close()
        decoderSession.close()
        ortEnv.close()
    }
}
