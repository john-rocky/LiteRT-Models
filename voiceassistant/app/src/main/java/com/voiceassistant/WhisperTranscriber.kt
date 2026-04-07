package com.voiceassistant

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
 * Whisper-tiny speech-to-text. Encoder TFLite GPU + decoder ONNX CPU.
 */
class WhisperTranscriber(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "Whisper"
        private const val ENCODER_MODEL = "whisper_encoder.tflite"
        private const val DECODER_MODEL = "whisper_decoder.onnx"
        private const val MAX_TOKENS = 224
    }

    private val nAudioState: Int
    private val sotToken: Int
    private val eotToken: Int
    private val transcribeToken: Int
    private val noTimestampsToken: Int
    private val langTokens: Map<String, Int>
    private val vocab: Map<Int, String>
    private val melProcessor: MelSpectrogram

    private val compiledModel: CompiledModel
    private val encoderInputBuffers: List<TensorBuffer>
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val decoderSession: OrtSession

    var lastEncodeMs = 0L; private set
    var lastDecodeMs = 0L; private set
    var acceleratorName = ""; private set

    init {
        val configJson = JSONObject(context.assets.open("whisper_config.json").bufferedReader().readText())
        nAudioState = configJson.getInt("n_audio_state")
        sotToken = configJson.getInt("sot_token")
        eotToken = configJson.getInt("eot_token")
        transcribeToken = configJson.getInt("transcribe_token")
        noTimestampsToken = configJson.getInt("no_timestamps_token")

        val langObj = configJson.getJSONObject("language_tokens")
        langTokens = mutableMapOf<String, Int>().apply {
            langObj.keys().forEach { key -> put(key, langObj.getInt(key)) }
        }

        val vocabJson = JSONObject(context.assets.open("whisper_vocab.json").bufferedReader().readText())
        vocab = mutableMapOf<Int, String>().apply {
            vocabJson.keys().forEach { key -> put(key.toInt(), vocabJson.getString(key)) }
        }

        melProcessor = MelSpectrogram(context)

        val encoderFile = resolveLargeFile(context, ENCODER_MODEL)
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
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, CPU fallback")
            val m = CompiledModel.create(encoderFile.absolutePath, CompiledModel.Options(Accelerator.CPU), null)
            acceleratorName = "CPU"
            m
        }
        encoderInputBuffers = compiledModel.createInputBuffers()

        val decoderFile = resolveLargeFile(context, DECODER_MODEL)
        decoderSession = ortEnv.createSession(decoderFile.absolutePath)
        Log.i(TAG, "Whisper ready (encoder=$acceleratorName)")
    }

    fun transcribe(samples: FloatArray, language: String = "en"): String {
        val mel = melProcessor.compute(samples)

        val t1 = System.nanoTime()
        encoderInputBuffers[0].writeFloat(mel)
        val encoderOutput = compiledModel.run(encoderInputBuffers)
        val audioFeatures = encoderOutput[0].readFloat()
        lastEncodeMs = (System.nanoTime() - t1) / 1_000_000

        val t2 = System.nanoTime()
        val langToken = langTokens[language] ?: langTokens["en"]!!
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
            val logitsTensor = results.get("logits").get() as OnnxTensor
            val logitsBuffer = logitsTensor.floatBuffer
            val vocabSize = logitsTensor.info.shape[2].toInt()
            val seqLen = logitsTensor.info.shape[1].toInt()
            val lastOffset = (seqLen - 1) * vocabSize
            var maxIdx = 0
            var maxVal = Float.NEGATIVE_INFINITY
            for (i in 0 until vocabSize) {
                val v = logitsBuffer.get(lastOffset + i)
                if (v > maxVal) { maxVal = v; maxIdx = i }
            }
            tokenTensor.close()
            results.close()
            if (maxIdx == eotToken) break
            tokens.add(maxIdx)
        }

        audioTensor.close()
        lastDecodeMs = (System.nanoTime() - t2) / 1_000_000

        val textTokens = tokens.subList(4, tokens.size)
        return decodeTokens(textTokens)
    }

    private fun decodeTokens(tokenIds: List<Int>): String {
        val sb = StringBuilder()
        for (id in tokenIds) {
            val text = vocab[id] ?: ""
            if (id >= 50257) continue
            sb.append(text)
        }
        return sb.toString().trim()
    }

    private fun resolveLargeFile(context: Context, name: String): java.io.File {
        val file = java.io.File(context.filesDir, name)
        if (!file.exists()) {
            try {
                context.assets.open(name).use { input ->
                    file.outputStream().use { output -> input.copyTo(output) }
                }
            } catch (_: Exception) {
                throw IllegalStateException("$name not found in files dir, push via adb")
            }
        }
        return file
    }

    override fun close() {
        encoderInputBuffers.forEach { it.close() }
        compiledModel.close()
        decoderSession.close()
    }
}
