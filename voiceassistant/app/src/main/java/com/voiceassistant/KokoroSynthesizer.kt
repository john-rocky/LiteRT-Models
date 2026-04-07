package com.voiceassistant

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Kokoro-82M v1.0 text-to-speech via ONNX Runtime.
 * Single ONNX graph: phoneme IDs + style + speed → 24kHz waveform.
 */
class KokoroSynthesizer(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "Kokoro"
        private const val MODEL_FILE = "model_fp16.onnx"
        private const val VOICES_DIR = "voices"
        const val SAMPLE_RATE = 24000
        const val STYLE_DIM = 256
        private const val MAX_INDEX = 510
    }

    private val appContext = context.applicationContext
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    val voiceIds: List<String>
    var providerName: String = "CPU"; private set
    var lastInferMs = 0L; private set

    init {
        val modelFile = resolveLargeFile(appContext, MODEL_FILE)
        Log.i(TAG, "Loading model: ${modelFile.absolutePath} (${modelFile.length() / 1_000_000} MB)")

        session = try {
            val nnapiOpts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                addNnapi()
            }
            val s = ortEnv.createSession(modelFile.absolutePath, nnapiOpts)
            providerName = "NNAPI"
            s
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI failed: ${e.message}, CPU fallback")
            val cpuOpts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortEnv.createSession(modelFile.absolutePath, cpuOpts)
        }

        val voicesDir = File(appContext.filesDir, VOICES_DIR)
        voiceIds = if (voicesDir.exists()) {
            voicesDir.listFiles { f -> f.extension == "bin" }
                ?.map { it.nameWithoutExtension }
                ?.sorted()
                ?: emptyList()
        } else emptyList()
        Log.i(TAG, "Voices: $voiceIds, provider: $providerName")
    }

    fun synthesize(tokenIds: IntArray, voiceId: String, speed: Float = 1.0f): FloatArray {
        val padded = LongArray(tokenIds.size + 2)
        padded[0] = 0L
        for (i in tokenIds.indices) padded[i + 1] = tokenIds[i].toLong()
        padded[padded.size - 1] = 0L
        val seqLen = padded.size

        val style = loadStyleVector(voiceId, seqLen.coerceAtMost(MAX_INDEX))

        val t1 = System.nanoTime()
        val idsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(padded), longArrayOf(1, seqLen.toLong()))
        val styleTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(style), longArrayOf(1, STYLE_DIM.toLong()))
        val speedTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1))

        val inputs = mapOf(
            "input_ids" to idsTensor,
            "style" to styleTensor,
            "speed" to speedTensor,
        )
        val results = session.run(inputs)
        val audioTensor = results.get(0) as OnnxTensor
        val buf = audioTensor.floatBuffer
        buf.rewind()
        val audio = FloatArray(buf.remaining())
        buf.get(audio)

        idsTensor.close()
        styleTensor.close()
        speedTensor.close()
        results.close()

        lastInferMs = (System.nanoTime() - t1) / 1_000_000
        return audio
    }

    private fun loadStyleVector(voiceId: String, seqLen: Int): FloatArray {
        val voiceFile = File(appContext.filesDir, "$VOICES_DIR/$voiceId.bin")
        if (!voiceFile.exists()) throw IllegalStateException("Voice missing: ${voiceFile.absolutePath}")
        val totalFloats = (voiceFile.length() / 4).toInt()
        val numIndices = totalFloats / STYLE_DIM
        val index = (seqLen - 1).coerceIn(0, numIndices - 1)
        val style = FloatArray(STYLE_DIM)
        voiceFile.inputStream().use { input ->
            val skipBytes = (index * STYLE_DIM * 4).toLong()
            var skipped = 0L
            while (skipped < skipBytes) {
                val n = input.skip(skipBytes - skipped)
                if (n <= 0) break
                skipped += n
            }
            val bytes = ByteArray(STYLE_DIM * 4)
            var read = 0
            while (read < bytes.size) {
                val n = input.read(bytes, read, bytes.size - read)
                if (n <= 0) break
                read += n
            }
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(style)
        }
        return style
    }

    private fun resolveLargeFile(context: Context, name: String): File {
        val file = File(context.filesDir, name)
        if (!file.exists()) throw IllegalStateException("$name not found in files dir, push via adb")
        return file
    }

    override fun close() {
        session.close()
    }
}
