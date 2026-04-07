package com.kokoro

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
 *
 * Architecture: StyleTTS2-based, single ONNX graph.
 *   Input:  input_ids [1, seq_len] int64 — phoneme token IDs (BOS=0 + tokens + EOS=0)
 *           style    [1, 256]         float32 — voice style vector at index seq_len
 *           speed    [1]              float32 — speed multiplier (1.0 = normal)
 *   Output: audio    [1, samples]     float32 — 24kHz mono waveform
 *
 * MVP: phonemes are precomputed offline (see scripts/convert_kokoro.py) and
 * shipped in assets/demo_phrases.json. eSpeak NG / open_jtalk integration
 * for runtime phonemization is phase 2.
 *
 * Push model + voices via adb (see scripts/install_to_device.sh).
 */
class KokoroSynthesizer(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "Kokoro"
        private const val MODEL_FILE = "model_fp16.onnx"
        private const val VOICES_DIR = "voices"
        const val SAMPLE_RATE = 24000
        const val STYLE_DIM = 256

        // Voice .bin layout: float32 [N_token_lengths, 1, 256]
        // Indexed by clamped token length to fetch the matching style vector.
        private const val MAX_INDEX = 510
    }

    private val appContext = context.applicationContext
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    /** Available voice IDs (file basenames without .bin). */
    val voiceIds: List<String>

    /** Active execution provider name: "NNAPI" or "CPU". */
    var providerName: String = "CPU"; private set

    var lastInferMs = 0L; private set

    init {
        val modelFile = resolveLargeFile(appContext, MODEL_FILE)
        Log.i(TAG, "Loading model: ${modelFile.absolutePath} (${modelFile.length() / 1_000_000} MB)")

        // Try NNAPI EP first (may route to GPU/DSP/NPU on Pixel 8a),
        // fall back to CPU XNNPACK if NNAPI rejects the graph.
        session = try {
            val nnapiOpts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                addNnapi()
            }
            val s = ortEnv.createSession(modelFile.absolutePath, nnapiOpts)
            providerName = "NNAPI"
            Log.i(TAG, "Loaded with NNAPI EP")
            s
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI EP failed (${e.message}), falling back to CPU")
            val cpuOpts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortEnv.createSession(modelFile.absolutePath, cpuOpts)
        }
        Log.i(TAG, "Model ready ($providerName), inputs=${session.inputNames}, outputs=${session.outputNames}")

        // Discover voices in /data/data/com.kokoro/files/voices/
        val voicesDir = File(appContext.filesDir, VOICES_DIR)
        voiceIds = if (voicesDir.exists()) {
            voicesDir.listFiles { f -> f.extension == "bin" }
                ?.map { it.nameWithoutExtension }
                ?.sorted()
                ?: emptyList()
        } else {
            Log.w(TAG, "No voices directory: ${voicesDir.absolutePath}")
            emptyList()
        }
        Log.i(TAG, "Voices: $voiceIds")
    }

    /**
     * Synthesize speech from precomputed phoneme token IDs.
     *
     * @param tokenIds Phoneme IDs from the vocab (without BOS/EOS — added internally)
     * @param voiceId  Voice file basename (e.g. "af_heart", "jf_alpha")
     * @param speed    Speed multiplier (default 1.0)
     * @return PCM float32 mono samples at 24kHz, range roughly [-1, 1]
     */
    fun synthesize(
        tokenIds: IntArray,
        voiceId: String,
        speed: Float = 1.0f,
    ): FloatArray {
        // BOS=0, tokens, EOS=0 (StyleTTS2 convention)
        val padded = LongArray(tokenIds.size + 2)
        padded[0] = 0L
        for (i in tokenIds.indices) padded[i + 1] = tokenIds[i].toLong()
        padded[padded.size - 1] = 0L
        val seqLen = padded.size

        // Voice style vector at index = padded length (clamped)
        val style = loadStyleVector(voiceId, seqLen.coerceAtMost(MAX_INDEX))

        val t1 = System.nanoTime()

        val idsTensor = OnnxTensor.createTensor(
            ortEnv, LongBuffer.wrap(padded),
            longArrayOf(1, seqLen.toLong())
        )
        val styleTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(style),
            longArrayOf(1, STYLE_DIM.toLong())
        )
        val speedTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(floatArrayOf(speed)),
            longArrayOf(1)
        )

        val inputs = mutableMapOf<String, OnnxTensor>()
        // Input names per HF model card: input_ids, style, speed
        inputs["input_ids"] = idsTensor
        inputs["style"] = styleTensor
        inputs["speed"] = speedTensor

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
        Log.i(TAG, "Synthesized ${audio.size} samples (${audio.size.toFloat() / SAMPLE_RATE}s) in ${lastInferMs}ms")
        return audio
    }

    /**
     * Load voice style vector. Each .bin is float32 [N, 1, 256] where N >= seqLen.
     * We seek directly to (seqLen - 1) * 256 floats and read 256 floats.
     */
    private fun loadStyleVector(voiceId: String, seqLen: Int): FloatArray {
        val voiceFile = File(appContext.filesDir, "$VOICES_DIR/$voiceId.bin")
        if (!voiceFile.exists()) {
            throw IllegalStateException("Voice file missing: ${voiceFile.absolutePath}")
        }
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
        if (!file.exists()) {
            // Try assets fallback for small builds
            try {
                context.assets.open(name).use { input ->
                    file.outputStream().use { output -> input.copyTo(output) }
                }
            } catch (_: Exception) {
                throw IllegalStateException(
                    "$name not found. Push via adb:\n" +
                    "  adb push $name /data/local/tmp/\n" +
                    "  adb shell run-as com.kokoro cp /data/local/tmp/$name /data/data/com.kokoro/files/"
                )
            }
        }
        return file
    }

    override fun close() {
        session.close()
        ortEnv.close()
    }
}
