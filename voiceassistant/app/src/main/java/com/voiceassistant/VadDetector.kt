package com.voiceassistant

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Silero VAD v5 wrapper used by the voice assistant for hands-free turn taking
 * (auto-record on speech start, auto-stop on trailing silence) and barge-in.
 *
 * Processes 16 kHz mono audio in fixed 512-sample chunks (32 ms each). The
 * model is a stateful LSTM — `state` must be carried across calls and reset
 * between independent audio streams via [reset].
 *
 * Inputs:
 *   input  float32 [1, 512]    — PCM samples in [-1, 1]
 *   state  float32 [2, 1, 128] — LSTM h/c, zero on first call
 *   sr     int64   scalar      — sample rate (16000)
 *
 * Outputs:
 *   output float32 [1, 1]      — speech probability
 *   stateN float32 [2, 1, 128] — updated LSTM state
 *
 * The model file `silero_vad.onnx` lives in the app's private files dir,
 * pushed by `scripts/install_to_device.sh` like the other models in this
 * module. The README documents the download URL.
 */
class VadDetector(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "Vad"
        private const val MODEL_FILE = "silero_vad.onnx"
        const val SAMPLE_RATE = 16_000
        /** Number of new audio samples per call (32 ms @ 16 kHz). */
        const val CHUNK_SAMPLES = 512
        /**
         * Silero v5 expects the model input to be the last [CONTEXT_SAMPLES]
         * samples of the previous chunk concatenated with the current chunk.
         * So the actual ONNX input is [batch, CONTEXT_SAMPLES + CHUNK_SAMPLES]
         * = [1, 576] at 16 kHz, not [1, 512].
         *
         * This is **not** documented in the model's tensor shapes (which show
         * a generic [-1, -1]). It is hidden inside silero_vad's `OnnxWrapper`.
         * Without this prepended context the model returns ~0 probability for
         * everything, including loud, clean speech.
         */
        private const val CONTEXT_SAMPLES = 64
        private const val STATE_SIZE = 2 * 1 * 128
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val inputName: String
    private val stateName: String
    private val srName: String
    private val outputName: String
    private val newStateName: String

    private var state = FloatArray(STATE_SIZE)
    private var context = FloatArray(CONTEXT_SAMPLES)
    /** Reusable input buffer of size CONTEXT_SAMPLES + CHUNK_SAMPLES. */
    private val modelInput = FloatArray(CONTEXT_SAMPLES + CHUNK_SAMPLES)
    private val srTensor: OnnxTensor

    init {
        val modelFile = resolveLargeFile(context, MODEL_FILE)
        Log.i(TAG, "Loading: ${modelFile.absolutePath} (${modelFile.length() / 1024} KB)")
        session = env.createSession(modelFile.absolutePath)

        val ins = session.inputNames.toList()
        val outs = session.outputNames.toList()
        inputName = ins.first { it == "input" }
        stateName = ins.first { it == "state" }
        srName = ins.first { it == "sr" }
        outputName = outs.first { it == "output" }
        newStateName = outs.first { it == "stateN" }

        // Silero v5 sr is a scalar int64 (shape []). Use the primitive overload —
        // the buffer-based createTensor with longArrayOf() shape combination is
        // an unusual case and may not behave the same on all ORT versions.
        srTensor = OnnxTensor.createTensor(env, SAMPLE_RATE.toLong())
    }

    /** Reset LSTM state and audio context. Call between independent audio streams. */
    fun reset() {
        state = FloatArray(STATE_SIZE)
        context = FloatArray(CONTEXT_SAMPLES)
    }

    /**
     * Run VAD on one [CHUNK_SAMPLES]-sample chunk of new audio.
     *
     * Internally this concatenates the previously-saved [CONTEXT_SAMPLES] of
     * audio in front of the new chunk before feeding the ONNX session, then
     * stores the last [CONTEXT_SAMPLES] samples of the chunk as context for
     * the next call. Without this prepended context Silero v5 returns ~0
     * probability for all inputs.
     *
     * @param chunk Exactly [CHUNK_SAMPLES] PCM float samples in [-1, 1]
     * @return Speech probability in [0, 1]
     */
    fun processChunk(chunk: FloatArray): Float {
        require(chunk.size == CHUNK_SAMPLES) {
            "VAD chunk must be exactly $CHUNK_SAMPLES samples, got ${chunk.size}"
        }

        // Build [context | chunk] into the reusable model-input buffer.
        System.arraycopy(context, 0, modelInput, 0, CONTEXT_SAMPLES)
        System.arraycopy(chunk, 0, modelInput, CONTEXT_SAMPLES, CHUNK_SAMPLES)

        val inputTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(modelInput),
            longArrayOf(1, (CONTEXT_SAMPLES + CHUNK_SAMPLES).toLong())
        )
        val stateTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(state), longArrayOf(2, 1, 128)
        )

        try {
            val result = session.run(
                mapOf(
                    inputName to inputTensor,
                    stateName to stateTensor,
                    srName to srTensor,
                )
            )
            val outTensor = result.get(outputName).get() as OnnxTensor
            val outBuf = outTensor.floatBuffer
            val prob = if (outBuf.remaining() > 0) outBuf.get(0) else 0f

            val nsTensor = result.get(newStateName).get() as OnnxTensor
            nsTensor.floatBuffer.get(state)

            // Save the last CONTEXT_SAMPLES samples of this chunk as context
            // for the next call (this is exactly what silero_vad's OnnxWrapper
            // does internally).
            System.arraycopy(chunk, CHUNK_SAMPLES - CONTEXT_SAMPLES, context, 0, CONTEXT_SAMPLES)

            result.close()
            return prob
        } finally {
            inputTensor.close()
            stateTensor.close()
        }
    }

    private fun resolveLargeFile(context: Context, name: String): File {
        val file = File(context.filesDir, name)
        if (!file.exists()) {
            throw IllegalStateException(
                "$name not found in files dir. Download it from the GitHub Release " +
                "and push via adb (see voiceassistant/README.md)."
            )
        }
        return file
    }

    override fun close() {
        srTensor.close()
        session.close()
    }
}
