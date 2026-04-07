package com.voiceassistant

import android.content.Context
import android.util.Log

/**
 * Full voice assistant pipeline:
 *   audio → Whisper STT → SmolLM2 chat → English phonemize → Kokoro TTS → audio
 *
 * English only for MVP. Japanese support would require detecting input
 * language from Whisper, swapping the LM prompt, and using JapanesePhonemizer.
 */
class VoiceAssistant(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "VoiceAssistant"
        private const val DEFAULT_VOICE = "af_heart"
    }

    private val appContext = context.applicationContext
    private val whisper: WhisperTranscriber
    private val lm: LanguageModel
    private val kokoro: KokoroSynthesizer
    private val phonemizer: EnglishPhonemizer

    /** Per-stage timings in ms, populated after each call to [process]. */
    var lastWhisperMs = 0L; private set
    var lastLmMs = 0L; private set
    var lastTtsMs = 0L; private set

    val voiceIds: List<String>
    val ttsProvider: String

    init {
        Log.i(TAG, "Initializing VoiceAssistant pipeline...")
        whisper = WhisperTranscriber(context)
        lm = LanguageModel(context)
        kokoro = KokoroSynthesizer(context)
        val vocab = Phonemizer.loadVocab(context)
        phonemizer = EnglishPhonemizer.load(context, vocab)
        voiceIds = kokoro.voiceIds
        ttsProvider = kokoro.providerName
        Log.i(TAG, "Pipeline ready (whisper=${whisper.acceleratorName}, tts=$ttsProvider)")
    }

    data class Result(
        val transcript: String,
        val response: String,
        val audio: FloatArray,
    )

    /**
     * Streaming pipeline. As the LM generates tokens, sentence chunks are
     * detected, phonemized, synthesized, and emitted via [onAudioChunk] for
     * immediate playback. This drops the time-to-first-audio from
     * "LM_full_time + TTS_full_time" down to roughly
     * "LM_first_sentence + TTS_first_sentence" (~600 ms for short replies).
     *
     * @param samples       16 kHz mono PCM
     * @param voiceId       Kokoro voice
     * @param onTranscript  Called once with the Whisper transcript
     * @param onResponse    Called incrementally as the LM grows the response
     * @param onAudioChunk  Called for each synthesized sentence audio buffer.
     *                      Caller should write to a streaming AudioTrack.
     * @return Final Result with full transcript and response (audio is empty;
     *         it has already been delivered chunk-by-chunk).
     */
    fun processStreaming(
        samples: FloatArray,
        voiceId: String = DEFAULT_VOICE,
        onTranscript: (String) -> Unit,
        onResponse: (String) -> Unit,
        onAudioChunk: (FloatArray) -> Unit,
    ): Result {
        // Stage 1: Whisper STT
        val transcript = whisper.transcribe(samples, "en")
        lastWhisperMs = whisper.lastEncodeMs + whisper.lastDecodeMs
        Log.i(TAG, "STT: '$transcript' (${lastWhisperMs}ms)")
        onTranscript(transcript)

        if (transcript.isBlank()) return Result("(no speech detected)", "", FloatArray(0))

        // Stage 2 + 3 + 4 interleaved: LM token stream → sentence chunks → TTS → audio chunks
        val sentenceBuffer = StringBuilder()
        var lastEmittedLength = 0
        val ttsTotalNs = AtomicLongRef()
        val firstChunkNs = AtomicLongRef()
        val processStartNs = System.nanoTime()

        val response = lm.generate(transcript) { cumulative ->
            // Compute the new substring since last callback
            val delta = cumulative.substring(lastEmittedLength)
            lastEmittedLength = cumulative.length
            sentenceBuffer.append(delta)
            onResponse(cumulative)

            // Drain any complete sentences
            while (true) {
                val sentence = extractCompleteSentence(sentenceBuffer) ?: break
                val tokenIds = phonemizer.phonemize(sentence)
                if (tokenIds.isEmpty()) continue
                val t0 = System.nanoTime()
                val audio = kokoro.synthesize(tokenIds, voiceId, 1.0f)
                ttsTotalNs.value += System.nanoTime() - t0
                if (firstChunkNs.value == 0L) firstChunkNs.value = System.nanoTime() - processStartNs
                onAudioChunk(audio)
            }
        }
        lastLmMs = lm.lastGenerateMs

        // Flush trailing partial sentence (no terminator)
        val tail = sentenceBuffer.toString().trim()
        if (tail.isNotEmpty()) {
            val tokenIds = phonemizer.phonemize(tail)
            if (tokenIds.isNotEmpty()) {
                val t0 = System.nanoTime()
                val audio = kokoro.synthesize(tokenIds, voiceId, 1.0f)
                ttsTotalNs.value += System.nanoTime() - t0
                if (firstChunkNs.value == 0L) firstChunkNs.value = System.nanoTime() - processStartNs
                onAudioChunk(audio)
            }
        }

        lastTtsMs = ttsTotalNs.value / 1_000_000
        val firstChunkMs = firstChunkNs.value / 1_000_000
        Log.i(TAG, "Streaming done: STT ${lastWhisperMs}ms, LM ${lastLmMs}ms, " +
            "TTS total ${lastTtsMs}ms, time-to-first-audio ${firstChunkMs}ms")

        return Result(transcript, response, FloatArray(0))
    }

    /** Pull off the first complete sentence (terminated by . ! ?) from the buffer. */
    private fun extractCompleteSentence(buf: StringBuilder): String? {
        for (i in buf.indices) {
            val c = buf[i]
            if (c == '.' || c == '!' || c == '?') {
                val sentence = buf.substring(0, i + 1).trim()
                buf.delete(0, i + 1)
                if (sentence.isNotEmpty()) return sentence
            }
        }
        return null
    }

    /** Mutable Long holder usable from non-inline lambdas. */
    private class AtomicLongRef { var value: Long = 0L }

    /**
     * Run the full pipeline on a recorded audio buffer (16kHz mono float PCM).
     *
     * @param samples 16 kHz mono PCM float samples
     * @param voiceId Kokoro voice (default af_heart)
     * @param onStage Optional callback after each stage: (stageName, partialResult)
     */
    fun process(
        samples: FloatArray,
        voiceId: String = DEFAULT_VOICE,
        onStage: ((stage: String, value: String) -> Unit)? = null,
    ): Result {
        // Stage 1: Whisper STT
        val transcript = whisper.transcribe(samples, "en")
        lastWhisperMs = whisper.lastEncodeMs + whisper.lastDecodeMs
        Log.i(TAG, "STT: '$transcript' (${lastWhisperMs}ms)")
        onStage?.invoke("transcript", transcript)

        if (transcript.isBlank()) {
            return Result("(no speech detected)", "", FloatArray(0))
        }

        // Stage 2: SmolLM2 chat
        val response = lm.generate(transcript)
        lastLmMs = lm.lastGenerateMs
        Log.i(TAG, "LM: '$response' (${lastLmMs}ms, ${lm.lastTokenCount} tokens)")
        onStage?.invoke("response", response)

        if (response.isBlank()) {
            return Result(transcript, "(no response)", FloatArray(0))
        }

        // Stage 3: English phonemize
        val tokenIds = phonemizer.phonemize(response)
        if (tokenIds.isEmpty()) {
            Log.w(TAG, "Phonemizer produced 0 tokens for: $response")
            return Result(transcript, response, FloatArray(0))
        }

        // Stage 4: Kokoro TTS
        val audio = kokoro.synthesize(tokenIds, voiceId, 1.0f)
        lastTtsMs = kokoro.lastInferMs

        // Diagnostics: report the actual sample range so we can detect silent output
        var minV = Float.POSITIVE_INFINITY
        var maxV = Float.NEGATIVE_INFINITY
        var sumSq = 0.0
        for (s in audio) {
            if (s < minV) minV = s
            if (s > maxV) maxV = s
            sumSq += s.toDouble() * s
        }
        val rms = if (audio.isNotEmpty()) kotlin.math.sqrt(sumSq / audio.size) else 0.0
        Log.i(TAG, "TTS: ${audio.size} samples (${lastTtsMs}ms), tokens=${tokenIds.size}, " +
            "range=[${"%.4f".format(minV)}, ${"%.4f".format(maxV)}], rms=${"%.4f".format(rms)}")

        // Also dump the first synthesized audio to /sdcard for off-device verification
        try {
            saveWav(audio)
        } catch (e: Exception) {
            Log.w(TAG, "saveWav failed: ${e.message}")
        }

        return Result(transcript, response, audio)
    }

    private fun saveWav(samples: FloatArray) {
        val sr = KokoroSynthesizer.SAMPLE_RATE
        val pcm = ByteArray(samples.size * 2)
        var i = 0
        for (s in samples) {
            val v = (s.coerceIn(-1f, 1f) * 32767f).toInt()
            pcm[i++] = (v and 0xff).toByte()
            pcm[i++] = ((v shr 8) and 0xff).toByte()
        }
        val dataSize = pcm.size
        val out = java.io.ByteArrayOutputStream()
        // WAV header
        out.write("RIFF".toByteArray())
        writeInt(out, 36 + dataSize)
        out.write("WAVE".toByteArray())
        out.write("fmt ".toByteArray())
        writeInt(out, 16)
        writeShort(out, 1)            // PCM
        writeShort(out, 1)            // mono
        writeInt(out, sr)
        writeInt(out, sr * 2)         // byte rate
        writeShort(out, 2)            // block align
        writeShort(out, 16)           // bits/sample
        out.write("data".toByteArray())
        writeInt(out, dataSize)
        out.write(pcm)

        val target = java.io.File(appContext.filesDir, "voiceassistant_last.wav")
        target.writeBytes(out.toByteArray())
        Log.i(TAG, "Saved WAV to ${target.absolutePath} (${target.length()} bytes)")
    }

    private fun writeInt(out: java.io.OutputStream, v: Int) {
        out.write(v and 0xff)
        out.write((v shr 8) and 0xff)
        out.write((v shr 16) and 0xff)
        out.write((v shr 24) and 0xff)
    }
    private fun writeShort(out: java.io.OutputStream, v: Int) {
        out.write(v and 0xff)
        out.write((v shr 8) and 0xff)
    }

    override fun close() {
        whisper.close()
        lm.close()
        kokoro.close()
    }
}
