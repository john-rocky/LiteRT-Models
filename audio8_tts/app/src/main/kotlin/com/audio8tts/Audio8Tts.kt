package com.audio8tts

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.sqrt

/**
 * Text + voice -> speech: the part of the app the screen ([MainActivity]) and the device check (`Audio8DeviceCheck`)
 * share.
 *
 * It owns one [Audio8Engine] (the LiteRT graphs) and adds what a user-facing app needs around it: which model files are
 * missing, the bundled and recorded voices, the text rules ([TextRules]), cancellation, and a close that may be called
 * twice. [speak] and [registerVoice] run graphs and must come from one thread at a time (the engine reuses its input
 * and output buffers); [close] belongs on that thread too, or after the last call has returned.
 */
class Audio8Tts private constructor(
    /** The graphs; the probe and autorun modes of [MainActivity] drive it directly. */
    val engine: Audio8Engine,
    /** The folder the model files and voices are read from (the app's external files dir). */
    val modelDir: File,
    /** Wall time of [load]: tokenizer, graph creation, first runs and, in codec mode "auto", the GPU codec check. */
    val loadSeconds: Double,
) : AutoCloseable {

    /** Whether a text can be spoken, and if not, the line the screen shows. */
    sealed class TextCheck {
        /** Shown on screen when the text is refused; null when it can be spoken. */
        abstract val message: String?

        class Ok(val tokens: Int) : TextCheck() {
            override val message: String? = null
        }

        object Empty : TextCheck() {
            override val message = "Type a sentence first."
        }

        /** [tokens] counts at most the first [TextRules.MAX_CHARS] characters; [truncated] says when more were left. */
        class TooLong(val tokens: Int, val truncated: Boolean) : TextCheck() {
            override val message = "Too long: ${if (truncated) "more than " else ""}$tokens tokens. One Speak reads up " +
                "to ${TextRules.MAX_TEXT_TOKENS} tokens (about 20 s of speech); split the text."
        }
    }

    /** Thrown by [speak] for a text [checkText] refuses; the message is the on-screen line. */
    class TextRefused(val check: TextCheck) : IllegalArgumentException(check.message)

    /** Thrown by [load] when required files are not in the model folder; [names] lists them. */
    class MissingModelFiles(val dir: File, val names: List<String>) :
        IllegalStateException("Model files missing in ${dir.path}: ${names.joinToString(", ")}")

    /** A reference voice: the transcript of its clip and the clip's codec codes [10][frames]. */
    class Voice(val id: String, val transcript: String, val codes: Array<IntArray>) {
        val frames: Int get() = codes[0].size
        val seconds: Double get() = frames * Audio8Engine.FRAME / Audio8Engine.SR.toDouble()
    }

    /** One spoken text. */
    class Speech(
        val text: String,
        val voice: String,
        val seed: Long,
        /** 16-bit mono PCM at 44.1 kHz without a header, as AudioTrack takes it; [wavBytes] adds the header. */
        val pcm: ByteArray,
        val audioSeconds: Double,
        /** Text in, samples out: prompt, prefill, the frame loop and the codec decode. */
        val generateSeconds: Double,
        val codecSeconds: Double,
        /** Root mean square of the samples in [-1, 1]; silence reads about 0. */
        val rms: Double,
        val gen: Audio8Engine.Gen,
        val codecInfo: Map<String, Any>,
        val codecCalls: List<Audio8Engine.CodecCall>,
    ) {
        /** Real-time factor: [generateSeconds] / [audioSeconds]; below 1 the speech is ready before it would end. */
        val rtf: Double get() = generateSeconds / audioSeconds

        /** The samples as a complete RIFF/WAVE file. */
        fun wavBytes(): ByteArray = Wav.mono16Bytes(pcm, Audio8Engine.SR)
    }

    private val running = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)

    /** True while [speak] or [registerVoice] runs. */
    val busy: Boolean get() = running.get()
    val isClosed: Boolean get() = closed.get()
    val codecBackend: String get() = engine.codecBackend
    val threads: Int get() = engine.threads

    /** Empty and too-long texts are refused before any graph runs; the count is the tokenizer's, on the cleaned text. */
    fun checkText(text: String): TextCheck = TextRules.check(text) { engine.tokenizer.encode(it).size }

    /**
     * Speaks [text] in [voice] and returns the samples and the times. Throws [TextRefused] for a text [checkText]
     * refuses and [CancellationException] when [cancel] is set before the samples are ready; the object stays usable
     * after either. [onFrame] gets (frames so far, ms since the call) after every frame and [onDecoding] runs once when
     * the codec starts; both run on the calling thread.
     */
    fun speak(
        text: String,
        voice: Voice,
        seed: Long = DEFAULT_SEED,
        greedy: Boolean = false,
        cancel: AtomicBoolean? = null,
        onFrame: ((Int, Double) -> Unit)? = null,
        onDecoding: (() -> Unit)? = null,
    ): Speech {
        check(!closed.get()) { "Audio8Tts is closed" }
        val textCheck = checkText(text)
        if (textCheck !is TextCheck.Ok) throw TextRefused(textCheck)
        check(running.compareAndSet(false, true)) { "another speak or registration is running" }
        try {
            val t0 = SystemClock.elapsedRealtimeNanos()
            val g = engine.generate(text, voice.transcript, voice.codes, Sampler(greedy, seed), cancel = cancel,
                onFrame = onFrame)
            if (g.stoppedBy == "cancelled") throw CancellationException("cancelled after ${g.frames.size} frames")
            if (g.frames.isEmpty()) error("the model ended the speech before its first frame")
            onDecoding?.invoke()
            val c0 = SystemClock.elapsedRealtimeNanos()
            val info = LinkedHashMap<String, Any>()
            val calls = ArrayList<Audio8Engine.CodecCall>()
            val wav = engine.decodeAudio(g.frames, info, calls, cancel)
            val t1 = SystemClock.elapsedRealtimeNanos()
            // A short text decodes in one codec call, which cannot stop halfway: a Cancel during it lands here.
            if (cancel?.get() == true) throw CancellationException("cancelled during the codec decode")
            var sq = 0.0
            for (v in wav) sq += v * v
            return Speech(text, voice.id, seed, Wav.floatToPcm16(wav), wav.size / Audio8Engine.SR.toDouble(),
                (t1 - t0) / 1e9, (t1 - c0) / 1e9, sqrt(sq / wav.size), g, info, calls)
        } finally {
            running.set(false)
        }
    }

    /**
     * Encodes a recorded clip (44.1 kHz mono floats; the encoder takes the first 10.03 s) into codec codes and saves it
     * as the voice [id] (default [MY_VOICE]) with its [transcript]. The encoder graph is created for this call and
     * closed after it.
     */
    fun registerVoice(audio: FloatArray, transcript: String, id: String = MY_VOICE): Pair<Voice, Audio8Engine.Registration> {
        check(!closed.get()) { "Audio8Tts is closed" }
        check(File(modelDir, Audio8Engine.ENCODER).isFile) { "${Audio8Engine.ENCODER} is missing in ${modelDir.path}" }
        check(running.compareAndSet(false, true)) { "another speak or registration is running" }
        try {
            val r = engine.encodeReference(audio)
            return saveVoice(modelDir, id, transcript, r.codes, audio) to r
        } finally {
            running.set(false)
        }
    }

    /** The test vectors of prompt_constants.json, plus the fixed fragments re-encoded from their text. */
    fun tokenizerSelfTest(): JSONObject {
        val rows = JSONArray()
        var pass = true
        for (v in engine.constants.vectors) {
            val got = engine.tokenizer.encode(v.text)
            val ok = got.contentEquals(v.ids)
            pass = pass && ok
            rows.put(JSONObject().put("name", v.name).put("ok", ok).apply {
                if (!ok) put("got", JSONArray(got.toList())).put("want", JSONArray(v.ids.toList()))
            })
        }
        var fragOk = true
        for ((k, text) in engine.constants.fragmentTexts) {
            val got = engine.tokenizer.encode(text)
            val ok = got.contentEquals(engine.constants.fragments.getValue(k))
            fragOk = fragOk && ok
            if (!ok) rows.put(JSONObject().put("name", "fragment:$k").put("ok", false).put("got", JSONArray(got.toList())))
        }
        Log.i(TAG, "TOKENIZER_SELFTEST vectors_pass=$pass fragments_pass=$fragOk $rows")
        return JSONObject().put("pass", pass && fragOk).put("vectors", engine.constants.vectors.size)
            .put("fragments_pass", fragOk).put("rows", rows)
    }

    /** Releases the graphs once; later calls do nothing. */
    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        engine.close()
    }

    companion object {
        private const val TAG = "Audio8Tts"
        const val DEFAULT_SEED = 42L
        const val MY_VOICE = "my_voice"
        private const val PREFS = "audio8"

        /** The two voices of the model repository's voices/ folder. */
        val BUNDLED_VOICES = listOf("ja_funasr_example", "en_librispeech_1272")

        /** Files [load] cannot do without. */
        val REQUIRED_FILES = listOf(Audio8Engine.SLOW, Audio8Engine.FAST, Audio8Engine.CODEC_GPU_128,
            Audio8Engine.CODEC_CPU_128, "tokenizer.json")

        /** Files one part of the app needs: the T192 GPU codec (speech over 5.9 s) and the encoder (recording a voice). */
        val OPTIONAL_FILES = listOf(Audio8Engine.CODEC_GPU_192, Audio8Engine.ENCODER)

        fun missingFiles(modelDir: File): List<String> = REQUIRED_FILES.filter { !File(modelDir, it).isFile }

        /**
         * Creates the graphs (slow and fast AR on the CPU with [threads] threads; the codec per [codecMode]: "auto" =
         * GPU fp16 only if its output matches the CPU int8 codec on a bundled voice's codes, "gpu", "cpu"). Throws
         * [MissingModelFiles] before anything is created when a required file is absent.
         */
        fun load(
            modelDir: File,
            gpuCacheDir: File,
            constants: PromptConstants,
            threads: Int = 4,
            codecMode: String = "auto",
            kvOwner: String = "output",
            slowWeightCache: String? = null,
        ): Audio8Tts {
            val missing = missingFiles(modelDir)
            if (missing.isNotEmpty()) throw MissingModelFiles(modelDir, missing)
            val t0 = SystemClock.elapsedRealtimeNanos()
            // The GPU codec runs only after its output has matched the CPU int8 codec on real codes. Without a bundled
            // voice there is nothing to check it on, so the codec stays on the CPU (round 1 of the demo lane measured a
            // GPU codec on the Galaxy S26 that returns the same wav for every input without raising an error).
            val checkCodes = BUNDLED_VOICES.firstNotNullOfOrNull { id ->
                runCatching { loadVoice(modelDir, id).codes }.getOrNull()
            }
            val mode = if (codecMode == "auto" && checkCodes == null) "cpu" else codecMode
            val engine = Audio8Engine(modelDir, gpuCacheDir, constants, threads, kvOwner, slowWeightCache, mode, checkCodes)
            return Audio8Tts(engine, modelDir, (SystemClock.elapsedRealtimeNanos() - t0) / 1e9)
        }

        /** The line shown when the microphone permission is refused; recording stops there, speaking does not. */
        const val MIC_DENIED = "Microphone permission denied: recording a voice is off. The sample voices still work."

        /**
         * Why "Record my voice" cannot start, or null when it can: the encoder file is missing, or the microphone
         * permission is not granted ([MIC_DENIED]; the screen then asks for it once).
         */
        fun recordingProblem(context: Context, modelDir: File): String? = when {
            !File(modelDir, Audio8Engine.ENCODER).isFile ->
                "Recording a voice needs ${Audio8Engine.ENCODER} in ${modelDir.path} (README, step 3)."
            context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED -> MIC_DENIED
            else -> null
        }

        fun voiceDir(modelDir: File, id: String): File = File(modelDir, "voices/$id")

        fun hasVoice(modelDir: File, id: String): Boolean =
            File(voiceDir(modelDir, id), "meta.json").isFile && File(voiceDir(modelDir, id), "codes.npy").isFile

        /** voices/<id>/meta.json (the transcript, reference_text) and codes.npy ([10, frames]). */
        fun loadVoice(modelDir: File, id: String): Voice {
            val d = voiceDir(modelDir, id)
            val text = JSONObject(File(d, "meta.json").readText()).getString("reference_text")
            val npy = Npy.loadInts(File(d, "codes.npy"))
            require(npy.shape.size == 2 && npy.shape[0] == Audio8Engine.NUM_CB) { "codes.npy shape ${npy.shape.toList()}" }
            val n = npy.shape[1]
            return Voice(id, text, Array(Audio8Engine.NUM_CB) { q -> IntArray(n) { npy.data[q * n + it] } })
        }

        /** Writes voices/<id>/ in the bundled voices' format, plus the clip as reference.wav; replaces an older one. */
        fun saveVoice(modelDir: File, id: String, transcript: String, codes: Array<IntArray>, audio: FloatArray): Voice {
            val n = codes[0].size
            val tmp = File(modelDir, "voices/.$id.tmp")
            tmp.deleteRecursively()
            check(tmp.mkdirs()) { "could not create ${tmp.path}" }
            Npy.saveU2(File(tmp, "codes.npy"), intArrayOf(Audio8Engine.NUM_CB, n),
                IntArray(Audio8Engine.NUM_CB * n) { codes[it / n][it % n] })
            File(tmp, "meta.json").writeText(JSONObject().put("reference_text", transcript)
                .put("shape", JSONArray(listOf(Audio8Engine.NUM_CB, n))).put("sample_rate", Audio8Engine.SR)
                .put("source", "recorded on this phone (${Build.MODEL}) by the Audio8 TTS sample")
                .put("encoder", Audio8Engine.ENCODER).put("recorded_epoch_ms", System.currentTimeMillis()).toString(2))
            Wav.writeMono16(File(tmp, "reference.wav"), Wav.floatToPcm16(audio), Audio8Engine.SR)
            val dir = voiceDir(modelDir, id)
            dir.deleteRecursively()
            check(tmp.renameTo(dir)) { "could not write ${dir.path}" }
            return Voice(id, transcript, codes)
        }

        /**
         * The codec mode for a launch without an explicit choice: "auto", or "cpu" once the GPU codec has failed its
         * check on this OS build with this LiteRT version and codec file. The check costs about 5 s of load on a phone
         * where it fails, and the outcome does not change between launches.
         */
        fun rememberedCodecMode(context: Context, modelDir: File): String =
            if (context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).contains(gpuFailureKey(modelDir))) "cpu" else "auto"

        /** Keeps a failed GPU codec check of an "auto" load for [rememberedCodecMode]; a passed check is not kept. */
        fun rememberCodecCheck(context: Context, tts: Audio8Tts) {
            val e = tts.engine
            if (e.codecMode != "auto" || e.codecBackend != "CPU") return
            val why = e.gpuError ?: "GPU codec not used"
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString(gpuFailureKey(tts.modelDir), "$why (${System.currentTimeMillis()})").apply()
        }

        /** The remembered failure, or null. */
        fun rememberedGpuFailure(context: Context, modelDir: File): String? =
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(gpuFailureKey(modelDir), null)

        private fun gpuFailureKey(modelDir: File): String = "gpu_codec_failed|${Build.FINGERPRINT}|" +
            "${BuildConfig.LITERT_VERSION}|${File(modelDir, Audio8Engine.CODEC_GPU_128).length()}"
    }
}
