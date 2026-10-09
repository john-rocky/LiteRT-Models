package com.audio8tts

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The sample's device check: the [Audio8Tts] the screen uses, on the model files pushed to the app's external files
 * dir. One logcat line per step under the tag `audio8-check`,
 *   RESULT step=<name> ok=<bool> key=value ...
 * then a last line `RESULT ok=<all> ...` with the pre-registered line: models, load, speak-ja, speak-en, cancel,
 * regenerate and release ok; RTF <= 2.0 on the first (cold) speak; at least 3 failure-path steps ok. The register step
 * (a voice registered from the speak-ja wav, then used) counts in `ok=<all>`, not in the line.
 *
 *   adb shell am instrument -w -e class com.audio8tts.Audio8DeviceCheck \
 *       com.audio8tts.test/androidx.test.runner.AndroidJUnitRunner
 *   adb logcat -d -s audio8-check | grep RESULT
 *
 * The speak steps write their wav files to <external files dir>/check/.
 */
@RunWith(AndroidJUnit4::class)
class Audio8DeviceCheck {
    private class Step(val name: String, val ok: Boolean, val failurePath: Boolean, val exercised: Boolean)

    private val steps = ArrayList<Step>()
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val modelDir = ctx.getExternalFilesDir(null)!!
    private val outDir = File(modelDir, "check")

    @Test
    fun check() {
        Log.i(TAG, "START device=${Build.MANUFACTURER}_${Build.MODEL} android=${Build.VERSION.RELEASE} " +
            "litert=${BuildConfig.LITERT_VERSION} dir=${modelDir.path} thermal=${thermal()}")
        outDir.mkdirs()
        val constants = PromptConstants(ctx.assets.open("prompt_constants.json").bufferedReader().readText())

        step("models") {
            val names = Audio8Tts.REQUIRED_FILES + Audio8Tts.OPTIONAL_FILES
            val sizes = names.associateWith { File(modelDir, it).let { f -> if (f.isFile) f.length() else -1L } }
            val missing = Audio8Tts.missingFiles(modelDir)
            val voices = Audio8Tts.BUNDLED_VOICES.filter { Audio8Tts.hasVoice(modelDir, it) }
            (missing.isEmpty() && voices.size == Audio8Tts.BUNDLED_VOICES.size) to linkedMapOf(
                "required_missing" to missing.ifEmpty { "none" },
                "bytes_total" to sizes.values.filter { it > 0 }.sum(),
                "voices" to voices.size,
                "files" to sizes.entries.joinToString(",") { "${it.key}:${it.value}" },
            )
        }

        var loaded: Audio8Tts? = null
        step("load") {
            val t = Audio8Tts.load(modelDir, ctx.cacheDir, constants, threads = 4, codecMode = "auto")
            Audio8Tts.rememberCodecCheck(ctx, t)
            loaded = t
            val selfTest = t.tokenizerSelfTest().getBoolean("pass")
            selfTest to linkedMapOf(
                "load_s" to t.loadSeconds,
                "codec" to t.codecBackend,
                "codec_mode" to t.engine.codecMode,
                "gpu_corr" to t.engine.gpuCheck?.get("corr_gpu_vs_cpu_int8"),
                "gpu_error" to t.engine.gpuError,
                "threads" to t.threads,
                "tokenizer_self_test" to selfTest,
                "create_ms" to t.engine.createMs,
                "warmup_ms" to t.engine.warmupMs,
            )
        }

        var firstRtf = Double.NaN
        val t = loaded
        if (t == null) {
            for (n in listOf("speak-ja", "speak-en", "empty-text", "long-text", "cancel", "regenerate", "release")) {
                record(n, false, n in FAILURE_PATHS, true, linkedMapOf("skipped" to "load failed"))
            }
        } else {
            val voiceJa = Audio8Tts.loadVoice(modelDir, JA_VOICE)
            val voiceEn = Audio8Tts.loadVoice(modelDir, EN_VOICE)
            val ja = speakStep("speak-ja", t, TEXT_JA, voiceJa)
            firstRtf = ja?.rtf ?: Double.NaN
            speakStep("speak-en", t, TEXT_EN, voiceEn)

            step("empty-text") {
                val c = t.checkText(" 　\n ")
                val thrown = runCatching { t.speak("", voiceJa) }.exceptionOrNull()
                (c is Audio8Tts.TextCheck.Empty && thrown is Audio8Tts.TextRefused && thrown.message == c.message &&
                    !t.busy) to linkedMapOf(
                    "check" to c.javaClass.simpleName,
                    "thrown" to thrown?.javaClass?.simpleName,
                    "message" to c.message,
                )
            }

            step("long-text") {
                val c = t.checkText(TEXT_LONG)
                val thrown = runCatching { t.speak(TEXT_LONG, voiceEn) }.exceptionOrNull()
                (c is Audio8Tts.TextCheck.TooLong && c.tokens > TextRules.MAX_TEXT_TOKENS &&
                    thrown is Audio8Tts.TextRefused && !t.busy) to linkedMapOf(
                    "tokens" to (c as? Audio8Tts.TextCheck.TooLong)?.tokens,
                    "limit" to TextRules.MAX_TEXT_TOKENS,
                    "thrown" to thrown?.javaClass?.simpleName,
                    "message" to c.message,
                )
            }

            step("cancel") {
                val cancel = AtomicBoolean(false)
                val reached = CountDownLatch(1)
                var outcome: Throwable? = null
                val thread = Thread {
                    outcome = runCatching {
                        t.speak(TEXT_CANCEL, voiceEn, cancel = cancel,
                            onFrame = { n, _ -> if (n >= CANCEL_AT_FRAME) reached.countDown() })
                    }.exceptionOrNull()
                }
                val t0 = SystemClock.elapsedRealtimeNanos()
                thread.start()
                val gotThere = reached.await(120, TimeUnit.SECONDS)
                val c0 = SystemClock.elapsedRealtimeNanos()
                cancel.set(true)
                thread.join(120_000)
                val c1 = SystemClock.elapsedRealtimeNanos()
                (gotThere && !thread.isAlive && outcome is CancellationException && !t.busy && !t.isClosed) to linkedMapOf(
                    "cancel_at_frame" to CANCEL_AT_FRAME,
                    "cancel_after_s" to (c0 - t0) / 1e9,
                    "stopped_in_ms" to (c1 - c0) / 1e6,
                    "outcome" to outcome?.javaClass?.simpleName,
                    "message" to outcome?.message,
                    "busy_after" to t.busy,
                )
            }

            val again = speakStep("regenerate", t, TEXT_JA, voiceJa) { sp ->
                linkedMapOf("same_frames_as_speak_ja" to (ja != null && sameFrames(ja.gen.frames, sp.gen.frames)))
            }
            Log.i(TAG, "regenerate after cancel: ${if (again != null) "ok" else "failed"}")

            // "Record my voice" without a microphone: the speak-ja wav and its text go through the same registration
            // (encoder, codes.npy + meta.json + reference.wav), are read back like any voice and speak the English text.
            step("register") {
                val (ref, rate) = Wav.monoFloat(File(outDir, "speak-ja.wav"))
                check(rate == Audio8Engine.SR) { "speak-ja.wav is $rate Hz" }
                val (voice, r) = t.registerVoice(ref, TEXT_JA, CHECK_VOICE)
                val dir = Audio8Tts.voiceDir(modelDir, CHECK_VOICE)
                val saved = dir.list().orEmpty().sorted()
                val back = Audio8Tts.loadVoice(modelDir, CHECK_VOICE)
                val sameCodes = back.frames == voice.frames && back.codes.indices.all { back.codes[it].contentEquals(voice.codes[it]) }
                val sp = t.speak(TEXT_EN, back)
                val wav = File(outDir, "register-speak.wav")
                wav.writeBytes(sp.wavBytes())
                dir.deleteRecursively()
                (sameCodes && back.transcript == TEXT_JA && saved == listOf("codes.npy", "meta.json", "reference.wav") &&
                    sp.audioSeconds in 1.0..15.0 && sp.rms > RMS_FLOOR) to linkedMapOf(
                    "ref_s" to ref.size / Audio8Engine.SR.toDouble(),
                    "frames" to voice.frames,
                    "encoder_create_ms" to r.createMs,
                    "encode_ms" to r.encodeMs,
                    "encoder_close_ms" to r.closeMs,
                    "saved" to saved,
                    "codes_read_back_equal" to sameCodes,
                    "speak_frames" to sp.gen.frames.size,
                    "speak_generate_s" to sp.generateSeconds,
                    "speak_audio_s" to sp.audioSeconds,
                    "speak_rtf" to sp.rtf,
                    "speak_rms" to sp.rms,
                    "wav" to wav.name,
                    "removed" to !dir.exists(),
                )
            }

            step("release") {
                t.close()
                val second = runCatching { t.close() }.exceptionOrNull()
                val after = runCatching { t.speak(TEXT_EN, voiceEn) }.exceptionOrNull()
                (second == null && t.isClosed && after is IllegalStateException) to linkedMapOf(
                    "second_close" to (second?.javaClass?.simpleName ?: "no_exception"),
                    "speak_after_close" to after?.javaClass?.simpleName,
                    "message" to after?.message,
                )
            }
        }

        step("missing-model") {
            val empty = File(ctx.cacheDir, "check-empty-model-dir").apply { deleteRecursively(); mkdirs() }
            val thrown = runCatching { Audio8Tts.load(empty, ctx.cacheDir, constants) }.exceptionOrNull()
            val names = (thrown as? Audio8Tts.MissingModelFiles)?.names.orEmpty()
            (thrown is Audio8Tts.MissingModelFiles && names == Audio8Tts.REQUIRED_FILES &&
                thrown.message.orEmpty().contains(Audio8Engine.SLOW)) to linkedMapOf(
                "thrown" to thrown?.javaClass?.simpleName,
                "missing" to names.size,
                "message" to thrown?.message,
            )
        }

        val granted = ctx.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        step("mic-denied", exercised = !granted) {
            val problem = Audio8Tts.recordingProblem(ctx, modelDir)
            val bundledLoad = Audio8Tts.BUNDLED_VOICES.all { runCatching { Audio8Tts.loadVoice(modelDir, it) }.isSuccess }
            (granted || (problem == Audio8Tts.MIC_DENIED && bundledLoad)) to linkedMapOf(
                "granted" to granted,
                "exercised" to !granted,
                "message" to problem,
                "bundled_voices_load" to bundledLoad,
            )
        }

        val coreOk = CORE.all { n -> steps.any { it.name == n && it.ok } }
        val failurePathsOk = steps.filter { it.failurePath && it.exercised && it.ok }.map { it.name }
        val rtfOk = firstRtf <= MAX_FIRST_RTF
        val linePass = coreOk && rtfOk && failurePathsOk.size >= 3
        val allOk = steps.all { it.ok }
        val failed = steps.filter { !it.ok }.joinToString(",") { it.name }.ifEmpty { "none" }
        Log.i(TAG, "RESULT ok=$allOk steps=${steps.size} failed=$failed core_ok=$coreOk " +
            "first_rtf=${fmt(firstRtf)} rtf_line=${fmt(MAX_FIRST_RTF)} failure_paths_ok=${failurePathsOk.size} " +
            "failure_paths=${failurePathsOk.joinToString(",")} line_pass=$linePass thermal=${thermal()}")
        assertTrue("device check: failed=$failed line_pass=$linePass", allOk && linePass)
    }

    /** One speak step: the wav is written, lasts 1 to 15 s and is not silent; returns the speech, or null. */
    private fun speakStep(
        name: String,
        t: Audio8Tts,
        text: String,
        voice: Audio8Tts.Voice,
        extra: (Audio8Tts.Speech) -> Map<String, Any?> = { emptyMap() },
    ): Audio8Tts.Speech? {
        var result: Audio8Tts.Speech? = null
        step(name) {
            val before = thermal()
            val tokens = (t.checkText(text) as? Audio8Tts.TextCheck.Ok)?.tokens
            val sp = t.speak(text, voice)
            val wav = File(outDir, "$name.wav")
            wav.writeBytes(sp.wavBytes())
            result = sp
            val ok = wav.length() == 44L + sp.pcm.size && sp.audioSeconds in 1.0..15.0 && sp.rms > RMS_FLOOR
            ok to (linkedMapOf<String, Any?>(
                "voice" to voice.id,
                "tokens" to tokens,
                "frames" to sp.gen.frames.size,
                "stopped_by" to sp.gen.stoppedBy,
                "generate_s" to sp.generateSeconds,
                "audio_s" to sp.audioSeconds,
                "rtf" to sp.rtf,
                "codec_s" to sp.codecSeconds,
                "codec" to sp.codecInfo["codec_backend"],
                "prefill_ms" to sp.gen.stats["prefill_ms"],
                "slow_ms_median" to median(sp.gen.decodeMs),
                "fast10_ms_median" to median(sp.gen.fastMs),
                "rms" to sp.rms,
                "wav" to wav.name,
                "wav_bytes" to wav.length(),
                "thermal_before" to before,
                "thermal_after" to thermal(),
            ) + extra(sp))
        }
        return result
    }

    private fun step(name: String, exercised: Boolean = true, block: () -> Pair<Boolean, Map<String, Any?>>) {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val (ok, kv) = try {
            block()
        } catch (e: Throwable) {
            Log.e(TAG, "step $name threw", e)
            false to linkedMapOf<String, Any?>("error" to e.toString())
        }
        record(name, ok, name in FAILURE_PATHS, exercised, kv + ("step_s" to (SystemClock.elapsedRealtimeNanos() - t0) / 1e9))
    }

    private fun record(name: String, ok: Boolean, failurePath: Boolean, exercised: Boolean, kv: Map<String, Any?>) {
        steps.add(Step(name, ok, failurePath, exercised))
        Log.i(TAG, "RESULT step=$name ok=$ok " + kv.entries.joinToString(" ") { "${it.key}=${value(it.value)}" })
    }

    private fun value(v: Any?): String = when (v) {
        null -> "null"
        is Double -> fmt(v)
        is Float -> fmt(v.toDouble())
        is String -> if (v.any { it.isWhitespace() || it == '=' }) "\"${v.replace("\"", "'").replace('\n', ' ')}\"" else v
        is Map<*, *> -> v.entries.joinToString(",", "{", "}") { "${it.key}:${value(it.value)}" }
        is List<*> -> v.joinToString(",", "[", "]") { value(it) }
        else -> v.toString()
    }

    private fun fmt(d: Double): String = if (d.isNaN()) "NaN" else String.format(Locale.US, "%.3f", d)

    private fun median(x: DoubleArray): Double = if (x.isEmpty()) Double.NaN else x.sorted()[x.size / 2]

    private fun sameFrames(a: List<IntArray>, b: List<IntArray>): Boolean =
        a.size == b.size && a.indices.all { a[it].contentEquals(b[it]) }

    /** Thermal status and every CPU policy whose scaling_max_freq is below cpuinfo_max_freq (a frequency cap). */
    private fun thermal(): String {
        val status = ctx.getSystemService(PowerManager::class.java)?.currentThermalStatus ?: -1
        val caps = runCatching {
            File("/sys/devices/system/cpu/cpufreq").listFiles { f -> f.name.startsWith("policy") }.orEmpty()
                .sortedBy { it.name }.mapNotNull { d ->
                    val cur = File(d, "scaling_max_freq").readText().trim()
                    val max = File(d, "cpuinfo_max_freq").readText().trim()
                    if (cur != max) "${d.name}:$cur/$max" else null
                }
        }.getOrElse { listOf("unreadable") }
        return "status$status/caps:${caps.joinToString("+").ifEmpty { "none" }}"
    }

    companion object {
        private const val TAG = "audio8-check"
        private const val JA_VOICE = "ja_funasr_example"
        private const val EN_VOICE = "en_librispeech_1272"
        private const val CHECK_VOICE = "check_registered"
        const val TEXT_JA = "今日は天気が良いので、公園まで散歩に行きましょう。"
        const val TEXT_EN = "Hello from LiteRT. This voice was made on the phone, with no network at all."
        const val TEXT_CANCEL = "This sentence is long on purpose, so that the check can stop the speech in the middle " +
            "and then make sure that the next request still works."
        val TEXT_LONG = "Everything you hear was made on this phone, with no network at all. ".repeat(7)
        private const val CANCEL_AT_FRAME = 20
        private const val RMS_FLOOR = 0.01
        private const val MAX_FIRST_RTF = 2.0
        private val CORE = listOf("models", "load", "speak-ja", "speak-en", "cancel", "regenerate", "release")
        private val FAILURE_PATHS = setOf("empty-text", "long-text", "cancel", "release", "missing-model", "mic-denied")
    }
}
