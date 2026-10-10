package com.asrlitertlm

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device check for the app's paths on a phone with the bundles pushed. One line per step under the log tag
 * `asr-check`: `RESULT step=<name> ok=<bool> model=<id> ...`, then `RESULT ok=<all> scope=<test>`.
 *
 * [check] expects RECORD_AUDIO to be revoked when it starts (`adb shell pm revoke com.asrlitertlm
 * android.permission.RECORD_AUDIO`): its first step is the microphone path without the permission; it then grants the
 * permission itself. Per bundle on the phone: load, the three clips (the answer's text after normalization equals the
 * expected sentence; the CER is printed when it does not), the microphone (clip en played through the speaker and
 * picked up by the microphone, CER <= 0.15), release (a second release is harmless) and, when the load wrote the
 * runtime cache, a second load. Then the failure paths: switching models, a clip longer than 30 s, a silent
 * recording and a bundle that is not on the phone.
 *
 * [gpu] runs only with `-e gpu_model <id>`: the language model on the GPU, the audio encoder on the CPU.
 */
@RunWith(AndroidJUnit4::class)
class AsrDeviceCheck {
  private val instrumentation = InstrumentationRegistry.getInstrumentation()
  private val context = instrumentation.targetContext
  private val outcomes = mutableListOf<Boolean>()

  @Test
  fun check() {
    AsrSession.init(context)
    val present = ORDER.filter { AsrSession.isPresent(it) }
    report(
      "bundles",
      present.isNotEmpty(),
      null,
      "present" to present.joinToString(",") { it.id }.ifEmpty { "none" },
      "missing" to ORDER.filterNot { it in present }.joinToString(",") { it.id }.ifEmpty { "none" },
      "dir" to AsrSession.modelDir.path,
    )
    if (present.isEmpty()) return finish("check")

    val first = present.first()
    val firstCache = cacheFiles(first)
    val screen = launch(launchIntent(first))
    try {
      val firstLoad = awaitLoad(checkNotNull(AsrSession.lastLoad) { "the screen did not start a load" }, first)
      val loads = mutableMapOf(first to (firstLoad to firstCache))

      permissionStep(screen, first)
      instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.RECORD_AUDIO)
      report("grant", hasMicPermission(), null, "granted" to hasMicPermission())

      for (profile in present) {
        val load = loads[profile] ?: run {
          val before = cacheFiles(profile)
          awaitLoad(onScreen(screen) { it.selectModel(profile) }, profile)?.let { it to before }
        }
        if (load?.first == null) continue
        reportLoad("load", profile, load.first!!, load.second)
        clipSteps(profile, "clip")
        micStep(screen, profile)
        releaseStep(profile)
        if (cacheState(profile, load.second) == "written") reloadStep(profile)
      }
      failurePaths(screen, present)
    } finally {
      close(screen)
    }
    finish("check")
  }

  @Test
  fun gpu() {
    val profile = ModelProfile.byId(InstrumentationRegistry.getArguments().getString("gpu_model"))
    assumeTrue("run with -e gpu_model <id>", profile != null)
    AsrSession.init(context)
    val model = profile!!
    if (!AsrSession.isPresent(model)) {
      report("gpu-load", false, model, "reason" to "bundle not on the phone")
      return finish("gpu")
    }
    val before = cacheFiles(model)
    val screen = launch(launchIntent(model).putExtra(MainActivity.EXTRA_BACKEND, "gpu"))
    try {
      val info = awaitLoad(checkNotNull(AsrSession.lastLoad), model, step = "gpu-load")
      if (info != null) {
        reportLoad("gpu-load", model, info, before)
        clipSteps(model, "gpu-clip")
        releaseStep(model, step = "gpu-release")
        val beforeReload = cacheFiles(model)
        val reloaded = awaitLoad(AsrSession.load(model, LmBackend.GPU), model, step = "gpu-reload")
        if (reloaded != null) reportLoad("gpu-reload", model, reloaded, beforeReload)
        AsrSession.release().get(TIMEOUT_S, TimeUnit.SECONDS)
      }
    } finally {
      close(screen)
    }
    finish("gpu")
  }

  // ---- steps --------------------------------------------------------------------------------------------------

  /** The microphone path with RECORD_AUDIO revoked returns a sentence, records nothing, and a clip still works. */
  private fun permissionStep(screen: MainActivity, profile: ModelProfile) {
    val deniedAtStart = !hasMicPermission()
    val result = micTest(screen, durationMs = 2000, play = false, waitS = 10)
    val sentence = (result as? MicTestResult.Failed)?.message.orEmpty()
    val clip = Clip.ALL[1]
    val clipText = transcribeText(AsrSession.clipFile(clip), "clip:${clip.id}")
    val clipOk = clipText != null && TextMatch.matches(EXPECTED.getValue(clip.id), clipText.answer.text)
    report(
      "permission",
      deniedAtStart && sentence == MicRecorder.PERMISSION_MESSAGE && clipOk,
      profile,
      "denied_at_start" to deniedAtStart,
      "message" to sentence.ifEmpty { result.toString() },
      "clip_en_while_denied" to clipOk,
    )
  }

  private fun clipSteps(profile: ModelProfile, step: String) {
    for (clip in Clip.ALL) {
      val expected = EXPECTED.getValue(clip.id)
      val transcript = transcribeText(AsrSession.clipFile(clip), "clip:${clip.id}")
      if (transcript == null) {
        report(step, false, profile, "clip" to clip.label, "reason" to "no transcript")
        continue
      }
      val text = transcript.answer.text
      val match = TextMatch.matches(expected, text)
      report(
        step,
        match,
        profile,
        "clip" to clip.label,
        "backend" to transcript.lmBackend,
        "audio_s" to fmt(transcript.sentSeconds),
        "send_s" to fmt(transcript.sendMs / 1000),
        "rtf" to fmt(transcript.rtf, digits = 3),
        "conversation_s" to fmt(transcript.createConversationMs / 1000),
        "language" to transcript.answer.language.ifEmpty { "-" },
        "match" to match,
        "cer" to fmt(TextMatch.cer(expected, text), digits = 4),
        "text" to text,
      )
    }
  }

  /** Clip en through the speaker, picked up by the microphone (the mic_test launch extras' path). */
  private fun micStep(screen: MainActivity, profile: ModelProfile) {
    val result = micTest(screen, durationMs = MIC_TEST_MS, play = true, waitS = 90)
    val outcome = (result as? MicTestResult.Finished)?.outcome
    val transcript = (outcome as? TranscribeOutcome.Text)?.transcript
    if (transcript == null) {
      report("mic", false, profile, "result" to result.toString())
      return
    }
    val expected = EXPECTED.getValue(Clip.ALL[1].id)
    val cer = TextMatch.cer(expected, transcript.answer.text)
    report(
      "mic",
      cer <= MIC_MAX_CER,
      profile,
      "rec_s" to fmt(transcript.audioSeconds),
      "send_s" to fmt(transcript.sendMs / 1000),
      "rtf" to fmt(transcript.rtf, digits = 3),
      "cer" to fmt(cer, digits = 4),
      "language" to transcript.answer.language.ifEmpty { "-" },
      "text" to transcript.answer.text,
    )
  }

  private fun releaseStep(profile: ModelProfile, step: String = "release") {
    val first = runCatching { AsrSession.release().get(TIMEOUT_S, TimeUnit.SECONDS) }
    val second = runCatching { AsrSession.release().get(TIMEOUT_S, TimeUnit.SECONDS) }
    report(
      step,
      first.getOrNull() == true && second.getOrNull() == false && AsrSession.current == null,
      profile,
      "first" to (first.getOrNull() ?: first.exceptionOrNull()),
      "second" to (second.getOrNull() ?: second.exceptionOrNull()),
      "vmrss_mb" to statusMb("VmRSS"),
    )
  }

  /** A second load, now that the first wrote the runtime cache: the load time a user sees from then on. */
  private fun reloadStep(profile: ModelProfile) {
    val before = cacheFiles(profile)
    val info = awaitLoad(AsrSession.load(profile), profile, step = "reload") ?: return
    reportLoad("reload", profile, info, before)
    AsrSession.release().get(TIMEOUT_S, TimeUnit.SECONDS)
  }

  private fun failurePaths(screen: MainActivity, present: List<ModelProfile>) {
    // Switching: load one bundle, then pick another without releasing; the screen's path closes the first.
    val to = present.first()
    val from = present.firstOrNull { it != to }
    if (from == null) {
      // Not a result: the phone holds one bundle (with its runtime cache) at a time when storage is short.
      Log.i(TAG, "NOTE step=switch not_run=\"one bundle on the phone\" model=${to.id}")
    } else {
      awaitLoad(onScreen(screen) { it.selectModel(from) }, from, step = "switch")
      val switched = awaitLoad(onScreen(screen) { it.selectModel(to) }, to, step = "switch")
      val clip = Clip.ALL[1]
      val text = transcribeText(AsrSession.clipFile(clip), "clip:${clip.id}")?.answer?.text
      report(
        "switch",
        switched != null && AsrSession.current?.profile == to && text != null &&
          TextMatch.matches(EXPECTED.getValue(clip.id), text),
        to,
        "from" to from.id,
        "load_s" to (switched?.let { fmt(it.loadMs / 1000) } ?: "-"),
        "vmrss_mb" to statusMb("VmRSS"),
        "text" to (text ?: "-"),
      )
    }
    if (AsrSession.current == null) awaitLoad(AsrSession.load(to), to, step = "long-clip")

    // Longer than one window: the app sends the first 30 s and does not crash.
    val long = File(context.cacheDir, "long-check.wav")
    val pcm = (Clip.ALL + Clip.ALL[1]).map { Wav.pcm16(AsrSession.clipFile(it)).data }
    Wav.writeMono16(long, pcm.fold(ByteArray(0)) { all, part -> all + part }, MicRecorder.SAMPLE_RATE)
    val longText = transcribeText(long, "long-clip")
    report(
      "long-clip",
      longText != null && longText.audioSeconds > 31.0 && kotlin.math.abs(longText.sentSeconds - 30.0) < 0.01 &&
        longText.answer.text.isNotEmpty(),
      AsrSession.current?.profile,
      "audio_s" to fmt(longText?.audioSeconds ?: Wav.seconds(long)),
      "sent_s" to (longText?.let { fmt(it.sentSeconds) } ?: "-"),
      "send_s" to (longText?.let { fmt(it.sendMs / 1000) } ?: "-"),
      "text" to (longText?.answer?.text ?: "-"),
    )

    // A silent recording: "no speech", and the model is not called.
    val silent = File(context.cacheDir, "silence-check.wav")
    Wav.writeMono16(silent, ByteArray(MicRecorder.SAMPLE_RATE * 2 * 3), MicRecorder.SAMPLE_RATE)
    val silence = runCatching { AsrSession.transcribe(silent, "silence").get(TIMEOUT_S, TimeUnit.SECONDS) }
    report(
      "silence",
      silence.getOrNull() is TranscribeOutcome.NoSpeech,
      AsrSession.current?.profile,
      "result" to (silence.getOrNull() ?: silence.exceptionOrNull()),
    )

    // A bundle that is not on the phone: a sentence naming the file and where to push it; the loaded model stays.
    val loadedBefore = AsrSession.current?.profile
    val ghost = ModelProfile.QWEN3_ASR_1_7B.copy(fileName = "not-on-this-phone.litertlm")
    val missing =
      runCatching { AsrSession.load(ghost, bundle = AsrSession.bundleFile(ghost)).get(TIMEOUT_S, TimeUnit.SECONDS) }
    val message = ((missing.exceptionOrNull() as? ExecutionException)?.cause as? AsrException)?.message.orEmpty()
    report(
      "missing-bundle",
      missing.isFailure && message.contains("not-on-this-phone.litertlm") &&
        message.contains(AsrSession.modelDir.path) && AsrSession.current?.profile == loadedBefore,
      loadedBefore,
      "message" to message.ifEmpty { missing.toString() },
      "still_loaded" to (AsrSession.current?.profile?.id ?: "none"),
    )
    releaseStep(loadedBefore ?: to, step = "release-final")
    long.delete()
    silent.delete()
  }

  // ---- helpers ------------------------------------------------------------------------------------------------

  private fun launchIntent(profile: ModelProfile) =
    Intent(context, MainActivity::class.java)
      .putExtra(MainActivity.EXTRA_MODEL, profile.id)
      .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

  /** Starts the screen in the foreground (the microphone records only for a visible app) and waits until it runs. */
  private fun launch(intent: Intent): MainActivity = instrumentation.startActivitySync(intent) as MainActivity

  private fun close(screen: MainActivity) {
    instrumentation.runOnMainSync { screen.finish() }
    instrumentation.waitForIdleSync()
  }

  private fun <T> onScreen(screen: MainActivity, block: (MainActivity) -> T): T {
    val value = AtomicReference<T>()
    instrumentation.runOnMainSync { value.set(block(screen)) }
    return value.get()
  }

  private fun awaitLoad(future: Future<LoadInfo>?, profile: ModelProfile, step: String = "load"): LoadInfo? {
    if (future == null) {
      report(step, false, profile, "reason" to "the screen did not load it")
      return null
    }
    return try {
      future.get(LOAD_TIMEOUT_S, TimeUnit.SECONDS)
    } catch (e: ExecutionException) {
      report(step, false, profile, "message" to (e.cause?.message ?: e.toString()))
      null
    }
  }

  private fun transcribeText(wav: File, source: String): Transcript? =
    try {
      (AsrSession.transcribe(wav, source).get(TIMEOUT_S, TimeUnit.SECONDS) as? TranscribeOutcome.Text)?.transcript
    } catch (e: ExecutionException) {
      Log.e(TAG, "transcription failed: $source", e.cause)
      null
    }

  private fun micTest(screen: MainActivity, durationMs: Long, play: Boolean, waitS: Long): MicTestResult? {
    val done = CountDownLatch(1)
    val result = AtomicReference<MicTestResult>()
    instrumentation.runOnMainSync {
      screen.runMicTest(durationMs, play) {
        result.set(it)
        done.countDown()
      }
    }
    done.await(waitS, TimeUnit.SECONDS)
    return result.get()
  }

  private fun reportLoad(step: String, profile: ModelProfile, info: LoadInfo, before: Map<String, CacheFile>) {
    report(
      step,
      true,
      profile,
      "backend" to info.lmBackend,
      "load_s" to fmt(info.loadMs / 1000),
      "initialize_s" to fmt(info.initializeMs / 1000),
      "first_conversation_s" to fmt(info.firstConversationMs / 1000),
      "cache" to cacheState(profile, before),
      "cache_mb" to cacheFiles(profile).values.sumOf { it.bytes } / 1_000_000,
      "threads" to AsrEngine.CPU_THREADS,
      "vmhwm_mb" to statusMb("VmHWM"),
      "vmrss_mb" to statusMb("VmRSS"),
      "mem_available_mb" to memAvailableMb(),
    )
  }

  /** One runtime cache file: the runtime names them after the bundle file. */
  private data class CacheFile(val bytes: Long, val modified: Long)

  private fun cacheFiles(profile: ModelProfile): Map<String, CacheFile> =
    context.cacheDir
      .listFiles { file -> file.name.startsWith(profile.fileName) }
      .orEmpty()
      .associate { it.name to CacheFile(it.length(), it.lastModified()) }

  /** "loaded" when the load left the cache files as they were, "written" when it created or rewrote them. */
  private fun cacheState(profile: ModelProfile, before: Map<String, CacheFile>): String =
    if (cacheFiles(profile) == before && before.isNotEmpty()) "loaded" else "written"

  private fun hasMicPermission(): Boolean =
    context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

  private fun statusMb(key: String): Long =
    File("/proc/self/status").readLines().firstOrNull { it.startsWith("$key:") }
      ?.trim()?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()?.div(1024) ?: -1

  private fun memAvailableMb(): Long =
    File("/proc/meminfo").readLines().firstOrNull { it.startsWith("MemAvailable:") }
      ?.trim()?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()?.div(1024) ?: -1

  private fun report(step: String, ok: Boolean, profile: ModelProfile?, vararg fields: Pair<String, Any?>) {
    outcomes += ok
    val line = StringBuilder("RESULT step=$step ok=$ok model=${profile?.id ?: "-"}")
    for ((key, value) in fields) line.append(' ').append(key).append('=').append(quote(value.toString()))
    Log.i(TAG, line.toString())
  }

  private fun finish(scope: String) {
    val all = outcomes.isNotEmpty() && outcomes.all { it }
    Log.i(TAG, "RESULT ok=$all scope=$scope steps=${outcomes.size} failed=${outcomes.count { !it }}")
    assertTrue("a device check step failed (logcat -s $TAG)", all)
  }

  private fun quote(value: String): String =
    if (value.isNotEmpty() && value.none { it.isWhitespace() || it == '"' }) value
    else "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\""

  private fun fmt(value: Double, digits: Int = 2) = String.format(Locale.US, "%.${digits}f", value)

  private companion object {
    const val TAG = "asr-check"
    const val LOAD_TIMEOUT_S = 300L
    const val TIMEOUT_S = 120L
    const val MIC_TEST_MS = 9000L
    const val MIC_MAX_CER = 0.15

    /** Bundles in the order the check loads them. */
    val ORDER =
      listOf(ModelProfile.QWEN3_ASR_1_7B, ModelProfile.CONFUCIUS4_R2T2, ModelProfile.FUN_ASR_NANO_2512)

    /**
     * The expected sentences: Qwen3-ASR-1.7B's answers on the Mac CPU (litert-lm-api 0.17.1), which the S26 app runs
     * of 2026-10-03 matched. Fun-ASR-Nano-2512 and Confucius4-R2T2 give the same sentences on the Mac CPU.
     */
    val EXPECTED =
      mapOf(
        "fleurs_cmn_1698" to "宇宙中的一切都由物质构成，而所有的物质都由被称为原子的微小颗粒组成。",
        "fleurs_en_1698" to
          "Everything in the universe is made of matter. All matter is made of tiny particles called atoms.",
        "fleurs_ja_1698" to "宇宙に存在するあらゆるものは物質でできています。あらゆる物質は原子と呼ばれる微小な粒子でできています。",
      )
  }
}
