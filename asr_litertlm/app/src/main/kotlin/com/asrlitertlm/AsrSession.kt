package com.asrlitertlm

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.io.IOException
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

/** One bundled clip: FLEURS test sentence 1698 read in one language (google/fleurs, CC BY 4.0). */
data class Clip(val id: String, val label: String, val rawRes: Int) {
  companion object {
    val ALL =
      listOf(
        Clip("fleurs_cmn_1698", "zh", R.raw.fleurs_cmn_1698),
        Clip("fleurs_en_1698", "en", R.raw.fleurs_en_1698),
        Clip("fleurs_ja_1698", "ja", R.raw.fleurs_ja_1698),
      )
  }
}

/** The end of one transcription request. */
sealed interface TranscribeOutcome {
  data class Text(val transcript: Transcript) : TranscribeOutcome

  /** The audio was close to digital silence, so the model was not called. */
  data class NoSpeech(val levels: Wav.Levels, val seconds: Double) : TranscribeOutcome
}

/** What the session did, delivered to listeners on the main thread. */
sealed interface AsrEvent {
  data class Loading(val profile: ModelProfile, val lmBackend: LmBackend) : AsrEvent

  data class Loaded(val info: LoadInfo) : AsrEvent

  data class LoadFailed(val profile: ModelProfile, val message: String) : AsrEvent

  data class Transcribing(val source: String) : AsrEvent

  data class Transcribed(val source: String, val outcome: TranscribeOutcome) : AsrEvent

  data class TranscribeFailed(val source: String, val message: String) : AsrEvent

  data class Released(val profile: ModelProfile) : AsrEvent
}

/**
 * The app's one engine, shared by the screen and the device check. Every model call runs on one worker thread (an
 * Engine is not called from two threads), and one bundle is in memory at a time. The methods return futures for
 * callers that wait (the device check) and publish [AsrEvent]s for the screen.
 */
object AsrSession {
  private const val TAG = "AsrLitertlm"

  /** Recordings quieter than this (RMS) are reported as no speech instead of being sent to the model. */
  const val NO_SPEECH_RMS_DBFS = -60.0

  private val worker: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "asr-worker") }
  private val main = Handler(Looper.getMainLooper())
  private val listeners = CopyOnWriteArrayList<(AsrEvent) -> Unit>()

  @Volatile private var engine: AsrEngine? = null
  private lateinit var modelDirectory: File
  private lateinit var cacheDirectory: File
  private lateinit var appContext: Context

  /** The bundle in memory, or null. */
  @Volatile
  var current: LoadInfo? = null
    private set

  /** The last load request, so that a caller can wait for a load the screen started. */
  @Volatile
  var lastLoad: Future<LoadInfo>? = null
    private set

  /** Where the app looks for the bundles: `/sdcard/Android/data/com.asrlitertlm/files`. */
  val modelDir: File
    get() = modelDirectory

  fun init(context: Context) {
    synchronized(this) {
      if (engine != null) return
      appContext = context.applicationContext
      modelDirectory = appContext.getExternalFilesDir(null) ?: appContext.filesDir
      cacheDirectory = appContext.cacheDir
      engine = AsrEngine(cacheDirectory)
    }
  }

  fun bundleFile(profile: ModelProfile): File = File(modelDirectory, profile.fileName)

  fun isPresent(profile: ModelProfile): Boolean = bundleFile(profile).isFile

  /** The clip as a wav file the runtime can open (copied out of the APK once). */
  fun clipFile(clip: Clip): File {
    val file = File(cacheDirectory, "${clip.id}.wav")
    if (!file.isFile) {
      val partial = File(cacheDirectory, "${clip.id}.wav.part")
      appContext.resources.openRawResource(clip.rawRes).use { input ->
        partial.outputStream().use { input.copyTo(it) }
      }
      check(partial.renameTo(file)) { "could not write $file" }
    }
    return file
  }

  fun addListener(listener: (AsrEvent) -> Unit) {
    listeners.add(listener)
  }

  fun removeListener(listener: (AsrEvent) -> Unit) {
    listeners.remove(listener)
  }

  /** Loads [profile] (closing the bundle loaded before). [bundle] defaults to the file in [modelDir]. */
  fun load(
    profile: ModelProfile,
    lmBackend: LmBackend = LmBackend.CPU,
    bundle: File = bundleFile(profile),
  ): Future<LoadInfo> =
    worker
      .submit(
        Callable {
          publish(AsrEvent.Loading(profile, lmBackend))
          val asr = checkNotNull(engine) { "AsrSession.init() was not called" }
          try {
            asr.load(profile, bundle, lmBackend).also { info ->
              current = info
              Log.i(
                TAG,
                String.format(
                  Locale.US,
                  "LOAD model=%s backend=%s load_ms=%.1f initialize_ms=%.1f first_conversation_ms=%.1f bytes=%d",
                  profile.id,
                  lmBackend,
                  info.loadMs,
                  info.initializeMs,
                  info.firstConversationMs,
                  info.bundleBytes,
                ),
              )
              publish(AsrEvent.Loaded(info))
            }
          } catch (e: AsrException) {
            current = asr.loaded
            Log.e(TAG, "LOAD_FAILED model=${profile.id} backend=$lmBackend: ${e.message}", e.cause)
            publish(AsrEvent.LoadFailed(profile, e.message ?: e.toString()))
            throw e
          }
        }
      )
      .also { lastLoad = it }

  /**
   * Transcribes a 16 kHz mono 16-bit wav with the loaded bundle; [source] names the input in events and logs
   * ("clip:<id>", "mic"). Near-silent audio returns [TranscribeOutcome.NoSpeech] without calling the model.
   */
  fun transcribe(wav: File, source: String): Future<TranscribeOutcome> =
    worker.submit(
      Callable {
        publish(AsrEvent.Transcribing(source))
        val asr = checkNotNull(engine) { "AsrSession.init() was not called" }
        try {
          val pcm =
            try {
              Wav.pcm16(wav)
            } catch (e: IOException) {
              throw AsrException("Could not read ${wav.name}: ${e.message}", e)
            } catch (e: RuntimeException) { // not a wav, no data chunk, not 16-bit
              throw AsrException("Could not read ${wav.name}: ${e.message}", e)
            }
          val levels = Wav.levels(pcm.data)
          val outcome =
            if (levels.rmsDbfs < NO_SPEECH_RMS_DBFS) {
              TranscribeOutcome.NoSpeech(levels, pcm.seconds)
            } else {
              TranscribeOutcome.Text(asr.transcribe(wav, File(cacheDirectory, "sent.wav")))
            }
          log(source, outcome)
          publish(AsrEvent.Transcribed(source, outcome))
          outcome
        } catch (e: AsrException) {
          Log.e(TAG, "TRANSCRIBE_FAILED source=$source: ${e.message}", e.cause)
          publish(AsrEvent.TranscribeFailed(source, e.message ?: e.toString()))
          throw e
        }
      }
    )

  /** Closes the loaded bundle; the future holds false when none was loaded (a second call is harmless). */
  fun release(): Future<Boolean> =
    worker.submit(
      Callable {
        val profile = current?.profile
        val closed = checkNotNull(engine) { "AsrSession.init() was not called" }.release()
        current = null
        Log.i(TAG, "RELEASE model=${profile?.id} closed=$closed")
        if (closed && profile != null) publish(AsrEvent.Released(profile))
        closed
      }
    )

  private fun log(source: String, outcome: TranscribeOutcome) {
    when (outcome) {
      is TranscribeOutcome.Text -> {
        val transcript = outcome.transcript
        Log.i(
          TAG,
          String.format(
            Locale.US,
            "TRANSCRIBED source=%s model=%s backend=%s audio_s=%.2f sent_s=%.2f send_ms=%.1f rtf=%.3f raw=%s",
            source,
            transcript.profile.id,
            transcript.lmBackend,
            transcript.audioSeconds,
            transcript.sentSeconds,
            transcript.sendMs,
            transcript.rtf,
            transcript.raw.replace('\n', ' '),
          ),
        )
      }
      is TranscribeOutcome.NoSpeech ->
        Log.i(
          TAG,
          String.format(
            Locale.US,
            "NO_SPEECH source=%s seconds=%.2f rms_dbfs=%.1f peak_dbfs=%.1f",
            source,
            outcome.seconds,
            outcome.levels.rmsDbfs,
            outcome.levels.peakDbfs,
          ),
        )
    }
  }

  private fun publish(event: AsrEvent) {
    main.post { listeners.forEach { it(event) } }
  }
}
