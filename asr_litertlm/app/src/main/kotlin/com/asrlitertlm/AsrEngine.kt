package com.asrlitertlm

import android.os.SystemClock
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File

/** Which processor runs the language model. The audio encoder always runs on the CPU (the GPU delegate rejects it). */
enum class LmBackend {
  CPU,
  GPU,
}

/** A failure the app shows as a sentence instead of crashing. */
class AsrException(message: String, cause: Throwable? = null) : Exception(message, cause)

/** What one load cost. "Engine load" = Engine.initialize() plus the first conversation, which creates the encoder. */
data class LoadInfo(
  val profile: ModelProfile,
  val bundle: File,
  val bundleBytes: Long,
  val lmBackend: LmBackend,
  val initializeMs: Double,
  val firstConversationMs: Double,
) {
  val loadMs: Double
    get() = initializeMs + firstConversationMs
}

/** One transcription. [sendMs] is the wall time of sendMessage(), from sending the clip to the full answer. */
data class Transcript(
  val profile: ModelProfile,
  val lmBackend: LmBackend,
  val raw: String,
  val answer: Answer,
  /** Length of the audio the app received. */
  val audioSeconds: Double,
  /** Length of the audio sent to the model (at most [AsrEngine.MAX_AUDIO_SECONDS]). */
  val sentSeconds: Double,
  val sendMs: Double,
  val createConversationMs: Double,
) {
  /** Real-time factor: transcription time over the length of the audio the model heard. */
  val rtf: Double
    get() = sendMs / 1000.0 / sentSeconds
}

/**
 * One loaded `.litertlm` bundle and the calls every model in [ModelProfile.ALL] shares: an Engine with the language
 * model on the CPU (or the GPU) and the audio encoder on the CPU, a new Conversation per clip, one user message that
 * holds only the audio file, greedy sampling.
 *
 * Not thread-safe: the app calls it from one worker thread ([AsrSession]).
 */
class AsrEngine(private val cacheDir: File) : AutoCloseable {
  private var engine: Engine? = null

  /** The bundle in memory, or null. */
  var loaded: LoadInfo? = null
    private set

  /**
   * Loads [profile] from [bundle], closing the engine loaded before. A missing file or a runtime failure becomes an
   * [AsrException] with a sentence for the screen; a missing file leaves the loaded engine as it was.
   */
  fun load(profile: ModelProfile, bundle: File, lmBackend: LmBackend = LmBackend.CPU): LoadInfo {
    if (!bundle.isFile) throw AsrException(missingBundleMessage(profile, bundle.parentFile))
    // One engine at a time: two 2.7 GB bundles in memory at once would cross what the phone gives one app.
    release()
    val lm = if (lmBackend == LmBackend.GPU) Backend.GPU() else Backend.CPU(threadCount = CPU_THREADS)
    val config =
      EngineConfig(
        modelPath = bundle.path,
        backend = lm,
        audioBackend = Backend.CPU(threadCount = CPU_THREADS),
        cacheDir = cacheDir.path,
      )
    val newEngine = Engine(config)
    val start = SystemClock.elapsedRealtimeNanos()
    val initialized: Long
    try {
      newEngine.initialize()
      initialized = SystemClock.elapsedRealtimeNanos()
      // The runtime creates the audio encoder with the first conversation; do it here so it counts as loading.
      newEngine.createConversation(conversationConfig(profile)).close()
    } catch (t: Throwable) {
      if (newEngine.isInitialized()) runCatching { newEngine.close() }
      throw AsrException("Could not load ${profile.fileName} (${lmBackend.name}): ${t.message ?: t}", t)
    }
    val ready = SystemClock.elapsedRealtimeNanos()
    engine = newEngine
    return LoadInfo(
        profile = profile,
        bundle = bundle,
        bundleBytes = bundle.length(),
        lmBackend = lmBackend,
        initializeMs = (initialized - start) / 1e6,
        firstConversationMs = (ready - initialized) / 1e6,
      )
      .also { loaded = it }
  }

  /**
   * Transcribes a 16 kHz mono 16-bit wav. Audio longer than [MAX_AUDIO_SECONDS] is cut there first ([scratch] holds
   * the cut copy): each model reads one window of about 30 s per message.
   */
  fun transcribe(wav: File, scratch: File): Transcript {
    val currentEngine = engine ?: throw AsrException("No model is loaded.")
    val info = loaded ?: throw AsrException("No model is loaded.")
    val audioSeconds = Wav.seconds(wav)
    val sent = Wav.trimmed(wav, MAX_AUDIO_SECONDS, scratch)
    val sentSeconds = if (sent === wav) audioSeconds else Wav.seconds(sent)
    val created: Long
    val sendStart: Long
    val sendEnd: Long
    val raw: String
    val start = SystemClock.elapsedRealtimeNanos()
    try {
      currentEngine.createConversation(conversationConfig(info.profile)).use { conversation ->
        created = SystemClock.elapsedRealtimeNanos()
        val message = Message.user(Contents.of(Content.AudioFile(sent.absolutePath)))
        sendStart = SystemClock.elapsedRealtimeNanos()
        raw = conversation.sendMessage(message).toString()
        sendEnd = SystemClock.elapsedRealtimeNanos()
      }
    } catch (t: Throwable) {
      throw AsrException("${info.profile.displayName} could not transcribe this clip: ${t.message ?: t}", t)
    }
    return Transcript(
      profile = info.profile,
      lmBackend = info.lmBackend,
      raw = raw,
      answer = info.profile.parse(raw),
      audioSeconds = audioSeconds,
      sentSeconds = sentSeconds,
      sendMs = (sendEnd - sendStart) / 1e6,
      createConversationMs = (created - start) / 1e6,
    )
  }

  /**
   * Closes the engine and returns true, or returns false when none is loaded. Safe to call twice: the runtime's own
   * Engine.close() throws on a second call, so the reference is dropped before closing.
   */
  fun release(): Boolean {
    val current = engine ?: return false
    engine = null
    loaded = null
    current.close()
    return true
  }

  override fun close() {
    release()
  }

  companion object {
    /** CPU threads for the language model and the audio encoder (the model cards' Galaxy S26 rows). */
    const val CPU_THREADS = 4

    /** The longest audio sent in one message; longer audio is cut here. */
    const val MAX_AUDIO_SECONDS = 30.0

    fun conversationConfig(profile: ModelProfile) =
      ConversationConfig(
        samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0),
        maxOutputToken = profile.maxOutputTokens,
      )

    /** The sentence the picker and a failed load show for a bundle that is not on the phone. */
    fun missingBundleMessage(profile: ModelProfile, directory: File?): String =
      "${profile.fileName} is not on this phone. Download it from huggingface.co/${profile.hubRepo} and push it " +
        "to ${directory?.path ?: "the app's files directory"}/ with adb (see README)."
  }
}
