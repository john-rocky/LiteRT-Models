package com.d1omni

import android.content.ContentUris
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import android.provider.OpenableColumns
import androidx.compose.runtime.Immutable
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import java.io.ByteArrayInputStream
import java.io.File
import kotlin.math.max
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** The voice note Decide will send: its samples, where it came from, the saved file of a recording. */
@Immutable
class VoiceClip(
  val samples: ShortArray,
  val source: D1Source,
  val sha256: String,
  val bytes: Int,
  /** The picked file's name, the recording's file name, or the sample's. */
  val name: String,
  /** A recording's wav in `files/` (absolute path). */
  val path: String?,
  /** A recording's facts for the run JSON (start / stop wall ms, how it ended, the audio source, the limit). */
  val recording: Map<String, Any?>?,
)

/** The photo Decide will send: the file's bytes as picked, where it came from, a thumbnail for the screen. */
@Immutable
class PhotoPick(
  val bytes: ByteArray,
  val source: D1Source,
  val sha256: String,
  val name: String,
  val format: String,
  val width: Int,
  val height: Int,
  val thumbnail: ImageBitmap?,
)

/** One picture of the Recent photos sheet: its MediaStore Uri, name, and a thumbnail. */
@Immutable class RecentPhoto(val uri: Uri, val name: String, val thumbnail: ImageBitmap?)

/** One answer on the screen: the question as asked, and once its call returned, its answer. */
@Immutable data class AnswerUi(val qid: String, val instructions: String, val shown: D1Shown? = null)

/** The answers of one Decide: one per question, then the ms line once the work is done, and its run JSON. */
@Immutable
data class ResultUi(
  val answers: List<AnswerUi>,
  val ms: Long? = null,
  val msLine: String? = null,
  val json: String? = null,
) {
  val complete: Boolean
    get() = ms != null && answers.all { it.shown != null }
}

/** One input screen's questions (the editor's drafts), whether the editor is open, its answers and its message. */
@Immutable
data class InputUi(
  val drafts: List<QuestionDraft>,
  val editing: Boolean = false,
  val result: ResultUi? = null,
  val error: String? = null,
)

/** The summary: "3 inputs · 4 answers · N ms · airplane mode on", one line per input, the device line. */
@Immutable
data class SummaryUi(
  val headline: String,
  val inputs: Int,
  val answers: Int,
  val totalMs: Long,
  val airplane: Boolean,
  /** Each input's answers as shown, or null when it was not decided. */
  val lines: List<Pair<D1Input, List<D1Shown>?>>,
  val deviceLine: String,
)

/** The screen's state; [tab] 0..2 are the inputs ([D1Input] order), 3 the summary. */
@Immutable
data class UiState(
  val status: String = "Loading the tokenizer…",
  val error: Boolean = false,
  val ready: Boolean = false,
  val engineLine: String = "",
  val tab: Int = 0,
  val recording: Boolean = false,
  val recordedSamples: Int = 0,
  val levelDb: Float = -90f,
  val maxRecordSeconds: Double = 10.0,
  val deciding: D1Input? = null,
  val voice: VoiceClip? = null,
  val photo: PhotoPick? = null,
  /** The Recent photos sheet: null while closed, the newest pictures (up to four) while open. */
  val recentPhotos: List<RecentPhoto>? = null,
  val message: String = "",
  val messageSource: D1Source = D1Source.TYPED,
  val inputs: Map<D1Input, InputUi> = emptyMap(),
  val summary: SummaryUi? = null,
  /** A gate or timing launch (debug build) shows its status only. */
  val diagnostics: Boolean = false,
) {
  /** The pill: LOADING, RECORDING, DECIDING, then the shown screen's state (DONE when its answers are complete). */
  val pill: String
    get() =
      when {
        !ready -> "loading"
        recording -> "recording"
        deciding != null -> "deciding"
        tab in D1Input.entries.indices ->
          if (inputs[D1Input.entries[tab]]?.result?.complete == true) "done" else "ready"
        else -> if (summary != null && summary.inputs == D1Input.entries.size) "done" else "ready"
      }

  fun input(which: D1Input): InputUi = inputs.getValue(which)
}

/**
 * Owns the engine and runs every graph call on [D1Runtime.dispatcher]. A normal launch reads the tokenizer and the
 * contract, compiles every graph the three inputs need (L256, L128, the audio graph T1001, the vision tower, the
 * projector), makes one untimed pass over the bundled sample (`ENGINE_READY`) and shows the voice screen. Each screen
 * takes its own input (a recording or a WAV file, a photo, a typed message) and its own questions (the editor), and
 * Decide answers them on the phone (one run JSON per Decide); Load sample fills the three screens; the summary adds them
 * up. An autoplay intent runs Load sample and the three Decides in turn (the reproduction run); a gate or timing launch
 * (debug build) runs that protocol into its report instead.
 */
class MainViewModel(private val context: Context) : ViewModel() {
  private val sample: D1Sample = D1Sample.parse(raw(R.raw.sample))
  private val _uiState =
    MutableStateFlow(UiState(inputs = D1Input.entries.associateWith { InputUi(defaultDrafts(it)) }))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  private var engine: D1AppEngine? = null
  private var loadMs = 0L
  private var warmupMs = 0L
  private var started = false
  private var nextKey = 1000L
  private val player = D1AudioPlayer()
  private val recorder = D1Recorder()

  /** Held while the engine loads and while a Decide runs: one graph job at a time. */
  private val engineLock = Mutex()

  @Volatile private var pillBox: IntArray? = null
  @Volatile private var pillPadLeft = 0
  @Volatile private var screenSize = intArrayOf(0, 0)
  @Volatile private var densityAndScale = floatArrayOf(0f, 0f)
  private val answerLayouts = HashMap<D1Input, List<D1AnswerLayout>>()

  /** Starts what [launch] asks for, once per ViewModel; a later launch intent goes to [newIntent]. */
  fun start(launch: D1Launch) {
    if (started) return
    started = true
    when (launch) {
      is D1Launch.Autoplay -> {
        load(launch.backend, launch.precisions)
        autoplay(launch)
      }
      is D1Launch.Normal -> load(launch.backend, launch.precisions)
      else -> run(launch)
    }
  }

  /** A launch intent that reached the running activity (`singleTop`). */
  fun newIntent(launch: D1Launch) {
    when (launch) {
      is D1Launch.Autoplay -> {
        if (engine == null && !engineLock.isLocked) {
          D1Demo.autoplayStart()
          D1Demo.failed("the engine is not loaded (a gate or timing run, or a failed load): force-stop and launch again")
          return
        }
        autoplay(launch)
      }
      is D1Launch.Normal -> Unit
      is D1Launch.Invalid -> {
        D1Demo.failed(launch.reason)
        show(launch.reason, error = true)
      }
      else -> D1Demo.failed("the app is running; force-stop it and launch again")
    }
  }

  private fun run(launch: D1Launch) {
    _uiState.update { it.copy(diagnostics = true) }
    when (launch) {
      is D1Launch.Invalid -> {
        D1Demo.failed(launch.reason)
        show(launch.reason, error = true)
      }
      is D1Launch.Gate -> diagnostics("Fixture gate") { progress -> D1GateRunner(context).run(launch, progress) }
      is D1Launch.Timing -> diagnostics("Timing") { progress -> D1TimingRunner(context).run(launch, progress) }
      is D1Launch.VGate -> diagnostics("Picture gate") { progress -> D1VisionGate(context).gate(launch, progress) }
      is D1Launch.VTiming -> diagnostics("Picture timing") { progress -> D1VisionGate(context).timing(launch, progress) }
      is D1Launch.Normal,
      is D1Launch.Autoplay -> Unit
    }
  }

  // ---- Engine ----

  private fun load(backend: D1Backend, precisions: D1Precisions) {
    viewModelScope.launch {
      engineLock.withLock {
        try {
          val loaded =
            withContext(D1Runtime.dispatcher) {
              engine?.close()
              engine = null
              val e = D1AppEngine.load(context, backend, precisions) { line -> show(line) }
              engine = e
              show("Warming up (one untimed pass over the sample)…")
              warmupMs = warmup(e)
              e
            }
          loadMs = Math.round(loaded.loadMs)
          D1Demo.engineReady(loadMs, warmupMs)
          val maxSeconds = maxRecordSamples(loaded) / D1Audio.SAMPLE_RATE.toDouble()
          _uiState.update {
            it.copy(
              status = "Ready",
              error = false,
              ready = true,
              maxRecordSeconds = maxSeconds,
              engineLine =
                "${D1Text.accelerator(loaded.backends())} · " +
                  "L${loaded.decide.resident.sorted().joinToString(" + L")} · audio T${loaded.decide.audio.resident.joinToString()} · " +
                  "vision tower · loaded in %.1f s".format(java.util.Locale.ROOT, loadMs / 1000.0),
            )
          }
        } catch (failure: Exception) {
          loadFailed(D1Decider.describe(failure))
        } catch (failure: LinkageError) {
          loadFailed("Native runtime: ${D1Decider.describe(failure)}")
        } catch (failure: OutOfMemoryError) {
          loadFailed(D1Decider.describe(failure))
        }
      }
    }
  }

  /** The untimed pass after the load: the sample's three inputs through Decide once, so the first timed one is warm. */
  private fun warmup(e: D1AppEngine): Long {
    val start = System.nanoTime()
    val voice = sample.input(D1Input.VOICE)
    D1Decide.audio(e, D1Wav.parse(raw(rawId(voice.mediaFile)), voice.mediaFile ?: "wav"), voice.questions)
    val photo = sample.input(D1Input.PHOTO)
    D1Decide.image(e, raw(rawId(photo.mediaFile)), photo.questions)
    val message = sample.input(D1Input.MESSAGE)
    D1Decide.text(e, message.state as String, message.questions)
    return Math.round((System.nanoTime() - start) / 1e6)
  }

  private fun loadFailed(reason: String) {
    D1Demo.failed(reason)
    show(reason, error = true)
  }

  /** The longest recording the installed audio graphs hold (10 s with T1001), at most 30 s. */
  private fun maxRecordSamples(e: D1AppEngine?): Int {
    val largest = e?.decide?.audio?.installed?.maxOrNull() ?: D1AppEngine.AUDIO_BUCKET
    return minOf((largest - 1) * D1Audio.HOP, D1Audio.MAX_SECONDS * D1Audio.SAMPLE_RATE)
  }

  // ---- Screens ----

  fun selectTab(tab: Int) {
    _uiState.update { it.copy(tab = tab, summary = summary(it)) }
  }

  /** Load sample: the sample's voice note, photo and message in the three screens, each with its default questions. */
  fun loadSample() {
    if (busy()) return
    val voice = sample.input(D1Input.VOICE)
    val photo = sample.input(D1Input.PHOTO)
    val message = sample.input(D1Input.MESSAGE)
    val voiceBytes = raw(rawId(voice.mediaFile))
    val photoBytes = raw(rawId(photo.mediaFile))
    check(D1Answers.sha256(voiceBytes) == voice.mediaSha256) { "res/raw/${voice.mediaFile} is not the sample's file" }
    check(D1Answers.sha256(photoBytes) == photo.mediaSha256) { "res/raw/${photo.mediaFile} is not the sample's file" }
    val samples = D1Wav.parse(voiceBytes, voice.mediaFile ?: "wav")
    val bitmap = BitmapFactory.decodeByteArray(photoBytes, 0, photoBytes.size)
    _uiState.update { state ->
      state.copy(
        voice = VoiceClip(samples, D1Source.SAMPLE, D1Answers.sha256(voiceBytes), voiceBytes.size, voice.mediaFile ?: "", null, null),
        photo =
          PhotoPick(photoBytes, D1Source.SAMPLE, D1Answers.sha256(photoBytes), photo.mediaFile ?: "", D1Photo.format(photoBytes),
            bitmap?.width ?: 0, bitmap?.height ?: 0, bitmap?.asImageBitmap()),
        message = message.state as String,
        messageSource = D1Source.SAMPLE,
        inputs = D1Input.entries.associateWith { InputUi(defaultDrafts(it)) },
        summary = null,
      )
    }
    D1Demo.sampleLoaded(sample.id)
  }

  // ---- Voice ----

  /** Starts the microphone (the activity has the RECORD_AUDIO permission by now). */
  fun startRecording() {
    val state = _uiState.value
    if (!state.ready || busy()) return
    player.stop()
    val maxSamples = maxRecordSamples(engine)
    val startedWall = System.currentTimeMillis()
    _uiState.update {
      it.copy(recording = true, recordedSamples = 0, levelDb = -90f, inputs = clearResult(it, D1Input.VOICE, null))
    }
    try {
      recorder.start(
        maxSamples,
        onLevel = { db, count -> _uiState.update { it.copy(levelDb = db, recordedSamples = count) } },
        onDone = { recording -> viewModelScope.launch { recorded(recording, maxSamples, startedWall) } },
      )
      D1Demo.recordStart(maxSamples / D1Audio.SAMPLE_RATE.toDouble())
    } catch (failure: Exception) {
      _uiState.update { it.copy(recording = false, inputs = clearResult(it, D1Input.VOICE, D1Decider.describe(failure))) }
      D1Demo.failed("record: ${D1Decider.describe(failure)}")
    }
  }

  fun stopRecording() = recorder.stop()

  fun recordPermissionDenied() {
    _uiState.update {
      it.copy(inputs = clearResult(it, D1Input.VOICE, "The microphone permission was not given; pick a WAV file instead."))
    }
  }

  private suspend fun recorded(recording: D1Recording, maxSamples: Int, requestedWall: Long) {
    val samples = recording.samples
    if (recording.error != null || samples.size < MIN_RECORDING_SAMPLES) {
      val reason = recording.error ?: "The recording is too short (${D1Text.seconds(samples.size)}); record at least 0.5 s."
      _uiState.update { it.copy(recording = false, inputs = clearResult(it, D1Input.VOICE, reason)) }
      D1Demo.failed("record: $reason")
      return
    }
    val bytes = D1Wav.encode(samples)
    val file = File(context.filesDir, "recorded-${recording.startedWallMs}.wav")
    withContext(Dispatchers.IO) { file.writeBytes(bytes) }
    val sha = D1Answers.sha256(bytes)
    val facts =
      linkedMapOf<String, Any?>(
        "requested_wall_ms" to requestedWall,
        "started_wall_ms" to recording.startedWallMs,
        "stopped_wall_ms" to recording.stoppedWallMs,
        "end" to recording.end,
        "audio_source" to recording.audioSource,
        "sample_rate" to D1Audio.SAMPLE_RATE,
        "samples" to samples.size,
        "seconds" to samples.size / D1Audio.SAMPLE_RATE.toDouble(),
        "max_samples" to maxSamples,
        "wav" to file.absolutePath,
        "wav_sha256" to sha,
        "wav_bytes" to bytes.size,
      )
    _uiState.update {
      it.copy(
        recording = false,
        voice = VoiceClip(samples, D1Source.RECORDED, sha, bytes.size, file.name, file.absolutePath, facts),
        inputs = clearResult(it, D1Input.VOICE, null),
      )
    }
    D1Demo.recordStop(samples.size, file.absolutePath, sha, recording.end)
  }

  /** A WAV file from the system file picker: 16 kHz mono 16-bit PCM only (the reason is shown otherwise). */
  fun pickedAudio(uri: Uri) {
    if (busy()) return
    viewModelScope.launch {
      try {
        val (bytes, name) = withContext(Dispatchers.IO) { readUri(uri, MAX_AUDIO_BYTES) }
        val samples = D1Wav.parse(bytes, name)
        val sha = D1Answers.sha256(bytes)
        _uiState.update {
          it.copy(voice = VoiceClip(samples, D1Source.PICKED, sha, bytes.size, name, null, null), inputs = clearResult(it, D1Input.VOICE, null))
        }
        D1Demo.picked(D1Kind.AUDIO, sha, bytes.size, name)
      } catch (failure: Exception) {
        val reason = failure.message ?: D1Decider.describe(failure)
        _uiState.update { it.copy(inputs = clearResult(it, D1Input.VOICE, reason)) }
        D1Demo.failed("pick audio: $reason")
      }
    }
  }

  /** Plays the current voice note through the speaker, at the phone's own media volume. */
  fun play() {
    val clip = _uiState.value.voice ?: return
    if (_uiState.value.recording) return
    player.play(clip.samples)
  }

  // ---- Photo ----

  /** A picture from the photo picker. */
  fun pickedPhoto(uri: Uri) {
    if (busy()) return
    viewModelScope.launch {
      try {
        val (bytes, name) = withContext(Dispatchers.IO) { readUri(uri, MAX_PHOTO_BYTES) }
        val (thumbnail, size) = withContext(Dispatchers.Default) { thumbnail(bytes) }
        val sha = D1Answers.sha256(bytes)
        _uiState.update {
          it.copy(
            photo = PhotoPick(bytes, D1Source.PICKED, sha, name, D1Photo.format(bytes), size.first, size.second, thumbnail),
            inputs = clearResult(it, D1Input.PHOTO, null),
          )
        }
        D1Demo.picked(D1Kind.IMAGE, sha, bytes.size, name)
      } catch (failure: Exception) {
        val reason = failure.message ?: D1Decider.describe(failure)
        _uiState.update { it.copy(inputs = clearResult(it, D1Input.PHOTO, reason)) }
        D1Demo.failed("pick photo: $reason")
      }
    }
  }

  /**
   * The Recent photos sheet: the newest pictures on the phone (MediaStore, newest added first, up to [RECENT_PHOTOS]),
   * each with a thumbnail. The activity holds the read permission by now (READ_MEDIA_IMAGES, or READ_EXTERNAL_STORAGE
   * before Android 13).
   */
  fun openRecentPhotos() {
    if (busy()) return
    viewModelScope.launch {
      try {
        val photos = withContext(Dispatchers.IO) { recentPhotos() }
        _uiState.update { it.copy(recentPhotos = photos, inputs = clearError(it, D1Input.PHOTO)) }
        D1Demo.recentPhotos(photos.map { it.name })
      } catch (failure: Exception) {
        val reason = failure.message ?: D1Decider.describe(failure)
        _uiState.update { it.copy(recentPhotos = null, inputs = clearResult(it, D1Input.PHOTO, reason)) }
        D1Demo.failed("recent photos: $reason")
      }
    }
  }

  fun closeRecentPhotos() = _uiState.update { it.copy(recentPhotos = null) }

  fun photosPermissionDenied() {
    _uiState.update {
      it.copy(recentPhotos = null, inputs = clearResult(it, D1Input.PHOTO, "The photo permission was not given; use Browse… instead."))
    }
  }

  /** A picture of the sheet: read like a picked file. */
  fun pickedRecent(photo: RecentPhoto) {
    _uiState.update { it.copy(recentPhotos = null) }
    pickedPhoto(photo.uri)
  }

  private fun recentPhotos(): List<RecentPhoto> {
    val resolver = context.contentResolver
    val collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
    val projection = arrayOf(MediaStore.Images.Media._ID, MediaStore.Images.Media.DISPLAY_NAME)
    val order = "${MediaStore.Images.Media.DATE_ADDED} DESC, ${MediaStore.Images.Media._ID} DESC"
    val out = ArrayList<RecentPhoto>()
    resolver.query(collection, projection, null, null, order)?.use { cursor ->
      val idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
      val nameColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
      while (cursor.moveToNext() && out.size < RECENT_PHOTOS) {
        val uri = ContentUris.withAppendedId(collection, cursor.getLong(idColumn))
        val name = cursor.getString(nameColumn) ?: "photo"
        val thumbnail =
          runCatching {
              if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                resolver.loadThumbnail(uri, android.util.Size(THUMB_REQUEST_PX, THUMB_REQUEST_PX), null)
              } else {
                resolver.openInputStream(uri)?.use { stream ->
                  BitmapFactory.decodeStream(stream, null, BitmapFactory.Options().apply { inSampleSize = 4 })
                }
              }
            }
            .getOrNull()
        out.add(RecentPhoto(uri, name, thumbnail?.asImageBitmap()))
      }
    }
    return out
  }

  private fun clearError(state: UiState, input: D1Input): Map<D1Input, InputUi> =
    state.inputs + (input to state.input(input).copy(error = null))

  // ---- Message ----

  fun setMessage(text: String) {
    if (busy()) return
    _uiState.update { it.copy(message = text, messageSource = D1Source.TYPED, inputs = clearResult(it, D1Input.MESSAGE, null)) }
  }

  // ---- Questions ----

  fun toggleEditor(input: D1Input) = editInput(input, clear = false) { it.copy(editing = !it.editing) }

  fun setQuestionId(input: D1Input, key: Long, id: String) = editDraft(input, key) { it.copy(id = id) }

  fun setQuestionType(input: D1Input, key: Long, type: QuestionType) = editDraft(input, key) { it.copy(type = type) }

  fun setInstructions(input: D1Input, key: Long, text: String) = editDraft(input, key) { it.copy(instructions = text) }

  fun setOptions(input: D1Input, key: Long, text: String) = editDraft(input, key) { it.copy(options = text) }

  fun addQuestion(input: D1Input) =
    editInput(input) { it.copy(drafts = it.drafts + QuestionDraft(nextKey++, "q${it.drafts.size + 1}", QuestionType.NOUL, "", "")) }

  fun removeQuestion(input: D1Input, key: Long) = editInput(input) { it.copy(drafts = it.drafts.filterNot { d -> d.key == key }) }

  fun resetQuestions(input: D1Input) = editInput(input) { it.copy(drafts = defaultDrafts(input)) }

  private fun editDraft(input: D1Input, key: Long, change: (QuestionDraft) -> QuestionDraft) =
    editInput(input) { it.copy(drafts = it.drafts.map { d -> if (d.key == key) change(d) else d }) }

  private fun editInput(input: D1Input, clear: Boolean = true, change: (InputUi) -> InputUi) {
    if (busy()) return
    _uiState.update { state ->
      val current = state.input(input)
      val changed = change(current).let { if (clear) it.copy(result = null, error = null) else it }
      state.copy(inputs = state.inputs + (input to changed))
    }
  }

  private fun defaultDrafts(input: D1Input): List<QuestionDraft> {
    val drafts = D1Drafts.fromQuestions(sample.input(input).questions, nextKey)
    nextKey += drafts.size
    return drafts
  }

  // ---- Decide ----

  /** Decide on [input]'s screen (the button). */
  fun decide(input: D1Input) {
    if (busy()) return
    viewModelScope.launch { runDecide(input) }
  }

  /**
   * One Decide: the input and its questions checked, the work on the engine's thread (each answer shown when its call
   * returns), the ms line, then the run JSON once the answers are on screen. Returns the run JSON, or null after
   * showing why it could not run.
   */
  private suspend fun runDecide(input: D1Input): File? {
    val state = _uiState.value
    if (!state.ready) return null
    val questions =
      try {
        D1Drafts.toQuestions(state.input(input).drafts)
      } catch (failure: D1DraftException) {
        return refuse(input, D1Drafts.message(failure))
      }
    val voice = state.voice
    val photo = state.photo
    val message = state.message
    when (input) {
      D1Input.VOICE -> {
        if (voice == null) return refuse(input, "Record a voice note, pick a WAV file or load the sample first.")
        val most = maxRecordSamples(engine)
        if (voice.samples.size > most) {
          return refuse(
            input,
            "This clip is ${D1Text.seconds(voice.samples.size)}; the installed audio graph holds ${D1Text.seconds(most)}. " +
              "Install a longer one (scripts/install_to_device.sh with AUDIO=\"1001 2001\" or \"3001\") or use a shorter clip.",
          )
        }
      }
      D1Input.PHOTO -> if (photo == null) return refuse(input, "Pick a photo or load the sample first.")
      D1Input.MESSAGE -> if (message.isBlank()) return refuse(input, "Type a message or load the sample first.")
    }
    val source =
      when (input) {
        D1Input.VOICE -> requireNotNull(voice).source
        D1Input.PHOTO -> requireNotNull(photo).source
        D1Input.MESSAGE -> state.messageSource
      }
    val placeholders = questions.map { (qid, q) -> AnswerUi(qid, q.instructions) }
    _uiState.update {
      it.copy(
        deciding = input,
        inputs = it.inputs + (input to it.input(input).copy(editing = false, error = null, result = ResultUi(placeholders))),
        summary = null,
      )
    }
    synchronized(answerLayouts) { answerLayouts.remove(input) }
    if (input == D1Input.MESSAGE && source == D1Source.TYPED) D1Demo.typed(message.length)
    D1Demo.decideStart(input, source)
    val events = ArrayList<Map<String, Any?>>()
    val t0 = System.nanoTime()
    fun event(name: String, extra: Map<String, Any?> = emptyMap()) {
      events.add(
        linkedMapOf<String, Any?>("event" to name, "wall_ms" to System.currentTimeMillis(), "t_ms" to (System.nanoTime() - t0) / 1e6)
          .apply { putAll(extra) }
      )
    }
    event("decide_start")
    var json: File? = null
    engineLock.withLock {
      val e = engine ?: return refuse(input, "The engine is not ready: ${_uiState.value.status}")
      val stateStart = withContext(Dispatchers.IO) { D1Device.state(context) }
      try {
        val onAnswer: (D1RunQuestion) -> Unit = { q ->
          D1Demo.questionDone(input, q.qid, q.inferMs)
          event("q_done", mapOf("qid" to q.qid, "ms" to q.inferMs))
          _uiState.update { s ->
            val current = s.input(input)
            val result = current.result ?: ResultUi(placeholders)
            val answers = result.answers.map { a -> if (a.qid == q.qid) a.copy(shown = q.shown) else a }
            s.copy(inputs = s.inputs + (input to current.copy(result = result.copy(answers = answers))))
          }
        }
        val work =
          withContext(D1Runtime.dispatcher) {
            when (input) {
              D1Input.VOICE -> D1Decide.audio(e, requireNotNull(voice).samples, questions, onAnswer)
              D1Input.PHOTO -> D1Decide.image(e, requireNotNull(photo).bytes, questions, onAnswer)
              D1Input.MESSAGE -> D1Decide.text(e, message, questions, onAnswer)
            }
          }
        val msLine = D1Text.msLine(work.itemMs, work.buckets)
        event("work_done", mapOf("item_total_ms" to work.itemMs))
        _uiState.update { s ->
          val current = s.input(input)
          val result = (current.result ?: ResultUi(placeholders)).copy(ms = work.itemMs, msLine = msLine)
          s.copy(deciding = null, inputs = s.inputs + (input to current.copy(result = result)))
        }
        // Let the answers reach the screen and their layout report arrive before the file is written.
        delay(LAYOUT_SETTLE_MS)
        val stateEnd = withContext(Dispatchers.IO) { D1Device.state(context) }
        val run =
          D1Run.build(
            D1RunInput(
              input = input,
              source = source,
              media = media(input, voice, photo, message),
              state = if (input == D1Input.MESSAGE) message else null,
              work = work,
              shownMs = msLine,
              deviceModel = Build.MODEL,
              deviceManufacturer = Build.MANUFACTURER,
              deviceShownAs = D1Device.marketName(),
              androidRelease = Build.VERSION.RELEASE,
              accelerator = D1Text.accelerator(e.backends()),
              precision = precisionByKind(e),
              precisionRequested = LinkedHashMap(e.precisions.requested),
              graphs = e.graphs(),
              memoryAtReady = e.memoryAtReady,
              engineLoadMs = loadMs,
              warmupMs = warmupMs,
              airplaneMode = D1Device.airplaneMode(context),
              cgroup = stateStart["cgroup"] as String,
              cgroupEnd = stateEnd["cgroup"] as String,
              layout = layout(input),
              events = events.toList(),
              stateStart = stateStart,
              stateEnd = stateEnd,
            )
          )
        val file = withContext(Dispatchers.IO) { D1Demo.writeRun(context, run) }
        json = file
        _uiState.update { s ->
          val current = s.input(input)
          s.copy(inputs = s.inputs + (input to current.copy(result = current.result?.copy(json = file.path))), summary = summary(s))
        }
        D1Demo.decideDone(input, work.itemMs, file.path)
      } catch (failure: Exception) {
        decideFailed(input, D1Decider.describe(failure))
      } catch (failure: LinkageError) {
        decideFailed(input, "native runtime ${D1Decider.describe(failure)}")
      } catch (failure: OutOfMemoryError) {
        decideFailed(input, D1Decider.describe(failure))
      }
    }
    return json
  }

  private fun refuse(input: D1Input, reason: String): File? {
    _uiState.update { it.copy(deciding = null, inputs = clearResult(it, input, reason)) }
    D1Demo.failed("decide ${input.wireName}: $reason")
    return null
  }

  private fun decideFailed(input: D1Input, reason: String) {
    D1Demo.failed("decide ${input.wireName}: $reason")
    _uiState.update { it.copy(deciding = null, inputs = clearResult(it, input, reason)) }
  }

  /** The input's media facts for the run JSON (null for a message: its text is the state). */
  private fun media(input: D1Input, voice: VoiceClip?, photo: PhotoPick?, message: String): Map<String, Any?>? =
    when (input) {
      D1Input.VOICE ->
        requireNotNull(voice).let {
          linkedMapOf(
            "name" to it.name,
            "sha256" to it.sha256,
            "bytes" to it.bytes,
            "samples" to it.samples.size,
            "seconds" to it.samples.size / D1Audio.SAMPLE_RATE.toDouble(),
            "path" to it.path,
            "recording" to it.recording,
          )
        }
      D1Input.PHOTO ->
        requireNotNull(photo).let {
          linkedMapOf("name" to it.name, "sha256" to it.sha256, "bytes" to it.bytes.size, "format" to it.format,
            "picked_hw" to listOf(it.height, it.width))
        }
      D1Input.MESSAGE ->
        linkedMapOf("chars" to message.length, "sha256" to D1Answers.sha256(message.toByteArray(Charsets.UTF_8)))
    }

  /** The GPU precision each kind of graph runs at (null for a kind on the CPU). */
  private fun precisionByKind(e: D1AppEngine): LinkedHashMap<String, String?> {
    val graphs = e.graphs()
    fun of(prefix: String): String? =
      graphs.filter { it.graph.startsWith(prefix) }.map { it.precision?.wireName }.distinct().singleOrNull()
    return linkedMapOf("decide" to of("decide_"), "audio" to of("audio_"), "vision" to of("vision_tower"))
  }

  private fun clearResult(state: UiState, input: D1Input, error: String?): Map<D1Input, InputUi> =
    state.inputs + (input to state.input(input).copy(result = null, error = error))

  private fun busy(): Boolean = _uiState.value.recording || _uiState.value.deciding != null

  // ---- Summary ----

  private fun summary(state: UiState): SummaryUi? {
    val done = D1Input.entries.filter { state.inputs[it]?.result?.complete == true }
    if (done.isEmpty()) return null
    val totalMs = done.sumOf { requireNotNull(state.input(it).result?.ms) }
    val answers = done.sumOf { state.input(it).result?.answers?.size ?: 0 }
    val airplane = D1Device.airplaneMode(context)
    val lines =
      D1Input.entries.map { input ->
        input to state.input(input).result?.takeIf { it.complete }?.answers?.mapNotNull { it.shown }
      }
    val accelerator = engine?.let { D1Text.accelerator(it.backends()) } ?: "GPU"
    return SummaryUi(D1Text.summary(done.size, answers, totalMs, airplane), done.size, answers, totalMs, airplane, lines,
      D1Text.deviceLine(D1Device.marketName(), accelerator))
  }

  // ---- Layout reports (for the run JSON) ----

  fun onScreen(width: Int, height: Int, density: Float, fontScale: Float) {
    screenSize = intArrayOf(width, height)
    densityAndScale = floatArrayOf(density, fontScale)
  }

  fun onPillLayout(box: IntArray, padLeft: Int) {
    pillBox = box
    pillPadLeft = padLeft
  }

  fun onAnswerLayout(input: D1Input, layouts: List<D1AnswerLayout>) {
    synchronized(answerLayouts) { answerLayouts[input] = layouts }
  }

  private fun layout(input: D1Input): D1RunLayout =
    D1RunLayout(
      screenSize[0],
      screenSize[1],
      pillBox,
      pillPadLeft,
      synchronized(answerLayouts) { answerLayouts[input].orEmpty() },
      densityAndScale[0],
      densityAndScale[1],
    )

  // ---- Autoplay (the reproduction run) ----

  private fun autoplay(launch: D1Launch.Autoplay) {
    D1Demo.autoplayStart()
    viewModelScope.launch {
      // Wait for the engine: the load holds the lock until ENGINE_READY.
      engineLock.withLock {}
      if (engine == null) {
        D1Demo.failed("autoplay: the engine did not load: ${_uiState.value.status}")
        return@launch
      }
      if (busy()) {
        D1Demo.failed("autoplay: busy (recording or deciding)")
        return@launch
      }
      delay(launch.delayMs)
      loadSample()
      val files = ArrayList<String>()
      for ((index, input) in D1Input.entries.withIndex()) {
        selectTab(index)
        delay(launch.gapMs)
        val file = runDecide(input)
        if (file == null) {
          D1Demo.failed("autoplay: Decide on ${input.wireName} did not finish")
          return@launch
        }
        files.add(file.path)
      }
      delay(launch.gapMs)
      selectTab(D1Input.entries.size)
      D1Demo.autoplayDone(files)
    }
  }

  // ---- Helpers ----

  private fun diagnostics(title: String, body: ((String) -> Unit) -> D1GateRunner.Summary) {
    show("$title…")
    viewModelScope.launch {
      val summary = withContext(D1Runtime.dispatcher) { body { line -> show("$title: $line") } }
      show("$title: ${summary.status} · ${summary.path}", error = summary.error != null)
    }
  }

  private fun show(status: String, error: Boolean = false) {
    _uiState.update { it.copy(status = status, error = error) }
  }

  private fun raw(id: Int): ByteArray = context.resources.openRawResource(id).use { it.readBytes() }

  private fun rawId(name: String?): Int =
    when (name) {
      "sample_voice_note.wav" -> R.raw.sample_voice_note
      "img_dogs_01.png" -> R.raw.img_dogs_01
      else -> throw IllegalStateException("no bundled file $name")
    }

  /** The bytes behind [uri] (at most [limit]) and its display name. */
  private fun readUri(uri: Uri, limit: Int): Pair<ByteArray, String> {
    val resolver = context.contentResolver
    val name =
      runCatching {
          resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
          }
        }
        .getOrNull() ?: uri.lastPathSegment ?: "picked"
    val bytes =
      requireNotNull(resolver.openInputStream(uri)) { "cannot open $name" }.use { stream ->
        val out = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(1 shl 16)
        while (true) {
          val read = stream.read(buffer)
          if (read < 0) break
          out.write(buffer, 0, read)
          require(out.size() <= limit) { "$name is larger than ${limit / (1 shl 20)} MB" }
        }
        out.toByteArray()
      }
    return bytes to name
  }

  /** A display copy of a picture (at most about 1,024 px, EXIF orientation applied) and its full size. */
  private fun thumbnail(bytes: ByteArray): Pair<ImageBitmap?, Pair<Int, Int>> {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    require(bounds.outWidth > 0 && bounds.outHeight > 0) { "this file is not a picture Android can decode" }
    var sampleSize = 1
    while (max(bounds.outWidth, bounds.outHeight) / (sampleSize * 2) >= THUMBNAIL_PX) sampleSize *= 2
    val bitmap =
      BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sampleSize })
        ?: return null to (bounds.outWidth to bounds.outHeight)
    val orientation =
      runCatching {
          ExifInterface(ByteArrayInputStream(bytes)).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        .getOrDefault(ExifInterface.ORIENTATION_NORMAL)
    val matrix = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> matrix.apply { setRotate(90f); postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> matrix.apply { setRotate(-90f); postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
    }
    val shown =
      if (matrix.isIdentity) bitmap else Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    val swap = orientation in 5..8
    val size = if (swap) bounds.outHeight to bounds.outWidth else bounds.outWidth to bounds.outHeight
    return shown.asImageBitmap() to size
  }

  override fun onCleared() {
    recorder.close()
    player.close()
    val current = engine
    engine = null
    if (current != null) {
      viewModelScope.launch(D1Runtime.dispatcher) { current.close() }
    }
  }

  companion object {
    /** A recording under this many samples (0.5 s) is refused. */
    const val MIN_RECORDING_SAMPLES = D1Audio.MIN_SAMPLES
    private const val MAX_AUDIO_BYTES = 64 shl 20
    private const val MAX_PHOTO_BYTES = 64 shl 20
    private const val THUMBNAIL_PX = 1024
    /** Pictures in the Recent photos sheet, and the size its thumbnails are asked for. */
    const val RECENT_PHOTOS = 4
    private const val THUMB_REQUEST_PX = 256
    private const val LAYOUT_SETTLE_MS = 400L

    fun getFactory(context: Context): ViewModelProvider.Factory = viewModelFactory {
      initializer { MainViewModel(context.applicationContext) }
    }
  }
}
