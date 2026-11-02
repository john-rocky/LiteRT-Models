package com.d1_omni_2b

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.roundToInt
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** A bundled example (assets/examples.json): its state, picture, clips and questions. */
class Example(val id: String, val title: String, val state: Any?, val picture: String?, val clips: List<String>, val questions: Map<*, *>)

/** One answered request: the answers by card, and the state text their spans index. */
class Answered(
  val answers: Map<Long, Map<String, Any?>>,
  val stateText: String,
  val tokens: Int,
  val bucket: Int,
  val accelerator: String,
  val totalMs: Double,
  val loadMs: Double,
)

/** Where a recording or a picked WAV file goes: the state's clips, or one question's spoken instructions. */
sealed class ClipTarget {
  object State : ClipTarget()

  data class Spoken(val questionKey: Long) : ClipTarget()
}

data class UiState(
  val status: String = "Starting…",
  val ready: Boolean = false,
  val busy: Boolean = false,
  val error: String? = null,
  val examples: List<Pair<String, String>> = emptyList(),
  val exampleId: String? = null,
  val stateText: String = "",
  val stateIsJson: Boolean = false,
  val photo: Bitmap? = null,
  val clips: List<ClipDraft> = emptyList(),
  val questions: List<QuestionDraft> = emptyList(),
  val answered: Answered? = null,
  /** The target of the recording in progress, and its length so far. */
  val recording: ClipTarget? = null,
  val recordedSeconds: Double = 0.0,
  /** The key of the clip playing. */
  val playing: Long? = null,
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
  private val worker = Executors.newSingleThreadExecutor { Thread(it, "d1-litert").apply { isDaemon = true } }.asCoroutineDispatcher()
  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state

  /** Where the model files are copied (see README.md). */
  val modelDir: File = File(application.getExternalFilesDir(null), "model")
  private val runsDir = File(application.getExternalFilesDir(null), "runs")
  /** Every clip the editor holds, as the WAV file it was read from (recordings are written here first). */
  private val clipsDir = File(application.filesDir, "clips")
  private var accelerators: Accelerators? = null
  private var host: Host? = null
  private var decide: PhoneDecideBackend? = null
  private var tower: LiteRtTower? = null
  private var audio: LiteRtAudio? = null
  private var picture: Picture? = null
  private var pictureSha256: String? = null
  private var nextKey = 1L
  private val examples: List<Example> = loadExamples(application)
  private val loaded = CompletableDeferred<Boolean>()
  private val recorder = Recorder()
  private val player = Player()
  /** The clip checks before the model is loaded: the extractor's frames for a clip's length. */
  private val extractor = FeatureExtractor()

  init {
    modelDir.mkdirs()
    runsDir.mkdirs()
    clipsDir.mkdirs()
    _state.update { it.copy(examples = examples.map { e -> e.id to e.title }) }
    selectExample(examples.first().id)
    viewModelScope.launch { load() }
  }

  /** The longest sequence this phone runs: 6,144 tokens need a 16 GB phone (a 12 GB phone runs out of memory). */
  private fun maxTokens(): Int {
    val info = ActivityManager.MemoryInfo()
    (getApplication<Application>().getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
    return if (info.totalMem >= 14L * 1024 * 1024 * 1024) 6144 else 4096
  }

  private suspend fun load() {
    try {
      _state.update { it.copy(status = "Loading the tokenizer and the table…") }
      var note: String? = null
      val d =
        withContext(worker) {
          ModelFiles.require(modelDir, ModelFiles.HOST, "the host")
          val acc = Accelerators.create(getApplication())
          accelerators = acc
          val npu = DecideBackends.npu(modelDir, maxTokens(), acc)
          if (npu == null) {
            note =
              if (acc.npu) "NPU not used: the NPU files (${Graphs.NPU_DIR}/) are not in the model folder"
              else "NPU not used: ${acc.note}"
          }
          val d: PhoneDecideBackend = npu ?: DecideBackends.gpu(modelDir, maxTokens(), acc).also {
            require(it.available) {
              "Missing model files in ${modelDir.absolutePath}: no decision graph (${Graphs.PREFIX}_decide_*.tflite). " +
                "Copy them with scripts/push_model.sh (README.md); nothing was loaded."
            }
          }
          // Nothing is loaded yet: the decision graph is the largest stage, so it must fit with the reserve to spare.
          Memory.ensure("decision graph (${d.form})", d.needMb + Memory.RESERVE_MB, "Close other apps and open this app again.")
          tower = LiteRtTower(modelDir, acc)
          audio = LiteRtAudio(modelDir, acc)
          d
        }
      _state.update { it.copy(status = if (d is NpuThenGpuDecide) "Compiling the decision graph for the NPU…" else "Compiling the decision graph on the GPU…") }
      var active = d
      val bucket =
        withContext(worker) {
          try {
            val b = active.bucketFor(256)
            active.prepare(b)
            b
          } catch (failure: Throwable) {
            if (active !is NpuThenGpuDecide) throw failure
            // The NPU files did not load on this phone: the GPU files take over, and the screen says why.
            Log.e(Accelerators.TAG, "NPU load failed", failure)
            active.close()
            val gpu = DecideBackends.gpu(modelDir, maxTokens(), accelerators!!)
            require(gpu.available) {
              "The NPU files did not load (${failure.message}) and no GPU decision graph (${Graphs.PREFIX}_decide_L*_fp16fc.tflite) is in the model folder."
            }
            note = "NPU not used: its files did not load (${failure.message}); the decision graph runs on the GPU at FP32"
            Memory.ensure("decision graph (${gpu.form})", gpu.needMb + Memory.RESERVE_MB, "Close other apps and open this app again.")
            active = gpu
            val b = gpu.bucketFor(256)
            gpu.prepare(b)
            b
          }
        }
      val h = withContext(worker) { Host.load(modelDir, active, tower, audio) }
      host = h
      decide = active
      val graphs =
        when {
          active is NpuThenGpuDecide -> "decision graph on the NPU (${active.form})"
          active.form.startsWith("8") -> "GPU FP32 · all lengths"
          else -> "GPU FP32 · L$bucket"
        }
      val missingAudio = audio!!.missing.takeIf { it.isNotEmpty() }?.let { " · missing ${it.joinToString()}" } ?: ""
      _state.update {
        it.copy(
          ready = true,
          status = "Ready · ${Build.MODEL} · $graphs (${(active.lastLoadMs / 1000).format(1)} s to compile)$missingAudio" +
            (note?.let { n -> "\n$n" } ?: ""),
        )
      }
      loaded.complete(true)
    } catch (failure: Throwable) {
      Log.e(TAG, "load failed", failure)
      _state.update { it.copy(status = "Model unavailable", error = failure.message ?: failure.toString()) }
      loaded.complete(false)
    }
  }

  // ---------------------------------------------------------------------------------- editor

  fun selectExample(id: String) {
    val example = examples.first { it.id == id }
    try {
      val clips = example.clips.map { assetClip(it) }
      val drafts = example.questions.map { (name, raw) ->
        QuestionDraft.fromJson(nextKey++, name as String, raw as Map<*, *>) { media -> assetClip(media["path"] as String) }
      }
      val isJson = example.state != null && example.state !is String
      val text =
        when {
          example.state == null -> ""
          isJson -> D1Json.writeIndented(example.state, 2)
          else -> example.state as String
        }
      val photo = example.picture?.let { name -> getApplication<Application>().assets.open(name).use { BitmapFactory.decodeStream(it) } }
      setPhotoBitmap(photo)
      _state.update {
        it.copy(exampleId = id, stateText = text, stateIsJson = isJson, clips = clips, questions = drafts, answered = null, error = null)
      }
    } catch (failure: Exception) {
      _state.update { it.copy(error = "The example '$id' could not be read: ${failure.message}") }
    }
  }

  fun setStateText(text: String) = _state.update { it.copy(stateText = text, answered = null, exampleId = null) }

  fun setStateIsJson(json: Boolean) = _state.update { it.copy(stateIsJson = json, answered = null) }

  fun updateQuestion(key: Long, change: (QuestionDraft) -> QuestionDraft) =
    _state.update { s -> s.copy(questions = s.questions.map { if (it.key == key) change(it) else it }, answered = null) }

  fun addQuestion() =
    _state.update { s ->
      val names = s.questions.map { it.name }.toSet()
      val name = generateSequence(1) { it + 1 }.map { "q$it" }.first { it !in names }
      s.copy(questions = s.questions + QuestionDraft(nextKey++, name, QType.NOUL, ""), answered = null)
    }

  fun removeQuestion(key: Long) = _state.update { s -> s.copy(questions = s.questions.filter { it.key != key }, answered = null) }

  fun showError(message: String) = _state.update { it.copy(error = message) }

  // ---------------------------------------------------------------------------------- clips

  /**
   * A clip from a WAV file's bytes, read as the model's own code reads it, checked against the longest clip the
   * audio encoder takes, and kept as a file of the app's: refused with the reason (and nothing kept) when it is not
   * 16-bit PCM, is empty, is longer than 30.015 s, or has a rate the app does not resample.
   */
  private fun readClip(bytes: ByteArray, label: String): ClipDraft {
    val clip = Wav.clip(bytes, label)
    Requests.checkClip(clip, extractor, LiteRtAudio.BUCKETS.last())
    val key = nextKey++
    val file = File(clipsDir, "$key-${label.replace(Regex("[^A-Za-z0-9._-]"), "_")}".let { if (it.endsWith(".wav")) it else "$it.wav" })
    file.writeBytes(bytes)
    return ClipDraft(key, file, label, clip, Requests.sha256(bytes))
  }

  private fun assetClip(name: String): ClipDraft =
    readClip(getApplication<Application>().assets.open(name).use { it.readBytes() }, name)

  private fun place(draft: ClipDraft, target: ClipTarget) =
    when (target) {
      is ClipTarget.State -> _state.update { it.copy(clips = it.clips + draft, answered = null, exampleId = null, error = null) }
      is ClipTarget.Spoken -> updateQuestion(target.questionKey) { it.copy(spoken = draft) }.also { _state.update { s -> s.copy(error = null) } }
    }

  /** A WAV file the user picked, for the state or as a spoken question. */
  fun addClip(uri: Uri?, target: ClipTarget) {
    if (uri == null) return
    viewModelScope.launch {
      val resolver = getApplication<Application>().contentResolver
      val name =
        runCatching {
            resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
          }
          .getOrNull() ?: "clip.wav"
      try {
        val draft =
          withContext(worker) {
            val bytes = resolver.openInputStream(uri)?.use { it.readBytes() } ?: error("the file could not be opened")
            readClip(bytes, name)
          }
        place(draft, target)
      } catch (failure: Exception) {
        _state.update { it.copy(error = "Could not use $name: ${failure.message}") }
      }
    }
  }

  /**
   * A clip from a file (run files): the same reading, checks and message as a picked file. Returns the refusal, or
   * null when the clip was added.
   */
  private fun addClipFile(file: File, target: ClipTarget): String? =
    try {
      place(readClip(file.readBytes(), file.name), target)
      null
    } catch (failure: Exception) {
      "Could not use ${file.name}: ${failure.message}".also { message -> _state.update { it.copy(error = message) } }
    }

  /**
   * A run file's question cards, in the model's question JSON (as in examples.json; a spoken question's clip is a WAV
   * file of the runs folder): they replace the cards. Returns the refusal, or null when the cards were set.
   */
  private fun setRunQuestions(questions: Map<*, *>): String? =
    try {
      val drafts =
        questions.map { (name, raw) ->
          QuestionDraft.fromJson(nextKey++, name as String, raw as Map<*, *>) { media ->
            val clip = media["path"] as String
            readClip(File(runsDir, clip).readBytes(), clip)
          }
        }
      _state.update { it.copy(questions = drafts, answered = null, exampleId = null, error = null) }
      null
    } catch (failure: Exception) {
      "Could not use the questions: ${failure.message}".also { message -> _state.update { it.copy(error = message) } }
    }

  /** A run file's photo (a picture of the runs folder), read as a picked photo is; null removes the photo. */
  private suspend fun setRunPhoto(file: File?): String? {
    if (file == null) {
      removePhoto()
      return null
    }
    return try {
      val bitmap = withContext(worker) { decodePhoto(Uri.fromFile(file)) }
      setPhotoBitmap(bitmap)
      _state.update { it.copy(answered = null, exampleId = null, error = null) }
      null
    } catch (failure: Throwable) {
      "Could not read the photo ${file.name}: ${failure.message}".also { message -> _state.update { it.copy(error = message) } }
    }
  }

  fun removeClip(key: Long) {
    if (_state.value.playing == key) stopPlaying()
    _state.update { s -> s.copy(clips = s.clips.filter { it.key != key }, answered = null, exampleId = null) }
  }

  fun removeSpoken(questionKey: Long) = updateQuestion(questionKey) { it.copy(spoken = null) }

  /** Starts recording for [target] (the caller holds the microphone permission); it stops at 30 s or at [stopRecording]. */
  fun startRecording(target: ClipTarget) {
    if (_state.value.recording != null) return
    stopPlaying()
    try {
      recorder.start(
        Requests.RECORD_SECONDS * Recorder.RATE,
        onProgress = { n -> _state.update { it.copy(recordedSeconds = n.toDouble() / Recorder.RATE) } },
      ) { rec ->
        viewModelScope.launch {
          _state.update { it.copy(recording = null) }
          if (rec.error != null) {
            _state.update { it.copy(error = "The recording failed: ${rec.error}") }
            return@launch
          }
          try {
            val draft = withContext(worker) { readClip(Wav.write(rec.samples, Recorder.RATE), "recording-${System.currentTimeMillis() / 1000}.wav") }
            place(draft, target)
          } catch (failure: Exception) {
            _state.update { it.copy(error = "The recording could not be used: ${failure.message}") }
          }
        }
      }
      _state.update { it.copy(recording = target, recordedSeconds = 0.0, error = null) }
    } catch (failure: Exception) {
      _state.update { it.copy(error = "Could not record: ${failure.message}") }
    }
  }

  fun stopRecording() = recorder.stop()

  fun play(draft: ClipDraft) {
    try {
      player.play(draft.clip) { _state.update { s -> if (s.playing == draft.key) s.copy(playing = null) else s } }
      _state.update { it.copy(playing = draft.key) }
    } catch (failure: Exception) {
      _state.update { it.copy(error = "Could not play the clip: ${failure.message}") }
    }
  }

  fun stopPlaying() {
    player.stop()
    _state.update { it.copy(playing = null) }
  }

  // ---------------------------------------------------------------------------------- photo

  /** A photo of the user's: decoded upright, its long side shrunk to [PHOTO_SIDE] pixels. */
  fun setPhoto(uri: Uri?) {
    if (uri == null) return
    viewModelScope.launch {
      try {
        val bitmap = withContext(worker) { decodePhoto(uri) }
        setPhotoBitmap(bitmap)
        _state.update { it.copy(answered = null, exampleId = null, error = null) }
      } catch (failure: Throwable) {
        _state.update { it.copy(error = "Could not read the photo: ${failure.message}") }
      }
    }
  }

  fun removePhoto() {
    setPhotoBitmap(null)
    _state.update { it.copy(answered = null, exampleId = null) }
  }

  private fun setPhotoBitmap(bitmap: Bitmap?) {
    if (bitmap == null) {
      picture = null
      pictureSha256 = null
    } else {
      val argb = rawPixels(bitmap)
      picture = Picture.of(argb, bitmap.width, bitmap.height)
      // The SHA-256 of the RGB values the host reads (what numpy's tobytes() hashes for an [H, W, 3] uint8 picture).
      pictureSha256 = Requests.sha256(ByteArray(argb.size * 3) { i -> (argb[i / 3] shr (16 - 8 * (i % 3))).toByte() })
    }
    _state.update { it.copy(photo = bitmap) }
  }

  /**
   * The picture's stored values, as the model's own image code reads a file: no color management
   * (Bitmap.getPixels would convert a wide-gamut or ProPhoto picture to sRGB).
   */
  private fun rawPixels(bitmap: Bitmap): IntArray {
    val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
    val bytes = java.nio.ByteBuffer.allocate(argb.byteCount)
    argb.copyPixelsToBuffer(bytes)
    val data = bytes.array()
    val stride = argb.rowBytes
    return IntArray(argb.width * argb.height) { i ->
      val at = (i / argb.width) * stride + (i % argb.width) * 4
      (0xff shl 24) or ((data[at].toInt() and 0xff) shl 16) or ((data[at + 1].toInt() and 0xff) shl 8) or (data[at + 2].toInt() and 0xff)
    }
  }

  private fun decodePhoto(uri: Uri): Bitmap {
    val resolver = getApplication<Application>().contentResolver
    // Decoded upright (ImageDecoder applies the EXIF orientation), at a power-of-two sample size that
    // keeps the long side at least PHOTO_SIDE, then scaled to PHOTO_SIDE.
    val decoded =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        ImageDecoder.decodeBitmap(ImageDecoder.createSource(resolver, uri)) { decoder, info, _ ->
          decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
          decoder.setTargetSampleSize(sampleSize(max(info.size.width, info.size.height)))
        }
      } else {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
        val options = BitmapFactory.Options().apply { inSampleSize = sampleSize(max(bounds.outWidth, bounds.outHeight)) }
        resolver.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, options) } ?: error("not a picture")
      }
    val upright = if (decoded.config == Bitmap.Config.ARGB_8888) decoded else decoded.copy(Bitmap.Config.ARGB_8888, false)
    val (w, h) = shrunk(upright.width, upright.height)
    return if (w == upright.width && h == upright.height) upright else Bitmap.createScaledBitmap(upright, w, h, true)
  }

  private fun sampleSize(longSide: Int): Int {
    var n = 1
    while (longSide / (n * 2) >= PHOTO_SIDE) n *= 2
    return n
  }

  private fun shrunk(width: Int, height: Int): Pair<Int, Int> {
    val scale = PHOTO_SIDE.toDouble() / max(width, height)
    if (scale >= 1.0) return width to height
    return max(1, (width * scale).roundToInt()) to max(1, (height * scale).roundToInt())
  }

  // ---------------------------------------------------------------------------------- decide

  /**
   * The graph files a request needs that are not in the model folder, checked before anything runs: the picture
   * tower for a photo, and each clip's audio file.
   */
  private fun preflight(media: List<Any>, spoken: List<ClipDraft>) {
    val a = audio!!
    if (media.any { it is Picture }) {
      ModelFiles.require(modelDir, listOf(Graphs.TOWER), "a request with a photo")
    }
    val clips = media.filterIsInstance<AudioClip>() + spoken.map { it.clip }
    // A clip longer than the largest graph is left to the host, which refuses it with its length.
    val needed = clips.map { c -> extractor.numFrames(Resample.length(c.mono.size, c.rate, extractor.samplingRate)) }
      .filter { it <= a.maxFrames }
      .map { a.file(a.bucketFor(it)).name }
      .distinct()
    ModelFiles.require(modelDir, needed, "these clips")
  }

  /** Answers every card in one call; what the Decide button does. Returns the run's record (for run files). */
  suspend fun decide(): Map<String, Any?>? {
    val s = _state.value
    val h = host ?: return null
    if (!s.ready || s.busy || s.recording != null) return null
    val built =
      try {
        val request = Requests.build(s.stateText, s.stateIsJson, if (s.photo != null) picture else null, s.clips, s.questions)
        preflight(request.third, s.questions.mapNotNull { it.spoken })
        request
      } catch (failure: IllegalArgumentException) {
        _state.update { it.copy(error = failure.message) }
        return null
      }
    val (stateValue, questions, media) = built
    _state.update { it.copy(busy = true, error = null) }
    return try {
      val start = System.nanoTime()
      val result = withContext(worker) { h.systemOne(stateValue, questions, media) }
      val wallMs = (System.nanoTime() - start) / 1e6
      val d = decide!!
      val byKey = s.questions.associate { it.key to (result.answers[it.name] as Map<String, Any?>) }
      val text = if (stateValue != null) h.prompt.stateText(stateValue) else ""
      // Compile time of this request only (a graph compiled for an earlier request does not count again).
      val loadMs = result.timing.loadMs
      val answered = Answered(byKey, text, result.inputTokens, d.lastBucket, d.lastAccelerator, result.timing.totalMs - loadMs, loadMs)
      _state.update { it.copy(busy = false, answered = answered, stateText = if (s.stateIsJson) text else it.stateText) }
      record(s, stateValue, result, wallMs, loadMs, d)
    } catch (failure: Throwable) {
      Log.e(TAG, "decide failed", failure)
      _state.update { it.copy(busy = false, error = failure.message ?: failure.toString()) }
      linkedMapOf("device" to Build.MODEL, "error" to (failure.message ?: failure.toString()))
    }
  }

  /** The Decide record: the request (clips by file and SHA-256), the token ids, each clip's features, the answers and the times. */
  private fun record(s: UiState, stateValue: Any?, result: HostResult, wallMs: Double, loadMs: Double, d: PhoneDecideBackend): Map<String, Any?> {
    val drafts = IdentityClips(s)
    fun clipRef(c: AudioClip): Map<String, Any?> {
      val draft = drafts.of(c)
      return linkedMapOf("type" to "audio", "path" to draft?.file?.name, "label" to draft?.label, "sha256" to draft?.sha256)
    }
    val questions = LinkedHashMap<String, Any?>()
    for (q in s.questions) {
      val json = q.toJson()
      (json["instructions"] as? AudioClip)?.let { json["instructions"] = clipRef(it) }
      questions[q.name] = json
    }
    val media = ArrayList<Any?>()
    if (s.photo != null) media.add(linkedMapOf("type" to "image", "width" to s.photo.width, "height" to s.photo.height, "sha256" to pictureSha256))
    s.clips.forEach { media.add(clipRef(it.clip)) }
    val t = result.timing
    return linkedMapOf(
      "device" to Build.MODEL,
      "form" to d.form,
      "request" to linkedMapOf("state" to stateValue, "questions" to questions, "media" to media),
      "clips" to result.clips.map { c ->
        linkedMapOf(
          "path" to drafts.of(c.clip)?.file?.name,
          "sha256" to drafts.of(c.clip)?.sha256,
          "rate" to c.clip.rate,
          "channels" to c.clip.channels,
          "samples" to c.clip.mono.size,
          "samples_16k" to c.samples,
          "frames" to c.frames,
          "tokens" to c.tokens,
          "features_shape" to listOf(128, c.frames),
          "features_sha256" to c.features?.let { Requests.sha256(it) },
        )
      },
      "response" to linkedMapOf("answers" to result.answers, "usage" to linkedMapOf("input_tokens" to result.inputTokens, "output_tokens" to 0)),
      "sequences" to result.sequences.map { seq ->
        linkedMapOf("first_question" to seq.firstQuestion, "ids" to seq.ids, "state_len" to seq.stateLength, "question" to seq.question, "ends" to seq.ends)
      },
      "timing" to linkedMapOf(
        "prep_ms" to t.prepMs, "feature_ms" to t.featureMs, "tower_ms" to t.towerMs, "audio_ms" to t.audioMs,
        "decide_ms" to t.decideMs, "readout_ms" to t.readoutMs, "total_ms" to t.totalMs, "load_ms" to loadMs, "wall_ms" to wallMs,
        "bucket" to d.lastBucket, "accelerator" to d.lastAccelerator, "audio_calls" to t.audioCalls, "decide_calls" to t.decideCalls,
        "canvas" to (if (s.photo != null) tower?.lastCanvas?.let { c -> listOf(c.first, c.second) } else null),
      ),
    )
  }

  /** The editor's clips by the clip object the host saw. */
  private class IdentityClips(s: UiState) {
    private val map = java.util.IdentityHashMap<AudioClip, ClipDraft>()

    init {
      s.clips.forEach { map[it.clip] = it }
      s.questions.forEach { q -> q.spoken?.let { map[it.clip] = it } }
    }

    fun of(clip: AudioClip): ClipDraft? = map[clip]
  }

  fun decideFromButton() {
    viewModelScope.launch { decide() }
  }

  // ---------------------------------------------------------------------------------- run files

  /**
   * Runs a run file (runs/<name> in the app's files directory): its steps call the same functions the screen does.
   * Steps: {"example": id}, {"state": text}, {"json": true}, {"clips": ["a.wav", ...]} (the state's clips, from the
   * runs directory), {"photo": "p.png"} (null removes the photo), {"questions": {name: question, ...}} (the cards, in
   * the model's question JSON; a spoken question's instructions are {"type": "audio", "path": "q.wav"}),
   * {"spoken": {"question": name, "clip": "q.wav"}}, {"edit": {"question": name, "instructions": text}},
   * {"decide": true, "out": "out/name.json"}, {"wait_ms": n}.
   */
  fun run(name: String) {
    viewModelScope.launch {
      val file = File(runsDir, name)
      val log = ArrayList<Any?>()
      try {
        @Suppress("UNCHECKED_CAST") val doc = D1Json.parse(file.readBytes()) as Map<String, Any?>
        if (!loaded.await()) error(_state.value.error ?: "model unavailable")
        for (step in doc["steps"] as List<*>) {
          step as Map<*, *>
          (step["example"] as String?)?.let { selectExample(it) }
          (step["state"] as String?)?.let { setStateText(it) }
          (step["json"] as Boolean?)?.let { setStateIsJson(it) }
          (step["clips"] as List<*>?)?.let { names ->
            _state.update { it.copy(clips = emptyList()) }
            for (n in names) {
              addClipFile(File(runsDir, n as String), ClipTarget.State)?.let { log.add(linkedMapOf("clip" to n, "refused" to it)) }
            }
          }
          if (step.containsKey("photo")) {
            val photo = step["photo"] as String?
            setRunPhoto(photo?.let { File(runsDir, it) })?.let { log.add(linkedMapOf("photo" to photo, "refused" to it)) }
          }
          (step["questions"] as Map<*, *>?)?.let { questions ->
            setRunQuestions(questions)?.let { log.add(linkedMapOf("questions" to questions.keys.toList(), "refused" to it)) }
          }
          (step["spoken"] as Map<*, *>?)?.let { spoken ->
            val target = _state.value.questions.first { it.name == spoken["question"] }
            val clip = spoken["clip"] as String
            addClipFile(File(runsDir, clip), ClipTarget.Spoken(target.key))?.let { log.add(linkedMapOf("clip" to clip, "refused" to it)) }
          }
          (step["edit"] as Map<*, *>?)?.let { edit ->
            val target = _state.value.questions.first { it.name == edit["question"] }
            updateQuestion(target.key) { it.copy(instructions = edit["instructions"] as String, spoken = null) }
          }
          (step["wait_ms"] as JsonNumber?)?.let { kotlinx.coroutines.delay(it.toDouble().toLong()) }
          if (step["decide"] == true) {
            val record = decide() ?: linkedMapOf("device" to Build.MODEL, "refused" to _state.value.error)
            val out = step["out"] as String?
            if (out != null) {
              val target = File(runsDir, out)
              target.parentFile?.mkdirs()
              target.writeText(D1Json.write(record))
            }
            log.add(linkedMapOf("out" to out, "ok" to (!record.containsKey("error") && !record.containsKey("refused"))))
          }
        }
        File(runsDir, "$name.done").writeText(D1Json.write(linkedMapOf("run" to name, "steps" to log)))
      } catch (failure: Throwable) {
        Log.e(TAG, "run $name failed", failure)
        File(runsDir, "$name.done").writeText(D1Json.write(linkedMapOf("run" to name, "error" to (failure.message ?: failure.toString()), "steps" to log)))
      }
    }
  }

  override fun onCleared() {
    recorder.close()
    player.close()
    decide?.close()
    tower?.close()
    audio?.close()
    accelerators?.close()
    worker.close()
  }

  companion object {
    private const val TAG = "D1OmniDecide"

    /** The long side a photo is shrunk to: the picture then fits the smallest tower canvas (24 x 24 patches). */
    const val PHOTO_SIDE = 384

    private fun loadExamples(context: Context): List<Example> {
      @Suppress("UNCHECKED_CAST")
      val doc = D1Json.parse(context.assets.open("examples.json").use { it.readBytes() }) as Map<String, Any?>
      return (doc["examples"] as List<*>).map {
        it as Map<*, *>
        Example(
          it["id"] as String,
          it["title"] as String,
          it["state"],
          it["picture"] as String?,
          (it["clips"] as List<*>?)?.map { c -> c as String } ?: emptyList(),
          it["questions"] as Map<*, *>,
        )
      }
    }
  }
}

fun Double.format(digits: Int): String = String.format("%.${digits}f", this)
