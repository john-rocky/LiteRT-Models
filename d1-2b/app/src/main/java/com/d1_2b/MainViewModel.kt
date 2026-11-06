package com.d1_2b

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
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

/** The question types the editor offers, and how each is written in the model's question JSON. */
enum class QType(val label: String) {
  NOUL("Yes / no"),
  CHOICE("Choice"),
  MULTIPLE("Several choices"),
  SCORE("Score"),
  TEXT("Text"),
  NUMBER("Number"),
  ESTIMATE("Number in a range"),
  DATE("Date"),
  BOX("Box"),
  BOXES("Boxes"),
}

data class OptionDraft(val name: String, val text: String)

/** One question card of the editor. */
data class QuestionDraft(
  val key: Long,
  val name: String,
  val type: QType,
  val instructions: String,
  val options: List<OptionDraft> = listOf(OptionDraft("yes", "Yes"), OptionDraft("no", "No")),
  val levels: List<String> = listOf("Low", "Medium", "High"),
  val low: String = "0",
  val high: String = "10",
  val step: String = "1",
  val unit: String = "",
  val max: String = "",
) {
  /** The question in the model's JSON (type, instructions, criteria, ...). */
  fun toJson(): LinkedHashMap<String, Any?> {
    val out = LinkedHashMap<String, Any?>()
    out["type"] =
      when (type) {
        QType.NOUL -> "noul"
        QType.CHOICE,
        QType.MULTIPLE -> "choice"
        QType.SCORE -> "score"
        QType.TEXT -> "text"
        QType.NUMBER,
        QType.ESTIMATE -> "number"
        QType.DATE -> "date"
        QType.BOX -> "box"
        QType.BOXES -> "boxes"
      }
    if (type == QType.MULTIPLE) out["multiple"] = true
    out["instructions"] = instructions
    when (type) {
      QType.CHOICE,
      QType.MULTIPLE ->
        out["criteria"] = LinkedHashMap<String, Any?>().apply { options.forEach { put(it.name.trim(), it.text.trim().ifEmpty { null }) } }
      QType.SCORE -> out["criteria"] = levels.map { it.trim() }
      QType.ESTIMATE -> {
        out["range"] = listOf(number("low", low), number("high", high))
        if (step.isNotBlank()) out["step"] = number("step", step)
      }
      QType.BOXES -> if (max.isNotBlank()) out["max"] = number("max", max)
      else -> Unit
    }
    if ((type == QType.NUMBER || type == QType.ESTIMATE) && unit.isNotBlank()) out["unit"] = unit.trim()
    return out
  }

  private fun number(field: String, text: String): JsonNumber {
    val trimmed = text.trim()
    require(NUMBER.matches(trimmed)) { "question '$name': $field \"$text\" is not a number" }
    return JsonNumber(trimmed)
  }

  companion object {
    private val NUMBER = Regex("-?(0|[1-9]\\d*)(\\.\\d+)?([eE][-+]?\\d+)?")

    /** A card from the question JSON of a bundled example or a run file. */
    fun fromJson(key: Long, name: String, raw: Map<*, *>): QuestionDraft {
      val kind = raw["type"] as String
      val multiple = raw["multiple"] == true
      val type =
        when {
          kind == "choice" && multiple -> QType.MULTIPLE
          kind == "number" && raw.containsKey("range") -> QType.ESTIMATE
          else -> QType.entries.first { it.name == kind.uppercase() || (it == QType.NOUL && kind == "noul") }
        }
      var draft = QuestionDraft(key, name, type, raw["instructions"] as String)
      when (val criteria = raw["criteria"]) {
        is Map<*, *> -> draft = draft.copy(options = criteria.map { (k, v) -> OptionDraft(k as String, (v as String?) ?: "") })
        is List<*> -> draft = draft.copy(levels = criteria.map { it as String })
      }
      (raw["range"] as List<*>?)?.let { draft = draft.copy(low = text(it[0]), high = text(it[1]), step = raw["step"]?.let { s -> text(s) } ?: "") }
      (raw["unit"] as String?)?.let { draft = draft.copy(unit = it) }
      raw["max"]?.let { draft = draft.copy(max = text(it)) }
      return draft
    }

    private fun text(value: Any?): String = (value as? JsonNumber)?.literal ?: value.toString()
  }
}

/** A bundled example (assets/examples.json). */
class Example(val id: String, val title: String, val state: Any?, val picture: String?, val questions: Map<*, *>)

/** One answered request: the answers by card, and the state text their spans index. */
class Answered(
  val answers: Map<Long, Map<String, Any?>>,
  val stateText: String,
  val tokens: Int,
  val bucket: Int,
  val totalMs: Double,
  val loadMs: Double,
)

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
  val questions: List<QuestionDraft> = emptyList(),
  val answered: Answered? = null,
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
  private val worker = Executors.newSingleThreadExecutor { Thread(it, "d1-litert").apply { isDaemon = true } }.asCoroutineDispatcher()
  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state

  /** Where the model files are copied (see README.md). */
  val modelDir: File = File(application.getExternalFilesDir(null), "model")
  private val runsDir = File(application.getExternalFilesDir(null), "runs")
  private var host: Host? = null
  private var decide: LiteRtDecide? = null
  private var tower: LiteRtTower? = null
  private var picture: Picture? = null
  private var nextKey = 1L
  private val examples: List<Example> = loadExamples(application)
  private val loaded = CompletableDeferred<Boolean>()

  init {
    modelDir.mkdirs()
    runsDir.mkdirs()
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
      val (h, d) =
        withContext(worker) {
          val missing = listOf("provider/tokenizer.json", "provider/config.json", "provider/calibration.json", "tables/embed_table.npy").filter { !File(modelDir, it).isFile }
          require(missing.isEmpty()) { "Missing model files in ${modelDir.absolutePath}: ${missing.joinToString()}. See the README." }
          val d = LiteRtDecide(modelDir, maxTokens())
          // Nothing is loaded yet: the decision graph is the largest stage, so it must fit with the reserve to spare.
          Memory.ensure("decision graph (${d.form})", d.needMb + Memory.RESERVE_MB, "Close other apps and open this app again.")
          val t = LiteRtTower(modelDir).takeIf { it.available }
          tower = t
          Host.load(modelDir, d, t) to d
        }
      host = h
      decide = d
      _state.update { it.copy(status = "Compiling the decision graph on the GPU…") }
      val (graphs, compileMs) = withContext(worker) { prepareFor(h, d) }
      _state.update {
        it.copy(ready = true, status = "Ready · ${Build.MODEL} · GPU FP32 · $graphs (${(compileMs / 1000).format(1)} s to compile)")
      }
      loaded.complete(true)
    } catch (failure: Throwable) {
      Log.e(TAG, "load failed", failure)
      _state.update { it.copy(status = "Model unavailable", error = failure.message ?: failure.toString()) }
      loaded.complete(false)
    }
  }

  /**
   * Compiles what the example on screen needs, so that its first Decide does not wait for a compile:
   * the decision graph of its longest sequence (else the 256-token one) and, with a photo, the picture
   * tower. Returns the status line's graph words and the compile time in milliseconds.
   */
  private fun prepareFor(h: Host, d: LiteRtDecide): Pair<String, Double> {
    val s = _state.value
    val photo = if (s.photo != null) picture else null
    val longest =
      runCatching {
          val (stateValue, questions) = request(s)
          h.sequences(stateValue, h.named(questions).values.toList(), photo).maxOf { it.ids.size }
        }
        .getOrDefault(256)
    val bucket = runCatching { d.bucketFor(longest) }.getOrElse { d.bucketFor(256) }
    d.prepare(bucket)
    var compileMs = d.lastLoadMs
    var graphs = if (d.form.startsWith("8")) "all lengths" else "L$bucket"
    val t = tower
    if (photo != null && t != null) {
      try {
        t.prepare(photo)
        compileMs += t.lastLoadMs
        graphs += " + picture tower"
      } catch (failure: Exception) {
        // Not fatal: the first photo request compiles it, or says why it cannot.
        Log.w(TAG, "the picture tower is left for the first photo request: ${failure.message}")
      }
    }
    return graphs to compileMs
  }

  // ---------------------------------------------------------------------------------- editor

  fun selectExample(id: String) {
    val example = examples.first { it.id == id }
    val drafts = example.questions.map { (name, raw) -> QuestionDraft.fromJson(nextKey++, name as String, raw as Map<*, *>) }
    val isJson = example.state != null && example.state !is String
    val text =
      when {
        example.state == null -> ""
        isJson -> D1Json.writeIndented(example.state, 2)
        else -> example.state as String
      }
    val photo = example.picture?.let { name -> getApplication<Application>().assets.open(name).use { BitmapFactory.decodeStream(it) } }
    setPhotoBitmap(photo)
    _state.update { it.copy(exampleId = id, stateText = text, stateIsJson = isJson, questions = drafts, answered = null, error = null) }
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
    picture = bitmap?.let { Picture.of(rawPixels(it), it.width, it.height) }
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

  /** The request the cards describe: (state, named questions). */
  private fun request(s: UiState): Pair<Any?, LinkedHashMap<String, Any?>> {
    val state: Any? =
      when {
        s.stateIsJson -> try {
          D1Json.parse(s.stateText)
        } catch (failure: Exception) {
          throw IllegalArgumentException("The state is not valid JSON: ${failure.message}")
        }
        s.stateText.isBlank() -> if (s.photo != null) null else throw IllegalArgumentException("Write a state or add a photo.")
        else -> s.stateText
      }
    require(s.questions.isNotEmpty()) { "Add a question." }
    val questions = LinkedHashMap<String, Any?>()
    for (q in s.questions) {
      require(q.name.isNotBlank()) { "Every question needs a name." }
      require(q.name !in questions) { "Two questions are named '${q.name}'." }
      questions[q.name] = q.toJson()
    }
    return state to questions
  }

  /** Answers every card in one call; what the Decide button does. Returns the run's record (for run files). */
  suspend fun decide(): Map<String, Any?>? {
    val s = _state.value
    val h = host ?: return null
    if (!s.ready || s.busy) return null
    val (stateValue, questions) =
      try {
        request(s)
      } catch (failure: IllegalArgumentException) {
        _state.update { it.copy(error = failure.message) }
        return null
      }
    _state.update { it.copy(busy = true, error = null) }
    return try {
      val start = System.nanoTime()
      val requestPicture = if (s.photo != null) picture else null
      val result = withContext(worker) { h.systemOne(stateValue, questions, requestPicture) }
      val wallMs = (System.nanoTime() - start) / 1e6
      val d = decide!!
      val byKey = s.questions.associate { it.key to (result.answers[it.name] as Map<String, Any?>) }
      val text = if (stateValue != null) h.prompt.stateText(stateValue) else ""
      // Compile time of this request only (a graph compiled for an earlier request does not count again).
      val loadMs = result.timing.loadMs
      val answered = Answered(byKey, text, result.inputTokens, d.lastBucket, result.timing.totalMs - loadMs, loadMs)
      _state.update { it.copy(busy = false, answered = answered, stateText = if (s.stateIsJson) text else it.stateText) }
      // The Decide line's numbers, and where the time went, in the log (adb logcat -s D1Decide).
      val tm = result.timing
      Log.i(
        TAG,
        "Answered in ${answered.totalMs.format(0)} ms · ${answered.tokens} tokens · L${answered.bucket} graph " +
          "(prep ${tm.prepMs.format(1)}, tower ${tm.towerMs.format(1)}, decide ${tm.decideMs.format(1)}, " +
          "readout ${tm.readoutMs.format(1)}, compile ${loadMs.format(1)} ms; total ${answered.totalMs.format(1)} ms)",
      )
      linkedMapOf(
        "device" to Build.MODEL,
        "form" to d.form,
        "request" to linkedMapOf("state" to stateValue, "questions" to questions, "picture" to (s.photo != null)),
        "response" to linkedMapOf("answers" to result.answers, "usage" to linkedMapOf("input_tokens" to result.inputTokens, "output_tokens" to 0)),
        "sequences" to result.sequences.map { seq ->
          linkedMapOf("first_question" to seq.firstQuestion, "ids" to seq.ids, "state_len" to seq.stateLength, "question" to seq.question, "ends" to seq.ends)
        },
        "timing" to linkedMapOf(
          "prep_ms" to result.timing.prepMs, "tower_ms" to result.timing.towerMs, "decide_ms" to result.timing.decideMs,
          "readout_ms" to result.timing.readoutMs, "total_ms" to result.timing.totalMs, "load_ms" to loadMs,
          "wall_ms" to wallMs, "bucket" to d.lastBucket,
          "canvas" to requestPicture?.let { tower?.lastCanvas?.let { c -> listOf(c.first, c.second) } },
        ),
      )
    } catch (failure: Throwable) {
      Log.e(TAG, "decide failed", failure)
      _state.update { it.copy(busy = false, error = failure.message ?: failure.toString()) }
      linkedMapOf("device" to Build.MODEL, "error" to (failure.message ?: failure.toString()))
    }
  }

  fun decideFromButton() {
    viewModelScope.launch { decide() }
  }

  // ---------------------------------------------------------------------------------- run files

  /**
   * Runs a run file (runs/<name> in the app's files directory): its steps call the same functions
   * the screen does. Steps: {"example": id}, {"state": text}, {"edit": {"question": name,
   * "instructions": text}}, {"decide": true, "out": "out/name.json"}, {"wait_ms": n}.
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
          (step["edit"] as Map<*, *>?)?.let { edit ->
            val target = _state.value.questions.first { it.name == edit["question"] }
            updateQuestion(target.key) { it.copy(instructions = edit["instructions"] as String) }
          }
          (step["wait_ms"] as JsonNumber?)?.let { kotlinx.coroutines.delay(it.toDouble().toLong()) }
          if (step["decide"] == true) {
            val record = decide()
            val out = step["out"] as String?
            if (record != null && out != null) {
              val target = File(runsDir, out)
              target.parentFile?.mkdirs()
              target.writeText(D1Json.write(record))
            }
            log.add(linkedMapOf("out" to out, "ok" to (record != null && !record.containsKey("error"))))
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
    decide?.close()
    tower?.close()
    worker.close()
  }

  companion object {
    private const val TAG = "D1Decide"

    /** The long side a photo is shrunk to: the picture then fits the smallest tower canvas (24 x 24 patches). */
    const val PHOTO_SIDE = 384

    private fun loadExamples(context: Context): List<Example> {
      @Suppress("UNCHECKED_CAST")
      val doc = D1Json.parse(context.assets.open("examples.json").use { it.readBytes() }) as Map<String, Any?>
      return (doc["examples"] as List<*>).map {
        it as Map<*, *>
        Example(it["id"] as String, it["title"] as String, it["state"], it["picture"] as String?, it["questions"] as Map<*, *>)
      }
    }
  }
}

fun Double.format(digits: Int): String = String.format("%.${digits}f", this)
