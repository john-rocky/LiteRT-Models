/*
 * Ports the sequence packing, slot reading and pointer reads of LiquidAI/d1-Omni-2B (revision 431f1d66:
 * runner.py, attention.py, pointer.py, prompt.readouts, media.Reader) to Kotlin, as the model repository's
 * host (host/d1_host.py) runs them; that code is under the LFM Open License v1.0 (see LICENSE and NOTICE).
 */

package com.d1_omni_2b

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.util.IdentityHashMap
import kotlin.math.exp
import kotlin.math.sqrt

/** The model's hidden width. */
const val HIDDEN = 2048

/** The decision graph behind the host: one packed sequence in, its hidden states out. */
interface DecideBackend {
  /** The longest sequence a graph takes; a longer one is refused before any graph runs. */
  val maxLength: Int

  /** Why [maxLength] is the limit when the model has longer graphs (for the refusal), else "". */
  val limitNote: String
    get() = ""

  /**
   * Runs one sequence: [embeds] holds [tokens] rows of [HIDDEN] floats, [question] each token's tag
   * (-1 on the state, k on question k). Returns the hidden states, row-major, at least [tokens] rows.
   */
  fun decide(embeds: FloatArray, tokens: Int, stateLength: Int, question: IntArray): FloatArray

  /** The part of the last [decide] spent compiling its graph, in milliseconds (0 when the graph was ready). */
  val lastLoadMs: Double
    get() = 0.0
}

/** The picture tower behind the host. */
interface TowerBackend {
  /** The picture's tokens, [patchRows * patchColumns / 4] rows of [HIDDEN] floats, in the model's order. */
  fun tower(picture: Picture): FloatArray

  /** The part of the last [tower] spent compiling its graph, in milliseconds (0 when the graph was ready). */
  val lastLoadMs: Double
    get() = 0.0
}

/** The audio encoder behind the host: one clip's log-mel in, its tokens out. */
interface AudioBackend {
  /** The longest clip a graph takes, in mel frames; a longer clip is refused before any graph runs. */
  val maxFrames: Int

  /**
   * Runs one clip: [features] is its log-mel [128, [frames]] row-major (one row per mel band). Returns its tokens,
   * ceil(frames / 8) rows of [HIDDEN] floats, in order.
   */
  fun audio(features: FloatArray, frames: Int): FloatArray

  /** The part of the last [audio] spent compiling its graph, in milliseconds (0 when the graph was ready). */
  val lastLoadMs: Double
    get() = 0.0
}

/** How the processor writes a clip (processor_config.json): `<|audio|>` once per encoder token, between the span tokens. */
class AudioMarkup(val token: String, val start: String, val end: String, val markers: Boolean) {
  /** chat_template.jinja's audio item. */
  val template: String
    get() = start + token + end

  fun of(tokens: Int): String = if (markers) start + token.repeat(tokens) + end else token.repeat(tokens)
}

/** One clip of a request at the extractor's rate: its log-mel `input_features` [128, frames] and its tokens. */
class ClipFeatures(val clip: AudioClip, val samples: Int, val frames: Int, val tokens: Int) {
  /** The features, row-major [128, frames]; computed once the request passed the length checks. */
  var features: FloatArray? = null
}

/** The tied embedding table as bfloat16 bits (tables/embed_table.npy, uint16 [rows, HIDDEN]), memory-mapped. */
class EmbedTable(file: File) : Closeable {
  private val channel = RandomAccessFile(file, "r").channel
  private val bits: ShortBuffer
  val rows: Int

  init {
    val head = channel.map(FileChannel.MapMode.READ_ONLY, 0, minOf(channel.size(), 4096L)).order(ByteOrder.LITTLE_ENDIAN)
    val magic = ByteArray(6).also { head.get(it) }
    require(magic.contentEquals(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte()))) {
      "${file.name} is not an .npy file"
    }
    val major = head.get().toInt()
    head.get()
    val headerLength = if (major == 1) head.short.toInt() and 0xffff else head.int
    val offset = (if (major == 1) 10 else 12) + headerLength
    val header = ByteArray(headerLength).also { head.get(it) }.toString(Charsets.ISO_8859_1)
    require("'<u2'" in header && "'fortran_order': False" in header) { "${file.name}: not little-endian uint16 C order" }
    val shape = Regex("'shape': \\((\\d+), (\\d+)\\)").find(header) ?: error("${file.name}: no 2-D shape")
    rows = shape.groupValues[1].toInt()
    require(shape.groupValues[2].toInt() == HIDDEN) { "${file.name}: rows of ${shape.groupValues[2]}, not $HIDDEN" }
    require(channel.size() >= offset + rows.toLong() * HIDDEN * 2) { "${file.name} is truncated" }
    bits = channel.map(FileChannel.MapMode.READ_ONLY, offset.toLong(), rows.toLong() * HIDDEN * 2).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
  }

  /** Row [id] widened to float32 (the 16 bits become the high half) into [out] at [at]. */
  fun row(id: Int, out: FloatArray, at: Int) {
    require(id in 0 until rows) { "token id $id outside the table" }
    val base = id * HIDDEN
    for (i in 0 until HIDDEN) out[at + i] = Float.fromBits((bits.get(base + i).toInt() and 0xffff) shl 16)
  }

  fun row(id: Int): FloatArray = FloatArray(HIDDEN).also { row(id, it, 0) }

  override fun close() = channel.close()
}

/** One packed sequence: a state and up to `set_size` questions (attention.pack_set). */
class PackedSequence(
  val firstQuestion: Int,
  val ids: IntArray,
  val stateLength: Int,
  val question: IntArray,
  /** Each question's answer slots, as positions in [ids]. */
  val ends: List<IntArray>,
  val positions: StatePositions?,
  /** The sequence's media in its order: the state's pictures and clips, then each packed question's spoken clip. */
  val media: List<Any> = emptyList(),
) {
  val clips: List<ClipFeatures>
    get() = media.filterIsInstance<ClipFeatures>()
}

/**
 * Where one call's time went, in milliseconds: [featureMs] is the clips' resampling and log-mel; [loadMs] is the part
 * of [towerMs], [audioMs] and [decideMs] spent compiling graphs.
 */
class HostTiming(
  var prepMs: Double = 0.0,
  var featureMs: Double = 0.0,
  var towerMs: Double = 0.0,
  var audioMs: Double = 0.0,
  var decideMs: Double = 0.0,
  var readoutMs: Double = 0.0,
  var loadMs: Double = 0.0,
  var decideCalls: Int = 0,
  var towerCalls: Int = 0,
  var audioCalls: Int = 0,
  var tokens: Int = 0,
) {
  val totalMs: Double
    get() = prepMs + featureMs + towerMs + audioMs + decideMs + readoutMs
}

/** One `system_one` call's result: the answers by question name and what the call read. */
class HostResult(
  val answers: LinkedHashMap<String, LinkedHashMap<String, Any?>>,
  val inputTokens: Int,
  val sequences: List<PackedSequence>,
  val timing: HostTiming,
  /** The request's clips: the state's in order, then the spoken questions' in question order. */
  val clips: List<ClipFeatures> = emptyList(),
)

/**
 * The d1-Omni-2B engine and its System One API over a [DecideBackend], a [TowerBackend] and an [AudioBackend]: the
 * provider's `model.system_one(state, questions, media)` with one picture at most and any number of clips; a
 * question's instructions may be a clip (a spoken question).
 */
class Host(
  val tokenizer: Tokenizer,
  val prompt: Prompt,
  private val calibration: Calibration?,
  private val table: EmbedTable,
  private val decide: DecideBackend,
  private val tower: TowerBackend?,
  /** `d1_set_size`: questions read together after one state. */
  private val setSize: Int = 16,
  private val imageTokenId: Int = 125030,
  private val audio: AudioBackend? = null,
  val extractor: FeatureExtractor? = null,
  private val audioMarkup: AudioMarkup? = null,
  private val audioTokenId: Int = 125031,
) {
  private val tableRows = HashMap<List<Int>, FloatArray>()

  /** The question JSON checked and read as the model's question classes, by name. */
  fun named(questions: Map<String, Any?>): LinkedHashMap<String, Question> {
    val out = LinkedHashMap<String, Question>()
    for ((name, raw) in questions) {
      try {
        out[name] = Questions.parse(raw)
      } catch (e: IllegalArgumentException) {
        throw IllegalArgumentException("question '$name': ${e.message}")
      }
    }
    return out
  }

  /** The picture's markup: its tokens between the vision start and end tokens. */
  fun markup(picture: Picture): String = "<|vision_start|>" + "<|image|>".repeat(picture.tokens) + "<|vision_end|>"

  /** A clip's markup for [tokens] encoder tokens (`replace_audio_token`). */
  fun markup(clip: ClipFeatures): String = requireNotNull(audioMarkup) { "this checkpoint takes no audio" }.of(clip.tokens)

  /** A clip planned at the extractor's rate: its length there, its frames and its tokens (no features yet). */
  fun plan(clip: AudioClip): ClipFeatures {
    val extractor = requireNotNull(extractor) { "this checkpoint takes no audio" }
    val samples = Resample.length(clip.mono.size, clip.rate, extractor.samplingRate)
    val frames = extractor.numFrames(samples)
    return ClipFeatures(clip, samples, frames, extractor.numEncoderTokens(frames))
  }

  /**
   * The packed sequences of one request, as the provider builds them (`SlotReader._sequences`): [media] the state's
   * items in order ([Picture] or [ClipFeatures]), [voices] each question's spoken clip or null.
   */
  fun sequences(state: Any?, qs: List<Question>, voices: List<ClipFeatures?>, media: List<Any>): List<PackedSequence> {
    if (qs.any { it is BoxQ || it is BoxesQ }) {
      val n = media.count { it is Picture }
      require(n == 1) { "a box or boxes question reads exactly one picture; this request has $n" }
    }
    val prefix = prompt.prefixText(state, media.joinToString("") { if (it is Picture) markup(it) else markup(it as ClipFeatures) })
    val layouts = qs.map { prompt.suffixLayout(it) }
    // Reader.place: a spoken question's clip in place of its instructions
    val suffixes = layouts.mapIndexed { i, (text, _) -> voices[i]?.let { text.replace(Prompt.SPOKEN, markup(it)) } ?: text }
    val header = prompt.questionHeader(state)
    val out = ArrayList<PackedSequence>()
    for (k in qs.indices step setSize) {
      val chunk = k until minOf(qs.size, k + setSize)
      val text = prefix + suffixes[k]
      val reads = chunk.any { qs[it] is Read }
      val encoded = tokenizer.encodeWithOffsets(text)
      // attention.pack_set
      val ids = encoded.ids.toMutableList()
      val state0 = ids.size - tokenizer.encode(suffixes[k]).size
      val tags = MutableList(state0) { -1 } + MutableList(ids.size - state0) { 1 }
      val question = tags.toMutableList()
      val ends = ArrayList<IntArray>()
      ends.add(positions(layouts[k].second, ids.size - state0, ids.size - 1))
      for ((n, j) in chunk.drop(1).withIndex()) {
        val own = tokenizer.encode(header + suffixes[j])
        ids.addAll(own.toList())
        repeat(own.size) { question.add(n + 2) }
        ends.add(positions(layouts[j].second, own.size, ids.size - 1))
      }
      val map = if (reads) StatePositions.of(prompt, state, prefix, text, encoded) else null
      val parts = media + chunk.mapNotNull { voices[it] }
      out.add(PackedSequence(k, ids.toIntArray(), state0, question.toIntArray(), ends, map, parts))
    }
    return out
  }

  private fun positions(own: IntArray, length: Int, end: Int): IntArray {
    require(own.isNotEmpty() && own.last() == 0 && own.toList() == own.toSortedSet().toList() && own.all { it > -length && it <= 0 }) {
      "slots must be increasing positions inside their question, ending at its last token"
    }
    return IntArray(own.size) { end + own[it] }
  }

  /**
   * `model.system_one(state, questions, media)`: named questions over one state and its [media] in order (at most
   * one [Picture], any number of [AudioClip]s). A question whose `instructions` is an [AudioClip] is spoken.
   */
  fun systemOne(state: Any?, questions: Map<String, Any?>, media: List<Any> = emptyList()): HostResult {
    val timing = HostTiming()
    var t0 = System.nanoTime()
    val named = named(questions)
    val qs = named.values.toList()
    val pictures = media.filterIsInstance<Picture>()
    require(pictures.size <= 1) { "this app reads one picture at most; this request has ${pictures.size}" }
    val picture = pictures.firstOrNull()
    val stateMedia = media.map {
      when (it) {
        is Picture -> it
        is AudioClip -> plan(it)
        else -> throw IllegalArgumentException("a media item is a picture or a clip, not ${it.javaClass.simpleName}")
      }
    }
    val voices = questions.values.map { q -> ((q as? Map<*, *>)?.get("instructions") as? AudioClip)?.let { plan(it) } }
    val clips = stateMedia.filterIsInstance<ClipFeatures>() + voices.filterNotNull()
    val seqs = sequences(state, qs, voices, stateMedia)
    for (seq in seqs) {
      require(seq.ids.size <= decide.maxLength) {
        "a sequence of ${seq.ids.size} tokens (questions ${seq.firstQuestion + 1}..) is longer than the largest graph " +
          "on this phone (${decide.maxLength} tokens${decide.limitNote}): shorten the state, or ask fewer questions"
      }
    }
    if (clips.isNotEmpty()) {
      val backend = audio ?: throw IllegalStateException("a request with audio clips needs the audio encoder")
      val extractor = extractor!!
      for (seq in seqs) {
        for (c in seq.clips) {
          require(c.frames <= backend.maxFrames) {
            "a clip of ${c.frames} frames (${String.format(java.util.Locale.ROOT, "%.2f", c.samples.toDouble() / extractor.samplingRate)} s, " +
              "questions ${seq.firstQuestion + 1}..) is longer than the longest the audio encoder takes (${backend.maxFrames} " +
              "frames = ${PyFormat.g(extractor.seconds(backend.maxFrames))} s): the app never cuts a clip; shorten it"
          }
        }
      }
    }
    var t = System.nanoTime()
    timing.prepMs += (t - t0) / 1e6
    t0 = t
    // media.Reader._decoded: each clip resampled to the extractor's rate, then its log-mel
    for (c in clips) {
      c.features = extractor!!.features(Resample.resampled(c.clip.mono, c.clip.rate, extractor.samplingRate))
      check(c.features!!.size == extractor.settings.featureSize * c.frames) { "the features of ${c.clip.name} have the wrong frame count" }
    }
    t = System.nanoTime()
    timing.featureMs += (t - t0) / 1e6
    t0 = t
    var features: FloatArray? = null
    if (picture != null) {
      val backend = tower ?: throw IllegalStateException("a request with a picture needs the picture tower")
      features = backend.tower(picture)
      require(features.size == picture.tokens * HIDDEN) { "the tower returned ${features.size / HIDDEN} rows, expected ${picture.tokens}" }
      timing.towerCalls++
      timing.loadMs += backend.lastLoadMs
      t = System.nanoTime()
      timing.towerMs += (t - t0) / 1e6
      t0 = t
    }
    val probs = HashMap<Int, Any>()
    val clipTokens = IdentityHashMap<ClipFeatures, FloatArray>()
    // Shortest first, as the model's own host batches a request's sequences (equal lengths keep their order).
    for (seq in seqs.sortedBy { it.ids.size }) {
      val tokens = seq.ids.size
      // once a clip: a request's sequences share the state's clips
      val own = ArrayList<FloatArray>()
      for (c in seq.clips) {
        val out = clipTokens.getOrPut(c) {
          t = System.nanoTime()
          timing.prepMs += (t - t0) / 1e6
          t0 = t
          val backend = audio!!
          val rows = backend.audio(c.features!!, c.frames)
          require(rows.size == c.tokens * HIDDEN) {
            "the audio encoder returned ${rows.size / HIDDEN} rows for a clip of ${c.frames} frames, expected ${c.tokens}"
          }
          timing.audioCalls++
          timing.loadMs += backend.lastLoadMs
          t = System.nanoTime()
          timing.audioMs += (t - t0) / 1e6
          t0 = t
          rows
        }
        own.add(out)
      }
      val embeds = FloatArray(tokens * HIDDEN)
      var image = 0
      var sound = 0
      var clip = 0
      var row = 0
      for (p in 0 until tokens) {
        if (features != null && seq.ids[p] == imageTokenId) {
          System.arraycopy(features, image * HIDDEN, embeds, p * HIDDEN, HIDDEN)
          image++
        } else if (seq.ids[p] == audioTokenId && clip < own.size) {
          System.arraycopy(own[clip], row * HIDDEN, embeds, p * HIDDEN, HIDDEN)
          sound++
          row++
          if (row * HIDDEN == own[clip].size) {
            clip++
            row = 0
          }
        } else {
          table.row(seq.ids[p], embeds, p * HIDDEN)
        }
      }
      if (features != null) {
        require(image == picture!!.tokens) {
          "Image features and image tokens do not match, tokens: $image, features: ${picture.tokens}"
        }
      }
      val placeholders = seq.ids.count { it == audioTokenId }
      val rows = own.sumOf { it.size / HIDDEN }
      require(placeholders == rows && sound == rows) {
        "Placeholder tokens and features do not match: $placeholders tokens $audioTokenId, $rows features"
      }
      t = System.nanoTime()
      timing.prepMs += (t - t0) / 1e6
      t0 = t
      val hidden = decide.decide(embeds, tokens, seq.stateLength, seq.question)
      timing.decideCalls++
      timing.loadMs += decide.lastLoadMs
      timing.tokens += tokens
      t = System.nanoTime()
      timing.decideMs += (t - t0) / 1e6
      t0 = t
      read(seq, qs, hidden, probs)
      t = System.nanoTime()
      timing.readoutMs += (t - t0) / 1e6
      t0 = t
    }
    val answers = LinkedHashMap<String, LinkedHashMap<String, Any?>>()
    val context = prompt.stateText(state)
    for ((index, entry) in named.entries.withIndex()) {
      val q = entry.value
      answers[entry.key] = Api.answer(q, probs.getValue(index), if (q is Read) context else "", calibration)
    }
    timing.readoutMs += (System.nanoTime() - t0) / 1e6
    return HostResult(answers, seqs.sumOf { it.ids.size }, seqs, timing, clips)
  }

  /** Reads one sequence's slots: option codes through the readout tables, reads through the pointer. */
  private fun read(seq: PackedSequence, qs: List<Question>, hidden: FloatArray, probs: HashMap<Int, Any>) {
    for ((k, own) in seq.ends.withIndex()) {
      val index = seq.firstQuestion + k
      val q = qs[index]
      val slots = prompt.questionSlots(q)
      if (q is Read) {
        val map = requireNotNull(seq.positions) { "a read needs the state's positions" }
        val start = pointerLogits(hidden, own[0], map)
        val end = pointerLogits(hidden, own[1], map)
        val cap = SPAN_CAPS.getValue(q.type)
        val (at, found) = StatePositions.spans(start, end, cap)
        val read = map.read(at[0], at[1], cap)
        val span = read["span"] as IntArray?
        if (span != null) read["span"] = listOf(map.text.codePointCount(0, span[0]), map.text.codePointCount(0, span[1]))
        read["found"] = found
        probs[index] = read
        continue
      }
      val tempered = !prompt.typed(q)
      val rows = slots.indices.map { s -> readout(slots[s], hidden, own[s], tempered) }
      probs[index] =
        when {
          q is Choice && q.multiple -> DoubleArray(rows.size) { rows[it][0] }
          q is BoxQ || q is BoxesQ -> rows
          else -> rows[0]
        }
    }
  }

  /** One read slot's scores over the state's positions: float32 dot products / sqrt(d), -inf off the mask. */
  private fun pointerLogits(hidden: FloatArray, slot: Int, map: StatePositions): FloatArray {
    val mask = map.pointerMask
    val scale = sqrt(HIDDEN.toDouble()).toFloat()
    val out = FloatArray(mask.size)
    val q = slot * HIDDEN
    for (t in mask.indices) {
      if (!mask[t]) {
        out[t] = Float.NEGATIVE_INFINITY
        continue
      }
      var sum = 0.0
      val base = t * HIDDEN
      for (i in 0 until HIDDEN) sum += hidden[q + i].toDouble() * hidden[base + i].toDouble()
      out[t] = sum.toFloat() / scale
    }
    return out
  }

  /** One code slot's option probabilities: the best form of each option, / temperature when [tempered], softmax. */
  private fun readout(slot: Question, hidden: FloatArray, position: Int, tempered: Boolean): DoubleArray {
    val readout = prompt.readoutTable(slot)
    val rows = tableRows.getOrPut(readout.tokens.toList()) {
      val all = FloatArray(readout.tokens.size * HIDDEN)
      readout.tokens.forEachIndexed { i, id -> table.row(id, all, i * HIDDEN) }
      all
    }
    val base = position * HIDDEN
    val logits = FloatArray(readout.tokens.size)
    for (j in readout.tokens.indices) {
      var sum = 0.0
      val r = j * HIDDEN
      for (i in 0 until HIDDEN) sum += hidden[base + i].toDouble() * rows[r + i].toDouble()
      logits[j] = sum.toFloat()
    }
    val temperature = if (tempered && calibration != null) calibration.temperature(slot) else 1.0
    val scores = DoubleArray(readout.forms.size) { o -> readout.forms[o].maxOf { logits[it] }.toDouble() / temperature }
    val top = scores.max()
    val exps = DoubleArray(scores.size) { exp(scores[it] - top) }
    val total = exps.sum()
    return DoubleArray(scores.size) { exps[it] / total }
  }

  companion object {
    /** A read's longest span, in tokens (pointer.SPAN_CAPS). */
    val SPAN_CAPS = mapOf("text" to 512, "number" to 32, "date" to 32)

    /** The host over the model directory's provider/ files and table, with these backends. */
    fun load(modelDir: File, decide: DecideBackend, tower: TowerBackend?, audio: AudioBackend? = null): Host {
      val provider = File(modelDir, "provider")
      val tokenizer = Tokenizer(File(provider, "tokenizer.json"))
      @Suppress("UNCHECKED_CAST")
      val config = D1Json.parse(File(provider, "config.json").readBytes()) as Map<String, Any?>
      val codes = (config["d1_codes"] as JsonNumber?)?.toInt() ?: 1
      val numbers = config["d1_numbers"] == true
      val version = (config["d1_prompt"] as JsonNumber?)?.toInt() ?: 1
      require(config["d1_attention"] == "set") { "this host serves `set` attention (d1-2B's)" }
      require(config["d1_box_edges"] == null) { "this host reads box edges at the median (no d1_box_edges)" }
      val setSize = (config["d1_set_size"] as JsonNumber?)?.toInt() ?: 16
      val imageToken = (config["image_token_id"] as JsonNumber).toInt()
      require(tokenizer.tokenId("<|image|>") == imageToken) { "the tokenizer's image token is not config.json's image_token_id" }
      val calibrationFile = File(provider, "calibration.json")
      val calibration = if (calibrationFile.isFile) Calibration.load(calibrationFile) else null
      // The clips' feature extractor and markup (Lfm3OmniProcessor's), checked against the template and config.json
      @Suppress("UNCHECKED_CAST")
      val processor = D1Json.parse(File(provider, "processor_config.json").readBytes()) as Map<String, Any?>
      require(processor["processor_class"] == "Lfm3OmniProcessor") {
        "the app writes clips as Lfm3OmniProcessor does, not ${processor["processor_class"]}"
      }
      val extractor = FeatureExtractor(ExtractorSettings.of(processor))
      val markup =
        AudioMarkup(
          processor["audio_token"] as String? ?: "<|audio|>",
          processor["vision_start_token"] as String? ?: "<|media_start|>",
          processor["vision_end_token"] as String? ?: "<|media_end|>",
          processor["audio_markers"] as Boolean? ?: true,
        )
      for (token in listOf(markup.token, markup.start, markup.end)) {
        require(tokenizer.tokenId(token) != null && tokenizer.encode(token).size == 1) { "the tokenizer has no single token $token" }
      }
      require(File(provider, "chat_template.jinja").readText().contains("\"${markup.template}\"")) {
        "chat_template.jinja no longer writes ${markup.template} for a clip"
      }
      val audioToken = (config["audio_token_id"] as JsonNumber).toInt()
      require(tokenizer.tokenId(markup.token) == audioToken) { "the processor's audio token is not config.json's audio_token_id" }
      val table = EmbedTable(File(modelDir, "tables/embed_table.npy"))
      return Host(
        tokenizer, Prompt(tokenizer, codes, numbers, version), calibration, table, decide, tower, setSize, imageToken,
        audio, extractor, markup, audioToken,
      )
    }
  }
}
