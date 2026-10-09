package com.d1omni

import android.content.Context
import android.graphics.BitmapFactory
import android.os.Build
import android.os.SystemClock
import androidx.compose.runtime.Immutable
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import java.io.File
import java.security.MessageDigest
import kotlin.math.roundToLong
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/** The pill beside the title: what the inbox demo is doing. */
enum class D1Pill(val wireName: String) {
  READY("ready"),
  PLAYING("playing"),
  DECIDING("deciding"),
  DONE("done"),
}

/** A card's state indicator: grey until its item starts, blue while it runs, green when answered. */
enum class D1CardState(val wireName: String) {
  PENDING("pending"),
  RUNNING("running"),
  DONE("done"),
  FAILED("failed"),
}

/** One answer row: the question's name, and once its call returned, its answer and ms. */
@Immutable data class D1RowUi(val qid: String, val shown: D1Shown? = null, val ms: Long? = null)

/**
 * One item's card: [header] ("Voice note · 8.7 s"), the media graph's ms once known, its state, the
 * voice note's playback (the sound's start in `System.nanoTime()` and its length; the bar follows),
 * the photo, the message, the rows and the total line.
 */
@Immutable
data class D1CardUi(
  val item: String,
  val kind: D1Kind,
  val header: String,
  val mediaMs: String? = null,
  val state: D1CardState = D1CardState.PENDING,
  val playStartNanos: Long? = null,
  val playDurationNanos: Long = 0L,
  val playEnded: Boolean = false,
  val thumbnail: ImageBitmap? = null,
  val message: String? = null,
  val rows: List<D1RowUi> = emptyList(),
  val total: String? = null,
)

/** The presentation screen of one inbox run (see `view/PresentationScreen`). */
@Immutable
data class D1PresentationUi(
  val runId: Long,
  val title: String,
  val pill: D1Pill,
  val cards: List<D1CardUi>,
  val footer: List<String>,
  /** What the layout measures before drawing: every header, word and footer line it can show. */
  val layoutInput: D1InboxLayoutInput,
  val failure: String? = null,
  val running: Boolean = true,
)

/** A fixture or media file the run read: its bytes, where they came from and their sha256. */
class D1Source(val bytes: ByteArray, val path: String, val source: String, val sha256: String)

/** What the engine's load recorded, for the run JSON. */
class D1EngineInfo(val loadMs: Long, val warmupMs: Long)

/**
 * The bundled demo files (`res/raw`): the fixture and its two media files, by the file name a
 * fixture or an autoplay intent gives.
 */
object D1Bundled {
  val RAW: Map<String, Int> =
    mapOf(
      "inbox_demo.json" to R.raw.inbox_demo,
      "aud_food_03.wav" to R.raw.aud_food_03,
      "img_dogs_01.png" to R.raw.img_dogs_01,
    )

  const val FIXTURE = "inbox_demo.json"

  /**
   * [name] read from `files/` (a plain name, or an absolute path inside `files/`), else the bundled
   * copy of that name; null when neither exists.
   */
  fun read(context: Context, name: String): D1Source? {
    val files = context.filesDir.canonicalFile
    val file =
      if (name.startsWith("/")) File(name).canonicalFile.takeIf { it.path.startsWith(files.path + File.separator) }
      else File(files, name)
    if (file != null && file.isFile) {
      val bytes = file.readBytes()
      return D1Source(bytes, file.path, "files", sha256(bytes))
    }
    val id = RAW[if (name.startsWith("/")) File(name).name else name] ?: return null
    val bytes = context.resources.openRawResource(id).use { it.readBytes() }
    return D1Source(bytes, "res/raw/${File(name).name}", "bundled", sha256(bytes))
  }

  fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}

/**
 * One run of the inbox demo on the presentation screen (an autoplay intent, or the Decide button):
 * [delayMs] after the intent arrived the presentation appears; [gapMs] before each item; a voice
 * note plays through the speaker first (the pill turns to ● PLAYING when the sound starts, the
 * card's bar follows it) and is answered after its last sample left the speaker; each item's work
 * (host steps, media graph, then one decision call per question, each row shown when its call
 * returns) runs on [D1Runtime.dispatcher]; then DONE with the total, the run JSON
 * (`files/d1omni-demo-<epoch ms>.json`) and `AUTOPLAY_DONE`. Every ms shown, logged and written is
 * one measurement rounded once.
 */
class D1AutoplayRunner(
  private val context: Context,
  private val engine: D1InboxEngine,
  private val info: D1EngineInfo,
  private val player: D1AudioPlayer,
  private val show: (D1PresentationUi?) -> Unit,
  private val update: ((D1PresentationUi) -> D1PresentationUi) -> Unit,
  private val layout: () -> D1DemoLayout?,
) {
  /** Thrown inside the run to end it with `failed <reason>`. */
  class Failure(reason: String) : Exception(reason)

  private val events = ArrayList<Map<String, Any?>>()
  private var startNanos = 0L

  private fun event(name: String, item: String? = null, extra: Map<String, Any?> = emptyMap()) {
    events.add(
      linkedMapOf<String, Any?>(
          "event" to name,
          "item" to item,
          "wall_ms" to System.currentTimeMillis(),
          "t_ms" to (System.nanoTime() - startNanos) / 1e6,
        )
        .apply { putAll(extra) }
    )
  }

  private fun card(item: String, change: (D1CardUi) -> D1CardUi) =
    update { ui -> ui.copy(cards = ui.cards.map { if (it.item == item) change(it) else it }) }

  /** Runs [fixtureName] (see [D1Bundled.read]); returns the run JSON file. */
  suspend fun run(fixtureName: String, delayMs: Long, gapMs: Long, receivedNanos: Long): File {
    startNanos = System.nanoTime()
    val stateStart = withContext(Dispatchers.IO) { D1Device.state(context) }
    val cgroup = stateStart["cgroup"] as String
    val fixtureSource =
      withContext(Dispatchers.IO) { D1Bundled.read(context, fixtureName) }
        ?: throw Failure("missing fixture $fixtureName (files/ or bundled)")
    val fixture =
      try {
        D1InboxFixture.parse(fixtureSource.bytes)
      } catch (failure: IllegalArgumentException) {
        throw Failure("fixture does not parse: ${failure.message}")
      }
    // Every media file read before the presentation: the voice note's length is in its header, the
    // photo is on its card. Reading and parsing the wav is part of the voice note's work (timed).
    class Prepared(val source: D1Source?, val samples: ShortArray?, val wavNanos: Long, val thumbnail: ImageBitmap?)
    val prepared = LinkedHashMap<String, Prepared>()
    for (item in fixture.items) {
      if (item.kind == D1Kind.TEXT) {
        prepared[item.item] = Prepared(null, null, 0L, null)
        continue
      }
      val name = requireNotNull(item.mediaFile)
      val source = withContext(Dispatchers.IO) { D1Bundled.read(context, name) } ?: throw Failure("missing media $name")
      if (source.sha256 != item.mediaSha256) {
        throw Failure("${item.item}: $name has sha256 ${source.sha256}, the fixture says ${item.mediaSha256}")
      }
      prepared[item.item] =
        if (item.kind == D1Kind.AUDIO) {
          val t0 = System.nanoTime()
          val samples = D1Wav.parse(source.bytes, name)
          Prepared(source, samples, System.nanoTime() - t0, null)
        } else {
          val bitmap =
            BitmapFactory.decodeByteArray(source.bytes, 0, source.bytes.size)
              ?: throw Failure("${item.item}: $name does not decode")
          Prepared(source, null, 0L, bitmap.asImageBitmap())
        }
    }
    val headers = fixture.items.associate { it.item to D1InboxText.header(it, prepared[it.item]?.samples?.size) }
    val backends = engine.backends()
    val accelerator = D1InboxText.accelerator(backends)
    val graphsLine =
      D1InboxText.graphsLine(engine.decide.resident, engine.decide.audio.resident, vision = true)
    val device = D1Device.marketName()
    val airplane = D1Device.airplaneMode(context)
    val footerStart = D1InboxText.footer(device, accelerator, graphsLine, null, airplane)
    val layoutInput =
      D1InboxLayoutInput.of(
        fixture,
        headers,
        footerStart.dropLast(1) + D1InboxText.footerTotalWidest(),
      )
    val cards =
      fixture.items.map { item ->
        D1CardUi(
          item.item,
          item.kind,
          headers.getValue(item.item),
          thumbnail = prepared[item.item]?.thumbnail,
          message = if (item.kind == D1Kind.TEXT) D1Prompt.serialize(item.state) else null,
          rows = item.questions.keys.map { D1RowUi(it) },
        )
      }
    // delay_ms counts from the intent's arrival to the presentation on screen.
    var waitedNanos = 0L
    suspend fun waitFor(ms: Long) {
      val t0 = System.nanoTime()
      if (ms > 0) delay(ms)
      waitedNanos += System.nanoTime() - t0
    }
    waitFor(delayMs - (SystemClock.elapsedRealtimeNanos() - receivedNanos) / NANOS_PER_MS)
    show(D1PresentationUi(System.currentTimeMillis(), fixture.title, D1Pill.READY, cards, footerStart, layoutInput))
    event("presentation")
    val items = ArrayList<D1DemoItem>()
    val itemNanos = ArrayList<Long>()
    for (item in fixture.items) {
      waitFor(gapMs)
      val prep = prepared.getValue(item.item)
      var playback: D1DemoPlayback? = null
      if (item.kind == D1Kind.AUDIO) playback = play(item, requireNotNull(prep.samples))
      update { it.copy(pill = D1Pill.DECIDING) }
      card(item.item) { it.copy(state = D1CardState.RUNNING, playEnded = true) }
      D1Demo.deciding(item.item)
      event("deciding", item.item)
      val result = withContext(D1Runtime.dispatcher) { work(item, prep.source, prep.samples, prep.wavNanos) }
      itemNanos.add(result.totalNanos)
      val totalMs = D1InboxText.itemMs(result.totalNanos)
      card(item.item) { it.copy(state = D1CardState.DONE, total = D1InboxText.itemTotal(item.questions.size, totalMs)) }
      D1Demo.itemDone(item.item, totalMs)
      event("item_done", item.item, mapOf("item_total_ms" to totalMs))
      items.add(result.toItem(item, prep.source, playback, headers.getValue(item.item), totalMs))
    }
    // The footer adds the cards' totals as shown; the run JSON also keeps the unrounded sum.
    val requestTotalMs = D1InboxText.requestTotalMs(itemNanos)
    val requestTotalNs = itemNanos.sum()
    val footer = D1InboxText.footer(device, accelerator, graphsLine, requestTotalMs, airplane)
    update { it.copy(pill = D1Pill.DONE, footer = footer, running = false) }
    event("done", extra = mapOf("request_total_ms" to requestTotalMs, "request_total_ns" to requestTotalNs))
    // Let the final frame reach the screen and the layout report arrive before it is written.
    delay(LAYOUT_SETTLE_MS)
    val stateEnd = withContext(Dispatchers.IO) { D1Device.state(context) }
    val run =
      D1DemoRun.build(
        D1DemoRunInput(
          fixtureId = fixture.id,
          fixturePath = fixtureSource.path,
          fixtureSha256 = fixtureSource.sha256,
          deviceModel = Build.MODEL,
          deviceManufacturer = Build.MANUFACTURER,
          deviceShownAs = device,
          androidRelease = Build.VERSION.RELEASE,
          accelerator = accelerator,
          precision = precisionByKind(),
          precisionRequested = LinkedHashMap(engine.precisions.requested),
          graphs = engine.graphs(),
          memoryAtReady = engine.memoryAtReady,
          engineLoadMs = info.loadMs,
          warmupMs = info.warmupMs,
          title = fixture.title,
          footerLines = footer,
          delayMs = delayMs,
          gapMs = gapMs,
          requestTotalMs = requestTotalMs,
          requestTotalNs = requestTotalNs,
          items = items,
          airplaneMode = airplane,
          cgroup = cgroup,
          cgroupEnd = stateEnd["cgroup"] as String,
          layout = layout(),
          events = events.toList(),
          stateStart = stateStart,
          stateEnd = stateEnd,
        )
      )
    return withContext(Dispatchers.IO) { D1Demo.writeRun(context, run) }
  }

  /** The GPU precision each kind of graph runs at (null for a kind on the CPU). */
  private fun precisionByKind(): LinkedHashMap<String, String?> {
    val graphs = engine.graphs()
    fun of(prefix: String): String? =
      graphs.filter { it.graph.startsWith(prefix) }.map { it.precision?.wireName }.distinct().singleOrNull()
    return linkedMapOf("decide" to of("decide_"), "audio" to of("audio_"), "vision" to of("vision_tower"))
  }

  /** Plays the voice note and waits for its last sample to leave the speaker. */
  private suspend fun play(item: D1InboxItem, samples: ShortArray): D1DemoPlayback {
    val durationNanos = samples.size * 1_000_000_000L / D1Audio.SAMPLE_RATE
    // One clock pair for the playback: a System.nanoTime() of the player maps to the wall clock through it.
    val requestWall = System.currentTimeMillis()
    val requestNanos = System.nanoTime()
    fun wall(nanos: Long): Long = requestWall + ((nanos - requestNanos) / NANOS_PER_MS_D).roundToLong()
    event("play", item.item)
    val playback =
      withContext(Dispatchers.IO) {
        player.playBlocking(samples) { start ->
          // The watcher thread: the sound has started (the AudioTimestamp's frame 0, or the moving head).
          update { ui ->
            ui.copy(
              pill = D1Pill.PLAYING,
              cards = ui.cards.map { if (it.item == item.item) it.copy(playStartNanos = start, playDurationNanos = durationNanos) else it },
            )
          }
          val headMs = ((start - player.lastPlayNanos) / NANOS_PER_MS_D).roundToLong()
          D1Demo.playing(item.item, headMs, wall(start))
          event("playing", item.item, mapOf("head_ms" to headMs))
        }
      }
    // "timeout" (the end was not seen within the clip's length + 3.2 s) is recorded, not fatal: the sound
    // has finished by then; demo/check_take.py fails such a run.
    event("playback_end", item.item, mapOf("end" to playback.end, "head_at_end" to playback.headAtEnd))
    return D1DemoPlayback(
      requestWall,
      wall(playback.playNanos),
      playback.startNanos?.let { wall(it) },
      samples.size * 1000L / D1Audio.SAMPLE_RATE,
      samples.size,
      playback.startSource,
      playback.end,
      (playback.endNanos - playback.playNanos) / NANOS_PER_MS_D,
    )
  }

  /** One item's work and its times (on [D1Runtime.dispatcher]). */
  private class Work(
    val totalNanos: Long,
    val startedWall: Long,
    val endedWall: Long,
    val hostMs: LinkedHashMap<String, Long>,
    val mediaGraphMs: LinkedHashMap<String, Long>,
    val prefixRows: Int,
    val questions: List<D1DemoQuestion>,
    val media: Map<String, Any?>,
  ) {
    fun toItem(item: D1InboxItem, source: D1Source?, playback: D1DemoPlayback?, header: String, totalMs: Long) =
      D1DemoItem(
        item.item,
        item.kind,
        item.mediaFile,
        source?.sha256,
        source?.source,
        playback,
        hostMs,
        mediaGraphMs,
        prefixRows,
        header,
        totalMs,
        startedWall,
        endedWall,
        questions,
        media,
      )
  }

  private fun work(item: D1InboxItem, source: D1Source?, samples: ShortArray?, wavNanos: Long): Work {
    val startedWall = System.currentTimeMillis()
    val start = System.nanoTime()
    val hostMs = LinkedHashMap<String, Long>()
    val mediaGraphMs = LinkedHashMap<String, Long>()
    val media = LinkedHashMap<String, Any?>()
    var prefix: FloatArray? = null
    var prefixRows = 0
    when (item.kind) {
      D1Kind.AUDIO -> {
        val audio = engine.decide.audio.audioPrefix(requireNotNull(samples))
        hostMs["wav"] = ms(wavNanos)
        hostMs["mel"] = ms(audio.waveformNanos + audio.melNanos)
        hostMs["inputs"] = ms(audio.inputsNanos + audio.rowsNanos)
        val graphMs = audio.call.totalMs.roundToLong()
        mediaGraphMs["audio"] = graphMs
        card(item.item) { it.copy(mediaMs = D1InboxText.mediaMs(item.kind, graphMs)) }
        prefix = audio.prefix
        prefixRows = audio.info.prefixRows
        media["info"] = audio.info.toJson()
        media["steps_ms"] = audio.times()
        media["audio_backend"] = audio.backend.wireName
        media["audio_precision"] = audio.precision?.wireName
      }
      D1Kind.IMAGE -> {
        val t0 = System.nanoTime()
        val decoded = D1Image.decode(requireNotNull(source).bytes)
        val decodeNanos = System.nanoTime() - t0
        val run = engine.vision.imagePrefix(decoded.rgb)
        hostMs["decode"] = ms(decodeNanos)
        hostMs["resize"] = ms(run.resizeNanos)
        hostMs["patches"] = ms(run.patchesNanos)
        hostMs["pos"] = ms(run.positionsNanos)
        hostMs["unshuffle"] = ms(run.unshuffleNanos)
        mediaGraphMs["tower"] = ms(run.towerNanos)
        mediaGraphMs["projector"] = ms(run.projectorNanos)
        val graphMs = ((run.towerNanos + run.projectorNanos) / NANOS_PER_MS_D).roundToLong()
        card(item.item) { it.copy(mediaMs = D1InboxText.mediaMs(item.kind, graphMs)) }
        prefix = run.prefix
        prefixRows = run.rows
        media["prefix"] = run.toJson()
        media["hw"] = listOf(decoded.rgb.height, decoded.rgb.width)
        media["png_chunks_stripped"] = decoded.strippedChunks
        media["bitmap_color_space"] = decoded.colorSpace
        media["tower_backend"] = engine.vision.towerBackend.wireName
        media["projector_backend"] = engine.vision.projectorBackend.wireName
      }
      D1Kind.TEXT -> Unit
    }
    val t1 = System.nanoTime()
    val rows = engine.rows(item, prefixRows)
    hostMs["encode"] = ms(System.nanoTime() - t1)
    val questions = ArrayList<D1DemoQuestion>()
    for ((name, row) in item.questions.keys.zip(rows)) {
      val call = engine.question(row, prefix)
      val inferMs = call.call.totalMs.roundToLong()
      val shown = D1InboxAnswer.shown(row.question, call.probabilities)
      card(item.item) { c -> c.copy(rows = c.rows.map { if (it.qid == name) it.copy(shown = shown, ms = inferMs) else it }) }
      D1Demo.questionDone(item.item, name, inferMs)
      questions.add(
        D1DemoQuestion(
          name,
          row.question,
          call.probabilities,
          shown,
          D1InboxText.rowMs(inferMs),
          call.answer,
          row.ids,
          row.markers,
          row.prefixRows,
          call.bucket,
          inferMs,
          ms(call.totalNanos),
          call.backend,
          call.precision,
        )
      )
    }
    val totalNanos = wavNanos + (System.nanoTime() - start)
    return Work(totalNanos, startedWall, System.currentTimeMillis(), hostMs, mediaGraphMs, prefixRows, questions, media)
  }

  private fun ms(nanos: Long): Long = (nanos / NANOS_PER_MS_D).roundToLong()

  companion object {
    private const val NANOS_PER_MS = 1_000_000L
    private const val NANOS_PER_MS_D = 1e6
    private const val LAYOUT_SETTLE_MS = 400L

    /**
     * The untimed pass after the load: every item of [fixture] through its work once (no screen, no
     * playback), so that the timed run starts with the JIT and the GPU warm. Returns its ms.
     */
    fun warmup(context: Context, engine: D1InboxEngine, fixture: D1InboxFixture): Long {
      val start = System.nanoTime()
      for (item in fixture.items) {
        var prefix: FloatArray? = null
        var rows = 0
        when (item.kind) {
          D1Kind.AUDIO -> {
            val source = requireNotNull(D1Bundled.read(context, requireNotNull(item.mediaFile)))
            val audio = engine.decide.audio.audioPrefix(D1Wav.parse(source.bytes, item.mediaFile))
            prefix = audio.prefix
            rows = audio.info.prefixRows
          }
          D1Kind.IMAGE -> {
            val source = requireNotNull(D1Bundled.read(context, requireNotNull(item.mediaFile)))
            val run = engine.vision.imagePrefix(D1Image.decode(source.bytes).rgb)
            prefix = run.prefix
            rows = run.rows
          }
          D1Kind.TEXT -> Unit
        }
        for (row in engine.rows(item, rows)) engine.question(row, prefix)
      }
      return ((System.nanoTime() - start) / NANOS_PER_MS_D).roundToLong()
    }
  }
}
