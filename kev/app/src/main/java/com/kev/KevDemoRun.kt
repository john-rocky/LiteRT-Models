package com.kev

/**
 * One question of a demo run: the result, its answer, the strings on screen and whole-ms phases.
 */
class KevDemoQuestion(
  val result: KevQuestionResult,
  /** The question's `to_answers` entry. */
  val answer: Map<*, *>,
  /** The numbers on the card ([KevAnswerView.shownCompact]). */
  val shown: Map<String, Any?>,
  /** The ms text on the card, from [inferMs]. */
  val shownMs: String,
  val tokenizeMs: Long,
  /** Input writes + `run()` + read-back, the card's number. */
  val inferMs: Long,
  val headMs: Long,
  /** From the card turning to running until its answer. */
  val totalMs: Long,
)

/** Where the presentation screen drew things, in screen pixels (filled on the device). */
class KevDemoLayout(
  val screenWidth: Int,
  val screenHeight: Int,
  /** Top of the title and bottom of the footer. */
  val contentTop: Int,
  val contentBottom: Int,
  val cards: List<Card>,
  val ticketTextPx: Float,
  val density: Float,
  val fontScale: Float,
) {
  /** A card's state indicator box and the size of its answer text. */
  class Card(
    val qid: String,
    val left: Int,
    val top: Int,
    val width: Int,
    val height: Int,
    val textPx: Float,
  )
}

/** A graph file in `files/`: its name, window and size. */
class KevGraphFile(val file: String, val window: Int, val bytes: Long)

/** Everything a demo run JSON records. */
class KevDemoRunInput(
  val fixtureId: String,
  val fixturePath: String,
  val deviceModel: String,
  val deviceManufacturer: String,
  /** The device name in the footer. */
  val deviceShownAs: String,
  val deviceAndroidRelease: String,
  val litert: String,
  /** "GPU FP32" or "CPU 4 threads". */
  val accelerator: String,
  /** The largest window the questions ran on: the graph that holds the longest row. */
  val graph: KevGraphFile,
  /** The windows the questions ran on, in ascending order. */
  val windowsUsed: List<Int>,
  /** The graphs compiled when the run ended, by ascending window. */
  val resident: List<KevGraphFile>,
  /** The windows compiled for this run's request, in order. */
  val compiled: List<Int>,
  /** `ActivityManager.MemoryInfo.availMem` right before each of those compiles, in bytes. */
  val availableBytesBeforeCompile: List<Long>,
  /** The windows closed for this run's request. */
  val closed: List<Int>,
  /** A second graph was wanted but the available memory was below the limit. */
  val secondRefused: Boolean,
  /** Tokenizer + head + graph compile when the engine became ready. */
  val engineLoadMs: Long,
  /** The untimed full pass after loading. */
  val warmupMs: Long,
  val title: String,
  val footerLines: List<String>,
  val delayMs: Long,
  val gapMs: Long,
  val tokenizeMs: Long,
  /** From the start of tokenizing to the last head, without the presentation waits. */
  val requestTotalMs: Long,
  val questions: List<KevDemoQuestion>,
  val airplaneMode: Boolean,
  /** The cpuset line of `/proc/self/cgroup` when the run started and when it ended. */
  val cgroup: String,
  val cgroupEnd: String,
  val layout: KevDemoLayout?,
)

/**
 * The demo run JSON (`files/kev-demo-<epoch ms>.json`): device and runtime, the graphs, the shown
 * strings and timings of every question, its row (IDs, readout indices, sha256 of the int32 IDs,
 * the window it ran on) and where the screen drew the cards. Key names follow the demo recording
 * scripts; `graph` keeps the `file` / `L` / `bytes` of the largest window used and adds `windows`
 * (every window the questions ran on), `resident` (the compiled graphs), `compiled` (each compile
 * for this request with the available memory right before it), `closed` and `second_refused`.
 */
object KevDemoRun {
  /** Card indicator colours: pending grey, running blue, done green. */
  val PALETTE: Map<String, String> =
    linkedMapOf("pending" to "#5F6368", "running" to "#1565C0", "done" to "#2E7D32")

  /** Top-level keys every run JSON has. */
  val KEYS =
    listOf(
      "fixture_id",
      "fixture",
      "device",
      "runtime",
      "graph",
      "engine_load_ms",
      "warmup_ms",
      "title",
      "footer_lines",
      "delay_ms",
      "gap_ms",
      "tokenize_ms",
      "request_total_ms",
      "questions",
      "airplane_mode",
      "cgroup",
      "cgroup_end",
      "layout",
    )

  /** Keys of every `questions[]` entry. */
  val QUESTION_KEYS =
    listOf(
      "qid",
      "type",
      "keys",
      "probs",
      "shown",
      "shown_ms",
      "answer",
      "row_ids",
      "row_len",
      "decide_idx",
      "opt_idx",
      "window",
      "ids_sha256",
      "tokenize_ms",
      "infer_ms",
      "head_ms",
      "total_ms",
    )

  /** Keys of `layout` (pixel values exist only on the device). */
  val LAYOUT_KEYS =
    listOf("screen_px", "content_px", "palette", "cards", "ticket_text_px", "density", "font_scale")

  fun build(input: KevDemoRunInput): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "fixture_id" to input.fixtureId,
      "fixture" to linkedMapOf("id" to input.fixtureId, "path" to input.fixturePath),
      "device" to
        linkedMapOf(
          "model" to input.deviceModel,
          "manufacturer" to input.deviceManufacturer,
          "shown_as" to input.deviceShownAs,
          "android_release" to input.deviceAndroidRelease,
        ),
      "runtime" to linkedMapOf("litert" to input.litert, "accelerator" to input.accelerator),
      "graph" to
        linkedMapOf(
          "file" to input.graph.file,
          "L" to input.graph.window,
          "bytes" to input.graph.bytes,
          "windows" to input.windowsUsed,
          "resident" to input.resident.map { graph(it) },
          "compiled" to
            input.compiled.mapIndexed { index, window ->
              linkedMapOf(
                "L" to window,
                "avail_mem_bytes" to input.availableBytesBeforeCompile.getOrNull(index),
              )
            },
          "closed" to input.closed,
          "second_refused" to input.secondRefused,
        ),
      "engine_load_ms" to input.engineLoadMs,
      "warmup_ms" to input.warmupMs,
      "title" to input.title,
      "footer_lines" to input.footerLines,
      "delay_ms" to input.delayMs,
      "gap_ms" to input.gapMs,
      "tokenize_ms" to input.tokenizeMs,
      "request_total_ms" to input.requestTotalMs,
      "questions" to input.questions.map { question(it) },
      "airplane_mode" to input.airplaneMode,
      "cgroup" to input.cgroup,
      "cgroup_end" to input.cgroupEnd,
      "layout" to layout(input.layout),
    )

  private fun graph(graph: KevGraphFile): LinkedHashMap<String, Any?> =
    linkedMapOf("file" to graph.file, "L" to graph.window, "bytes" to graph.bytes)

  private fun question(question: KevDemoQuestion): LinkedHashMap<String, Any?> {
    val result = question.result
    val row = result.row
    return linkedMapOf(
      "qid" to result.meta.id,
      "type" to result.meta.type.wireName,
      "keys" to result.meta.keys,
      "probs" to result.probabilities,
      "shown" to question.shown,
      "shown_ms" to question.shownMs,
      "answer" to question.answer,
      "row_ids" to row.ids,
      "row_len" to row.length,
      "decide_idx" to row.decideIndex,
      "opt_idx" to row.optionIndices,
      "window" to result.window,
      "ids_sha256" to KevPipeline.idsSha256(row.ids),
      "tokenize_ms" to question.tokenizeMs,
      "infer_ms" to question.inferMs,
      "head_ms" to question.headMs,
      "total_ms" to question.totalMs,
    )
  }

  private fun layout(layout: KevDemoLayout?): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "screen_px" to layout?.let { listOf(it.screenWidth, it.screenHeight) },
      "content_px" to
        layout?.let { linkedMapOf("top" to it.contentTop, "bottom" to it.contentBottom) },
      "palette" to PALETTE,
      "cards" to
        layout?.cards?.map {
          linkedMapOf(
            "qid" to it.qid,
            "indicator_px" to
              linkedMapOf(
                "left" to it.left,
                "top" to it.top,
                "width" to it.width,
                "height" to it.height,
              ),
            "text_px" to it.textPx,
          )
        },
      "ticket_text_px" to layout?.ticketTextPx,
      "density" to layout?.density,
      "font_scale" to layout?.fontScale,
    )
}
