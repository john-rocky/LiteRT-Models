package com.d1omni

/** One question of a demo run: its row, its probabilities, what the screen showed and its times. */
class D1DemoQuestion(
  val qid: String,
  val question: D1Question,
  /** The probabilities in option order ([yes, no] for a noul), float64, unrounded. */
  val probabilities: DoubleArray,
  val shown: D1Shown,
  /** The ms string on the row ("84 ms"), from [inferMs]. */
  val shownMs: String,
  /** `prompt.answer()` of [probabilities]. */
  val answer: Map<String, Any?>,
  val ids: IntArray,
  val markers: IntArray,
  val prefixRows: Int,
  val bucket: Int,
  /** The decision call: input writes + `run()` + read-back, whole ms (the row's number). */
  val inferMs: Long,
  /** The question from its inputs to its answer (inputs, call, read-out, answer), whole ms. */
  val totalMs: Long,
  /** Where its graph ran and at which GPU precision (null on the CPU). */
  val backend: D1Backend,
  val precision: D1Precision?,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "qid" to qid,
      "type" to question.type.wireName,
      "keys" to D1InboxAnswer.keys(question),
      "probs" to probabilities.toList(),
      "shown" to linkedMapOf("answer" to shown.answer, "prob" to shown.prob),
      "shown_ms" to shownMs,
      "answer" to answer,
      "row_ids" to ids.toList(),
      "row_len" to ids.size,
      "markers" to markers.toList(),
      "P" to prefixRows,
      "bucket" to bucket,
      "ids_sha256" to D1InboxAnswer.idsSha256(ids),
      "infer_ms" to inferMs,
      "total_ms" to totalMs,
      "backend" to backend.wireName,
      "precision" to precision?.wireName,
    )
}

/**
 * The speaker playback of an audio item: wall-clock times (epoch ms, one clock pair: the play request,
 * the `play()` call, the first moving head or output timestamp) and the clip.
 */
class D1DemoPlayback(
  val requestWallMs: Long,
  val playWallMs: Long,
  val headStartedWallMs: Long?,
  val durationMs: Long,
  val samples: Int,
  val startSource: String?,
  val end: String,
  val playToEndMs: Double,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "request_wall_ms" to requestWallMs,
      "play_wall_ms" to playWallMs,
      "head_started_wall_ms" to headStartedWallMs,
      "duration_ms" to durationMs,
      "samples" to samples,
      "start_source" to startSource,
      "end" to end,
      "play_to_end_ms" to playToEndMs,
    )
}

/** One item of a demo run: its media, the host steps' and the media graphs' ms, its questions. */
class D1DemoItem(
  val item: String,
  val kind: D1Kind,
  val mediaFile: String?,
  /** sha256 of the media bytes the app read, and where it read them (`files` or `bundled`). */
  val mediaSha256: String?,
  val mediaSource: String?,
  val playback: D1DemoPlayback?,
  /** Host steps in whole ms: audio {wav, mel, inputs}, image {decode, resize, patches, pos, unshuffle}, both {encode}. */
  val hostMs: LinkedHashMap<String, Long>,
  /** Media graph calls in whole ms: audio {audio}, image {tower, projector}. */
  val mediaGraphMs: LinkedHashMap<String, Long>,
  val prefixRows: Int,
  val shownHeader: String,
  /** The item's work: its host steps, media graphs and decision calls (no playback, no wait), whole ms. */
  val itemTotalMs: Long,
  /** Wall clock (epoch ms) when the item's work started (after the playback for a voice note). */
  val processingStartedWallMs: Long,
  val processingEndedWallMs: Long,
  val questions: List<D1DemoQuestion>,
  /** Media details: the clip's sizes, the picture's layout and crops. */
  val media: Map<String, Any?>,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "item" to item,
      "kind" to kind.wireName,
      "media_file" to mediaFile,
      "media_sha256" to mediaSha256,
      "media_source" to mediaSource,
      "playback" to playback?.toJson(),
      "host_ms" to hostMs,
      "media_graph_ms" to mediaGraphMs,
      "prefix_rows" to prefixRows,
      "shown_header" to shownHeader,
      "item_total_ms" to itemTotalMs,
      "processing_started_wall_ms" to processingStartedWallMs,
      "processing_ended_wall_ms" to processingEndedWallMs,
      "questions" to questions.map { it.toJson() },
      "media" to media,
    )
}

/** One resident graph: its file, bytes, where it runs, at which precision, its compile time. */
class D1DemoGraph(
  val graph: String,
  val file: String,
  val bytes: Long,
  val backend: D1Backend,
  val precision: D1Precision?,
  val compileMs: Double,
  val gpuFailure: String?,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "graph" to graph,
      "file" to file,
      "bytes" to bytes,
      "precision" to precision?.wireName,
      "backend" to backend.wireName,
      "compile_ms" to compileMs,
      "gpu_failure" to gpuFailure,
    )
}

/** Where the presentation drew the pill and each card's state indicator, in screen px (filled on the device). */
class D1DemoLayout(
  val screenWidth: Int,
  val screenHeight: Int,
  val contentTop: Int,
  val contentBottom: Int,
  /** The pill: left, top, width, height and its left padding (the label starts after it). */
  val pill: IntArray,
  val pillPadLeft: Int,
  /** Each card's indicator box by item: left, top, width, height. */
  val indicators: LinkedHashMap<String, IntArray>,
  val answerTextPx: Float,
  val density: Float,
  val fontScale: Float,
  val plan: D1InboxPlan?,
)

/** Everything a demo run JSON records (see [D1DemoRun.build]). */
class D1DemoRunInput(
  val fixtureId: String,
  val fixturePath: String,
  val fixtureSha256: String,
  val deviceModel: String,
  val deviceManufacturer: String,
  val deviceShownAs: String,
  val androidRelease: String,
  val accelerator: String,
  /** The GPU precision of each kind of graph (decide / audio / vision), null for one on the CPU. */
  val precision: LinkedHashMap<String, String?>,
  /** What the launch asked for: the `precision` extras by kind (null when not given). */
  val precisionRequested: LinkedHashMap<String, String?>,
  val graphs: List<D1DemoGraph>,
  val memoryAtReady: Map<String, Any?>,
  val engineLoadMs: Long,
  val warmupMs: Long,
  val title: String,
  val footerLines: List<String>,
  val delayMs: Long,
  val gapMs: Long,
  /** The footer's total: the items' [D1DemoItem.itemTotalMs] added up. */
  val requestTotalMs: Long,
  /** The same work unrounded: the items' nanoseconds added up. */
  val requestTotalNs: Long,
  val items: List<D1DemoItem>,
  val airplaneMode: Boolean,
  val cgroup: String,
  val cgroupEnd: String,
  val layout: D1DemoLayout?,
  val events: List<Map<String, Any?>>,
  val stateStart: Map<String, Any?>,
  val stateEnd: Map<String, Any?>,
)

/**
 * The demo run JSON (`files/d1omni-demo-<epoch ms>.json`), Android-free: the keys of the Kev demo's
 * run JSON (`KevDemoRun`) with an item level between the run and its questions. Every ms is a whole
 * number that the screen, logcat and this file share; probabilities are unrounded float64.
 * `request_total_ms` = the three items' work (each item's host steps, media graphs and decision
 * calls, without the playback and the presentation's waits) as the cards show it, each item rounded
 * to whole ms on its own, added up; `request_total_ns` = the same work unrounded.
 */
object D1DemoRun {
  /** Pill colours by state, and card indicator colours by state. */
  val PILL_PALETTE: Map<String, String> =
    linkedMapOf("ready" to "#5F6368", "playing" to "#E53935", "deciding" to "#1565C0", "done" to "#2E7D32")
  val CARD_PALETTE: Map<String, String> =
    linkedMapOf("pending" to "#5F6368", "running" to "#1565C0", "done" to "#2E7D32")

  val KEYS =
    listOf(
      "fixture_id",
      "fixture",
      "device",
      "runtime",
      "graphs",
      "engine_load_ms",
      "warmup_ms",
      "title",
      "footer_lines",
      "delay_ms",
      "gap_ms",
      "request_total_ms",
      "request_total_ns",
      "items",
      "airplane_mode",
      "cgroup",
      "cgroup_end",
      "layout",
      "accelerators",
      "events",
      "state_start",
      "state_end",
    )

  val ITEM_KEYS =
    listOf(
      "item",
      "kind",
      "media_file",
      "media_sha256",
      "media_source",
      "playback",
      "host_ms",
      "media_graph_ms",
      "prefix_rows",
      "shown_header",
      "item_total_ms",
      "processing_started_wall_ms",
      "processing_ended_wall_ms",
      "questions",
      "media",
    )

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
      "markers",
      "P",
      "bucket",
      "ids_sha256",
      "infer_ms",
      "total_ms",
      "backend",
      "precision",
    )

  val LAYOUT_KEYS =
    listOf(
      "screen_px",
      "content_px",
      "palette",
      "pill_palette",
      "pill_px",
      "cards",
      "density",
      "font_scale",
      "plan",
    )

  fun build(input: D1DemoRunInput): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "fixture_id" to input.fixtureId,
      "fixture" to linkedMapOf("id" to input.fixtureId, "path" to input.fixturePath, "sha256" to input.fixtureSha256),
      "device" to
        linkedMapOf(
          "model" to input.deviceModel,
          "manufacturer" to input.deviceManufacturer,
          "shown_as" to input.deviceShownAs,
          "android_release" to input.androidRelease,
        ),
      "runtime" to
        linkedMapOf(
          "litert" to D1Decider.LITERT_VERSION,
          "accelerator" to input.accelerator,
          "precision" to input.precision,
          "precision_requested" to input.precisionRequested,
        ),
      "graphs" to
        linkedMapOf(
          "resident" to input.graphs.map { it.toJson() },
          "memory_at_ready" to input.memoryAtReady,
        ),
      "engine_load_ms" to input.engineLoadMs,
      "warmup_ms" to input.warmupMs,
      "title" to input.title,
      "footer_lines" to input.footerLines,
      "delay_ms" to input.delayMs,
      "gap_ms" to input.gapMs,
      "request_total_ms" to input.requestTotalMs,
      "request_total_ns" to input.requestTotalNs,
      "items" to input.items.map { it.toJson() },
      "airplane_mode" to input.airplaneMode,
      "cgroup" to input.cgroup,
      "cgroup_end" to input.cgroupEnd,
      "layout" to layout(input.layout),
      "accelerators" to
        input.graphs.map {
          linkedMapOf("graph" to it.graph, "backend" to it.backend.wireName, "precision" to it.precision?.wireName,
            "gpu_failure" to it.gpuFailure)
        },
      "events" to input.events,
      "state_start" to input.stateStart,
      "state_end" to input.stateEnd,
    )

  private fun box(values: IntArray): LinkedHashMap<String, Any?> =
    linkedMapOf("left" to values[0], "top" to values[1], "width" to values[2], "height" to values[3])

  private fun layout(layout: D1DemoLayout?): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "screen_px" to layout?.let { listOf(it.screenWidth, it.screenHeight) },
      "content_px" to layout?.let { linkedMapOf("top" to it.contentTop, "bottom" to it.contentBottom) },
      "palette" to CARD_PALETTE,
      "pill_palette" to PILL_PALETTE,
      "pill_px" to layout?.let { box(it.pill).apply { put("pad_left", it.pillPadLeft) } },
      "cards" to
        layout?.indicators?.map { (item, values) ->
          linkedMapOf("item" to item, "indicator_px" to box(values), "text_px" to layout.answerTextPx)
        },
      "density" to layout?.density,
      "font_scale" to layout?.fontScale,
      "plan" to layout?.plan?.toJson(),
    )
}
