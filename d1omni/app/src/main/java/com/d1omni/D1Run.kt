package com.d1omni

/** Where Decide's input came from: recorded with the microphone, picked from the phone, typed, or the bundled sample. */
enum class D1Source(val wireName: String) {
  RECORDED("recorded"),
  PICKED("picked"),
  TYPED("typed"),
  SAMPLE("sample"),
}

/** One resident graph: its file, bytes, where it runs, at which precision, its compile time. */
class D1RunGraph(
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

/** One answer's text as drawn: its size, its box on the screen (px), its lines and whether it was cut. */
class D1AnswerLayout(
  val qid: String,
  /** "answer" (the answer's word) or "prob" (its probability). */
  val part: String,
  val text: String,
  val sp: Float,
  val fontPx: Float,
  /** left, top, width, height in screen px. */
  val box: IntArray,
  val lines: Int,
  val overflow: Boolean,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "qid" to qid,
      "part" to part,
      "text" to text,
      "sp" to sp,
      "font_px" to fontPx,
      "box_px" to linkedMapOf("left" to box[0], "top" to box[1], "width" to box[2], "height" to box[3]),
      "lines" to lines,
      "overflow" to overflow,
    )
}

/** Where the screen drew the pill and the answers (filled on the device after the answers are drawn). */
class D1RunLayout(
  val screenWidth: Int,
  val screenHeight: Int,
  /** The pill: left, top, width, height and its left padding (the label starts after it). */
  val pill: IntArray?,
  val pillPadLeft: Int,
  val answers: List<D1AnswerLayout>,
  val density: Float,
  val fontScale: Float,
)

/** Everything one Decide's run JSON records (see [D1Run.build]). */
class D1RunInput(
  val input: D1Input,
  val source: D1Source,
  /** The media as the app read it: sha256, bytes, name, the recording's facts; null for a message. */
  val media: Map<String, Any?>?,
  /** The request's state: the message's text, or null. */
  val state: Any?,
  val work: D1Work,
  val shownMs: String,
  val deviceModel: String,
  val deviceManufacturer: String,
  val deviceShownAs: String,
  val androidRelease: String,
  val accelerator: String,
  val precision: LinkedHashMap<String, String?>,
  val precisionRequested: LinkedHashMap<String, String?>,
  val graphs: List<D1RunGraph>,
  val memoryAtReady: Map<String, Any?>,
  val engineLoadMs: Long,
  val warmupMs: Long,
  val airplaneMode: Boolean,
  val cgroup: String,
  val cgroupEnd: String,
  val layout: D1RunLayout?,
  val events: List<Map<String, Any?>>,
  val stateStart: Map<String, Any?>,
  val stateEnd: Map<String, Any?>,
)

/**
 * The run JSON of one Decide (`files/d1omni-run-<epoch ms>.json`), Android-free: the input and where it came from, its
 * media (sha256, bytes; a recording's length and saved wav; a photo's format and EXIF orientation), the questions as
 * asked, per question its row (ids, markers, P, bucket, the sha256 of the int32 ids), its unrounded probabilities,
 * `prompt.answer()`, the strings and ms on screen; the item's host steps and media graphs, its work (`item_total_ms`
 * as shown, `item_total_ns` unrounded); the device, runtime, resident graphs, memory at ready, airplane mode, cgroup;
 * where the pill and the answers were drawn. Every ms is a whole number that the screen, logcat and this file share.
 */
object D1Run {
  const val FORMAT = "d1omni-run/1"

  val KEYS =
    listOf(
      "run",
      "input",
      "kind",
      "source",
      "media",
      "state",
      "questions",
      "host_ms",
      "media_graph_ms",
      "prefix_rows",
      "item_total_ms",
      "item_total_ns",
      "shown_ms",
      "processing_started_wall_ms",
      "processing_ended_wall_ms",
      "media_info",
      "device",
      "runtime",
      "graphs",
      "engine_load_ms",
      "warmup_ms",
      "airplane_mode",
      "cgroup",
      "cgroup_end",
      "layout",
      "accelerators",
      "events",
      "state_start",
      "state_end",
    )

  val QUESTION_KEYS =
    listOf(
      "qid",
      "question",
      "type",
      "keys",
      "probs",
      "shown",
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

  val LAYOUT_KEYS = listOf("screen_px", "pill_palette", "pill_px", "answers", "density", "font_scale")

  fun question(q: D1RunQuestion): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "qid" to q.qid,
      "question" to D1Drafts.toJson(q.question),
      "type" to q.question.type.wireName,
      "keys" to D1Answers.keys(q.question),
      "probs" to q.probabilities.toList(),
      "shown" to linkedMapOf("answer" to q.shown.answer, "prob" to q.shown.prob),
      "answer" to q.answer,
      "row_ids" to q.ids.toList(),
      "row_len" to q.ids.size,
      "markers" to q.markers.toList(),
      "P" to q.prefixRows,
      "bucket" to q.bucket,
      "ids_sha256" to D1Answers.idsSha256(q.ids),
      "infer_ms" to q.inferMs,
      "total_ms" to q.totalMs,
      "backend" to q.backend.wireName,
      "precision" to q.precision?.wireName,
    )

  fun build(input: D1RunInput): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "run" to FORMAT,
      "input" to input.input.wireName,
      "kind" to input.input.kind.wireName,
      "source" to input.source.wireName,
      "media" to input.media,
      "state" to input.state,
      "questions" to input.work.questions.map { question(it) },
      "host_ms" to input.work.hostMs,
      "media_graph_ms" to input.work.mediaGraphMs,
      "prefix_rows" to input.work.prefixRows,
      "item_total_ms" to input.work.itemMs,
      "item_total_ns" to input.work.totalNanos,
      "shown_ms" to input.shownMs,
      "processing_started_wall_ms" to input.work.startedWall,
      "processing_ended_wall_ms" to input.work.endedWall,
      "media_info" to input.work.media,
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
      "graphs" to linkedMapOf("resident" to input.graphs.map { it.toJson() }, "memory_at_ready" to input.memoryAtReady),
      "engine_load_ms" to input.engineLoadMs,
      "warmup_ms" to input.warmupMs,
      "airplane_mode" to input.airplaneMode,
      "cgroup" to input.cgroup,
      "cgroup_end" to input.cgroupEnd,
      "layout" to layout(input.layout),
      "accelerators" to
        input.graphs.map {
          linkedMapOf(
            "graph" to it.graph,
            "backend" to it.backend.wireName,
            "precision" to it.precision?.wireName,
            "gpu_failure" to it.gpuFailure,
          )
        },
      "events" to input.events,
      "state_start" to input.stateStart,
      "state_end" to input.stateEnd,
    )

  private fun layout(layout: D1RunLayout?): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "screen_px" to layout?.let { listOf(it.screenWidth, it.screenHeight) },
      "pill_palette" to D1Text.PILL_PALETTE,
      "pill_px" to
        layout?.pill?.let {
          linkedMapOf("left" to it[0], "top" to it[1], "width" to it[2], "height" to it[3], "pad_left" to layout.pillPadLeft)
        },
      "answers" to layout?.answers?.map { it.toJson() },
      "density" to layout?.density,
      "font_scale" to layout?.fontScale,
    )
}
