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
  /** The GPU precision of the graph the question ran on. */
  val precision: KevPrecision,
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

/**
 * A graph file in `files/`: the graph, its size, the GPU precision it is compiled with and, for a
 * pair, whether it holds one copy of the weights (constant tensor sharing).
 */
class KevGraphFile(
  val graph: KevGraphKey,
  val bytes: Long,
  val precision: KevPrecision,
  val share: Boolean? = null,
)

/**
 * How a demo request ran: the `graph` mode asked for, the form taken, both predictions and what the
 * plan saw (the available memory and the resident graphs).
 */
class KevDemoPlan(
  val requested: KevGraphMode,
  val form: KevForm,
  val prediction: KevPrediction,
  val inputs: KevPlanInputs?,
)

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
  /**
   * "GPU FP32", "GPU FP16 (FP32 accum)", "GPU" (graphs at different precisions) or "CPU 4 threads".
   */
  val accelerator: String,
  /** The questions' GPU precision by its launch name, `fp32` or `fp16acc`, or `mixed`. */
  val precision: String,
  /** The `precision` extra of the launch, or null when each graph ran at its own default. */
  val precisionRequested: String?,
  /**
   * The largest window the questions ran on (the graph that holds the longest row), or the pair.
   */
  val graph: KevGraphFile,
  val form: KevForm,
  /** The row windows the questions ran on, in ascending order (empty for the pair). */
  val windowsUsed: List<Int>,
  /** The graphs compiled when the run ended: windows ascending, then the pair. */
  val resident: List<KevGraphFile>,
  /** The graphs compiled for this run's request, in order. */
  val compiled: List<KevGraphKey>,
  /** `ActivityManager.MemoryInfo.availMem` right before each of those compiles, in bytes. */
  val availableBytesBeforeCompile: List<Long>,
  /** The graphs closed for this run's request. */
  val closed: List<KevGraphKey>,
  /** A second graph was wanted but the available memory was below the limit. */
  val secondRefused: Boolean,
  val plan: KevDemoPlan,
  /** The pair's state call (null for rows). */
  val state: KevStateResult?,
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
 * The demo run JSON (`files/kev-demo-<epoch ms>.json`): device and runtime, the graphs, the plan,
 * the shown strings and timings of every question, its row (IDs, readout indices, sha256 of the
 * int32 IDs, the window it ran on) and where the screen drew the cards. Key names follow the demo
 * recording scripts; `graph` keeps the `file` / `L` / `bytes` of the largest window used (`L` is
 * null for the pair, whose `Ls` / `Lq` are under `pair`) and adds `form`, `precision` (that
 * graph's), `windows` (every window the questions ran on), `resident` (the compiled graphs with
 * their precision), `compiled` (each compile for this request with the available memory right
 * before it), `closed` and `second_refused`. The row IDs and indices are the causal row's (state +
 * branch) in both forms; a pair run adds `state` (`tokens`, `window` = Ls, `ms`) and each
 * question's `branch_len`. Each question has the `precision` of its graph.
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
      "plan",
      "state",
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
      "form",
      "window",
      "precision",
      "branch_len",
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
      "runtime" to
        linkedMapOf(
          "litert" to input.litert,
          "accelerator" to input.accelerator,
          "precision" to input.precision,
          "precision_requested" to input.precisionRequested,
        ),
      "graph" to
        linkedMapOf(
          "file" to input.graph.graph.file,
          "L" to (input.graph.graph as? KevGraphKey.Window)?.window,
          "bytes" to input.graph.bytes,
          "form" to input.form.wireName,
          "precision" to input.graph.precision.wireName,
          "share" to input.graph.share,
          "pair" to (input.graph.graph as? KevGraphKey.Pair)?.let { shape(it) },
          "windows" to input.windowsUsed,
          "resident" to input.resident.map { graph(it) },
          "compiled" to
            input.compiled.mapIndexed { index, graph ->
              key(graph).apply {
                put("avail_mem_bytes", input.availableBytesBeforeCompile.getOrNull(index))
              }
            },
          "closed" to input.closed.map { key(it) },
          "second_refused" to input.secondRefused,
        ),
      "plan" to
        linkedMapOf(
          "requested" to input.plan.requested.wireName,
          "form" to input.plan.form.wireName,
          "predicted_ms" to
            linkedMapOf(
              "rows" to input.plan.prediction.rowsMs,
              "pair" to input.plan.prediction.pairMs,
            ),
          "avail_mem_bytes" to input.plan.inputs?.availableBytes,
          "resident" to input.plan.inputs?.resident?.map { it.label },
          "share_mode" to input.plan.inputs?.pairShare?.wireName,
          "pair_shared_predicted" to input.plan.prediction.pairShared,
        ),
      "state" to
        input.state?.let {
          linkedMapOf("tokens" to it.tokens, "window" to it.window, "ms" to Math.round(it.ms))
        },
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
    linkedMapOf<String, Any?>("file" to graph.graph.file)
      .apply { putAll(key(graph.graph)) }
      .apply {
        put("bytes", graph.bytes)
        put("precision", graph.precision.wireName)
        if (graph.graph is KevGraphKey.Pair) put("share", graph.share)
      }

  /** A graph as `{"L"}` (window) or `{"Ls", "Lq"}` (pair). */
  private fun key(graph: KevGraphKey): LinkedHashMap<String, Any?> =
    when (graph) {
      is KevGraphKey.Window -> linkedMapOf("L" to graph.window)
      is KevGraphKey.Pair -> shape(graph)
    }

  private fun shape(pair: KevGraphKey.Pair): LinkedHashMap<String, Any?> =
    linkedMapOf("Ls" to pair.shape.stateLength, "Lq" to pair.shape.questionLength)

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
      "form" to result.form.wireName,
      "window" to result.window,
      "precision" to question.precision.wireName,
      "branch_len" to result.branchLength,
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
