package com.kev

/**
 * The Android-free part of the timing protocol: on a row graph, [WARMUP_CALLS] untimed calls, then
 * [TIMED_REPEATS] timed calls (a `request` set: that many requests of its rows back to back), where
 * a call is input writes + `run()` + read-back of `hidden`; on the shared-state pair, the same
 * counts of requests (the state call once, then one question step per row); and the request path of
 * one request from its text (tokenize, the plan's graph calls, head, `to_answers`). [shouldStop] is
 * asked after every call; once it says yes, the remaining calls are left out. Each call (or
 * request) also records its start on [wallClock] (the device's wall clock in milliseconds) and the
 * GPU clock ceiling [ceiling] reads right before it (a pair request: also right after it), next to
 * its ms; both are read outside the timed interval.
 */
class KevTimingCore(
  private val pipeline: KevPipeline,
  /** The row graph the row sets run on (null when the run times the pair). */
  private val runner: RowRunner?,
  private val wallClock: () -> Long = System::currentTimeMillis,
  /** The GPU clock ceiling in MHz, or null when it cannot be read. */
  private val ceiling: () -> Int? = { null },
  private val shouldStop: () -> Boolean,
) {
  var stoppedEarly = false
    private set

  /** One set of `timing_rows.json` (its L must be the row graph's window). */
  fun timeSet(set: KevTimingSet): LinkedHashMap<String, Any?> {
    val runner = requireNotNull(runner) { "No row graph" }
    require(set.window == runner.length) {
      "Set ${set.name} is for L${set.window}, the graph is L${runner.length}"
    }
    val padded = set.rows.map { KevPaddedRow.of(it.ids, runner.length) to it.ids.size }
    val warmup = ArrayList<Double>()
    val warmupStarts = ArrayList<Long>()
    val warmupCeilings = ArrayList<Int?>()
    var finite = true
    for (call in 0 until WARMUP_CALLS) {
      if (stoppedEarly) break
      val (inputs, length) = padded[call % padded.size]
      warmupCeilings.add(ceiling())
      warmupStarts.add(wallClock())
      val (ms, rowFinite) = call(inputs, length)
      warmup.add(ms)
      finite = finite && rowFinite
    }
    val perCall = ArrayList<Double>()
    val callStarts = ArrayList<Long>()
    val callCeilings = ArrayList<Int?>()
    val perRequest = ArrayList<Double>()
    for (repeat in 0 until TIMED_REPEATS) {
      if (stoppedEarly) break
      var request = 0.0
      var complete = true
      for ((inputs, length) in padded) {
        if (stoppedEarly) {
          complete = false
          break
        }
        callCeilings.add(ceiling())
        callStarts.add(wallClock())
        val (ms, rowFinite) = call(inputs, length)
        finite = finite && rowFinite
        perCall.add(ms)
        request += ms
      }
      if (complete) perRequest.add(request)
    }
    return linkedMapOf(
      "kind" to set.kind,
      "L" to set.window,
      "synthetic" to set.synthetic,
      "rows" to
        set.rows.map {
          linkedMapOf(
            "key" to it.key,
            "row_len" to it.ids.size,
            "ids_sha256" to KevPipeline.idsSha256(it.ids),
          )
        },
      "warmup_ms" to warmup,
      "warmup_starts_ms" to warmupStarts,
      "warmup_max_clock_mhz" to warmupCeilings,
      "per_call_ms" to KevStats.of(perCall)?.toJson(),
      "per_request_ms" to (if (set.rows.size > 1) KevStats.of(perRequest)?.toJson() else null),
      "calls_ms" to perCall,
      "call_starts_ms" to callStarts,
      "call_max_clock_mhz" to callCeilings,
      "finite" to finite,
    )
  }

  /**
   * One set of `timing_rows.json` on the pair (its L is not used): a `request` set is one request
   * whose rows share their state part (the row before its `[question]` token), a `single` set is
   * one request per row. [WARMUP_CALLS] untimed requests, then [TIMED_REPEATS] passes over the
   * set's requests; a request = the state call, then one question step per row.
   */
  fun timePairSet(set: KevTimingSet, pair: PairRunner): LinkedHashMap<String, Any?> {
    val requests = pairRequests(set, pair)
    val warmup = ArrayList<Double>()
    val warmupStarts = ArrayList<Long>()
    val warmupCeilings = ArrayList<Int?>()
    var finite = true
    for (call in 0 until WARMUP_CALLS) {
      if (stoppedEarly) break
      val request = pairRequest(requests[call % requests.size], pair)
      warmup.add(request.totalMs)
      warmupStarts.add(request.startMs)
      warmupCeilings.add(request.ceilingBefore)
      finite = finite && request.finite
    }
    val perRequest = ArrayList<Double>()
    val requestStarts = ArrayList<Long>()
    val requestCeilings = ArrayList<Int?>()
    val requestCeilingsAfter = ArrayList<Int?>()
    val perState = ArrayList<Double>()
    val perQuestion = ArrayList<Double>()
    val questionStarts = ArrayList<Long>()
    for (repeat in 0 until TIMED_REPEATS) {
      for (split in requests) {
        if (stoppedEarly) break
        val request = pairRequest(split, pair)
        finite = finite && request.finite
        if (request.complete) {
          perRequest.add(request.totalMs)
          requestStarts.add(request.startMs)
          requestCeilings.add(request.ceilingBefore)
          requestCeilingsAfter.add(request.ceilingAfter)
          perState.add(request.stateMs)
        }
        perQuestion.addAll(request.questionMs)
        questionStarts.addAll(request.questionStartsMs)
      }
    }
    return linkedMapOf(
      "kind" to set.kind,
      "form" to KevForm.PAIR.wireName,
      "Ls" to pair.stateLength,
      "Lq" to pair.questionLength,
      "synthetic" to set.synthetic,
      "rows" to
        set.rows.map {
          linkedMapOf(
            "key" to it.key,
            "row_len" to it.ids.size,
            "ids_sha256" to KevPipeline.idsSha256(it.ids),
          )
        },
      "requests" to
        requests.map { linkedMapOf("state_len" to it.state.size, "branch_lens" to it.lengths) },
      "warmup_request_ms" to warmup,
      "warmup_starts_ms" to warmupStarts,
      "warmup_max_clock_mhz" to warmupCeilings,
      "request_ms" to KevStats.of(perRequest)?.toJson(),
      "state_ms" to KevStats.of(perState)?.toJson(),
      "question_ms" to KevStats.of(perQuestion)?.toJson(),
      "request_calls_ms" to perRequest,
      "request_starts_ms" to requestStarts,
      "request_max_clock_mhz" to requestCeilings,
      "request_max_clock_mhz_after" to requestCeilingsAfter,
      "state_calls_ms" to perState,
      "question_calls_ms" to perQuestion,
      "question_starts_ms" to questionStarts,
      "finite" to finite,
    )
  }

  /** A timing set's request split for the pair: the state part and each row's branch. */
  private class PairSplit(val state: IntArray, val branches: List<IntArray>) {
    val lengths: List<Int>
      get() = branches.map { it.size }
  }

  private fun pairRequests(set: KevTimingSet, pair: PairRunner): List<PairSplit> {
    val splits =
      set.rows.map { row ->
        val cut = row.ids.indexOf(KevEncoder.QUESTION_ID)
        require(cut > 0) { "Row ${row.key} has no question token" }
        row.ids.copyOf(cut) to row.ids.copyOfRange(cut, row.ids.size)
      }
    val requests =
      if (set.kind == "request") {
        require(splits.all { it.first.contentEquals(splits[0].first) }) {
          "Set ${set.name}: the rows do not share one state"
        }
        listOf(PairSplit(splits[0].first, splits.map { it.second }))
      } else {
        splits.map { PairSplit(it.first, listOf(it.second)) }
      }
    for (request in requests) {
      require(request.state.size <= pair.stateLength) {
        "Set ${set.name}: a state is ${request.state.size} tokens > Ls ${pair.stateLength}"
      }
      require(request.branches.all { it.size <= pair.questionLength }) {
        "Set ${set.name}: a branch is over Lq ${pair.questionLength}"
      }
    }
    return requests
  }

  /**
   * One timed pair request: the state call, then each question step; its start on the wall clock
   * and the GPU ceiling read right before and right after it.
   */
  private class PairCall(
    val ceilingBefore: Int?,
    val ceilingAfter: Int?,
    val startMs: Long,
    val totalMs: Double,
    val stateMs: Double,
    val questionMs: List<Double>,
    val questionStartsMs: List<Long>,
    val finite: Boolean,
    val complete: Boolean,
  )

  private fun pairRequest(split: PairSplit, pair: PairRunner): PairCall {
    val state = KevPaddedRow.of(split.state, pair.stateLength)
    val ceilingBefore = ceiling()
    val startMs = wallClock()
    val start = System.nanoTime()
    pair.runState(state.ids, state.valid)
    val stateEnd = System.nanoTime()
    val questionMs = ArrayList<Double>()
    val questionStarts = ArrayList<Long>()
    val outputs = ArrayList<Pair<FloatArray, Int>>()
    var complete = true
    for (branch in split.branches) {
      if (stoppedEarly) {
        complete = false
        break
      }
      val inputs = KevPaddedRow.of(branch, pair.questionLength)
      questionStarts.add(wallClock())
      val questionStart = System.nanoTime()
      val hidden = pair.runQuestion(inputs.ids, inputs.valid)
      questionMs.add(KevPipeline.millis(System.nanoTime() - questionStart))
      outputs.add(hidden to branch.size)
      if (shouldStop()) stoppedEarly = true
    }
    val totalMs = KevPipeline.millis(System.nanoTime() - start)
    val ceilingAfter = ceiling()
    // Finiteness on the real positions, after the clock stops.
    val finite = outputs.all { (hidden, length) ->
      KevPipeline.nonFiniteCount(hidden, 0, length * KevPointerHead.HIDDEN_SIZE) == 0
    }
    return PairCall(
      ceilingBefore,
      ceilingAfter,
      startMs,
      totalMs,
      KevPipeline.millis(stateEnd - start),
      questionMs,
      questionStarts,
      finite,
      complete,
    )
  }

  /** [request] from its text with every question on the row graph. */
  fun timeRequestPath(id: String, request: KevRequest): LinkedHashMap<String, Any?> {
    val runner = requireNotNull(runner) { "No row graph" }
    val rows = pipeline.prepare(request).rows
    if (rows.maxOf { it.length } > runner.length)
      return linkedMapOf("record" to id, "skipped" to "a row is over L${runner.length}")
    return timeRequestPath(id, request, KevRunners.Rows(List(rows.size) { runner }))
  }

  /**
   * [request] from its text on [runners] (the graphs its plan compiled): tokenize, the state call
   * for the pair, one call and the head per question, `to_answers`.
   */
  fun timeRequestPath(
    id: String,
    request: KevRequest,
    runners: KevRunners,
  ): LinkedHashMap<String, Any?> {
    val prepared0 = pipeline.prepare(request)
    val rowLengths = prepared0.rows.map { it.length }
    val warmup = ArrayList<Double>()
    val warmupStarts = ArrayList<Long>()
    val warmupCeilings = ArrayList<Int?>()
    val total = ArrayList<Double>()
    val starts = ArrayList<Long>()
    val ceilings = ArrayList<Int?>()
    val tokenize = ArrayList<Double>()
    val state = ArrayList<Double>()
    val infer = ArrayList<Double>()
    val head = ArrayList<Double>()
    var lastAnswers: Map<String, Any?>? = null
    for (iteration in 0 until WARMUP_CALLS + TIMED_REPEATS) {
      if (stoppedEarly) break
      val startCeiling = ceiling()
      val startMs = wallClock()
      val start = System.nanoTime()
      val prepared = pipeline.prepare(request)
      val result = pipeline.runRequest(prepared, runners)
      val answers = pipeline.answers(prepared, result.questions)
      val ms = KevPipeline.millis(System.nanoTime() - start)
      if (shouldStop()) stoppedEarly = true
      if (iteration < WARMUP_CALLS) {
        warmup.add(ms)
        warmupStarts.add(startMs)
        warmupCeilings.add(startCeiling)
        continue
      }
      total.add(ms)
      starts.add(startMs)
      ceilings.add(startCeiling)
      tokenize.add(prepared.tokenizeMs)
      result.state?.let { state.add(it.ms) }
      infer.addAll(result.questions.map { it.inferMs })
      head.addAll(result.questions.map { it.headMs })
      lastAnswers = answers
    }
    val pair = runners is KevRunners.Pair
    return linkedMapOf(
      "record" to id,
      "questions" to request.questions.size,
      "form" to (if (pair) KevForm.PAIR else KevForm.ROW).wireName,
      "row_lens" to rowLengths,
      "state_len" to prepared0.encoded.stateIds.size,
      "branch_lens" to prepared0.encoded.branches.map { it.ids.size },
      "steps" to
        if (pair) {
          "tokenize the request, the state call, then per question: pad, question step, head; " +
            "then to_answers"
        } else {
          "tokenize the request, then per question: pad, graph call, head; then to_answers"
        },
      "warmup_ms" to warmup,
      "warmup_starts_ms" to warmupStarts,
      "warmup_max_clock_mhz" to warmupCeilings,
      "request_ms" to KevStats.of(total)?.toJson(),
      "request_calls_ms" to total,
      "request_starts_ms" to starts,
      "request_max_clock_mhz" to ceilings,
      "tokenize_ms" to KevStats.of(tokenize)?.toJson(),
      "state_ms" to KevStats.of(state)?.toJson(),
      "infer_ms_per_call" to KevStats.of(infer)?.toJson(),
      "head_ms_per_question" to KevStats.of(head)?.toJson(),
      "last_answers" to lastAnswers,
    )
  }

  /** One timed call; finiteness is checked on the real positions after the clock stops. */
  private fun call(inputs: KevPaddedRow, length: Int): Pair<Double, Boolean> {
    val runner = requireNotNull(runner) { "No row graph" }
    val start = System.nanoTime()
    val hidden = runner.run(inputs.ids, inputs.valid)
    val ms = KevPipeline.millis(System.nanoTime() - start)
    val finite = KevPipeline.nonFiniteCount(hidden, 0, length * KevPointerHead.HIDDEN_SIZE) == 0
    if (shouldStop()) stoppedEarly = true
    return ms to finite
  }

  companion object {
    /** Untimed calls (or requests) before the timed ones. */
    const val WARMUP_CALLS = 5

    /** Timed calls per single-row set, and timed requests per request set and request path. */
    const val TIMED_REPEATS = 20

    /** The bundled request with five questions (rows of 128–142 tokens). */
    const val REQUEST_PATH_RECORD = "own_fiveq_09"

    /** How the report describes the protocol. */
    const val PROTOCOL =
      "$WARMUP_CALLS warm-up calls, then $TIMED_REPEATS timed calls (a request set: $TIMED_REPEATS requests " +
        "of its rows back to back); ms = input writes + run() + read-back of hidden; median as numpy"

    /** How the report describes the pair's protocol. */
    const val PAIR_PROTOCOL =
      "$WARMUP_CALLS warm-up requests, then $TIMED_REPEATS timed passes over the set's requests; a request = " +
        "the state call (input writes + run() + the write of state_valid, which waits for the state on the " +
        "GPU) and one question step per row (input writes + run() + read-back of hidden); median as numpy"
  }
}
