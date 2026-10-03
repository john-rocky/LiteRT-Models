package com.kev

/**
 * The Android-free part of the timing protocol on [runner]: [WARMUP_CALLS] untimed calls, then
 * [TIMED_REPEATS] timed calls (a `request` set: that many requests of its rows back to back), where
 * a call is input writes + `run()` + read-back of `hidden`; and the request path of one request
 * from its text (tokenize, one call per question, head, `to_answers`). [shouldStop] is asked after
 * every call; once it says yes, the remaining calls are left out.
 */
class KevTimingCore(
  private val pipeline: KevPipeline,
  private val runner: RowRunner,
  private val shouldStop: () -> Boolean,
) {
  var stoppedEarly = false
    private set

  /** One set of `timing_rows.json` (its L must be [runner]'s window). */
  fun timeSet(set: KevTimingSet): LinkedHashMap<String, Any?> {
    require(set.window == runner.length) { "Set ${set.name} is for L${set.window}, the graph is L${runner.length}" }
    val padded = set.rows.map { KevPaddedRow.of(it.ids, runner.length) to it.ids.size }
    val warmup = ArrayList<Double>()
    var finite = true
    for (call in 0 until WARMUP_CALLS) {
      if (stoppedEarly) break
      val (inputs, length) = padded[call % padded.size]
      val (ms, rowFinite) = call(inputs, length)
      warmup.add(ms)
      finite = finite && rowFinite
    }
    val perCall = ArrayList<Double>()
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
      "rows" to set.rows.map { linkedMapOf("key" to it.key, "row_len" to it.ids.size, "ids_sha256" to KevPipeline.idsSha256(it.ids)) },
      "warmup_ms" to warmup,
      "per_call_ms" to KevStats.of(perCall)?.toJson(),
      "per_request_ms" to (if (set.rows.size > 1) KevStats.of(perRequest)?.toJson() else null),
      "calls_ms" to perCall,
      "finite" to finite,
    )
  }

  /** [request] from its text: tokenize, one call and the head per question, `to_answers`. */
  fun timeRequestPath(id: String, request: KevRequest): LinkedHashMap<String, Any?> {
    val rowLengths = pipeline.prepare(request).rows.map { it.length }
    if (rowLengths.max() > runner.length) return linkedMapOf("record" to id, "skipped" to "a row is over L${runner.length}")
    val warmup = ArrayList<Double>()
    val total = ArrayList<Double>()
    val tokenize = ArrayList<Double>()
    val infer = ArrayList<Double>()
    val head = ArrayList<Double>()
    var lastAnswers: Map<String, Any?>? = null
    for (iteration in 0 until WARMUP_CALLS + TIMED_REPEATS) {
      if (stoppedEarly) break
      val start = System.nanoTime()
      val prepared = pipeline.prepare(request)
      val results = prepared.rows.indices.map { pipeline.run(prepared, it, runner) }
      val answers = pipeline.answers(prepared, results)
      val ms = KevPipeline.millis(System.nanoTime() - start)
      if (shouldStop()) stoppedEarly = true
      if (iteration < WARMUP_CALLS) {
        warmup.add(ms)
        continue
      }
      total.add(ms)
      tokenize.add(prepared.tokenizeMs)
      infer.addAll(results.map { it.inferMs })
      head.addAll(results.map { it.headMs })
      lastAnswers = answers
    }
    return linkedMapOf(
      "record" to id,
      "questions" to request.questions.size,
      "row_lens" to rowLengths,
      "steps" to "tokenize the request, then per question: pad, graph call, head; then to_answers",
      "warmup_ms" to warmup,
      "request_ms" to KevStats.of(total)?.toJson(),
      "tokenize_ms" to KevStats.of(tokenize)?.toJson(),
      "infer_ms_per_call" to KevStats.of(infer)?.toJson(),
      "head_ms_per_question" to KevStats.of(head)?.toJson(),
      "last_answers" to lastAnswers,
    )
  }

  /** One timed call; finiteness is checked on the real positions after the clock stops. */
  private fun call(inputs: KevPaddedRow, length: Int): Pair<Double, Boolean> {
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
  }
}
