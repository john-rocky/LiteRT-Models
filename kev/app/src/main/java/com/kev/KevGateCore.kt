package com.kev

import kotlin.math.abs

/** The graph a gate runs on: one row window, or the shared-state pair. */
sealed interface KevGateGraph {
  class Row(val runner: RowRunner) : KevGateGraph

  class Pair(val runner: PairRunner) : KevGateGraph
}

/**
 * The Android-free part of the debug gate: the tokenizer on the probes, then every request of the
 * gate asset through `to_record` and the encoder (rows against the oracle), and the questions the
 * [graph] takes (at most [limit] of them when it is above 0) through the graph and the head
 * (probabilities against the oracle). A row window takes the rows that fit it; the pair takes the
 * requests whose state fits Ls and whose branches all fit Lq, running each request's state once
 * (the state part = the row before its `[question]` token). [shouldStop] is asked after every
 * question's graph call. [rows] grows as the run goes, so a partial report can be written at any
 * time.
 */
class KevGateCore(
  private val pipeline: KevPipeline,
  private val graph: KevGateGraph,
  private val limit: Int,
  private val shouldStop: () -> Boolean,
  private val log: (String) -> Unit,
) {
  /** The gate on one row window. */
  constructor(
    pipeline: KevPipeline,
    runner: RowRunner,
    limit: Int,
    shouldStop: () -> Boolean,
    log: (String) -> Unit,
  ) : this(pipeline, KevGateGraph.Row(runner), limit, shouldStop, log)

  /** One entry per asset question, in asset order. */
  val rows = ArrayList<Any?>()

  var stoppedEarly = false
    private set

  private var requests = 0
  private var inputTokensIdentical = 0
  private var questions = 0
  private var idsIdentical = 0
  private var indicesIdentical = 0
  private var run = 0
  private val skippedRows = ArrayList<Any?>()
  private var nonFiniteRows = 0
  private var argmaxEqual = 0
  private var nearTieRows = 0
  private var argmaxEqualOutsideNearTies = 0
  private var rowsOutsideNearTies = 0
  private var answersEqual = 0
  private var maxAbsDp = 0.0
  private var sumAbsDp = 0.0
  private var options = 0
  private val nearTieFlips = ArrayList<Any?>()
  private val inferMs = ArrayList<Double>()
  private val headMs = ArrayList<Double>()
  private var probeCases = 0
  private var probesRawEqual = 0
  private var probesUserEqual = 0
  private var requestsRun = 0
  private val notFitting = ArrayList<Any?>()
  private val stateMs = ArrayList<Double>()

  /**
   * The device tokenizer against the IDs transformers gives each probe, raw and via `user_tokens`.
   */
  fun probes(probes: List<KevTokenizerProbe>): LinkedHashMap<String, Any?> {
    val mismatches = ArrayList<Any?>()
    for (probe in probes) {
      val raw = pipeline.tokenizer.encode(probe.text)
      val user = pipeline.encoder.userTokens(probe.text)
      val rawSame = raw.contentEquals(probe.rawIds)
      val userSame = user.contentEquals(probe.userIds)
      if (rawSame) probesRawEqual++
      if (userSame) probesUserEqual++
      if ((!rawSame || !userSame) && mismatches.size < MAX_MISMATCHES) {
        mismatches.add(
          linkedMapOf(
            "id" to probe.id,
            "category" to probe.category,
            "expected_raw" to probe.rawIds,
            "actual_raw" to raw,
          )
        )
      }
    }
    probeCases = probes.size
    log("GATE_PROBES raw_equal=$probesRawEqual user_equal=$probesUserEqual cases=$probeCases")
    return linkedMapOf(
      "cases" to probeCases,
      "raw_equal" to probesRawEqual,
      "user_equal" to probesUserEqual,
      "mismatches" to mismatches,
    )
  }

  /**
   * Checks every asset question; [onProgress] gets the rows run so far every [PROGRESS_EVERY] rows.
   */
  fun run(items: List<KevGateItem>, onProgress: (Int) -> Unit) {
    for (item in items) {
      val prepared = pipeline.prepare(KevRequest.fromJson(item.request))
      requests++
      if (prepared.inputTokens == item.inputTokens) inputTokensIdentical++
      val entries =
        item.questions.mapIndexed { index, question ->
          compareRow(item, prepared.rows[index], question).also { rows.add(it) }
        }
      when (val graph = graph) {
        is KevGateGraph.Row ->
          for ((index, question) in item.questions.withIndex()) {
            val entry = entries[index]
            val row = prepared.rows[index]
            when {
              row.length > graph.runner.length ->
                skip(
                  entry,
                  "needs L${row.window(KevFiles.WINDOWS) ?: "> ${KevFiles.WINDOWS.last()}"}",
                )
              stoppedEarly || (limit > 0 && run >= limit) ->
                skip(entry, if (stoppedEarly) "stopped" else "limit")
              else ->
                runQuestion(entry, question, onProgress) {
                  pipeline.run(prepared, index, graph.runner)
                }
            }
          }
        is KevGateGraph.Pair -> runPair(item, prepared, entries, graph.runner, onProgress)
      }
    }
  }

  /** One request on the pair: the state once, then each question the limit leaves. */
  private fun runPair(
    item: KevGateItem,
    prepared: KevPrepared,
    entries: List<LinkedHashMap<String, Any?>>,
    pair: PairRunner,
    onProgress: (Int) -> Unit,
  ) {
    val miss = pairMiss(prepared, pair)
    if (miss != null) {
      notFitting.add(linkedMapOf("record" to item.id, "reason" to miss))
      entries.forEach { skip(it, miss) }
      return
    }
    var state: KevStateResult? = null
    for ((index, question) in item.questions.withIndex()) {
      val entry = entries[index]
      if (stoppedEarly || (limit > 0 && run >= limit)) {
        skip(entry, if (stoppedEarly) "stopped" else "limit")
        continue
      }
      val ran =
        state
          ?: pipeline.runState(prepared, pair).also {
            state = it
            stateMs.add(it.ms)
            requestsRun++
          }
      entry["state_tokens"] = ran.tokens
      entry["state_ms"] = ran.ms
      runQuestion(entry, question, onProgress) { pipeline.runBranch(prepared, index, pair) }
    }
  }

  /** Why [pair] cannot take [prepared] (its state over Ls or a branch over Lq), or null. */
  private fun pairMiss(prepared: KevPrepared, pair: PairRunner): String? {
    val state = prepared.encoded.stateIds.size
    if (state > pair.stateLength) return "state $state > Ls ${pair.stateLength}"
    val index = prepared.encoded.branches.indexOfFirst { it.ids.size > pair.questionLength }
    if (index < 0) return null
    val branch = prepared.encoded.branches[index].ids.size
    return "branch ${prepared.meta[index].id} $branch > Lq ${pair.questionLength}"
  }

  /** One question through [infer], then the progress line and the stop file. */
  private fun runQuestion(
    entry: LinkedHashMap<String, Any?>,
    question: KevGateQuestion,
    onProgress: (Int) -> Unit,
    infer: () -> KevQuestionResult,
  ) {
    runRow(question, entry, infer)
    if (run % PROGRESS_EVERY == 0) {
      log("GATE_ROW run=$run key=${entry["key"]} infer_ms=${entry["infer_ms"]}")
      onProgress(run)
    }
    if (shouldStop()) {
      stoppedEarly = true
      log("GATE_STOP after $run rows")
    }
  }

  private fun compareRow(
    item: KevGateItem,
    row: KevRow,
    question: KevGateQuestion,
  ): LinkedHashMap<String, Any?> {
    val idsSame = row.ids.contentEquals(question.rowIds)
    val indicesSame =
      row.decideIndex == question.decideIndex &&
        row.optionIndices.contentEquals(question.optionIndices)
    questions++
    if (idsSame) idsIdentical++
    if (indicesSame) indicesIdentical++
    val entry =
      linkedMapOf<String, Any?>(
        "key" to "${item.id}/${question.qid}",
        "source" to item.source,
        "row_len" to row.length,
        "ids_sha256" to KevPipeline.idsSha256(row.ids),
        "ids_identical" to idsSame,
        "indices_identical" to indicesSame,
      )
    if (!idsSame) {
      entry["ids_differ_at"] = KevGateChecks.firstDifference(question.rowIds, row.ids)
      entry["oracle_row_len"] = question.rowIds.size
    }
    return entry
  }

  private fun skip(entry: LinkedHashMap<String, Any?>, reason: String) {
    entry["status"] = "skipped"
    entry["reason"] = reason
    skippedRows.add(
      linkedMapOf("key" to entry["key"], "row_len" to entry["row_len"], "reason" to reason)
    )
  }

  private fun runRow(
    question: KevGateQuestion,
    entry: LinkedHashMap<String, Any?>,
    infer: () -> KevQuestionResult,
  ) {
    run++
    entry["status"] = "run"
    entry["form"] = formName
    entry["window"] =
      when (val graph = graph) {
        is KevGateGraph.Row -> graph.runner.length
        is KevGateGraph.Pair -> graph.runner.questionLength
      }
    entry["near_tie"] = question.nearTie
    entry["oracle_top2_gap"] = question.top2Gap
    val result =
      try {
        infer()
      } catch (failure: KevNonFiniteException) {
        nonFiniteRows++
        entry["finite"] = false
        entry["nonfinite_values"] = failure.count
        return
      }
    entry["branch_len"] = result.branchLength
    val device = result.probabilities
    val argmaxSame =
      KevAnswers.firstArgmax(device) == KevAnswers.firstArgmax(question.probabilities)
    val maxDp = KevGateChecks.maxAbsDifference(device, question.probabilities)
    if (argmaxSame) argmaxEqual++
    if (question.nearTie) {
      nearTieRows++
      if (!argmaxSame) nearTieFlips.add("${entry["key"]} (gap ${question.top2Gap})")
    } else {
      rowsOutsideNearTies++
      if (argmaxSame) argmaxEqualOutsideNearTies++
    }
    maxAbsDp = maxOf(maxAbsDp, maxDp)
    for (option in device.indices) sumAbsDp += abs(device[option] - question.probabilities[option])
    options += device.size
    inferMs.add(result.inferMs)
    headMs.add(result.headMs)
    val answer = KevAnswers.toAnswers(listOf(device), listOf(result.meta)).getValue(question.qid)
    if (KevJson.write(answer) == KevJson.write(question.answer)) answersEqual++
    entry.putAll(
      linkedMapOf(
        "finite" to true,
        "argmax_key" to question.keys[KevAnswers.firstArgmax(device)],
        "oracle_argmax_key" to question.keys[KevAnswers.firstArgmax(question.probabilities)],
        "argmax_equal" to argmaxSame,
        "max_abs_dp" to maxDp,
        "probs" to device,
        "infer_ms" to result.inferMs,
        "head_ms" to result.headMs,
      )
    )
  }

  /**
   * PASS: every probe and row identical, rows run with no NaN, the argmax outside near-ties, max
   * |Δp| ≤ [KevGateChecks.MAX_ABS_DP] and mean |Δp| ≤ [KevGateChecks.MEAN_ABS_DP], not stopped.
   */
  fun passed(): Boolean =
    !stoppedEarly &&
      probeCases > 0 &&
      probesRawEqual == probeCases &&
      probesUserEqual == probeCases &&
      idsIdentical == questions &&
      indicesIdentical == questions &&
      inputTokensIdentical == requests &&
      run > 0 &&
      nonFiniteRows == 0 &&
      argmaxEqualOutsideNearTies == rowsOutsideNearTies &&
      maxAbsDp <= KevGateChecks.MAX_ABS_DP &&
      meanAbsDp() <= KevGateChecks.MEAN_ABS_DP

  fun meanAbsDp(): Double = if (options == 0) 0.0 else sumAbsDp / options

  /** The counts so far, for the report. */
  fun summary(): LinkedHashMap<String, Any?> {
    val warm = if (inferMs.size > WARMUP_ROWS) inferMs.drop(WARMUP_ROWS) else inferMs
    val pairCounts: Map<String, Any?> =
      if (graph is KevGateGraph.Pair) {
        linkedMapOf(
          "requests_run" to requestsRun,
          "requests_not_fitting" to notFitting.size,
          "not_fitting" to notFitting,
          "state_ms" to KevStats.of(stateMs)?.toJson(),
        )
      } else {
        emptyMap()
      }
    return linkedMapOf<String, Any?>(
        "form" to formName,
        "requests" to requests,
        "input_tokens_identical" to inputTokensIdentical,
        "questions" to questions,
        "ids_identical" to idsIdentical,
        "indices_identical" to indicesIdentical,
        "rows_run" to run,
        "rows_skipped" to skippedRows.size,
        "skipped_rows" to skippedRows,
        "nonfinite_rows" to nonFiniteRows,
        "argmax_equal" to argmaxEqual,
        "near_tie_rows" to nearTieRows,
        "near_tie_argmax_flips" to nearTieFlips,
        "argmax_equal_outside_near_ties" to argmaxEqualOutsideNearTies,
        "rows_outside_near_ties" to rowsOutsideNearTies,
        "max_abs_dp" to maxAbsDp,
        "mean_abs_dp_all_options" to meanAbsDp(),
        "answers_equal_oracle" to answersEqual,
        "warmup_rows_excluded" to minOf(WARMUP_ROWS, inferMs.size),
        "infer_ms" to KevStats.of(warm)?.toJson(),
        "infer_ms_row_1" to inferMs.firstOrNull(),
        "head_ms" to KevStats.of(headMs)?.toJson(),
      )
      .apply { putAll(pairCounts) }
  }

  private val formName: String
    get() = if (graph is KevGateGraph.Pair) KevForm.PAIR.wireName else KevForm.ROW.wireName

  companion object {
    /** Rows run before the infer-time median starts. */
    const val WARMUP_ROWS = 5

    /** Rows between progress lines and partial reports. */
    const val PROGRESS_EVERY = 25

    private const val MAX_MISMATCHES = 5
  }
}
