package com.kev

import kotlin.math.abs

/**
 * The Android-free part of the debug gate: the tokenizer on the probes, then every request of the
 * gate asset through `to_record` and the encoder (rows against the oracle), and the rows that fit
 * [runner]'s window (at most [limit] of them when it is above 0) through the graph and the head
 * (probabilities against the oracle). [shouldStop] is asked after every graph call. [rows] grows as
 * the run goes, so a partial report can be written at any time.
 */
class KevGateCore(
  private val pipeline: KevPipeline,
  private val runner: RowRunner,
  private val limit: Int,
  private val shouldStop: () -> Boolean,
  private val log: (String) -> Unit,
) {
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
      for ((index, question) in item.questions.withIndex()) {
        val entry = compareRow(item, prepared.rows[index], question)
        rows.add(entry)
        when {
          prepared.rows[index].length > runner.length ->
            skip(entry, "needs L${prepared.rows[index].window ?: "> ${KevEncoder.WINDOWS.last()}"}")
          stoppedEarly || (limit > 0 && run >= limit) ->
            skip(entry, if (stoppedEarly) "stopped" else "limit")
          else -> {
            runRow(prepared, index, question, entry)
            if (run % PROGRESS_EVERY == 0) {
              log("GATE_ROW run=$run key=${entry["key"]} infer_ms=${entry["infer_ms"]}")
              onProgress(run)
            }
            if (shouldStop()) {
              stoppedEarly = true
              log("GATE_STOP after $run rows")
            }
          }
        }
      }
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
    prepared: KevPrepared,
    index: Int,
    question: KevGateQuestion,
    entry: LinkedHashMap<String, Any?>,
  ) {
    run++
    entry["status"] = "run"
    entry["window"] = runner.length
    entry["near_tie"] = question.nearTie
    entry["oracle_top2_gap"] = question.top2Gap
    val result =
      try {
        pipeline.run(prepared, index, runner)
      } catch (failure: KevNonFiniteException) {
        nonFiniteRows++
        entry["finite"] = false
        entry["nonfinite_values"] = failure.count
        return
      }
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
    val answer =
      KevAnswers.toAnswers(listOf(device), listOf(prepared.meta[index])).getValue(question.qid)
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
    return linkedMapOf(
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
  }

  companion object {
    /** Rows run before the infer-time median starts. */
    const val WARMUP_ROWS = 5

    /** Rows between progress lines and partial reports. */
    const val PROGRESS_EVERY = 25

    private const val MAX_MISMATCHES = 5
  }
}
