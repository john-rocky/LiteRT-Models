// SPDX-License-Identifier: Apache-2.0
package com.laya

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt

/** Debug-only device evidence collector. Numeric acceptance belongs to the supervising round. */
class LayaGateRunner(private val context: Context, private val engine: LayaEngine) {
  data class Report(val path: String, val status: String, val error: String? = null)

  fun run(backend: LayaEngine.Backend, window: Int): Report {
    check(BuildConfig.DEBUG) { "The fixture gate is available only in the debug build" }
    require(window == 256 || window == 512) { "Expected gate window 256 or 512" }
    val output = File(context.filesDir, "laya_gate_${backend.name.lowercase()}_$window.json")
    val report =
      linkedMapOf<String, Any?>(
        "schema_version" to 1,
        "device" to "${Build.MANUFACTURER} ${Build.MODEL}",
        "android_release" to Build.VERSION.RELEASE,
        "android_sdk" to Build.VERSION.SDK_INT,
        "build_fingerprint" to Build.FINGERPRINT,
        "application_id" to BuildConfig.APPLICATION_ID,
        "version_name" to BuildConfig.VERSION_NAME,
        "debuggable" to BuildConfig.DEBUG,
        "litert_version" to LayaEngine.LITERT_VERSION,
        "accelerator" to backend.name.lowercase(),
        "gpu_precision" to if (backend == LayaEngine.Backend.GPU) "FP32" else null,
        "graph_storage" to engine.storage.argument,
        "main_model" to engine.mainFilename(window),
        "embedding_table_sha256" to engine.embeddingTableSha256,
        "window" to window,
        "temperature" to 1.0,
        "tokenizer_load_ms" to engine.tokenizerLoadMs,
        "embedding_map_ms" to engine.embeddingLoadMs,
        "numerical_acceptance" to "NOT_EVALUATED: supervisor supplies round-2 tolerances",
        "timing_definition" to
          "main and action write + run enqueue + output readback; cold first call excluded from warm summary",
      )
    val results = mutableListOf<Map<String, Any?>>()
    report["rows"] = results
    var setupError: String? = null
    try {
      val fixtureFile = File(context.filesDir, "fixtures/gate_rows_s$window.json")
      check(fixtureFile.isFile) { "Missing ${fixtureFile.path}; run scripts/install_to_device.sh" }
      val fixture = LayaJson.asObject(LayaJson.parse(fixtureFile))
      require((fixture["window"] as Number).toInt() == window) { "Fixture window mismatch" }
      val rows = LayaJson.asArray(fixture["rows"]).map { LayaJson.asObject(it) }
      check(rows.isNotEmpty()) { "Gate fixture has no rows" }
      report["expected_rows"] = rows.size
      val compileStarted = System.nanoTime()
      engine.initialize(backend, window)
      report["compile_ms"] = ms(System.nanoTime() - compileStarted)
      var firstCall = true
      rows.forEach { row ->
        val detail = linkedMapOf<String, Any?>("row_id" to row["row_id"])
        results += detail
        try {
          val question = LayaJson.asObject(row["question"])
          val expectedIds = integers(row["sequence_ids"])
          val expectedMarkers = integers(row["marker_positions"])
          val preparedStarted = System.nanoTime()
          val sequence =
            engine.prepare(row["state"], question, window, row["question_id"].toString())
          detail["prepare_ms"] = ms(System.nanoTime() - preparedStarted)
          detail["sequence_ids"] = sequence.ids.toList()
          detail["marker_positions"] = sequence.markers.toList()
          detail["ids_exact"] = sequence.ids.contentEquals(expectedIds)
          detail["markers_exact"] = sequence.markers.contentEquals(expectedMarkers)
          detail["first_id_difference"] = firstDifference(sequence.ids, expectedIds)
          detail["first_marker_difference"] = firstDifference(sequence.markers, expectedMarkers)
          detail["qtype"] = sequence.question.qtype
          detail["K"] = sequence.markers.size
          require(sequence.question.qtype == (row["qtype"] as Number).toInt()) { "qtype mismatch" }
          require(sequence.markers.size == (row["K"] as Number).toInt()) { "Option-count mismatch" }

          detail["cold_first_call"] = firstCall
          firstCall = false
          val raw = engine.runRaw(sequence, backend, window)
          detail["graph_run_completed"] = true
          detail["finite"] = raw.finite
          detail["token_logits"] = finiteList(raw.tokenLogits)
          detail["raw_logits"] = finiteList(raw.markerLogits)
          detail["raw_act_logits"] = finiteList(raw.actLogits)
          detail["act_features"] = finiteList(raw.actFeatures)
          detail["token_logit_scale"] = scale(raw.tokenLogits)
          detail["pooled_cls_scale"] = scale(raw.pooledCls)
          detail["marker_logit_difference"] =
            difference(raw.markerLogits, floats(row["raw_logits"]))
          detail["act_logit_difference"] = difference(raw.actLogits, floats(row["raw_act_logits"]))
          detail["main_timing"] = raw.mainTiming.toMap()
          detail["act_timing"] = raw.actTiming?.toMap()
          detail["embedding_lookup_ms"] = raw.embeddingLookupMs
          detail["feature_ms"] = raw.featureMs
          detail["write_run_read_ms"] = raw.graphMs
          detail["call_ms"] = raw.callMs
          if (raw.finite) {
            val decodeStarted = System.nanoTime()
            val actual =
              LayaDecoder.decode(
                raw.markerLogits,
                raw.actLogits,
                sequence.question,
                LayaCalibration.identity(),
              )
            val official = LayaJson.asObject(row["official_answer"])
            val probabilities = LayaDecoder.softmax(raw.markerLogits)
            val referenceProbabilities = floats(row["probabilities"])
            detail["decode_ms"] = ms(System.nanoTime() - decodeStarted)
            detail["answer"] = actual
            detail["official_answer"] = official
            detail["official_dict_exact"] = jsonEqual(actual, official)
            detail["probabilities"] = probabilities.toList()
            detail["argmax"] = argmax(probabilities)
            detail["reference_argmax"] = argmax(referenceProbabilities)
            detail["argmax_exact"] = argmax(probabilities) == argmax(referenceProbabilities)
            detail["probability_max_abs_error"] = maxError(probabilities, referenceProbabilities)
            detail["answer_max_numeric_error"] = numericError(actual, official)
          } else {
            detail["error"] =
              "Nonfinite model values; action/decode may be skipped; see tensor scales"
          }
        } catch (failure: Exception) {
          detail["error"] = "${failure.javaClass.simpleName}: ${failure.message}"
        } catch (failure: LinkageError) {
          detail["error"] = "${failure.javaClass.simpleName}: ${failure.message}"
        }
      }
    } catch (failure: Exception) {
      setupError = "${failure.javaClass.simpleName}: ${failure.message}"
    } catch (failure: LinkageError) {
      setupError = "${failure.javaClass.simpleName}: ${failure.message}"
    }
    val summary =
      linkedMapOf<String, Any?>(
        "rows_recorded" to results.size,
        "ids_exact" to results.count { it["ids_exact"] == true },
        "markers_exact" to results.count { it["markers_exact"] == true },
        "finite_rows" to results.count { it["finite"] == true },
        "argmax_exact" to results.count { it["argmax_exact"] == true },
        "official_dict_exact" to results.count { it["official_dict_exact"] == true },
        "row_errors" to results.count { it["error"] != null },
        "probability_max_abs_error" to
          results
            .mapNotNull { (it["probability_max_abs_error"] as? Number)?.toDouble() }
            .maxOrNull(),
        "cold_first_call" to
          results
            .firstOrNull { it["cold_first_call"] == true }
            ?.let {
              linkedMapOf(
                "row_id" to it["row_id"],
                "write_run_read_ms" to it["write_run_read_ms"],
                "call_ms" to it["call_ms"],
                "main_timing" to it["main_timing"],
                "act_timing" to it["act_timing"],
              )
            },
        "warm_write_run_read_ms" to
          timingSummary(
            results
              .filter { it["cold_first_call"] == false }
              .mapNotNull { (it["write_run_read_ms"] as? Number)?.toDouble() }
          ),
      )
    val status =
      if (
        setupError != null ||
          results.isEmpty() ||
          results.any {
            it["error"] != null || it["ids_exact"] != true || it["markers_exact"] != true
          }
      )
        "FAIL"
      else "MEASURED"
    report["status"] = status
    report["summary"] = summary
    report["error"] = setupError
    output.writeText(LayaJson.stringify(report))
    Log.i(
      TAG,
      "summary ${LayaJson.stringify(linkedMapOf(
      "status" to status, "accelerator" to backend.name.lowercase(), "window" to window,
      "graph_storage" to engine.storage.argument,
      "rows" to results.size, "ids_exact" to summary["ids_exact"],
      "markers_exact" to summary["markers_exact"], "finite_rows" to summary["finite_rows"],
      "argmax_exact" to summary["argmax_exact"],
      "max_delta_p" to summary["probability_max_abs_error"], "file" to output.name,
      "error" to setupError,
    ))}",
    )
    return Report(output.absolutePath, status, setupError)
  }

  private fun integers(value: Any?) =
    LayaJson.asArray(value).map { (it as Number).toInt() }.toIntArray()

  private fun floats(value: Any?) =
    LayaJson.asArray(value).map { (it as Number).toFloat() }.toFloatArray()

  private fun finiteList(values: FloatArray): List<Float?> =
    values.map { if (it.isFinite()) it else null }

  private fun ms(nanos: Long) = nanos / 1_000_000.0

  private fun firstDifference(actual: IntArray, reference: IntArray): Int? {
    for (index in 0 until minOf(actual.size, reference.size)) {
      if (actual[index] != reference[index]) return index
    }
    return if (actual.size == reference.size) null else minOf(actual.size, reference.size)
  }

  private fun scale(values: FloatArray): Map<String, Any?> {
    val finite = values.filter { it.isFinite() }
    return linkedMapOf(
      "count" to values.size,
      "nonfinite_count" to values.count { !it.isFinite() },
      "minimum" to finite.minOrNull(),
      "maximum" to finite.maxOrNull(),
      "max_absolute" to finite.maxOfOrNull { abs(it.toDouble()) },
    )
  }

  private fun difference(actual: FloatArray, reference: FloatArray): Map<String, Any?> {
    val valid =
      actual.size == reference.size &&
        actual.all { it.isFinite() } &&
        reference.all { it.isFinite() }
    val maxAbs = if (valid) maxError(actual, reference) else null
    val referenceScale = reference.maxOfOrNull { abs(it.toDouble()) } ?: 0.0
    return linkedMapOf(
      "observed_scale" to scale(actual),
      "reference_scale" to scale(reference),
      "max_absolute_error" to maxAbs,
      "max_absolute_error_over_reference_max_absolute" to
        if (maxAbs != null && referenceScale != 0.0) maxAbs / referenceScale else null,
      "root_mean_square_error" to
        if (valid)
          sqrt(
            actual.indices.sumOf {
              val delta = actual[it].toDouble() - reference[it].toDouble()
              delta * delta
            } / actual.size
          )
        else null,
    )
  }

  private fun maxError(actual: FloatArray, reference: FloatArray): Double {
    require(actual.size == reference.size) { "Probability/vector length mismatch" }
    return actual.indices.maxOfOrNull { abs(actual[it].toDouble() - reference[it].toDouble()) }
      ?: 0.0
  }

  private fun argmax(values: FloatArray): Int {
    require(values.isNotEmpty())
    var best = 0
    for (index in 1 until values.size) if (values[index] > values[best]) best = index
    return best
  }

  private fun jsonEqual(actual: Any?, reference: Any?): Boolean =
    when {
      actual is Number && reference is Number -> actual.toDouble() == reference.toDouble()
      actual is Map<*, *> && reference is Map<*, *> ->
        actual.keys == reference.keys && actual.keys.all { jsonEqual(actual[it], reference[it]) }
      actual is List<*> && reference is List<*> ->
        actual.size == reference.size && actual.indices.all { jsonEqual(actual[it], reference[it]) }
      else -> actual == reference
    }

  private fun numericError(actual: Any?, reference: Any?): Double =
    when {
      actual is Number && reference is Number -> abs(actual.toDouble() - reference.toDouble())
      actual is Map<*, *> && reference is Map<*, *> ->
        actual.keys.maxOfOrNull { numericError(actual[it], reference[it]) } ?: 0.0
      actual is List<*> && reference is List<*> ->
        actual.indices.maxOfOrNull { numericError(actual[it], reference.getOrNull(it)) } ?: 0.0
      else -> 0.0
    }

  private fun timingSummary(values: List<Double>): Map<String, Any?> {
    val sorted = values.sorted()
    val median =
      if (sorted.isEmpty()) null
      else if (sorted.size % 2 == 0) {
        (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2
      } else sorted[sorted.size / 2]
    return linkedMapOf(
      "count" to values.size,
      "median" to median,
      "minimum" to sorted.firstOrNull(),
      "maximum" to sorted.lastOrNull(),
    )
  }

  companion object {
    const val TAG = "LAYA_GATE"
  }
}
