// SPDX-License-Identifier: Apache-2.0
package com.julia1

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File

/**
 * Debug-only device evidence collector. Stage 1 rebuilds every fixture request with the on-device
 * tokenizer and sequence rules and compares ids, markers and question type with the reference host.
 * Stage 2 runs the graph on the captured rows and records the marker logits and timings in the same
 * layout as the conversion gate, so the host compares them with the author's runtime.
 */
class JuliaGateRunner(private val context: Context, private val engine: JuliaEngine) {
  data class Report(val path: String, val status: String, val error: String? = null)

  fun run(backend: JuliaEngine.Backend, window: Int): Report {
    check(BuildConfig.DEBUG) { "The fixture gate is available only in the debug build" }
    require(window in JuliaEngine.WINDOWS) { "Expected gate window 512 or 1024" }
    val output = File(context.filesDir, "julia1_gate_${backend.name.lowercase()}_$window.json")
    val report =
      linkedMapOf<String, Any?>(
        "schema_version" to 1,
        "device" to Build.MODEL,
        "manufacturer" to Build.MANUFACTURER,
        "android" to Build.VERSION.SDK_INT,
        "android_release" to Build.VERSION.RELEASE,
        "application_id" to BuildConfig.APPLICATION_ID,
        "debuggable" to BuildConfig.DEBUG,
        "litert" to JuliaEngine.LITERT_VERSION,
        "accel" to backend.name.lowercase(),
        "precision" to if (backend == JuliaEngine.Backend.GPU) "fp32" else "cpu",
        "graph" to JuliaEngine.graphFilename(window),
        "table" to JuliaEngine.TABLE_FILE,
        "seq" to window,
        "tokenizer_load_ms" to engine.tokenizerLoadMs,
        "embedding_map_ms" to engine.embeddingLoadMs,
        "timing_definition" to
          "write + run enqueue + output readback per row; the first 5 rows are warm-up",
      )
    var setupError: String? = null
    var stageOneFailed = false
    val rows = mutableListOf<Map<String, Any?>>()
    try {
      val requests = File(context.filesDir, "fixtures/gate_requests.json")
      if (requests.isFile) {
        val encoding = encodingStage(requests, window)
        report["encoding"] = encoding
        stageOneFailed = encoding["status"] != "PASS"
      } else {
        report["encoding"] = "skipped: fixtures/gate_requests.json not installed"
      }
      val fixture = File(context.filesDir, "fixtures/gate_rows_s$window.json")
      check(fixture.isFile) { "Missing ${fixture.path}; run scripts/install_to_device.sh" }
      val captured = JuliaJson.asArray(JuliaJson.parse(fixture)).map { JuliaJson.asObject(it) }
      check(captured.isNotEmpty()) { "Gate fixture has no rows" }
      val compileStarted = System.nanoTime()
      engine.initialize(backend, window)
      report["compile_ms"] = ms(System.nanoTime() - compileStarted)
      val totals = ArrayList<Double>()
      captured.forEach { row ->
        val detail = linkedMapOf<String, Any?>("id" to row["id"])
        rows += detail
        try {
          val ids = integers(row["ids"])
          val markers = integers(row["markers"])
          val qtype = (row["qtype"] as Number).toInt()
          val question = placeholderQuestion(qtype, markers.size)
          val sequence = JuliaSequence(ids, markers, question)
          val raw = engine.runRaw(sequence, backend, window)
          detail["finite"] = raw.finite
          detail["logits"] = raw.markerLogits.map { if (it.isFinite()) it else null }
          detail["lookup_ms"] = raw.embeddingLookupMs
          detail["write_ms"] = raw.timing.writeMs
          detail["run_ms"] = raw.timing.enqueueMs
          detail["read_ms"] = raw.timing.readbackMs
          detail["write_run_read_ms"] = raw.timing.totalMs
          totals += raw.timing.totalMs
          if (raw.finite) {
            val answer = JuliaDecoder.decode(raw.markerLogits, question)
            detail["probabilities"] = answer.probabilities.toList()
            detail["argmax"] = answer.argmax
          }
        } catch (failure: Exception) {
          detail["error"] = "${failure.javaClass.simpleName}: ${failure.message}"
        } catch (failure: LinkageError) {
          detail["error"] = "${failure.javaClass.simpleName}: ${failure.message}"
        }
      }
      val warm = totals.drop(WARMUP).sorted()
      report["summary"] =
        linkedMapOf(
          "rows" to captured.size,
          "finite_rows" to rows.count { it["finite"] == true },
          "row_errors" to rows.count { it["error"] != null },
          "warm_median_write_run_read_ms" to warm.getOrNull(warm.size / 2),
          "warm_min_ms" to warm.firstOrNull(),
          "warm_max_ms" to warm.lastOrNull(),
          "first_call_ms" to totals.firstOrNull(),
          "warm_rows" to warm.size,
        )
    } catch (failure: Exception) {
      setupError = "${failure.javaClass.simpleName}: ${failure.message}"
    } catch (failure: LinkageError) {
      setupError = "${failure.javaClass.simpleName}: ${failure.message}"
    }
    report["rows"] = rows
    val status =
      if (
        setupError != null ||
          stageOneFailed ||
          rows.isEmpty() ||
          rows.any { it["error"] != null || it["finite"] != true }
      ) {
        "FAIL"
      } else {
        "MEASURED"
      }
    report["status"] = status
    report["error"] = setupError
    val partial = File(context.filesDir, output.name + ".partial")
    partial.writeText(JuliaJson.stringify(report))
    partial.renameTo(output)
    val summary = report["summary"]
    Log.i(
      TAG,
      "summary ${JuliaJson.stringify(linkedMapOf(
        "status" to status, "accel" to backend.name.lowercase(), "window" to window,
        "encoding" to (report["encoding"] as? Map<*, *>)?.get("status"),
        "rows" to rows.size, "finite_rows" to (summary as? Map<*, *>)?.get("finite_rows"),
        "warm_median_ms" to (summary as? Map<*, *>)?.get("warm_median_write_run_read_ms"),
        "file" to output.name, "error" to setupError,
      ))}",
    )
    return Report(output.absolutePath, status, setupError)
  }

  /** Every fixture request through the tokenizer and builder; ids must match the reference host. */
  private fun encodingStage(file: File, window: Int): Map<String, Any?> {
    val fixture = JuliaJson.asObject(JuliaJson.parse(file))
    val requests = JuliaJson.asArray(fixture["rows"]).map { JuliaJson.asObject(it) }
    val fitsKey = "fits_$window"
    var identical = 0
    var rejected = 0
    var mismatched = 0
    var unexpected = 0
    val examples = mutableListOf<Map<String, Any?>>()
    val started = System.nanoTime()
    requests.forEach { row ->
      val request = JuliaJson.asObject(row["request"])
      val expectedIds = integers(row["ids"])
      val expectedMarkers = integers(row["markers"])
      val expectedQtype = (row["qtype"] as Number).toInt()
      val fits = row[fitsKey] as? Boolean ?: (expectedIds.size <= window)
      try {
        val question = requestQuestion(request)
        val sequence = engine.prepare(request["state"], question, window)
        val same =
          sequence.ids.contentEquals(expectedIds) &&
            sequence.markers.contentEquals(expectedMarkers) &&
            sequence.qtype == expectedQtype
        when {
          !fits -> {
            unexpected++
            examples += linkedMapOf("id" to row["id"], "problem" to "accepted an overflowing row")
          }
          same -> identical++
          else -> {
            mismatched++
            if (examples.size < 5) {
              examples +=
                linkedMapOf(
                  "id" to row["id"],
                  "first_id_difference" to firstDifference(sequence.ids, expectedIds),
                  "markers" to sequence.markers.toList(),
                  "expected_markers" to expectedMarkers.toList(),
                  "qtype" to sequence.qtype,
                )
            }
          }
        }
      } catch (rejectedRow: EncodingException) {
        if (fits) {
          mismatched++
          if (examples.size < 5) {
            examples += linkedMapOf("id" to row["id"], "problem" to rejectedRow.message)
          }
        } else {
          rejected++
        }
      }
    }
    return linkedMapOf(
      "status" to
        if (mismatched == 0 && unexpected == 0 && requests.isNotEmpty()) "PASS" else "FAIL",
      "requests" to requests.size,
      "identical" to identical,
      "rejected_over_window" to rejected,
      "mismatched" to mismatched,
      "accepted_overflow" to unexpected,
      "elapsed_ms" to ms(System.nanoTime() - started),
      "examples" to examples,
    )
  }

  /** The author's request row: state, question, type and rendered options. */
  private fun requestQuestion(request: Map<String, Any?>): JuliaQuestion {
    val type = request["type"] as? String ?: "choice"
    val options = JuliaJson.asArray(request["options"]).map { it as String }
    val criteria: Any? =
      when (type) {
        "choice" -> options.indices.associateTo(LinkedHashMap()) { "option$it" to options[it] }
        "score" -> options
        else -> linkedMapOf("false" to options[0], "true" to options[1])
      }
    return JuliaQuestion(type, request["question"] as String, criteria)
  }

  /** Captured rows carry ids and markers only; the decoder needs a question with K options. */
  private fun placeholderQuestion(qtype: Int, optionCount: Int): JuliaQuestion =
    when (qtype) {
      0 -> JuliaQuestion("choice", "", (0 until optionCount).associate { "k$it" to "option $it" })
      1 -> JuliaQuestion("score", "", (0 until optionCount).map { "level $it" })
      else -> JuliaQuestion("noul", "", null)
    }

  private fun integers(value: Any?) =
    JuliaJson.asArray(value).map { (it as Number).toInt() }.toIntArray()

  private fun ms(nanos: Long) = nanos / 1_000_000.0

  private fun firstDifference(actual: IntArray, reference: IntArray): Int? {
    for (index in 0 until minOf(actual.size, reference.size)) {
      if (actual[index] != reference[index]) {
        return index
      }
    }
    return if (actual.size == reference.size) null else minOf(actual.size, reference.size)
  }

  companion object {
    const val TAG = "JULIA1_GATE"
    private const val WARMUP = 5
  }
}
