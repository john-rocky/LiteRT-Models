package com.gliclass

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.PowerManager
import android.util.Log
import java.io.File
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-asset gate, called only on the ViewModel's model dispatcher. For every captured official
 * request in `gate_fixtures.json` (the committed asset holds 152; the conversion run can regenerate
 * the full 552-request asset, see scripts/TEST_DATA.md): (1) the on-device tokenizer, linearization
 * and padding reproduce the captured Python inputs exactly; (2) the graph on the chosen backend
 * gives the official single-label and multi-label (threshold 0.5) decisions, with max |Δlogit|
 * against the fp32 oracle; (3) timing medians per window. The report is written as
 * `files/<report>.partial` while running and renamed to `files/<report>` at the end.
 */
class GliclassGateRunner(
  private val context: Context,
  private val classifier: () -> GliclassClassifier,
) {
  /** Outcome of one backend's gate and the path of its JSON report. */
  data class Summary(val backend: String, val passed: Boolean, val path: String, val error: String?)

  /** Runs the gate on [requestedBackend] ("gpu" or "cpu") and writes `files/<reportName>`. */
  fun run(requestedBackend: String?, reportName: String?): Summary {
    check(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
      "Gate fixtures are available only in the debug APK."
    }
    val backend =
      GliclassClassifier.Backend.valueOf((requestedBackend ?: "gpu").uppercase(Locale.ROOT))
    val name = reportName ?: "app_gate_${backend.name.lowercase(Locale.ROOT)}.json"
    require(name.matches(REPORT_NAME) && !name.endsWith(".partial")) {
      "Invalid report name $name"
    }
    val root =
      context.assets.open(GliclassGateFixtures.ASSET_NAME).bufferedReader().use {
        JSONObject(it.readText())
      }
    return runBackend(backend, GliclassGateFixtures.parse(root), File(context.filesDir, name))
  }

  private fun runBackend(
    backend: GliclassClassifier.Backend,
    fixtures: List<GliclassGateFixtures.Fixture>,
    destination: File,
  ): Summary {
    val partial = File(destination.parentFile, "${destination.name}.partial")
    destination.delete()
    val rows = JSONArray()
    val header =
      JSONObject()
        .put("set", "gliclass_app_gate")
        .put("litert_version", GliclassClassifier.LITERT_VERSION)
        .put("accelerator", backend.name)
        .put("precision", backend.precision)
        .put("device_model", Build.MODEL)
        .put("manufacturer", Build.MANUFACTURER)
        .put("android_sdk", Build.VERSION.SDK_INT)
        .put("build_fingerprint", Build.FINGERPRINT)
        .put("build_type", BuildConfig.BUILD_TYPE)
        .put("fixture_count", fixtures.size)
        .put("threshold", GliclassDecoder.DEFAULT_THRESHOLD)
        .put("warmup_passes_per_window", WARMUP_PASSES)
        .put(
          "cpu_threads",
          if (backend == GliclassClassifier.Backend.CPU) {
            GliclassClassifier.CPU_THREADS
          } else {
            JSONObject.NULL
          },
        )
        .put("status", "RUNNING")
        .put("compile_status", "NOT_RUN")
        .put("thermal_status_start", thermalStatus())
        .put(
          "graph_timing",
          "First input write through output readback, one timed run per fixture at its window",
        )
        .put("rows", rows)
    write(partial, header)
    var failureText: String? = null
    var inputsIdentical = 0
    var idsIdentical = 0
    var positionsIdentical = 0
    var singleEqual = 0
    var multiEqual = 0
    var finiteRows = 0
    var passedRows = 0
    var maxLogit = 0.0
    val graphByWindow = linkedMapOf<Int, MutableList<Double>>()
    val tokenizeEmbed = ArrayList<Double>()
    val decode = ArrayList<Double>()
    val writes = ArrayList<Double>()
    val enqueues = ArrayList<Double>()
    val readbacks = ArrayList<Double>()
    try {
      val helper = classifier()
      helper.initialize(backend)
      header.put("compile_status", "PASS")
      // Warm the full path of every window the asset uses before timing (first-dispatch and JIT
      // costs). The committed subset fits s128 only; the full asset also needs s256.
      for (window in GliclassInputs.WINDOWS) {
        val first = fixtures.firstOrNull { it.window == window } ?: continue
        repeat(WARMUP_PASSES) {
          helper.warmUp(first.text, first.labels, first.prompt, backend, window)
        }
      }
      header.put("compile_ms", JSONObject(helper.compileMs.toMap()))
      write(partial, header)
      for ((index, fixture) in fixtures.withIndex()) {
        val row = JSONObject().put("id", fixture.id).put("source", fixture.source)
        rows.put(row)
        try {
          val prepared = helper.inspectInputs(fixture.text, fixture.labels, fixture.prompt)
          val inputs = GliclassGateFixtures.compareInputs(fixture, prepared, PAD_ID)
          val inputsSame = inputs.getBoolean("identical")
          val idsSame = inputs.getJSONObject("input_ids").getBoolean("identical")
          val positionsSame = inputs.getJSONObject("label_positions").getBoolean("identical")
          if (inputsSame) {
            inputsIdentical++
          }
          if (idsSame) {
            idsIdentical++
          }
          if (positionsSame) {
            positionsIdentical++
          }
          val result =
            helper.classify(
              fixture.text,
              fixture.labels,
              fixture.prompt,
              GliclassDecoder.Mode.SINGLE_LABEL,
              GliclassDecoder.DEFAULT_THRESHOLD,
              backend,
            )
          val single = result.decision
          val multi =
            GliclassDecoder.decide(
              single.logits,
              fixture.labels,
              GliclassDecoder.Mode.MULTI_LABEL,
              GliclassDecoder.DEFAULT_THRESHOLD,
            )
          val singleSame = single.predictions.map { it.label } == fixture.officialSingle
          val multiSame = multi.predictions.map { it.label } == fixture.officialMulti
          val finite = single.logits.all { it.isFinite() }
          if (singleSame) {
            singleEqual++
          }
          if (multiSame) {
            multiEqual++
          }
          if (finite) {
            finiteRows++
          }
          val difference =
            GliclassGateFixtures.maxAbsDifference(single.logits, fixture.oracleLogits)
          maxLogit = maxOf(maxLogit, difference)
          val passed = inputsSame && singleSame && multiSame && finite
          if (passed) {
            passedRows++
          }
          graphByWindow.getOrPut(result.window) { ArrayList() }.add(result.timing.graphMs)
          tokenizeEmbed.add(result.timing.tokenizeEmbedMs)
          decode.add(result.timing.decodeMs)
          writes.add(result.timing.writeMs)
          enqueues.add(result.timing.enqueueMs)
          readbacks.add(result.timing.readbackMs)
          row
            .put("status", if (passed) "PASS" else "FAIL")
            .put("window", result.window)
            .put("encoded_tokens", result.encodedTokens)
            .put("label_count", fixture.labels.size)
            .put("inputs", if (inputsSame) JSONObject().put("identical", true) else inputs)
            .put("single_equal_official", singleSame)
            .put("multi_equal_official", multiSame)
            .put("all_logits_finite", finite)
            .put("max_abs_dlogit_vs_oracle", difference)
            .put("single", GliclassGateFixtures.predictionsJson(single))
            .put("multi", GliclassGateFixtures.predictionsJson(multi))
            .put("logits", JSONArray(single.logits.map { it.toDouble() }))
            .put(
              "ms",
              JSONObject()
                .put("tokenize_embed", result.timing.tokenizeEmbedMs)
                .put("graph", result.timing.graphMs)
                .put("write", result.timing.writeMs)
                .put("enqueue", result.timing.enqueueMs)
                .put("readback", result.timing.readbackMs)
                .put("decode", result.timing.decodeMs),
            )
        } catch (failure: Exception) {
          failureText = describe(failure)
          row.put("status", "FAIL").put("error", failureText)
        }
        if ((index + 1) % PARTIAL_EVERY == 0) {
          write(partial, header.put("rows_done", index + 1))
        }
      }
    } catch (failure: Exception) {
      failureText = describe(failure)
      header.put("compile_status", "FAIL").put("compile_message", failureText)
    } catch (failure: LinkageError) {
      failureText = describe(failure)
      header.put("compile_status", "FAIL").put("compile_message", "Native runtime: $failureText")
    }
    val passed = passedRows == fixtures.size && failureText == null
    header
      .put("status", if (passed) "PASS" else "FAIL")
      .put("rows_done", rows.length())
      .put("passed_rows", passedRows)
      .put("inputs_identical", inputsIdentical)
      .put("ids_identical", idsIdentical)
      .put("label_positions_identical", positionsIdentical)
      .put("single_equal_official", singleEqual)
      .put("multi_equal_official", multiEqual)
      .put("finite_rows", finiteRows)
      .put("max_abs_dlogit_vs_oracle", maxLogit)
      .put(
        "graph_ms_median_by_window",
        JSONObject().apply { graphByWindow.forEach { (w, values) -> put("s$w", median(values)) } },
      )
      .put(
        "rows_by_window",
        JSONObject().apply { graphByWindow.forEach { (w, values) -> put("s$w", values.size) } },
      )
      .put(
        "ms_median",
        JSONObject()
          .put("tokenize_embed", median(tokenizeEmbed))
          .put("write", median(writes))
          .put("enqueue", median(enqueues))
          .put("readback", median(readbacks))
          .put("decode", median(decode)),
      )
      .put("thermal_status_end", thermalStatus())
      .put("error", failureText ?: JSONObject.NULL)
    write(partial, header)
    check(partial.renameTo(destination)) {
      "Could not save gate result ${destination.absolutePath}"
    }
    Log.i(
      GliclassGateFixtures.GATE_LOG_TAG,
      JSONObject()
        .put("accelerator", backend.name)
        .put("status", header.getString("status"))
        .put("passed_rows", passedRows)
        .put("rows", fixtures.size)
        .put("max_abs_dlogit_vs_oracle", maxLogit)
        .put("path", destination.absolutePath)
        .put("error", failureText ?: JSONObject.NULL)
        .toString(),
    )
    return Summary(backend.name, passed, destination.absolutePath, failureText)
  }

  private fun median(values: List<Double>): Any =
    if (values.isEmpty()) JSONObject.NULL else values.sorted()[values.size / 2]

  private fun thermalStatus(): Int =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      context.getSystemService(PowerManager::class.java).currentThermalStatus
    } else {
      GliclassDiagnostics.THERMAL_STATUS_UNAVAILABLE
    }

  private fun write(file: File, content: JSONObject) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(content.toString(GliclassGateFixtures.REPORT_JSON_INDENT) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
  }

  private fun describe(failure: Throwable) =
    "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}"

  companion object {
    /** Untimed full passes per window before the timed rows. */
    const val WARMUP_PASSES = 5

    /** Rows between `.partial` report updates. */
    private const val PARTIAL_EVERY = 50

    private const val PAD_ID = 50283

    /** Report file names stay inside `files/`: letters, digits, dot, underscore and hyphen. */
    private val REPORT_NAME = Regex("[A-Za-z0-9._-]+")
  }
}
