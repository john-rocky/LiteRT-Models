package com.gliformer

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.Process
import android.os.SystemClock
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant
import java.util.UUID
import kotlin.math.abs
import kotlin.math.sqrt
import kotlinx.coroutines.delay
import org.json.JSONArray
import org.json.JSONObject

/** Device parity runner with matched-table acceptance and separate fp32 diagnostics. */
class GliformerGateRunner(private val context: Context, private val intent: Intent) {
  val reportName: String = intent.getStringExtra("report") ?: "gate_report.json"
  private val report = JSONObject()
  private val reportFile = File(context.filesDir, reportName)
  private val rows = JSONArray()
  private val failures = JSONArray()

  init {
    require(reportName.matches(Regex("[a-zA-Z0-9_-]+\\.json"))) { "Invalid report basename" }
  }

  suspend fun run(onExtractor: (GliformerExtractor) -> Unit): JSONObject {
    val continueFile = File(context.filesDir, "$reportName.continue")
    if (continueFile.exists())
      check(continueFile.delete()) { "Cannot clear stale memory-snapshot marker" }
    report
      .put("status", "RUNNING")
      .put("stage", "starting")
      .put("run_id", intent.getStringExtra("runId") ?: UUID.randomUUID().toString())
      .put("started_at", Instant.now().toString())
      .put("device", device())
      .put("pid", Process.myPid())
      .put("battery_before", battery())
      .put("memory_before", memory())
      .put("rows", rows)
      .put("failures", failures)
    save()
    try {
      val backendName = intent.getStringExtra("backend") ?: "gpu"
      require(backendName in listOf("gpu", "cpu")) { "backend must be gpu or cpu" }
      val backend =
        if (backendName == "gpu") GliformerExtractor.Backend.GPU else GliformerExtractor.Backend.CPU
      val window = intent.getIntExtra("window", 128)
      require(window == 128 || window == 256) { "Device gates support s128 or s256" }
      val storageName = intent.getStringExtra("table") ?: "fp16"
      require(storageName in listOf("fp16", "fp32")) { "table must be fp16 or fp32" }
      val storage =
        if (storageName == "fp16") GliformerInputs.EmbeddingTable.Storage.FP16
        else GliformerInputs.EmbeddingTable.Storage.FP32
      val forceWindow = intent.getBooleanExtra("forceWindow", false)
      val includeShort = intent.getBooleanExtra("includeShort", false)
      val repetitions = intent.getIntExtra("repetitions", 5)
      val warmupPasses = intent.getIntExtra("warmupPasses", 12)
      require(repetitions >= 5 && warmupPasses >= 12) {
        "At least 5 measured repetitions and 12 warm-up passes are required"
      }
      val corpus = assetJson("corpus.json").getJSONArray("rows")
      val rawById = index(assetJson("graph_inputs/manifest.json").getJSONArray("rows"))
      val fp16Manifest = assetJson("references_fp16.json")
      val fp32Manifest = assetJson("references_fp32.json")
      val fp16ByWindow = indexWindows(fp16Manifest)
      val fp32ByWindow =
        indexWindows(fp32Manifest) + indexWindows(assetJson("references_fp32_forced_s256.json"))
      val referenceManifest = if (storageName == "fp16") fp16Manifest else fp32Manifest
      val selected =
        (0 until corpus.length())
          .map { corpus.getJSONObject(it) }
          .filter {
            (includeShort || it.getString("group") == "corpus") &&
              if (forceWindow) it.getInt("window") <= window else it.getInt("window") == window
          }
      require(selected.isNotEmpty()) { "No cases selected" }
      report
        .put("backend", backendName)
        .put("window", window)
        .put("force_window", forceWindow)
        .put("include_short_duplicates", includeShort)
        .put("table_storage", storageName)
        .put("graph_storage", "wfp16")
        .put("gpu_precision", if (backendName == "gpu") "FP32" else JSONObject.NULL)
        .put("head_backend", if (window > 128) "CPU" else JSONObject.NULL)
        .put("repetitions", repetitions)
        .put("warmup_passes", warmupPasses)
        .put(
          "timing_scope",
          "Serial full-pipeline samples after complete startup warm-up; graph times include output readback; comparison/report work is outside samples",
        )
        .put("selected_inputs", selected.size)
        .put("completed_inputs", 0)
        .put("python_score_tolerance", 1e-5)
        .put("oracle_score_tolerance", 1e-3)
        .put("python_reference_table_storage", referenceManifest.getString("host_table_storage"))
        .put(
          "python_reference_runtime_version",
          referenceManifest.getJSONObject("packages").getString("ai-edge-litert"),
        )
        .put(
          "acceptance_reference",
          "Matched host-table storage and exact graph window; fp16 is the app default",
        )
        .put("fp32_reference_role", "Preserved diagnostic only when default fp16 table is selected")
        .put(
          "reference_note",
          "Matched-fp16 score tolerance 1e-5; oracle 1e-3; fp32 diagnostics reported separately",
        )
      val extractor = GliformerExtractor(context.filesDir, storage)
      onExtractor(extractor)
      val tokenizerRows = JSONArray()
      report.put("stage", "tokenizer_parity").put("tokenizer_rows", tokenizerRows)
      save()
      var tokenizerPassed = 0
      var rawComparisons = 0
      for (i in 0 until corpus.length()) {
        val fixture = corpus.getJSONObject(i)
        val id = fixture.getString("id")
        val prepared = extractor.prepare(fixture.getString("text"))
        val checks =
          compareCaptured(prepared, fixture.getJSONObject("captured"), fixture.getInt("window"))
        val raw = rawById.getValue(id).getJSONObject("tensors")
        graphArrays(prepared).forEach { (name, actual) ->
          val expected = raw.getJSONObject(name)
          val bytes = assetBytes(expected.getString("file"))
          require(
            bytes.size == expected.getInt("bytes") && sha256(bytes) == expected.getString("sha256")
          ) {
            "$id $name captured fixture checksum mismatch"
          }
          checks.put("raw_$name", compareFloats(actual, floats(bytes)))
          rawComparisons++
        }
        val pass = allChecksPass(checks)
        if (pass) tokenizerPassed++
        else failures.put("$id on-device tokenizer/routing differs from original capture")
        tokenizerRows.put(
          JSONObject()
            .put("id", id)
            .put("window", prepared.window.sequenceLength)
            .put("status", if (pass) "PASS" else "FAIL")
            .put("checks", checks)
        )
      }
      val unicode = assetJson("unicode.json")
      val unicodePrepared = extractor.prepare(unicode.getString("text"))
      val unicodeChecks =
        compareCaptured(
          unicodePrepared,
          unicode.getJSONObject("captured"),
          unicode.getInt("window"),
        )
      val unicodePass = allChecksPass(unicodeChecks)
      report
        .put(
          "unicode_tokenizer",
          JSONObject()
            .put("status", if (unicodePass) "PASS" else "FAIL")
            .put("text", unicode.getString("text"))
            .put("checks", unicodeChecks),
        )
        .put("tokenizer_passed", tokenizerPassed)
        .put("tokenizer_total", corpus.length())
        .put("raw_graph_tensor_comparisons", rawComparisons)
      if (!unicodePass)
        failures.put("Supplementary Unicode tokenizer preparation differs from Python")
      if (tokenizerPassed != 80 || !unicodePass) {
        report.put("status", "FAIL").put("stage", "tokenizer_failed")
        finish()
        return report
      }

      report.put("stage", "loading")
      save()
      val loadStart = SystemClock.elapsedRealtimeNanos()
      extractor.initialize(backend, window)
      report
        .put("load_ms", elapsedMs(loadStart))
        .put("memory_after_load", memory())
        .put("battery_after_load", battery())
        .put("stage", "warming")
      save()
      val warmupStart = SystemClock.elapsedRealtimeNanos()
      extractor.warmUp(backend, window, warmupPasses)
      report
        .put("warmup_total_ms", elapsedMs(warmupStart))
        .put("memory_after_warmup", memory())
        .put("stage", "inference")
      if (intent.getBooleanExtra("waitForMemorySnapshot", false)) {
        report.put("stage", "WAITING_FOR_MEMORY_SNAPSHOT")
        save()
        val deadline = SystemClock.elapsedRealtime() + 30_000L
        while (!continueFile.exists() && SystemClock.elapsedRealtime() < deadline) delay(250)
        report.put("memory_snapshot_handshake_timed_out", !continueFile.exists())
        if (continueFile.exists()) check(continueFile.delete())
        report.put("stage", "inference")
      }
      save()
      val logitsDirectory =
        File(context.filesDir, "gate_logits/${reportName.removeSuffix(".json")}")
      logitsDirectory.mkdirs()
      var oracleIdentical = 0
      var pythonIdentical = 0
      var pythonScoresPassed = 0
      var oracleScoresPassed = 0
      var allFinite = true
      var maxPythonError = 0.0
      var maxOracleError = 0.0
      var maxLogitError = 0.0
      var maxFp16Error = 0.0
      var maxFp32Error = 0.0
      var fp32Identical = 0
      for (fixture in selected) {
        val id = fixture.getString("id")
        report.put("active_input", id)
        save()
        val key = "${id}_s$window"
        val fp16Reference = fp16ByWindow.getValue(key)
        val fp32Reference = fp32ByWindow.getValue(key)
        val reference = if (storageName == "fp16") fp16Reference else fp32Reference
        val referenceBytes =
          assetBytes(reference.optString("asset_file", "logits/${reference.getString("file")}"))
        require(sha256(referenceBytes) == reference.getString("sha256")) {
          "$id reference logits checksum mismatch"
        }
        val referenceLogits = floats(referenceBytes)
        val timings = JSONArray()
        var firstResult: GliformerExtractor.Result? = null
        var oracleMatch = true
        var pythonMatch = true
        var rowFinite = true
        var rowPythonError = 0.0
        var rowOracleError = 0.0
        var rowLogitError = 0.0
        var rowFp16Error = 0.0
        var rowFp32Error = 0.0
        var fp32Match = true
        var routesMatch = true
        val comparisons = JSONArray()
        repeat(repetitions) { repetition ->
          val result = extractor.extract(fixture.getString("text"), backend, window)
          if (firstResult == null) firstResult = result
          val routeChecks =
            compareCaptured(result.prepared, fixture.getJSONObject("captured"), window)
          routesMatch = routesMatch && allChecksPass(routeChecks)
          val python = compareEntities(result.entities, reference.getJSONArray("python_entities"))
          val oracle = compareEntities(result.entities, fixture.getJSONArray("oracle_entities"))
          val fp16 = compareEntities(result.entities, fp16Reference.getJSONArray("python_entities"))
          val fp32 = compareEntities(result.entities, fp32Reference.getJSONArray("python_entities"))
          rowFp16Error = maxOf(rowFp16Error, fp16.getDouble("max_score_difference"))
          rowFp32Error = maxOf(rowFp32Error, fp32.getDouble("max_score_difference"))
          fp32Match = fp32Match && fp32.getBoolean("span_sets_equal")
          pythonMatch = pythonMatch && python.getBoolean("span_sets_equal")
          oracleMatch = oracleMatch && oracle.getBoolean("span_sets_equal")
          rowPythonError = maxOf(rowPythonError, python.getDouble("max_score_difference"))
          rowOracleError = maxOf(rowOracleError, oracle.getDouble("max_score_difference"))
          val stats = logitStats(result.logits, referenceLogits, result.prepared.words.size * 15)
          rowFinite = rowFinite && stats.getBoolean("all_finite")
          rowLogitError = maxOf(rowLogitError, stats.getDouble("valid_max_absolute_error"))
          comparisons.put(
            JSONObject()
              .put("repetition", repetition + 1)
              .put("python", python)
              .put("oracle", oracle)
              .put("fp16_reference", fp16)
              .put("fp32_reference", fp32)
              .put("logits", stats)
              .put("routing_identical", allChecksPass(routeChecks))
          )
          timings.put(
            JSONObject()
              .put("tokenize_lookup_ms", result.timings.tokenizeLookupMs)
              .put("graph_readback_ms", result.timings.graphMs)
              .put("decode_ms", result.timings.decodeMs)
              .put("total_ms", result.timings.totalMs)
              .put("encoder_readback_ms", result.timings.encoderMs)
              .put("head_readback_ms", result.timings.headMs)
          )
        }
        val actual = requireNotNull(firstResult)
        val rawLogits = File(logitsDirectory, "$id.bin")
        val packedBytes = ByteBuffer.allocate(actual.logits.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        actual.logits.forEach { packedBytes.putFloat(it) }
        rawLogits.writeBytes(packedBytes.array())
        val pythonScorePass = pythonMatch && rowPythonError <= 1e-5
        val oracleScorePass = oracleMatch && rowOracleError <= 1e-3
        val pass = pythonScorePass && oracleScorePass && rowFinite && routesMatch
        if (oracleMatch) oracleIdentical++
        if (pythonMatch) pythonIdentical++
        if (pythonScorePass) pythonScoresPassed++
        if (oracleScorePass) oracleScoresPassed++
        allFinite = allFinite && rowFinite
        maxPythonError = maxOf(maxPythonError, rowPythonError)
        maxOracleError = maxOf(maxOracleError, rowOracleError)
        maxLogitError = maxOf(maxLogitError, rowLogitError)
        maxFp16Error = maxOf(maxFp16Error, rowFp16Error)
        maxFp32Error = maxOf(maxFp32Error, rowFp32Error)
        if (fp32Match) fp32Identical++
        if (!pass)
          failures.put(
            "$id strict gate: Python spans=$pythonMatch score_error=$rowPythonError; oracle spans=$oracleMatch score_error=$rowOracleError; finite=$rowFinite routes=$routesMatch"
          )
        rows.put(
          JSONObject()
            .put("id", id)
            .put("status", if (pass) "PASS" else "FAIL")
            .put("window", actual.window)
            .put("reference_window", reference.getInt("window"))
            .put("reference_window_matches", actual.window == reference.getInt("window"))
            .put("text_words", actual.prepared.words.size)
            .put("encoded_tokens", actual.prepared.encodedLength)
            .put("all_finite", rowFinite)
            .put("routing_identical", routesMatch)
            .put("python_span_sets_identical", pythonMatch)
            .put("oracle_span_sets_identical", oracleMatch)
            .put("python_score_pass", pythonScorePass)
            .put("oracle_score_pass", oracleScorePass)
            .put("max_score_difference_python", rowPythonError)
            .put("max_score_difference_oracle", rowOracleError)
            .put("max_valid_logit_difference_python", rowLogitError)
            .put("max_score_difference_fp16_reference", rowFp16Error)
            .put("max_score_difference_fp32_reference", rowFp32Error)
            .put("fp32_span_sets_identical", fp32Match)
            .put("fp32_reference_window", fp32Reference.getInt("window"))
            .put("entities", entities(actual.entities))
            .put("repetitions", comparisons)
            .put("timing_samples", timings)
            .put("timing_summary", summarizeTimings(timings))
            .put("logits_file", rawLogits.relativeTo(context.filesDir).path)
            .put("logits_shape", JSONArray(listOf(1, 1, actual.prepared.window.textCapacity, 15)))
            .put("logits_bytes", packedBytes.array().size)
            .put("logits_sha256", sha256(packedBytes.array()))
        )
        report
          .put("completed_inputs", rows.length())
          .put("oracle_span_sets_identical", oracleIdentical)
          .put("python_span_sets_identical", pythonIdentical)
          .put("python_scores_passed", pythonScoresPassed)
          .put("oracle_scores_passed", oracleScoresPassed)
          .put("all_finite", allFinite)
          .put("max_score_difference_python", maxPythonError)
          .put("max_score_difference_oracle", maxOracleError)
          .put("max_valid_logit_difference_python", maxLogitError)
          .put("max_score_difference_fp16_reference", maxFp16Error)
          .put("max_score_difference_fp32_reference", maxFp32Error)
          .put("fp32_span_sets_identical", fp32Identical)
        save()
      }
      report
        .put("status", if (failures.length() == 0) "PASS" else "FAIL")
        .put("stage", "finished")
        .put(
          "span_gate_status",
          if (oracleIdentical == selected.size && pythonIdentical == selected.size && allFinite)
            "PASS"
          else "FAIL",
        )
        .put(
          "strict_score_gate_status",
          if (pythonScoresPassed == selected.size && oracleScoresPassed == selected.size) "PASS"
          else "FAIL",
        )
      finish()
    } catch (failure: Throwable) {
      report
        .put("status", "ERROR")
        .put("stage", "exception")
        .put("error", failure.toString())
        .put("stack_trace", failure.stackTraceToString())
      finish()
    }
    return report
  }

  private fun compareCaptured(
    prepared: GliformerInputs.Prepared,
    captured: JSONObject,
    expectedWindow: Int,
  ): JSONObject {
    val checks = JSONObject()
    val n = prepared.window.sequenceLength
    val t = prepared.window.textCapacity
    fun integers(name: String, actual: IntArray, expected: IntArray) {
      checks.put(
        name,
        compareFloats(
          actual.map { it.toFloat() }.toFloatArray(),
          expected.map { it.toFloat() }.toFloatArray(),
        ),
      )
    }
    checks.put("window", boolCheck(n == expectedWindow))
    checks.put(
      "encoded_length",
      boolCheck(prepared.encodedLength == captured.getInt("encoded_length")),
    )
    checks.put("word_count", boolCheck(prepared.words.size == captured.getInt("text_word_length")))
    integers(
      "input_ids",
      prepared.inputIds,
      captured.getJSONArray("input_ids").getJSONArray(0).ints().copyOf(n),
    )
    val first = captured.getJSONArray("word_first_subtokens").ints()
    val parents = captured.getJSONArray("schema_positions").ints()
    val entityPositions = captured.getJSONArray("entity_positions").ints()
    integers("first_subtokens", prepared.firstSubtokenPositions, first)
    integers("parent_positions", prepared.parentPositions, parents)
    integers("entity_positions", prepared.entityPositions, entityPositions)
    integers(
      "word_starts_codepoints",
      prepared.words.map { it.start }.toIntArray(),
      captured.getJSONArray("start_map").ints(),
    )
    integers(
      "word_ends_codepoints",
      prepared.words.map { it.end }.toIntArray(),
      captured.getJSONArray("end_map").ints(),
    )
    checks.put(
      "word_texts",
      boolCheck(prepared.words.map { it.text } == captured.getJSONArray("tokens").strings()),
    )
    val wordMask = IntArray(prepared.encodedLength)
    prepared.firstSubtokenPositions.forEachIndexed { word, token -> wordMask[token] = word + 1 }
    integers("words_mask", wordMask, captured.getJSONArray("words_mask").getJSONArray(0).ints())
    val attention = captured.getJSONArray("attention_mask").getJSONArray(0).ints()
    checks.put(
      "attention_mask",
      compareFloats(
        prepared.attentionMask,
        FloatArray(n) { attention.getOrElse(it) { 0 }.toFloat() },
      ),
    )
    val textRoute = FloatArray(t * n)
    first.forEachIndexed { word, token -> textRoute[word * n + token] = 1f }
    val parentRoute = FloatArray(n)
    parents.forEach { parentRoute[it] = 1f }
    val labelRoute = FloatArray(5 * n)
    entityPositions.forEachIndexed { label, token -> labelRoute[label * n + token] = 1f }
    checks
      .put("text_routing", compareFloats(prepared.textRouting, textRoute))
      .put("parent_routing", compareFloats(prepared.parentRouting, parentRoute))
      .put("label_routing", compareFloats(prepared.labelRouting, labelRoute))
      .put(
        "text_mask",
        compareFloats(prepared.textMask, FloatArray(t) { if (it < first.size) 1f else 0f }),
      )
    return checks
  }

  private fun graphArrays(value: GliformerInputs.Prepared) =
    linkedMapOf(
      "attention_mask" to value.attentionMask,
      "text_routing" to value.textRouting,
      "parent_routing" to value.parentRouting,
      "label_routing" to value.labelRouting,
      "text_mask" to value.textMask,
    )

  private fun compareFloats(actual: FloatArray, expected: FloatArray): JSONObject {
    var mismatches = abs(actual.size - expected.size)
    var maximum = 0.0
    var first = -1
    for (i in 0 until minOf(actual.size, expected.size)) {
      val difference = abs(actual[i].toDouble() - expected[i].toDouble())
      maximum = maxOf(maximum, difference)
      if (actual[i] != expected[i]) {
        mismatches++
        if (first == -1) first = i
      }
    }
    return JSONObject()
      .put("equal", mismatches == 0)
      .put("mismatches", mismatches)
      .put("max_absolute_error", maximum)
      .put("first_mismatch", first)
      .put("actual_elements", actual.size)
      .put("expected_elements", expected.size)
  }

  private fun compareEntities(
    actual: List<GliformerDecoder.Entity>,
    expected: JSONArray,
  ): JSONObject {
    val actualMap = actual.associateBy { Triple(it.label, it.start, it.end) }
    val expectedMap =
      (0 until expected.length()).associate { i ->
        expected.getJSONObject(i).let {
          Triple(it.getString("label"), it.getInt("start"), it.getInt("end")) to
            it.getDouble("score")
        }
      }
    val maximum =
      actualMap.keys.intersect(expectedMap.keys).maxOfOrNull {
        abs(actualMap.getValue(it).score.toDouble() - expectedMap.getValue(it))
      } ?: 0.0
    return JSONObject()
      .put("span_sets_equal", actualMap.keys == expectedMap.keys && actualMap.size == actual.size)
      .put("max_score_difference", maximum)
      .put("actual_count", actual.size)
      .put("expected_count", expected.length())
      .put("missing", JSONArray((expectedMap.keys - actualMap.keys).map { it.toString() }))
      .put("extra", JSONArray((actualMap.keys - expectedMap.keys).map { it.toString() }))
  }

  private fun logitStats(actual: FloatArray, reference: FloatArray, count: Int): JSONObject {
    require(actual.size >= count && reference.size >= count)
    var maxError = 0.0
    var actualNorm = 0.0
    var referenceNorm = 0.0
    for (i in 0 until count) {
      val a = actual[i].toDouble()
      val b = reference[i].toDouble()
      maxError = maxOf(maxError, abs(a - b))
      actualNorm += a * a
      referenceNorm += b * b
    }
    return JSONObject()
      .put("all_finite", actual.all { it.isFinite() })
      .put("actual_min", actual.minOrNull()!!.toDouble())
      .put("actual_max", actual.maxOrNull()!!.toDouble())
      .put("valid_elements", count)
      .put("valid_max_absolute_error", maxError)
      .put("valid_actual_l2", sqrt(actualNorm))
      .put("valid_reference_l2", sqrt(referenceNorm))
      .put(
        "valid_norm_ratio",
        if (referenceNorm > 0.0) sqrt(actualNorm / referenceNorm) else JSONObject.NULL,
      )
  }

  private fun summarizeTimings(samples: JSONArray): JSONObject {
    val output = JSONObject()
    for (name in
      listOf(
        "tokenize_lookup_ms",
        "graph_readback_ms",
        "decode_ms",
        "total_ms",
        "encoder_readback_ms",
        "head_readback_ms",
      )) {
      val values =
        (0 until samples.length()).map { samples.getJSONObject(it).getDouble(name) }.sorted()
      val middle = values.size / 2
      val median =
        if (values.size % 2 == 1) values[middle] else (values[middle - 1] + values[middle]) / 2
      output.put(
        name,
        JSONObject()
          .put("median", median)
          .put("min", values.first())
          .put("max", values.last())
          .put("spread_max_minus_min", values.last() - values.first())
          .put("count", values.size),
      )
    }
    return output
  }

  private fun entities(values: List<GliformerDecoder.Entity>) =
    JSONArray(
      values.map {
        JSONObject()
          .put("label", it.label)
          .put("text", it.text)
          .put("start", it.start)
          .put("end", it.end)
          .put("score", it.score.toDouble())
      }
    )

  private fun memory(): JSONObject {
    val info = Debug.MemoryInfo()
    Debug.getMemoryInfo(info)
    val result =
      JSONObject()
        .put("total_pss_kib", info.totalPss)
        .put("total_pss_bytes", info.totalPss.toLong() * 1024)
        .put("native_pss_kib", info.nativePss)
        .put("dalvik_pss_kib", info.dalvikPss)
        .put("source", "android.os.Debug.MemoryInfo plus /proc/self/status")
    try {
      val status = File("/proc/self/status").readText()
      for (name in listOf("VmHWM", "VmRSS", "VmPeak", "VmSize")) {
        val match = Regex("(?m)^$name:\\s+(\\d+)\\s+kB").find(status)
        result.put(
          "${name}_bytes",
          match?.groupValues?.get(1)?.toLong()?.times(1024) ?: JSONObject.NULL,
        )
      }
    } catch (failure: Exception) {
      result.put("proc_status_error", failure.toString())
    }
    return result
  }

  private fun battery(): JSONObject {
    val value = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    return JSONObject()
      .put("level", value?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1)
      .put("scale", value?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1)
      .put(
        "temperature_c",
        (value?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1000) ?: -1000) / 10.0,
      )
      .put("plugged", value?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1)
  }

  private fun device() =
    JSONObject()
      .put("manufacturer", Build.MANUFACTURER)
      .put("model", Build.MODEL)
      .put("fingerprint", Build.FINGERPRINT)
      .put("sdk", Build.VERSION.SDK_INT)
      .put("litert_version", "2.2.0")
      .put("debuggable", context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0)
      .put("build_variant", "debug")

  private fun finish() {
    report
      .put("finished_at", Instant.now().toString())
      .put("memory_after_gate", memory())
      .put("battery_after", battery())
    save()
  }

  private fun save() {
    report.put("updated_at", Instant.now().toString())
    val temporary = File(context.filesDir, "$reportName.tmp")
    temporary.writeText(report.toString(2) + "\n")
    check(temporary.renameTo(reportFile)) { "Cannot atomically publish $reportName" }
  }

  private fun allChecksPass(checks: JSONObject): Boolean =
    checks.keys().asSequence().all { checks.getJSONObject(it).getBoolean("equal") }

  private fun boolCheck(value: Boolean) = JSONObject().put("equal", value)

  private fun indexWindows(manifest: JSONObject): Map<String, JSONObject> =
    listOf("rows", "forced_rows")
      .flatMap { name ->
        val values = manifest.optJSONArray(name) ?: JSONArray()
        (0 until values.length()).map { values.getJSONObject(it) }
      }
      .associate { "${it.getString("id")}_s${it.getInt("window")}" to it }

  private fun index(rows: JSONArray) =
    (0 until rows.length()).associate {
      rows.getJSONObject(it).let { row -> row.getString("id") to row }
    }

  private fun assetBytes(path: String): ByteArray {
    if (path == "corpus.json" || path == "unicode.json") {
      return context.assets.open("gate/$path").use { it.readBytes() }
    }
    val directory = File(context.filesDir, "gate_fixtures").canonicalFile
    val file = File(directory, path).canonicalFile
    require(file.toPath().startsWith(directory.toPath())) { "Invalid fixture path" }
    require(file.isFile) { "Missing fixture $path; run install_to_device.sh --fixtures DATA" }
    return file.readBytes()
  }

  private fun assetJson(path: String) = JSONObject(assetBytes(path).toString(Charsets.UTF_8))

  private fun floats(bytes: ByteArray): FloatArray =
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let { buffer ->
      FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

  private fun JSONArray.ints() = IntArray(length()) { getInt(it) }

  private fun JSONArray.strings() = (0 until length()).map { getString(it) }

  private fun sha256(bytes: ByteArray) =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  private fun elapsedMs(start: Long) = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
}
