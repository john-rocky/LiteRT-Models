// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.content.Context
import android.os.Build
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-only fixed-token/x0 replay. Device acceptance is decided from saved tensor measurements.
 */
class SoproGateRunner(private val context: Context) {
  data class Result(val path: String, val status: String, val error: String? = null)

  /** Free-running producer for later device TTFA/RTF and quality judging; no phone-side gate. */
  fun stream(
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
    onlyId: String?,
    manifestPath: String,
    seedBase: Long,
  ): Result {
    val directory =
      File(context.filesDir, "stream_results/${System.currentTimeMillis()}").apply { mkdirs() }
    val reportFile = File(directory, "index.json")
    val rows = JSONArray()
    val report =
      JSONObject()
        .put("mode", "stream")
        .put("status", "RUNNING")
        .put("seed_base", seedBase)
        .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
        .put("build_fingerprint", Build.FINGERPRINT)
        .put("debuggable", BuildConfig.DEBUG)
        .put("rows", rows)
        .put("sink", "PCM producer; no AudioTrack playback in headless gate")
    fun save() {
      reportFile.writeText(report.toString(2))
    }
    save()
    try {
      val manifest = privateFile(manifestPath)
      val utterances = JSONObject(manifest.readText()).getJSONArray("utterances")
      SoproEngine(context.filesDir, placement, backend, precision).use { engine ->
        for (i in 0 until utterances.length()) {
          val entry = utterances.getJSONObject(i)
          val id = entry.getString("id")
          if (onlyId != null && id != onlyId) continue
          val referenceId = entry.getString("reference_id")
          val reference =
            readArray(
              manifest.parentFile!!,
              entry.getJSONObject("arrays").getJSONObject("reference_wav24"),
            )
              as FloatArray
          var samples = 0
          // Headless equivalent of the tap begins immediately before the public engine call.
          val tap = System.nanoTime()
          val output =
            engine.synthesizeStreaming(
              entry.getString("text"),
              entry.getString("lang"),
              reference,
              { samples += it.size },
              seed = seedBase + i,
              tapNanos = tap,
              referenceId = referenceId,
              fixedReferenceLevelDb = entry.getDouble("reference_level_db"),
            )
          val stats =
            requireNotNull(output.stats)
              .toJson()
              .put("trim", JSONObject(output.trim.asMap()))
              .put("utterance_id", id)
              .put("stop_reason", output.stopReason)
          val files =
            WavFiles.write(directory, entry.getString("lang"), referenceId, output.wav, stats)
          rows.put(
            JSONObject()
              .put("id", id)
              .put("text", entry.getString("text"))
              .put("lang", entry.getString("lang"))
              .put("reference_id", referenceId)
              .put("wav", files.wav.name)
              .put("sidecar", files.sidecar.name)
              .put("seed", seedBase + i)
              .put("sink_samples", samples)
              .put("stats", stats)
              .put("status", "PRODUCED")
          )
          save()
        }
      }
      require(rows.length() > 0) { "No stream utterances matched ${onlyId ?: "all"}" }
      report.put("status", "PRODUCED")
      save()
      return Result(reportFile.absolutePath, "PRODUCED")
    } catch (failure: Throwable) {
      report.put("status", "FAIL").put("error", failure.stackTraceToString())
      save()
      return Result(reportFile.absolutePath, "FAIL", failure.message)
    }
  }

  fun run(
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
    onlyId: String?,
    manifestPath: String,
  ): Result {
    val reportFile =
      File(
        context.filesDir,
        "gate_results/${precision.name.lowercase()}_${backend.name.lowercase()}_" +
          "${System.currentTimeMillis()}.json",
      )
    reportFile.parentFile?.mkdirs()
    val rows = JSONArray()
    val report =
      JSONObject()
        .put("status", "RUNNING")
        .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
        .put("android", Build.VERSION.RELEASE)
        .put("build_fingerprint", Build.FINGERPRINT)
        .put("app_version", BuildConfig.VERSION_NAME)
        .put("debuggable", BuildConfig.DEBUG)
        .put("precision", precision.name.lowercase())
        .put("default_backend", backend.name.lowercase())
        .put("placement", JSONObject(placement.mapValues { it.value.name.lowercase() }))
        .put("timing_condition", "debug build; informational, no timing gate")
        .put("rows", rows)
    fun save() {
      reportFile.writeText(report.toString(2))
    }
    try {
      val manifestFile = privateFile(manifestPath)
      val manifest = JSONObject(manifestFile.readText())
      val utterances = manifest.getJSONArray("utterances")
      SoproEngine(context.filesDir, placement, backend, precision).use { engine ->
        for (i in 0 until utterances.length()) {
          val utterance = utterances.getJSONObject(i)
          val id = utterance.getString("id")
          if (onlyId != null && id != onlyId) continue
          val arrays = utterance.getJSONObject("arrays")
          fun floats(key: String): FloatArray =
            readArray(manifestFile.parentFile!!, arrays.getJSONObject(key)) as FloatArray
          fun ints(key: String): IntArray =
            readArray(manifestFile.parentFile!!, arrays.getJSONObject(key)) as IntArray
          val output =
            engine.teacher(
              ints("text_ids"),
              floats("reference_wav24"),
              utterance.getDouble("reference_level_db"),
              ints("ref_semantic_tokens"),
              ints("sampled_semantic_tokens"),
              floats("x0"),
            )
          val logits = FloatArray(output.logits.sumOf { it.size })
          var offset = 0
          output.logits.forEach {
            it.copyInto(logits, offset)
            offset += it.size
          }
          val expectedRefTokens = ints("ref_semantic_tokens")
          val referenceTokensExact =
            expectedRefTokens.indices.count { expectedRefTokens[it] == output.reference.tokens[it] }
          val measurements =
            JSONObject()
              .put("id_emb", compare(floats("id_emb"), output.reference.idEmbedding))
              .put("cond_vec", compare(floats("cond_vec"), output.reference.conditioning))
              .put("logits", compare(floats("logits"), logits))
              .put(
                "solved_mel_normalized",
                compare(floats("solved_mel_normalized"), output.solvedMel),
              )
              .put("istft_features", compare(floats("istft_features"), output.features))
              .put("raw_segment_wav", compare(floats("raw_segment_wav"), output.raw))
              .put("final_wav", compare(floats("final_wav"), output.wav))
              .put("oracle_raw_segment_wav", compare(floats("oracle_raw_segment_wav"), output.raw))
          val expectedTrim = utterance.getJSONObject("trim")
          val trim = JSONObject(output.trim.asMap())
          val trimExact =
            listOf("lead_cut_samples", "trail_end_after_lead_samples", "final_samples").all {
              trim.getInt(it) == expectedTrim.getInt(it)
            }
          val graphMs = JSONArray()
          output.graphTimings.forEach { t ->
            graphMs.put(
              JSONObject()
                .put("graph", t.graph)
                .put("signature", t.signature)
                .put("backend", t.backend)
                .put("write_ms", t.writeMs)
                .put("run_ms", t.runMs)
                .put("readback_ms", t.readbackMs)
                .put("total_ms", t.totalMs)
            )
          }
          rows.put(
            JSONObject()
              .put("id", id)
              .put("token_count", output.tokens.size)
              .put("prefix_length", output.prefixLength)
              .put("reference_tokens_exact", referenceTokensExact)
              .put("reference_tokens_total", expectedRefTokens.size)
              .put("raw_samples", output.raw.size)
              .put("final_samples", output.wav.size)
              .put("measurements", measurements)
              .put("trim", trim)
              .put("trim_exact", trimExact)
              .put("graph_ms", graphMs)
              .put("elapsed_ms", output.elapsedMs)
              .put("status", "MEASURED")
          )
          save()
        }
      }
      require(rows.length() > 0) { "No gate utterances matched ${onlyId ?: "all"}" }
      report.put("status", "MEASURED")
      save()
      return Result(reportFile.absolutePath, "MEASURED")
    } catch (failure: Throwable) {
      report.put("status", "FAIL").put("error", failure.stackTraceToString())
      save()
      return Result(reportFile.absolutePath, "FAIL", failure.message)
    }
  }

  private fun privateFile(relative: String): File {
    val file = File(context.filesDir, relative).canonicalFile
    require(file.path.startsWith(context.filesDir.canonicalPath + File.separator))
    return file
  }

  private fun readArray(root: File, spec: JSONObject): Any {
    val file = File(root, spec.getString("path")).canonicalFile
    require(file.path.startsWith(root.canonicalPath + File.separator))
    val bytes = file.readBytes()
    require(
      MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) } ==
        spec.getString("sha256")
    )
    val shape = spec.getJSONArray("shape")
    var count = 1
    for (i in 0 until shape.length()) count *= shape.getInt(i)
    require(bytes.size == count * 4)
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    return when (spec.getString("dtype")) {
      "float32" -> FloatArray(count) { buffer.float }
      "int32" -> IntArray(count) { buffer.int }
      else -> error("Unsupported fixture type ${spec.getString("dtype")}")
    }
  }

  private fun compare(expected: FloatArray, actual: FloatArray): JSONObject {
    val report =
      JSONObject().put("expected_elements", expected.size).put("actual_elements", actual.size)
    if (expected.size != actual.size)
      return report.put("same_shape", false).put("finite", actual.all { it.isFinite() })
    var maxDiff = 0.0
    var sumA = 0.0
    var sumB = 0.0
    var aa = 0.0
    var bb = 0.0
    var ab = 0.0
    var maximum = 0.0
    var minimum = Double.POSITIVE_INFINITY
    var finite = true
    for (i in expected.indices) {
      val a = expected[i].toDouble()
      val b = actual[i].toDouble()
      finite = finite && a.isFinite() && b.isFinite()
      maxDiff = maxOf(maxDiff, abs(a - b))
      sumA += a
      sumB += b
      aa += a * a
      bb += b * b
      ab += a * b
      maximum = maxOf(maximum, abs(a))
      minimum = minOf(minimum, a)
    }
    if (!finite) return report.put("same_shape", true).put("finite", false)
    val count = expected.size.coerceAtLeast(1).toDouble()
    val denominator =
      sqrt(maxOf(0.0, aa - sumA * sumA / count) * maxOf(0.0, bb - sumB * sumB / count))
    val corr =
      if (denominator == 0.0) if (maxDiff == 0.0) 1.0 else 0.0
      else (ab - sumA * sumB / count) / denominator
    return report
      .put("same_shape", true)
      .put("finite", true)
      .put("max_abs_diff", maxDiff)
      .put("corr", corr)
      .put("reference_max_abs", maximum)
      .put("reference_rms", sqrt(aa / count))
  }
}
