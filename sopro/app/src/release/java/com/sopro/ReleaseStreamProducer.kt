// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/** Non-debuggable free-running stream measurement; no teacher/DSP/GPU fixture gates. */
class ReleaseStreamProducer(private val context: Context) {
  fun run(
    config: JSONObject,
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
    onlyId: String?,
    manifestPath: String,
    seed: Long,
  ): SoproGateEntry.Result {
    val root = MeasurementFiles.root(context, "r8_out", config.optString("tag", "stream"))
    val contractSet = config.optString("contract_set", "r9")
    require(contractSet in listOf("r6", "r9"))
    val styleVariant = PlacementConfig.StyleVariant.parse(config.optString("style_variant", "fp32"))
    val rows = JSONArray()
    val report =
      MeasurementFiles.base("stream", config.optString("apk_sha256", "unknown"))
        .put("status", "RUNNING")
        .put("rows", rows)
        .put("precision", precision.name.lowercase())
        .put("backend", backend.name.lowercase())
        .put("contract_set", contractSet)
        .put("style_variant", styleVariant.name.lowercase())
    fun save() = MeasurementFiles.save(root, report)
    save()
    try {
      val inputFile = File(context.filesDir, manifestPath).canonicalFile
      require(inputFile.path.startsWith(context.filesDir.canonicalPath + File.separator))
      val inputRoot = inputFile.parentFile!!
      val manifest = JSONObject(inputFile.readText())
      fun reference(id: String) = manifest.getJSONObject("references").getJSONObject(id)
      fun referenceWave(id: String): FloatArray {
        val spec = reference(id).getJSONObject("arrays").getJSONObject("reference_wav24")
        val path = File(inputRoot, spec.getString("path")).canonicalFile
        require(path.path.startsWith(inputRoot.canonicalPath + File.separator))
        require(MeasurementFiles.sha(path) == spec.getString("sha256"))
        require(spec.getString("dtype") == "float32")
        val bytes = path.readBytes()
        require(bytes.size % 4 == 0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { buffer.float }
      }
      val hero = ReferenceAudioDecoder.decodeBundled(context).wave
      SoproEngine(context.filesDir, placement, backend, precision, contractSet, styleVariant).use {
        engine ->
        val utterances = manifest.getJSONArray("utterances")
        var order = 0
        for (i in 0 until utterances.length()) {
          val item = utterances.getJSONObject(i)
          val id = item.getString("id")
          if (item.getString("suite") == "long") continue
          val utteranceSeed = seed + order++
          if (onlyId != null && onlyId != id) continue
          val rid = item.getString("reference_id")
          val wave = if (rid == "hero") hero else referenceWave(rid)
          report.put("current_id", id)
          save()
          var samples = 0
          val result =
            engine.synthesizeStreaming(
              item.getString("text"),
              item.getString("lang"),
              wave,
              { samples += it.size },
              seed = utteranceSeed,
              tapNanos = System.nanoTime(),
              referenceId = rid,
              fixedReferenceLevelDb =
                if (rid == "hero") null else reference(rid).getDouble("reference_level_db"),
            )
          val stats =
            requireNotNull(result.stats)
              .toJson()
              .put("trim", JSONObject(result.trim.asMap()))
              .put("contract_set", contractSet)
              .put("style_variant", styleVariant.name.lowercase())
          val files = WavFiles.write(root, item.getString("lang"), rid, result.wav, stats)
          val arrays =
            JSONObject()
              .put(
                "raw_segment_wav",
                MeasurementFiles.dump(root, "$id/raw_segment_wav", result.raw),
              )
              .put("final_wav", MeasurementFiles.dump(root, "$id/final_wav", result.wav))
          rows.put(
            JSONObject()
              .put("phone", MeasurementFiles.phone())
              .put("build", "release")
              .put("apk_sha256", config.optString("apk_sha256", "unknown"))
              .put("id", id)
              .put("text", item.getString("text"))
              .put("lang", item.getString("lang"))
              .put("reference_id", rid)
              .put("seed", utteranceSeed)
              .put("wav", files.wav.name)
              .put("sidecar", files.sidecar.name)
              .put("stats", stats)
              .put("arrays", arrays)
              .put("creation_ms", JSONObject(engine.creationMs.toMap()))
              .put("sink_samples", samples)
          )
          save()
        }
      }
      require(rows.length() > 0)
      report.put("status", "PRODUCED").put("finished_unix_ms", System.currentTimeMillis())
      save()
      return SoproGateEntry.Result(File(root, "index.json").absolutePath, "PRODUCED")
    } catch (failure: Throwable) {
      report
        .put("status", "ERROR")
        .put("error", failure.stackTraceToString())
        .put("finished_unix_ms", System.currentTimeMillis())
      save()
      return SoproGateEntry.Result(File(root, "index.json").absolutePath, "ERROR", failure.message)
    }
  }
}
