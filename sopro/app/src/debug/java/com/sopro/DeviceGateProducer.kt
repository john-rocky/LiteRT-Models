// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.content.Context
import android.os.Build
import android.os.Process
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

/** Raw-array producer only. All parity, domain and quality decisions belong to the Mac judge. */
class DeviceGateProducer(private val context: Context) {
  private lateinit var root: File
  private lateinit var inputRoot: File
  private lateinit var manifest: JSONObject
  private lateinit var report: JSONObject
  private lateinit var reportFile: File
  private val rows = JSONArray()
  private var contractSet = "r9"
  private var styleVariant = PlacementConfig.StyleVariant.FP32

  private fun save() {
    val temporary = File(root, "index.json.tmp")
    temporary.writeText(report.toString(2))
    check(temporary.renameTo(reportFile))
  }

  private fun sha(file: File): String {
    val hash = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { stream ->
      val buffer = ByteArray(65536)
      while (true) {
        val n = stream.read(buffer)
        if (n < 0) break
        hash.update(buffer, 0, n)
      }
    }
    return hash.digest().joinToString("") { "%02x".format(it) }
  }

  private fun load(spec: JSONObject, base: File = inputRoot): Any {
    val file = File(base, spec.getString("path")).canonicalFile
    require(file.path.startsWith(base.canonicalPath + File.separator))
    require(sha(file) == spec.getString("sha256")) { "Fixture checksum mismatch ${file.name}" }
    val bytes = file.readBytes()
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    return if (spec.getString("dtype") == "int32") IntArray(bytes.size / 4) { buffer.int }
    else FloatArray(bytes.size / 4) { buffer.float }
  }

  private fun dump(path: String, array: Any, shape: IntArray? = null): JSONObject {
    val file = File(root, "$path.bin")
    file.parentFile?.mkdirs()
    val count =
      when (array) {
        is FloatArray -> array.size
        is IntArray -> array.size
        else -> error("Array type")
      }
    val buffer = ByteBuffer.allocate(65536).order(ByteOrder.LITTLE_ENDIAN)
    FileOutputStream(file).use { stream ->
      for (i in 0 until count) {
        if (buffer.remaining() < 4) {
          stream.write(buffer.array(), 0, buffer.position())
          buffer.clear()
        }
        when (array) {
          is FloatArray -> buffer.putFloat(array[i])
          is IntArray -> buffer.putInt(array[i])
        }
      }
      stream.write(buffer.array(), 0, buffer.position())
    }
    return JSONObject()
      .put("path", "$path.bin")
      .put("dtype", if (array is FloatArray) "float32" else "int32")
      .put("shape", JSONArray((shape ?: intArrayOf(count)).toList()))
      .put("sha256", sha(file))
      .put("bytes", count * 4)
  }

  private fun reference(id: String) = manifest.getJSONObject("references").getJSONObject(id)

  private fun refFloat(id: String, key: String) =
    load(reference(id).getJSONObject("arrays").getJSONObject(key)) as FloatArray

  private fun refInts(id: String, key: String) =
    load(reference(id).getJSONObject("arrays").getJSONObject(key)) as IntArray

  private fun timings(values: List<SoproEngine.CallTiming>) =
    JSONArray(
      values.map { t ->
        JSONObject()
          .put("graph", t.graph)
          .put("signature", t.signature)
          .put("backend", t.backend)
          .put("write_ms", t.writeMs)
          .put("run_ms", t.runMs)
          .put("read_ms", t.readbackMs)
          .put("total_ms", t.totalMs)
      }
    )

  fun run(
    mode: String,
    config: JSONObject,
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
    onlyId: String?,
    manifestPath: String,
    seed: Long,
  ): SoproGateEntry.Result {
    val tag = config.optString("tag", mode)
    contractSet = config.optString("contract_set", "r9")
    require(contractSet in listOf("r6", "r9"))
    styleVariant = PlacementConfig.StyleVariant.parse(config.optString("style_variant", "fp32"))
    require(tag.matches(Regex("[A-Za-z0-9_-]+")))
    root = File(context.filesDir, "r4_out/$tag").apply { mkdirs() }
    reportFile = File(root, "index.json")
    report =
      JSONObject()
        .put("phone", "${Build.MANUFACTURER} ${Build.MODEL}")
        .put("fingerprint", Build.FINGERPRINT)
        .put("pid", Process.myPid())
        .put("mode", mode)
        .put("status", "RUNNING")
        .put("rows", rows)
        .put("precision", precision.name.lowercase())
        .put("backend", backend.name.lowercase())
        .put("debuggable", BuildConfig.DEBUG)
        .put("producer_only", true)
        .put("started_unix_ms", System.currentTimeMillis())
        .put("apk_sha256", config.optString("apk_sha256", "unknown"))
        .put("contract_set", contractSet)
        .put("style_variant", styleVariant.name.lowercase())
        .put("build", config.optString("build", "debug"))
    save()
    try {
      val inputFile = File(context.filesDir, manifestPath).canonicalFile
      require(inputFile.path.startsWith(context.filesDir.canonicalPath + File.separator))
      inputRoot = inputFile.parentFile!!
      manifest = JSONObject(inputFile.readText())
      when (mode) {
        "dsp" -> dsp()
        "teacher" -> teacher(config, backend, precision, placement, onlyId)
        "stream" -> stream(backend, precision, placement, onlyId, seed)
        "gpu" -> gpu(config, backend, precision, placement)
        else -> error("Unknown producer mode $mode")
      }
      report.put("status", "PRODUCED").put("finished_unix_ms", System.currentTimeMillis())
      save()
      return SoproGateEntry.Result(reportFile.absolutePath, "PRODUCED")
    } catch (failure: Throwable) {
      report
        .put("status", "ERROR")
        .put("error", failure.stackTraceToString())
        .put("finished_unix_ms", System.currentTimeMillis())
      save()
      return SoproGateEntry.Result(reportFile.absolutePath, "ERROR", failure.message)
    }
  }

  private fun dsp() {
    val host = HostDsp.fromAssets(File(context.filesDir, "host_assets"))
    for (rid in listOf("ref1", "ref2", "hero")) {
      val reference =
        if (rid == "hero") host.fixedReference(ReferenceAudioDecoder.decodeBundled(context).wave)
        else
          HostDsp.NormalizedReference(
            refFloat(rid, "reference_wav24"),
            reference(rid).getDouble("reference_level_db"),
          )
      val wav16 = host.resample24to16(reference.wav)
      val arrays =
        JSONObject()
          .put("wav24", dump("$rid/wav24", reference.wav))
          .put("wav16", dump("$rid/wav16", wav16))
          .put(
            "speaker_mel",
            dump("$rid/speaker_mel", host.speakerMel(wav16), intArrayOf(1, 80, 1001)),
          )
          .put(
            "semantic_mel",
            dump("$rid/semantic_mel", host.semanticMel(wav16), intArrayOf(1, 80, 1002)),
          )
          .put(
            "ref_mel_normalized",
            dump(
              "$rid/ref_mel_normalized",
              host.acousticMelNormalized(reference.wav),
              intArrayOf(1, 100, 938),
            ),
          )
      rows.put(
        JSONObject()
          .put("id", rid)
          .put("reference_level_db", reference.levelDb)
          .put("arrays", arrays)
      )
      save()
    }
    val (cos, sin) = ArHost.rotaryCosSin()
    report
      .put("rope_cos", dump("rope/cos", cos, intArrayOf(1024, 64)))
      .put("rope_sin", dump("rope/sin", sin, intArrayOf(1024, 64)))
    val tokenizer = SpTokenizer(File(context.filesDir, "host_assets/tokenizer.model"))
    val cases = manifest.getJSONArray("tokenizer")
    val output = JSONArray()
    for (i in 0 until cases.length()) {
      val item = cases.getJSONObject(i)
      output.put(
        JSONObject()
          .put("id", item.getString("id"))
          .put("suite", item.getString("suite"))
          .put("lang", item.getString("lang"))
          .put(
            "ids",
            dump("tokenizer/$i", tokenizer.encode(item.getString("text"), item.getString("lang"))),
          )
      )
    }
    report.put("tokenizer", output)
    save()
  }

  private fun teacher(
    config: JSONObject,
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
    onlyId: String?,
  ) {
    val suite = config.optString("suite", "short")
    val useDsp = config.optBoolean("device_dsp", false)
    val recorded = HashSet<String>()
    SoproEngine(context.filesDir, placement, backend, precision, contractSet, styleVariant).use {
      engine ->
      var evidence = linkedMapOf<String, FloatArray>()
      var arKeys = mutableListOf<FloatArray>()
      var arValues = mutableListOf<FloatArray>()
      val captureAr = config.optBoolean("capture_ar", false)
      engine.recordOutputs = { name, signature, values ->
        if (name == "style_prefix") evidence["style_prefix"] = values[0] as FloatArray
        if (name == "semantic_encoder" && values[0] is FloatArray)
          evidence["semantic_digit_logits"] = values[0] as FloatArray
        if (captureAr && name == "ar_merged") {
          if (signature == "prefill") {
            evidence["ar_prefill_k"] = values[1] as FloatArray
            evidence["ar_prefill_v"] = values[2] as FloatArray
          } else {
            arKeys.add(values[1] as FloatArray)
            arValues.add(values[2] as FloatArray)
          }
        }
      }
      if (config.optBoolean("record_inputs", false))
        engine.recordInputs = { name, signature, args ->
          if (recorded.add(name)) {
            val inputs =
              JSONArray(args.mapIndexed { i, array -> dump("records/$name/input_$i", array) })
            val spec =
              JSONObject().put("graph", name).put("signature", signature).put("inputs", inputs)
            File(root, "records/$name/index.json").writeText(spec.toString(2))
          }
        }
      val utterances = manifest.getJSONArray("utterances")
      for (i in 0 until utterances.length()) {
        val item = utterances.getJSONObject(i)
        val id = item.getString("id")
        if (item.getString("suite") != suite || (onlyId != null && onlyId != id)) continue
        if (useDsp && id !in listOf("ref1_en01", "ref2_de02")) continue
        evidence = linkedMapOf()
        arKeys = mutableListOf()
        arValues = mutableListOf()
        report.put("current_id", id)
        save()
        val rid = item.getString("reference_id")
        val arrays = item.getJSONObject("arrays")
        fun ints(key: String) = load(arrays.getJSONObject(key)) as IntArray
        fun floats(key: String) = load(arrays.getJSONObject(key)) as FloatArray
        val wav24 = refFloat(rid, "reference_wav24")
        val level = reference(rid).getDouble("reference_level_db")
        val result =
          if (useDsp)
            engine.teacher(
              ints("text_ids"),
              wav24,
              level,
              refInts(rid, "ref_semantic_tokens"),
              ints("sampled_semantic_tokens"),
              floats("x0"),
            )
          else
            engine.teacherFromMels(
              ints("text_ids"),
              wav24,
              level,
              refFloat(rid, "speaker_mel"),
              refFloat(rid, "semantic_mel"),
              refFloat(rid, "ref_mel_normalized"),
              refInts(rid, "ref_semantic_tokens"),
              ints("sampled_semantic_tokens"),
              floats("x0"),
            )
        val logits = FloatArray(result.logits.sumOf { it.size })
        var offset = 0
        result.logits.forEach {
          it.copyInto(logits, offset)
          offset += it.size
        }
        val out =
          JSONObject()
            .put("raw_segment_wav", dump("$id/raw_segment_wav", result.raw))
            .put("final_wav", dump("$id/final_wav", result.wav))
            .put(
              "solved_mel_normalized",
              dump(
                "$id/solved_mel_normalized",
                result.solvedMel,
                intArrayOf(1, 100, result.solvedMel.size / 100),
              ),
            )
            .put("logits", dump("$id/logits", logits, intArrayOf(result.logits.size, 1, 4377)))
            .put("reference_tokens", dump("$id/reference_tokens", result.reference.tokens))
            .put("id_emb", dump("$id/id_emb", result.reference.idEmbedding, intArrayOf(1, 192)))
            .put(
              "cond_vec",
              dump("$id/cond_vec", result.reference.conditioning, intArrayOf(1, 512)),
            )
        evidence.forEach { (name, value) ->
          val shape =
            when (name) {
              "style_prefix" -> intArrayOf(1, 8, 512)
              "semantic_digit_logits" -> intArrayOf(1, 235, 27)
              else -> intArrayOf(1, 96, 256, 64)
            }
          out.put(name, dump("$id/$name", value, shape))
        }
        if (captureAr) {
          fun joined(parts: List<FloatArray>): FloatArray {
            val all = FloatArray(parts.sumOf { it.size })
            var index = 0
            parts.forEach {
              it.copyInto(all, index)
              index += it.size
            }
            return all
          }
          out.put(
            "ar_new_k",
            dump("$id/ar_new_k", joined(arKeys), intArrayOf(arKeys.size, 1, 96, 1, 64)),
          )
          out.put(
            "ar_new_v",
            dump("$id/ar_new_v", joined(arValues), intArrayOf(arValues.size, 1, 96, 1, 64)),
          )
        }
        rows.put(
          JSONObject()
            .put("id", id)
            .put("reference_id", rid)
            .put("arrays", out)
            .put("trim", JSONObject(result.trim.asMap()))
            .put("phone", report.getString("phone"))
            .put("build", report.getString("build"))
            .put("apk_sha256", report.getString("apk_sha256"))
            .put("graph_ms", timings(result.graphTimings))
            .put("creation_ms", JSONObject(engine.creationMs.toMap()))
            .put("wall_ms", result.elapsedMs)
            .put("rtf", result.elapsedMs / (result.wav.size / 24.0))
            .put("device_dsp", useDsp)
            .put("token_count", result.tokens.size)
            .put("prefix_length", result.prefixLength)
        )
        save()
      }
    }
    require(rows.length() > 0)
  }

  private fun stream(
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
    onlyId: String?,
    seed: Long,
  ) {
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
        val reference = if (rid == "hero") hero else refFloat(rid, "reference_wav24")
        report.put("current_id", id)
        save()
        var samples = 0
        val result =
          engine.synthesizeStreaming(
            item.getString("text"),
            item.getString("lang"),
            reference,
            { samples += it.size },
            seed = utteranceSeed,
            tapNanos = System.nanoTime(),
            referenceId = rid,
            fixedReferenceLevelDb =
              if (rid == "hero") null else this.reference(rid).getDouble("reference_level_db"),
          )
        val stats =
          requireNotNull(result.stats)
            .toJson()
            .put("trim", JSONObject(result.trim.asMap()))
            .put("contract_set", contractSet)
            .put("style_variant", styleVariant.name.lowercase())
        val files = WavFiles.write(root, item.getString("lang"), rid, result.wav, stats)
        val out =
          JSONObject()
            .put("raw_segment_wav", dump("$id/raw_segment_wav", result.raw))
            .put("final_wav", dump("$id/final_wav", result.wav))
        rows.put(
          JSONObject()
            .put("id", id)
            .put("text", item.getString("text"))
            .put("lang", item.getString("lang"))
            .put("reference_id", rid)
            .put("seed", utteranceSeed)
            .put("wav", files.wav.name)
            .put("sidecar", files.sidecar.name)
            .put("stats", stats)
            .put("arrays", out)
            .put("sink_samples", samples)
        )
        save()
      }
    }
    require(rows.length() > 0)
  }

  private fun gpu(
    config: JSONObject,
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
  ) {
    val name = config.getString("graph")
    val recordFile = File(context.filesDir, config.getString("record_manifest")).canonicalFile
    require(recordFile.path.startsWith(context.filesDir.canonicalPath + File.separator))
    val recorded = JSONObject(recordFile.readText())
    val inputs = recorded.getJSONArray("inputs")
    val args =
      List(inputs.length()) {
        load(inputs.getJSONObject(it), File(context.filesDir, config.getString("record_root")))
      }
    report.put("graph", name).put("creation_started", true)
    save()
    SoproEngine(
        context.filesDir,
        placement,
        backend,
        precision,
        contractSet,
        styleVariant,
        listOf(name),
      )
      .use { engine ->
        val start = System.nanoTime()
        try {
          engine.compileGraph(name)
        } finally {
          report.put("creation_ms", (System.nanoTime() - start) / 1e6)
          save()
        }
        report.put("compiled", true)
        save()
        val signature = recorded.getString("signature")
        val adapted = engine.adaptRecordedInputs(name, signature, args)
        report.put("input_contract_adapted", adapted !== args)
        val output = engine.invoke(name, adapted, signature)
        report
          .put("outputs", JSONArray(output.mapIndexed { i, value -> dump("output_$i", value) }))
          .put("graph_ms", timings(engine.timings))
        save()
      }
  }
}
