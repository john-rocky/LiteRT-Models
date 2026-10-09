package com.d1omni

import android.content.Context
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** One question of an image record: the Python host's encoded row and the question it encodes. */
class D1VisionRow(
  val key: String,
  val name: String,
  val question: D1Question,
  val ids: IntArray,
  val markers: IntArray,
  val prefixRows: Int,
  val options: Int,
  val calibrate: Boolean,
)

/** The Python host's values for one crop: its size, grid and the sha256 of its tower inputs. */
class D1CropReference(
  val height: Int,
  val width: Int,
  val gridHeight: Int,
  val gridWidth: Int,
  val pixelsSha256: String,
  val posSha256: String,
  val maskSha256: String,
)

/**
 * The Python host's values for one picture (`vision_dump_v.py`): the decoded RGB's sha256 (and a
 * file with its bytes in `files/`), the layout, every crop's tower inputs by sha256, P and the Mac
 * CPU's prefix rows (a file of P x 1024 float32, the reference of `prefix_vs_mac`).
 */
class D1VisionReference(
  val height: Int,
  val width: Int,
  val rgbSha256: String,
  val rgbFile: String?,
  val layout: D1Layout,
  val crops: List<D1CropReference>,
  val prefixRows: Int,
  val prefixFile: String?,
  val prefixSha256: String?,
)

/** One picture request of a rows file: the picture in `files/`, the state, the questions' rows. */
class D1VisionRecord(
  val id: String,
  val mediaFile: String,
  val mediaSha256: String,
  val state: Any?,
  val rows: List<D1VisionRow>,
  val reference: D1VisionReference?,
)

/** A timing set: [name], [kind] (`request`: the record's questions as one request) and the record. */
class D1VisionTimingSet(val name: String, val kind: String, val record: D1VisionRecord)

/**
 * The rows files of the picture gate and timing (`D/device/r3/rows_image_*.json`,
 * `timing_image.json`), Android-free.
 */
object D1VisionRows {
  fun parse(bytes: ByteArray): List<D1VisionRecord> {
    val root = D1Json.parse(bytes) as Map<*, *>
    require(root["kind"] == "image") { "a picture rows file has kind image, not ${root["kind"]}" }
    require((root["pad_id"] as JsonNumber).toInt() == 0) { "pad_id must be 0" }
    return (root["records"] as List<*>).map { record(it as Map<*, *>) }
  }

  fun parseTiming(bytes: ByteArray): List<D1VisionTimingSet> {
    val root = D1Json.parse(bytes) as Map<*, *>
    require(root["kind"] == "image_timing") { "a picture timing file has kind image_timing, not ${root["kind"]}" }
    require((root["pad_id"] as JsonNumber).toInt() == 0) { "pad_id must be 0" }
    return (root["sets"] as List<*>).map {
      val set = it as Map<*, *>
      D1VisionTimingSet(set["name"] as String, set["kind"] as String, record(set["record"] as Map<*, *>))
    }
  }

  private fun record(entry: Map<*, *>): D1VisionRecord {
    val id = entry["id"] as String
    val mediaFile = entry["media_file"] as String
    require(D1Launch.fileNameValid(mediaFile)) { "$id: media file $mediaFile is not a plain file name" }
    val questions = entry["questions"] as Map<*, *>
    val rows =
      (entry["expected"] as List<*>).map {
        val row = it as Map<*, *>
        val name = row["name"] as String
        val question = D1Prompt.asQuestion(questions[name])
        val qtype = (row["qtype"] as JsonNumber).toInt()
        require(question.type.index == qtype) { "$id/$name: qtype $qtype but a ${question.type} question" }
        val options = (row["K"] as JsonNumber).toInt()
        require(options == question.options) { "$id/$name: K $options, the question has ${question.options}" }
        val markers = ints(row["markers"])
        require(markers.size >= options) { "$id/$name: ${markers.size} markers for K = $options" }
        D1VisionRow(
          row["key"] as String,
          name,
          question,
          ints(row["ids"]),
          markers,
          (row["P"] as JsonNumber).toInt(),
          options,
          row["calibrate"] as Boolean,
        )
      }
    return D1VisionRecord(
      id,
      mediaFile,
      entry["media_sha256"] as String,
      entry["state"],
      rows,
      (entry["reference"] as Map<*, *>?)?.let { reference(it) },
    )
  }

  private fun reference(ref: Map<*, *>): D1VisionReference {
    val layout = ref["layout"] as Map<*, *>
    fun int(map: Map<*, *>, key: String) = (map[key] as JsonNumber).toInt()
    fun name(key: String): String? =
      (ref[key] as String?)?.also { require(D1Launch.fileNameValid(it)) { "$key $it is not a plain file name" } }
    return D1VisionReference(
      int(ref, "h"),
      int(ref, "w"),
      ref["rgb_sha256"] as String,
      name("rgb_file"),
      D1Layout(int(layout, "grid_cols"), int(layout, "grid_rows"), int(layout, "thumb_h"), int(layout, "thumb_w"),
        layout["tiled"] as Boolean),
      (ref["crops"] as List<*>).map {
        val crop = it as Map<*, *>
        val hw = ints(crop["hw"])
        val grid = ints(crop["grid"])
        D1CropReference(hw[0], hw[1], grid[0], grid[1], crop["pixels_sha256"] as String,
          crop["pos_sha256"] as String, crop["mask_sha256"] as String)
      },
      int(ref, "P"),
      name("prefix_file"),
      ref["prefix_sha256"] as String?,
    )
  }

  private fun ints(value: Any?): IntArray = (value as List<*>).map { (it as JsonNumber).toInt() }.toIntArray()
}

/** sha256 and comparisons of the gate's arrays, Android-free. */
object D1VisionChecks {
  fun sha256(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  /** sha256 of float32 values as little-endian bytes (numpy's `tobytes()` on this machine). */
  fun sha256(values: FloatArray): String = sha256(floatBytes(values))

  fun floatBytes(values: FloatArray): ByteArray {
    val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buffer.asFloatBuffer().put(values)
    return buffer.array()
  }

  fun floats(bytes: ByteArray): FloatArray {
    require(bytes.size % 4 == 0) { "${bytes.size} bytes are not float32 values" }
    val out = FloatArray(bytes.size / 4)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
  }

  /** Bytes that differ and the largest difference of two uint8 arrays of one size. */
  fun byteDiff(a: ByteArray, b: ByteArray): LinkedHashMap<String, Any?> {
    if (a.size != b.size) return linkedMapOf("sizes" to listOf(a.size, b.size))
    var count = 0
    var largest = 0
    var first = -1
    for (index in a.indices) {
      val d = Math.abs((a[index].toInt() and 0xff) - (b[index].toInt() and 0xff))
      if (d != 0) {
        count++
        largest = maxOf(largest, d)
        if (first < 0) first = index
      }
    }
    return linkedMapOf("bytes_differ" to count, "max_abs" to largest, "first_index" to first)
  }

  /** max |a − b|, mean |a − b| and ||a − b|| / ||b|| (float64) of two float arrays of one size. */
  fun floatDiff(a: FloatArray, b: FloatArray): LinkedHashMap<String, Any?> {
    if (a.size != b.size) return linkedMapOf("sizes" to listOf(a.size, b.size))
    var largest = 0.0
    var sum = 0.0
    var squares = 0.0
    var norm = 0.0
    var nonfinite = 0
    for (index in a.indices) {
      if (!a[index].isFinite()) {
        nonfinite++
        continue
      }
      val d = Math.abs(a[index].toDouble() - b[index].toDouble())
      largest = maxOf(largest, d)
      sum += d
      squares += d * d
      norm += b[index].toDouble() * b[index].toDouble()
    }
    return linkedMapOf(
      "max_abs" to largest,
      "mean_abs" to sum / a.size,
      "rel_rms" to if (norm > 0) Math.sqrt(squares / norm) else null,
      "nonfinite" to nonfinite,
      "bit_equal" to a.contentEquals(b),
    )
  }
}

/**
 * The picture path's debug runs on the device, on [D1Runtime.dispatcher] (`--ez vgate true` /
 * `--ez vtiming true`, debug build):
 * - gate: every record of a rows file — its picture decoded from `files/` (decode_match = the
 *   Python host's RGB bit for bit), the prefix rows (per crop: the tower inputs' sha256 against the
 *   Python host's, `pixels_match` / `pos_match` / `mask_match`; the prefix against the Mac CPU's as a
 *   reference value), then each question's row (the rows file's IDs, re-encoded by this app for
 *   `ids_match`) on the smallest resident decision graph that holds it, its probabilities; the time
 *   of every step; each compile's memory before and after;
 * - timing: per set, a wait for the GPU after the compiles, warm-up requests, then timed requests,
 *   each = reading and decoding the picture, the prefix, every question's decision graph call and
 *   read-out, with the time of each step.
 * The report is `files/<report>.partial` while running and `files/<report>` at the end; `files/STOP`
 * ends a run after the current record / request.
 */
class D1VisionGate(private val context: Context) {
  private val files = context.filesDir

  fun gate(args: D1Launch.VGate, progress: (String) -> Unit): D1GateRunner.Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val stop = File(files, D1GateRunner.STOP_FILE)
    stop.delete()
    destination.delete()
    val report = D1Device.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "d1omni_app_vgate",
        "status" to "RUNNING",
        "fixture" to args.fixture,
        "limit" to args.limit,
        "backend_requested" to args.backend.wireName,
        "precision_requested" to args.precision.wireName,
        "precision_vision_requested" to args.visionPrecision.wireName,
        "state_start" to D1Device.state(context),
        "call" to "graph call = write the inputs + CompiledModel.run() + readFloat(output); run() alone returns before the GPU work ends",
      )
    )
    var error: String? = null
    var stopped = false
    var engine: D1Engine? = null
    var vision: D1VisionEngine? = null
    val images = ArrayList<LinkedHashMap<String, Any?>>()
    val rowsOut = ArrayList<LinkedHashMap<String, Any?>>()
    try {
      progress("loading")
      val records = D1VisionRows.parse(File(files, args.fixture).readBytes())
      report["records_in_file"] = records.size
      report["rows_in_file"] = records.sumOf { it.rows.size }
      val loaded = D1Engine.load(context, args.backend, args.precision)
      engine = loaded
      report["engine_load_ms"] = loaded.loadMs
      report["installed"] = loaded.installed
      report["memory_before_vision"] = D1Device.memory(context)
      progress("compiling the vision tower and the projector")
      val opened = D1VisionEngine.open(context, args.backend, args.visionPrecision)
      vision = opened
      report["table_ms"] = opened.tableMs
      report["vision_compiles"] = opened.compiles.map { it.toJson() }
      report["memory_after_vision"] = D1Device.memory(context)
      val positions = records.flatMap { record -> record.rows.map { it.prefixRows + it.ids.size } }
      progress("compiling the decision graphs")
      report["prepare"] = prepared(loaded.prepare(positions))
      report["memory_after_prepare"] = D1Device.memory(context)
      report["resident"] = loaded.resident
      report["resident_backends"] = loaded.residentBackends.map { it.wireName }
      report["compiles"] = loaded.compiles.map { it.toJson() }
      report["vision_backends"] =
        linkedMapOf("vision_tower" to opened.towerBackend.wireName, "projector" to opened.projectorBackend.wireName)
      report["images"] = images
      report["rows"] = rowsOut
      write(partial, report)
      val limit = if (args.limit > 0) args.limit else Int.MAX_VALUE
      Log.i(
        D1GateRunner.LOG_TAG,
        "VGATE_START records=${records.size} resident=${loaded.resident} tower=${opened.towerBackend.wireName} projector=${opened.projectorBackend.wireName}",
      )
      for (record in records) {
        if (rowsOut.size >= limit) break
        if (stop.exists()) {
          stopped = true
          break
        }
        progress("${record.id}: decoding")
        val image = runRecord(loaded, opened, record, limit - rowsOut.size, rowsOut)
        images.add(image)
        write(partial, report.apply { putAll(summary(images, rowsOut)) })
      }
      report["compiles"] = loaded.compiles.map { it.toJson() }
      report["vision_compiles"] = opened.compiles.map { it.toJson() }
      report["memory_at_end"] = D1Device.memory(context)
    } catch (failure: Exception) {
      error = D1Decider.describe(failure)
    } catch (failure: LinkageError) {
      error = "Native runtime: ${D1Decider.describe(failure)}"
    } catch (failure: OutOfMemoryError) {
      error = D1Decider.describe(failure)
    }
    return finish(report, partial, destination, summary(images, rowsOut), error, stopped, engine, vision, "VGATE_DONE")
  }

  /** One record of the gate: decode, prefix, every question's row; its entry in `images`. */
  private fun runRecord(
    engine: D1Engine,
    vision: D1VisionEngine,
    record: D1VisionRecord,
    rowsLeft: Int,
    rowsOut: MutableList<LinkedHashMap<String, Any?>>,
  ): LinkedHashMap<String, Any?> {
    val reference = record.reference
    val image = linkedMapOf<String, Any?>("id" to record.id, "media_file" to record.mediaFile)
    val raw = File(files, record.mediaFile).readBytes()
    image["media_sha256_match"] = D1VisionChecks.sha256(raw) == record.mediaSha256
    var start = System.nanoTime()
    val decoded = D1Image.decode(raw)
    image["decode_ms"] = (System.nanoTime() - start) / 1e6
    val rgb = decoded.rgb
    image["hw"] = listOf(rgb.height, rgb.width)
    image["orientation"] = decoded.orientation
    image["bitmap_color_space"] = decoded.colorSpace
    image["png_chunks_stripped"] = decoded.strippedChunks
    val rgbSha = D1VisionChecks.sha256(rgb.data)
    image["rgb_sha256"] = rgbSha
    val decodeMatch = reference?.let { rgbSha == it.rgbSha256 && rgb.height == it.height && rgb.width == it.width }
    image["decode_match"] = decodeMatch
    val referenceRgb = reference?.rgbFile?.let { File(files, it) }?.takeIf { it.isFile }?.readBytes()
    if (decodeMatch == false && referenceRgb != null) {
      image["decode_diff"] = D1VisionChecks.byteDiff(rgb.data, referenceRgb)
    }
    val crops = ArrayList<LinkedHashMap<String, Any?>>()
    val observer =
      D1VisionPrefix.Observer { index, crop, patches, positions ->
        val ref = reference?.crops?.getOrNull(index)
        val pixels = D1VisionChecks.sha256(patches.pixels)
        val pos = D1VisionChecks.sha256(positions)
        val mask = D1VisionChecks.sha256(patches.mask)
        crops.add(
          linkedMapOf(
            "k" to index,
            "hw" to listOf(crop.height, crop.width),
            "grid" to listOf(patches.gridHeight, patches.gridWidth),
            "grid_match" to ref?.let { it.gridHeight == patches.gridHeight && it.gridWidth == patches.gridWidth &&
              it.height == crop.height && it.width == crop.width },
            "pixels_match" to ref?.let { it.pixelsSha256 == pixels },
            "pos_match" to ref?.let { it.posSha256 == pos },
            "mask_match" to ref?.let { it.maskSha256 == mask },
            "pixels_sha256" to pixels,
            "pos_sha256" to pos,
          )
        )
      }
    val prefixRun = vision.imagePrefix(rgb, observer)
    image["prefix"] = prefixRun.toJson()
    image["crop_checks"] = crops
    image["layout_match"] = reference?.let { it.layout == prefixRun.layout }
    image["crops_match"] = reference?.let { it.crops.size == crops.size }
    image["P"] = prefixRun.rows
    image["P_match"] = reference?.let { it.prefixRows == prefixRun.rows }
    reference?.prefixFile?.let { File(files, it) }?.takeIf { it.isFile }?.let { file ->
      val mac = D1VisionChecks.floats(file.readBytes())
      image["prefix_vs_mac"] = D1VisionChecks.floatDiff(prefixRun.prefix, mac).apply {
        put("reference_sha256_match", reference.prefixSha256 == null ||
          D1VisionChecks.sha256(file.readBytes()) == reference.prefixSha256)
      }
    }
    if (decodeMatch == false && referenceRgb != null) {
      // Decode and arithmetic apart: the host steps on the Python host's own RGB.
      val ref = requireNotNull(reference)
      val (refCrops, _) = D1Vision.crops(D1Rgb(ref.width, ref.height, referenceRgb))
      image["pixels_from_reference_rgb_match"] =
        refCrops.withIndex().all { (k, c) ->
          D1VisionChecks.sha256(D1Vision.toPatches(c).pixels) == ref.crops.getOrNull(k)?.pixelsSha256
        }
    }
    val tokenizer = engine.tokenizer
    var decisionMs = 0.0
    var ran = 0
    for (row in record.rows) {
      if (ran >= rowsLeft) break
      ran++
      require(row.prefixRows == prefixRun.rows) {
        "${row.key}: the row has P = ${row.prefixRows}, the picture gives ${prefixRun.rows}"
      }
      start = System.nanoTime()
      val encoded =
        D1Rows.rows(tokenizer, engine.contract, record.state, listOf(row.question), prefixRun.rows, D1Kind.IMAGE)
          .single()
      val encodeMs = (System.nanoTime() - start) / 1e6
      val bucket =
        requireNotNull(D1Contract.bucketFor(row.prefixRows + row.ids.size, engine.resident)) {
          "${row.key}: no resident graph holds ${row.prefixRows + row.ids.size} positions"
        }
      val inputs = D1Rows.buildInputs(row.ids, prefixRun.prefix, row.prefixRows, bucket, row.question.type)
      val wall = System.currentTimeMillis()
      val call = engine.call(bucket, inputs)
      val markerScores = D1Readout.markerScores(call.scores, row.prefixRows, row.markers, row.options)
      val probabilities = D1Readout.probabilities(markerScores, row.question, row.calibrate, engine.contract)
      decisionMs += call.totalMs
      val real = row.prefixRows + row.ids.size
      rowsOut.add(
        linkedMapOf(
          "key" to row.key,
          "id" to record.id,
          "n" to row.ids.size,
          "P" to row.prefixRows,
          "K" to row.options,
          "L" to bucket,
          "finite" to D1Readout.finite(markerScores),
          "nonfinite_real" to (0 until real).count { !call.scores[it].isFinite() },
          "probs" to probabilities.toList(),
          "scores_at_markers" to markerScores.map { it.toDouble() },
          "write_ms" to call.writeMs,
          "run_ms" to call.runMs,
          "read_ms" to call.readMs,
          "write_run_read_ms" to call.totalMs,
          "t_start_ms" to wall,
          "ids_match" to encoded.ids.contentEquals(row.ids),
          "markers_match" to encoded.markers.contentEquals(row.markers),
          "encode_ms" to encodeMs,
          "decode_match" to decodeMatch,
          "backend" to engine.graph(bucket).backend.wireName,
        )
      )
    }
    image["decision_ms"] = decisionMs
    image["rows"] = ran
    return image
  }

  fun timing(args: D1Launch.VTiming, progress: (String) -> Unit): D1GateRunner.Summary {
    val destination = File(files, args.report)
    val partial = File(files, "${args.report}.partial")
    val stop = File(files, D1GateRunner.STOP_FILE)
    stop.delete()
    destination.delete()
    val report = D1Device.header(context)
    report.putAll(
      linkedMapOf(
        "set" to "d1omni_app_vtiming",
        "status" to "RUNNING",
        "rows_file" to args.rows,
        "sets_requested" to args.sets,
        "warmup_calls_setting" to args.warmup,
        "reps" to args.reps,
        "cool_ms" to args.coolMs,
        "backend_requested" to args.backend.wireName,
        "precision_requested" to args.precision.wireName,
        "precision_vision_requested" to args.visionPrecision.wireName,
        "state_start" to D1Device.state(context),
        "calls_format" to CALL_COLUMNS,
      )
    )
    val results = LinkedHashMap<String, Any?>()
    var error: String? = null
    var stopped = false
    var engine: D1Engine? = null
    var vision: D1VisionEngine? = null
    try {
      val sets =
        D1VisionRows.parseTiming(File(files, args.rows).readBytes()).filter {
          args.sets == null || it.name in args.sets
        }
      require(sets.isNotEmpty()) { "no timing set ${args.sets ?: ""} in ${args.rows}" }
      val loaded = D1Engine.load(context, args.backend, args.precision)
      engine = loaded
      report["engine_load_ms"] = loaded.loadMs
      val base = D1Device.gpuState()
      report["gpu_state_before_compile"] = base?.toJson()
      progress("compiling the vision tower and the projector")
      val opened = D1VisionEngine.open(context, args.backend, args.visionPrecision)
      vision = opened
      report["table_ms"] = opened.tableMs
      report["timing"] = results
      for (set in sets) {
        if (stop.exists()) {
          stopped = true
          break
        }
        progress("timing ${set.name}: compiling")
        val positions = set.record.rows.map { it.prefixRows + it.ids.size }
        loaded.prepare(positions)
        progress("timing ${set.name}: cooling")
        val cool = coolDown(base, args.coolMs)
        val warmup = ArrayList<List<Any?>>()
        for (index in 0 until args.warmup) warmup.add(request(loaded, opened, set.record).first)
        val calls = ArrayList<List<Any?>>()
        val questionCalls = ArrayList<List<Double>>()
        var finite = true
        var probabilities: List<List<Double>>? = null
        progress("timing ${set.name}: ${args.reps} requests")
        for (round in 0 until args.reps) {
          if (stop.exists()) {
            stopped = true
            break
          }
          val (call, detail) = request(loaded, opened, set.record)
          calls.add(call)
          questionCalls.add(detail.questionMs)
          finite = finite && detail.finite
          probabilities = detail.probabilities
        }
        val entry =
          linkedMapOf<String, Any?>(
            "kind" to set.kind,
            "record" to set.record.id,
            "rows" to set.record.rows.size,
            "keys" to set.record.rows.map { it.key },
            "P" to set.record.rows.firstOrNull()?.prefixRows,
            "tokens" to set.record.rows.map { it.ids.size },
            "buckets" to set.record.rows.map { D1Contract.bucketFor(it.prefixRows + it.ids.size, loaded.resident) },
            "decision_backends" to loaded.residentBackends.map { it.wireName },
            "vision_backends" to listOf(opened.towerBackend.wireName, opened.projectorBackend.wireName),
            "precision" to args.precision.wireName,
            "precision_vision" to args.visionPrecision.wireName,
            "cool" to cool,
            "warmup_calls" to warmup,
            "timed_calls" to calls,
            "question_decision_ms" to questionCalls,
            "finite_markers" to finite,
            "last_probabilities" to probabilities,
            "stopped_early" to stopped,
          )
        for ((column, name) in CALL_COLUMNS.withIndex()) {
          if (column in 1..8) entry["${name}_stats"] = D1GateCore.stats(calls.map { it[column] as Double })
        }
        results[set.name] = entry
        write(partial, report)
        Log.i(
          D1GateRunner.LOG_TAG,
          "VTIMING_SET ${set.name} median=${(entry["total_ms_stats"] as Map<*, *>?)?.get("median")}",
        )
        if (stopped) break
      }
      report["compiles"] = loaded.compiles.map { it.toJson() }
      report["vision_compiles"] = opened.compiles.map { it.toJson() }
      report["resident"] = loaded.resident
      report["memory_at_end"] = D1Device.memory(context)
    } catch (failure: Exception) {
      error = D1Decider.describe(failure)
    } catch (failure: LinkageError) {
      error = "Native runtime: ${D1Decider.describe(failure)}"
    } catch (failure: OutOfMemoryError) {
      error = D1Decider.describe(failure)
    }
    return finish(report, partial, destination, linkedMapOf(), error, stopped, engine, vision, "VTIMING_DONE")
  }

  private class RequestDetail(
    val questionMs: List<Double>,
    val finite: Boolean,
    val probabilities: List<List<Double>>,
  )

  /**
   * One timed request: read and decode the picture, the prefix, every question's inputs + decision
   * graph call + read-out. [CALL_COLUMNS] with the kgsl clock ceiling read before it.
   */
  private fun request(
    engine: D1Engine,
    vision: D1VisionEngine,
    record: D1VisionRecord,
  ): Pair<List<Any?>, RequestDetail> {
    val clock = D1Device.gpuState()?.maxClockMhz
    val wall = System.currentTimeMillis()
    val start = System.nanoTime()
    val decoded = D1Image.decode(File(files, record.mediaFile))
    val decoded1 = System.nanoTime()
    val prefixRun = vision.imagePrefix(decoded.rgb)
    val prefixed = System.nanoTime()
    val questionMs = ArrayList<Double>()
    val probabilities = ArrayList<List<Double>>()
    var finite = true
    for (row in record.rows) {
      val begin = System.nanoTime()
      val bucket = requireNotNull(D1Contract.bucketFor(row.prefixRows + row.ids.size, engine.resident))
      val inputs = D1Rows.buildInputs(row.ids, prefixRun.prefix, row.prefixRows, bucket, row.question.type)
      val call = engine.call(bucket, inputs)
      val markerScores = D1Readout.markerScores(call.scores, row.prefixRows, row.markers, row.options)
      probabilities.add(D1Readout.probabilities(markerScores, row.question, row.calibrate, engine.contract).toList())
      finite = finite && D1Readout.finite(markerScores)
      questionMs.add((System.nanoTime() - begin) / 1e6)
    }
    val end = System.nanoTime()
    val call =
      listOf(
        wall,
        (end - start) / 1e6,
        (decoded1 - start) / 1e6,
        prefixRun.resizeNanos / 1e6,
        prefixRun.positionsNanos / 1e6,
        prefixRun.towerNanos / 1e6,
        prefixRun.unshuffleNanos / 1e6,
        prefixRun.projectorNanos / 1e6,
        (end - prefixed) / 1e6,
        clock,
      )
    return call to RequestDetail(questionMs, finite, probabilities)
  }

  /**
   * Waits at most [coolMs] until kgsl's clock ceiling is at least [base]'s, its thermal power level
   * 0 and its temperature at most 5 °C above [base]'s; a fixed sleep when kgsl is unreadable
   * (`D1TimingRunner`'s rule).
   */
  private fun coolDown(base: D1GpuState?, coolMs: Long): LinkedHashMap<String, Any?> {
    val record = linkedMapOf<String, Any?>("cool_ms" to coolMs)
    if (coolMs <= 0) return record.apply { put("mode", "none") }
    val start = System.currentTimeMillis()
    if (base == null || D1Device.gpuState() == null) {
      Thread.sleep(coolMs)
      return record.apply {
        put("mode", "fixed")
        put("waited_ms", System.currentTimeMillis() - start)
      }
    }
    fun recovered(now: D1GpuState?): Boolean =
      now != null &&
        now.maxClockMhz >= base.maxClockMhz &&
        (now.thermalPowerLevel ?: 0) == 0 &&
        now.tempMilliC <= base.tempMilliC + COOL_MARGIN_MILLI_C
    var now = D1Device.gpuState()
    while (!recovered(now) && System.currentTimeMillis() - start < coolMs) {
      Thread.sleep(COOL_POLL_MS)
      now = D1Device.gpuState()
    }
    return record.apply {
      put("mode", "kgsl")
      put("waited_ms", System.currentTimeMillis() - start)
      put("base", base.toJson())
      put("end", now?.toJson())
      put("recovered", recovered(now))
      put("thermal_status", D1Device.thermalStatus(context))
      put("cpu_caps", D1Device.cpuCaps())
    }
  }

  private fun prepared(prepared: D1Prepared): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "row_buckets" to prepared.rowBuckets,
      "compiled" to prepared.compiled,
      "closed" to prepared.closed,
      "avail_mem_bytes_before_second" to prepared.availableBeforeSecond,
      "second_refused" to prepared.secondRefused,
    )

  private fun summary(
    images: List<Map<String, Any?>>,
    rows: List<Map<String, Any?>>,
  ): LinkedHashMap<String, Any?> {
    fun count(list: List<Map<String, Any?>>, key: String, value: Any?) = list.count { it[key] == value }
    fun cropCount(key: String) =
      images.sumOf { image -> (image["crop_checks"] as List<*>? ?: emptyList<Any>()).count { (it as Map<*, *>)[key] == true } }
    val cropTotal = images.sumOf { (it["crop_checks"] as List<*>? ?: emptyList<Any>()).size }
    return linkedMapOf(
      "summary" to
        linkedMapOf(
          "images_run" to images.size,
          "rows_run" to rows.size,
          "finite_rows" to count(rows, "finite", true),
          "nonfinite_rows" to count(rows, "finite", false),
          "ids_match" to "${count(rows, "ids_match", true)}/${rows.size}",
          "markers_match" to "${count(rows, "markers_match", true)}/${rows.size}",
          "decode_match" to "${count(images, "decode_match", true)}/${images.size}",
          "pixels_match" to "${cropCount("pixels_match")}/$cropTotal",
          "pos_match" to "${cropCount("pos_match")}/$cropTotal",
          "mask_match" to "${cropCount("mask_match")}/$cropTotal",
          "grid_match" to "${cropCount("grid_match")}/$cropTotal",
          "P_match" to "${count(images, "P_match", true)}/${images.size}",
          "rows_by_bucket" to rows.groupingBy { it["L"] }.eachCount().mapKeys { "L${it.key}" },
        )
    )
  }

  private fun finish(
    report: LinkedHashMap<String, Any?>,
    partial: File,
    destination: File,
    summary: Map<String, Any?>,
    error: String?,
    stopped: Boolean,
    engine: D1Engine?,
    vision: D1VisionEngine?,
    doneTag: String,
  ): D1GateRunner.Summary {
    report.putAll(summary)
    val status =
      when {
        error != null -> "FAILED"
        stopped -> "STOPPED"
        else -> "DONE"
      }
    report["status"] = status
    report["stopped_early"] = stopped
    report["error"] = error
    report["state_end"] = D1Device.state(context)
    for (closeable in listOf(vision, engine)) {
      try {
        closeable?.close()
      } catch (failure: Exception) {
        report["close_error"] = D1Decider.describe(failure)
      }
    }
    write(partial, report)
    check(partial.renameTo(destination)) { "Could not save ${destination.absolutePath}" }
    Log.i(D1GateRunner.LOG_TAG, "$doneTag status=$status path=${destination.absolutePath}")
    if (error != null) D1Demo.failed(error)
    return D1GateRunner.Summary(status, destination.absolutePath, error)
  }

  private fun write(file: File, report: Map<String, Any?>) {
    val temporary = File(file.parentFile, "${file.name}.tmp")
    temporary.writeText(D1Json.writeIndented(report, 1) + "\n")
    check(temporary.renameTo(file)) { "Could not save ${file.absolutePath}" }
  }

  companion object {
    /** One timed request's columns, in milliseconds except the first and the last. */
    val CALL_COLUMNS =
      listOf(
        "wall_ms",
        "total_ms",
        "decode_ms",
        "resize_patches_ms",
        "pos_ms",
        "tower_ms",
        "unshuffle_ms",
        "projector_ms",
        "decision_ms",
        "kgsl_max_clock_mhz",
      )

    private const val COOL_MARGIN_MILLI_C = 5000
    private const val COOL_POLL_MS = 500L
  }
}
