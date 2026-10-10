package com.d1omni

/**
 * One row of a gate or timing rows file (`D/device/r1/rows_L<L>.json`, `timing_rows.json`): the
 * Python host's encoded row ([ids], [markers], [prefixRows] = P, [options] = K, [type], whether the
 * read-out divides by the temperature) and, for the gate, the request it came from ([state],
 * [question], [stateHash] = sha256 of `serialize(state)`), so that the app can encode it again.
 */
class D1GateRow(
  val key: String,
  val ids: IntArray,
  val markers: IntArray,
  val type: QuestionType,
  val options: Int,
  val prefixRows: Int,
  val calibrate: Boolean,
  val hasRequest: Boolean,
  val state: Any?,
  val question: D1Question?,
  val stateHash: String?,
) {
  companion object {
    fun fromJson(value: Any?): D1GateRow {
      val row = value as Map<*, *>
      val key = row["key"] as String
      val qtype = (row["qtype"] as JsonNumber).toInt()
      val type =
        requireNotNull(QuestionType.entries.firstOrNull { it.index == qtype }) {
          "$key: qtype $qtype is not 0 (choice), 1 (score) or 2 (noul)"
        }
      val question = row["question"]?.let { D1Prompt.asQuestion(it) }
      if (question != null) {
        require(question.type == type) { "$key: qtype $qtype but a ${question.type} question" }
      }
      val options = (row["K"] as JsonNumber).toInt()
      val markers = ints(row["markers"])
      require(options in 1..markers.size) { "$key: K = $options but ${markers.size} markers" }
      return D1GateRow(
        key,
        ints(row["ids"]),
        markers,
        type,
        options,
        (row["P"] as JsonNumber?)?.toInt() ?: 0,
        row["calibrate"] as Boolean? ?: true,
        row.containsKey("state"),
        row["state"],
        question,
        row["state_hash"] as String?,
      )
    }

    private fun ints(value: Any?): IntArray =
      (value as List<*>).map { (it as JsonNumber).toInt() }.toIntArray()
  }
}

/** A gate rows file: the bucket [length] every row runs on, and the rows in order. */
class D1GateRows(val length: Int, val padId: Int, val rows: List<D1GateRow>) {
  companion object {
    fun parse(bytes: ByteArray): D1GateRows {
      val root = D1Json.parse(bytes) as Map<*, *>
      val padId = (root["pad_id"] as JsonNumber).toInt()
      require(padId == 0) { "pad_id $padId: the decision graph's pad id is 0" }
      return D1GateRows(
        (root["L"] as JsonNumber).toInt(),
        padId,
        (root["rows"] as List<*>).map { D1GateRow.fromJson(it) },
      )
    }
  }
}

/** A timing set: [name], [kind] (`single` or `request`), the bucket and the rows in call order. */
class D1TimingSet(val name: String, val kind: String, val length: Int, val rows: List<D1GateRow>) {
  companion object {
    /** The sets of a timing rows file, in file order. */
    fun parse(bytes: ByteArray): List<D1TimingSet> {
      val root = D1Json.parse(bytes) as Map<*, *>
      require((root["pad_id"] as JsonNumber).toInt() == 0) { "pad_id must be 0" }
      return (root["sets"] as List<*>).map {
        val set = it as Map<*, *>
        D1TimingSet(
          set["name"] as String,
          set["kind"] as String,
          (set["L"] as JsonNumber).toInt(),
          (set["rows"] as List<*>).map { row -> D1GateRow.fromJson(row) },
        )
      }
    }
  }
}

/** The app's own encoding of a gate row's request, against the row's IDs, markers and state. */
class D1Recheck(
  val idsMatch: Boolean?,
  val markersMatch: Boolean?,
  val stateMatch: Boolean?,
  val encodeMs: Double?,
)

/**
 * The gate's work on one row, Android-free: the six inputs from the row's own IDs, the read-out of
 * the graph's scores (Kotlin, float64) and the app's own encoding of the row's request ([recheck]:
 * the IDs the tokenizer makes on this device against the Python host's).
 */
class D1GateCore(private val tokenizer: D1Tokenizer, private val contract: D1Contract) {
  /**
   * The six inputs of [row] for a bucket of [length] positions (`build_inputs`): a media row (P > 0)
   * takes its [prefix] rows (P x 1024, the app's own audio prefix), a text row none.
   */
  fun inputs(row: D1GateRow, length: Int, prefix: FloatArray? = null): D1Inputs {
    if (row.prefixRows == 0) {
      require(prefix == null) { "${row.key}: a text row has no prefix" }
      return D1Rows.buildInputs(row.ids, null, 0, length, row.type)
    }
    val rows = requireNotNull(prefix) { "${row.key}: a media row (P = ${row.prefixRows}) needs its prefix" }
    return D1Rows.buildInputs(row.ids, rows, row.prefixRows, length, row.type)
  }

  /** Encodes the row's request again with this app's tokenizer and `encode` (text rows). */
  fun recheck(row: D1GateRow): D1Recheck {
    val question = row.question ?: return D1Recheck(null, null, null, null)
    if (!row.hasRequest) return D1Recheck(null, null, null, null)
    val start = System.nanoTime()
    val encoded = D1Rows.rows(tokenizer, contract, row.state, listOf(question), 0, D1Kind.TEXT).single()
    val encodeMs = (System.nanoTime() - start) / 1e6
    val stateMatch =
      row.stateHash?.let { D1Contract.sha256(D1Prompt.serialize(row.state ?: "")) == it }
    return D1Recheck(
      encoded.ids.contentEquals(row.ids),
      encoded.markers.contentEquals(row.markers),
      stateMatch,
      encodeMs,
    )
  }

  /** The distribution of [row] from the graph's whole [scores] output (null without a question). */
  fun probabilities(row: D1GateRow, scores: FloatArray): DoubleArray {
    val markerScores = D1Readout.markerScores(scores, row.prefixRows, row.markers, row.options)
    val question =
      row.question
        ?: D1Question(
          row.type,
          "",
          when (row.type) {
            QuestionType.CHOICE -> (0 until row.options).associate { "o$it" to null }
            QuestionType.SCORE -> List(row.options) { "l$it" }
            QuestionType.NOUL -> null
          },
        )
    require(question.options == row.options) { "${row.key}: K = ${row.options}, the question has ${question.options}" }
    return D1Readout.probabilities(markerScores, question, row.calibrate, contract)
  }

  /** One gate row's report entry. */
  fun record(
    row: D1GateRow,
    call: D1Call,
    startWallMs: Long,
    recheck: D1Recheck,
  ): LinkedHashMap<String, Any?> {
    val markerScores = D1Readout.markerScores(call.scores, row.prefixRows, row.markers, row.options)
    val real = row.prefixRows + row.ids.size
    val nonfiniteReal = (0 until real).count { !call.scores[it].isFinite() }
    return linkedMapOf(
      "key" to row.key,
      "n" to row.ids.size,
      "P" to row.prefixRows,
      "K" to row.options,
      "finite" to D1Readout.finite(markerScores),
      "nonfinite_real" to nonfiniteReal,
      "probs" to probabilities(row, call.scores).toList(),
      "scores_at_markers" to markerScores.map { it.toDouble() },
      "write_ms" to call.writeMs,
      "run_ms" to call.runMs,
      "read_ms" to call.readMs,
      "write_run_read_ms" to call.totalMs,
      "t_start_ms" to startWallMs,
      "ids_match" to recheck.idsMatch,
      "markers_match" to recheck.markersMatch,
      "state_match" to recheck.stateMatch,
      "encode_ms" to recheck.encodeMs,
    )
  }

  companion object {
    /** Same definition as numpy.median (the mean of the two middle values for an even count). */
    fun median(values: List<Double>): Double {
      require(values.isNotEmpty()) { "median of nothing" }
      val sorted = values.sorted()
      val middle = sorted.size / 2
      return if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2.0
    }

    /** median / min / max / n of [values], null when empty. */
    fun stats(values: List<Double>): LinkedHashMap<String, Any?>? =
      if (values.isEmpty()) null
      else
        linkedMapOf(
          "median" to median(values),
          "min" to values.min(),
          "max" to values.max(),
          "n" to values.size,
        )
  }
}

/** The kind of a rows file: its `kind` member, `text` when it has none (the timing rows of round 1). */
fun d1RowsKind(bytes: ByteArray): String = ((D1Json.parse(bytes) as? Map<*, *>)?.get("kind") as? String) ?: "text"

/**
 * One clip of an audio rows file (`D/device/r2/rows_audio.json`, made by the Python host): the wav
 * in `files/` ([mediaFile], its [mediaSha256]), the request's [state] (null = the audio mode's `{}`)
 * and named [questions], the Python host's sizes of the clip ([expected]: n, frames, T, T_b, P), the
 * Python mel dumps in `files/` to compare the app's mel with ([melFiles]: `f32` / `f64` -> file),
 * and the encoded [rows] (P = the host's prefix rows, no temperature).
 */
class D1AudioRecord(
  val id: String,
  val mediaFile: String,
  val mediaSha256: String?,
  val state: Any?,
  val questions: LinkedHashMap<String, D1Question>,
  val expected: Map<*, *>?,
  val melFiles: Map<String, String>,
  val rows: List<D1GateRow>,
) {
  /** [expected]'s integer member [name], or null. */
  fun expectedInt(name: String): Int? = (expected?.get(name) as? JsonNumber)?.toInt()

  companion object {
    fun fromJson(value: Any?): D1AudioRecord {
      val record = value as Map<*, *>
      val id = record["id"] as String
      val media = record["media_file"] as String
      require(D1Launch.fileNameValid(media)) { "$id: media_file $media is not a plain file name" }
      val melFiles = LinkedHashMap<String, String>()
      ((record["mel_files"] as Map<*, *>?) ?: emptyMap<String, String>()).forEach { (form, file) ->
        require(D1Launch.fileNameValid(file as String)) { "$id: mel file $file is not a plain file name" }
        melFiles[form as String] = file
      }
      val questions = LinkedHashMap<String, D1Question>()
      (record["questions"] as Map<*, *>).forEach { (name, question) ->
        questions[name as String] = D1Prompt.asQuestion(question)
      }
      val rows = (record["expected"] as List<*>).map { D1GateRow.fromJson(it) }
      require(rows.isNotEmpty()) { "$id: no rows" }
      for (row in rows) require(!row.calibrate) { "${row.key}: an audio row is read without the temperature" }
      return D1AudioRecord(
        id,
        media,
        record["media_sha256"] as String?,
        record["state"],
        questions,
        record["info"] as Map<*, *>?,
        melFiles,
        rows,
      )
    }
  }
}

/** An audio rows file: `{kind: audio, pad_id: 0, records: [...]}`. */
class D1AudioRows(val padId: Int, val records: List<D1AudioRecord>) {
  val rowCount: Int
    get() = records.sumOf { it.rows.size }

  companion object {
    fun parse(bytes: ByteArray): D1AudioRows {
      val root = D1Json.parse(bytes) as Map<*, *>
      require(root["kind"] == "audio") { "not an audio rows file (kind ${root["kind"]})" }
      val padId = (root["pad_id"] as JsonNumber).toInt()
      require(padId == 0) { "pad_id $padId: the decision graph's pad id is 0" }
      return D1AudioRows(padId, (root["records"] as List<*>).map { D1AudioRecord.fromJson(it) })
    }
  }
}

/** An audio timing set: [name], [kind] (`request`: the clip's wav to every answer) and the clip. */
class D1AudioTimingSet(val name: String, val kind: String, val record: D1AudioRecord) {
  companion object {
    /** The sets of an audio timing rows file (`{kind: audio, pad_id: 0, sets: [{name, kind, record}]}`). */
    fun parse(bytes: ByteArray): List<D1AudioTimingSet> {
      val root = D1Json.parse(bytes) as Map<*, *>
      require(root["kind"] == "audio") { "not an audio timing file (kind ${root["kind"]})" }
      require((root["pad_id"] as JsonNumber).toInt() == 0) { "pad_id must be 0" }
      return (root["sets"] as List<*>).map {
        val set = it as Map<*, *>
        D1AudioTimingSet(set["name"] as String, set["kind"] as String, D1AudioRecord.fromJson(set["record"]))
      }
    }
  }
}

/** The audio gate's checks of one clip and of its rows, Android-free (see [D1GateCore]). */
object D1AudioCheck {
  /** [row] with the app's own prefix rows [prefixRows] (the markers are read at P + marker). */
  fun withPrefix(row: D1GateRow, prefixRows: Int): D1GateRow =
    D1GateRow(
      row.key,
      row.ids,
      row.markers,
      row.type,
      row.options,
      prefixRows,
      row.calibrate,
      row.hasRequest,
      row.state,
      row.question,
      row.stateHash,
    )

  /**
   * The app's own encoding of [row]'s question over [record]'s state (kind audio, a null state is
   * the mode's `{}`) after [prefixRows] media rows, against the row's ids and markers, and the
   * serialized state against the row's `state_hash`.
   */
  fun recheck(
    tokenizer: D1Tokenizer,
    contract: D1Contract,
    record: D1AudioRecord,
    row: D1GateRow,
    prefixRows: Int,
  ): D1Recheck {
    val question = row.question ?: return D1Recheck(null, null, null, null)
    val start = System.nanoTime()
    val encoded = D1Rows.rows(tokenizer, contract, record.state, listOf(question), prefixRows, D1Kind.AUDIO).single()
    val encodeMs = (System.nanoTime() - start) / 1e6
    val mode = contract.modes.getValue(D1Kind.AUDIO.wireName)
    val effective =
      record.state ?: mode.stateNoneBecomes.takeIf { it !== D1Contract.NOT_SET }
    val stateMatch = row.stateHash?.let { D1Contract.sha256(D1Prompt.serialize(effective ?: "")) == it }
    return D1Recheck(
      encoded.ids.contentEquals(row.ids),
      encoded.markers.contentEquals(row.markers),
      stateMatch,
      encodeMs,
    )
  }

  /** The app's clip sizes against the Python host's [D1AudioRecord.expected] (null when it has none). */
  fun infoMatches(info: D1AudioInfo, record: D1AudioRecord): Boolean? {
    if (record.expected == null) return null
    return record.expectedInt("n") == info.samples &&
      record.expectedInt("frames") == info.frames &&
      record.expectedInt("T") == info.stftFrames &&
      record.expectedInt("T_b") == info.bucket &&
      record.expectedInt("P") == info.prefixRows
  }

  /** Little-endian float32 values of [bytes]. */
  fun floats(bytes: ByteArray): FloatArray {
    require(bytes.size % Float.SIZE_BYTES == 0) { "${bytes.size} bytes are not float32 values" }
    val out = FloatArray(bytes.size / Float.SIZE_BYTES)
    java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
  }

  /** [values] as little-endian float32 bytes. */
  fun bytes(values: FloatArray): ByteArray {
    val buffer = java.nio.ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    buffer.asFloatBuffer().put(values)
    return buffer.array()
  }

  /**
   * The element-wise difference of the app's mel and a Python mel of the same shape: max and mean
   * |Δ|, where the max is, the elements over 1e-3, and whether every element is bit-equal.
   */
  fun melDifference(app: FloatArray, python: FloatArray): LinkedHashMap<String, Any?> {
    require(app.size == python.size) { "the app's mel has ${app.size} values, the Python mel ${python.size}" }
    var worst = 0.0
    var at = 0
    var total = 0.0
    var over = 0
    var equal = true
    for (i in app.indices) {
      val d = Math.abs(app[i].toDouble() - python[i].toDouble())
      if (d > worst) {
        worst = d
        at = i
      }
      total += d
      if (d > MEL_ELEMENT_BAR) over++
      if (app[i].toRawBits() != python[i].toRawBits()) equal = false
    }
    return linkedMapOf(
      "values" to app.size,
      "max_abs" to worst,
      "mean_abs" to if (app.isEmpty()) 0.0 else total / app.size,
      "at" to at,
      "over_1e-3" to over,
      "bit_equal" to equal,
    )
  }

  /** The element bar of the mel comparison (the conversion lane's `bar_elementwise`). */
  const val MEL_ELEMENT_BAR = 1e-3
}
