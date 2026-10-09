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
  /** The six inputs of [row] for a bucket of [length] positions (`build_inputs`). */
  fun inputs(row: D1GateRow, length: Int): D1Inputs {
    require(row.prefixRows == 0) { "${row.key}: a media row (P = ${row.prefixRows}) needs its prefix" }
    return D1Rows.buildInputs(row.ids, null, 0, length, row.type)
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
