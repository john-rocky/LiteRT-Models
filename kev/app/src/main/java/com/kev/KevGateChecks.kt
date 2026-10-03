package com.kev

import kotlin.math.abs

/** One question of the debug gate asset with the author's fp32 oracle. */
class KevGateQuestion(
  val qid: String,
  val type: String,
  val keys: List<String>,
  val rowIds: IntArray,
  val decideIndex: Int,
  val optionIndices: IntArray,
  /** The oracle's float32 probabilities. */
  val probabilities: DoubleArray,
  val answer: Map<*, *>,
) {
  /** Top-1 minus top-2 probability in float32, as the oracle computes it (1 for one option). */
  val top2Gap: Double
    get() {
      if (probabilities.size < 2) return 1.0
      val sorted = probabilities.sortedArrayDescending()
      return (sorted[0].toFloat() - sorted[1].toFloat()).toDouble()
    }

  /** Within [KevGateChecks.NEAR_TIE] of a tie: a different argmax here is reported apart. */
  val nearTie: Boolean
    get() = top2Gap <= KevGateChecks.NEAR_TIE
}

/** One request of the gate asset: the request JSON, its `usage.input_tokens` and its questions. */
class KevGateItem(
  val id: String,
  val source: String,
  val request: Any?,
  val inputTokens: Int,
  val questions: List<KevGateQuestion>,
)

/**
 * One tokenizer probe: a string and the IDs transformers gives it, raw and through `user_tokens`.
 */
class KevTokenizerProbe(
  val id: String,
  val category: String,
  val text: String,
  val rawIds: IntArray,
  val userIds: IntArray,
)

/** Robust summary of a list of milliseconds. */
class KevStats(val median: Double, val min: Double, val max: Double, val count: Int) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf("median" to median, "min" to min, "max" to max, "n" to count)

  companion object {
    /** Median as numpy computes it (the mean of the two middle values for an even count). */
    fun of(values: List<Double>): KevStats? {
      if (values.isEmpty()) return null
      val sorted = values.sorted()
      val middle = sorted.size / 2
      val median =
        if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2
      return KevStats(median, sorted.first(), sorted.last(), sorted.size)
    }
  }
}

/** Android-free parsing and comparisons of the debug gate (`KevGateRunner`) and its JVM tests. */
object KevGateChecks {
  /** The bundled asset: SemIf + invented requests with the oracle per question. */
  const val ASSET_NAME = "gate_fixtures.json"

  /** The bundled tokenizer probes (the same file as the JVM test resource). */
  const val PROBES_NAME = "tokenizer_probes.json"

  /** A top-2 gap at or below this is a near-tie. */
  const val NEAR_TIE = 0.02

  /** Gate bar on rows run in the graph: max and mean |Δp| over all options. */
  const val MAX_ABS_DP = 0.02
  const val MEAN_ABS_DP = 0.002

  /** Parses the asset and checks its declared record and question counts. */
  fun parseAsset(bytes: ByteArray): List<KevGateItem> {
    val asset = KevJson.parse(bytes) as Map<*, *>
    val items =
      (asset["items"] as List<*>).map { entry ->
        val item = entry as Map<*, *>
        KevGateItem(
          item["id"] as String,
          item["source"] as String,
          item["request"],
          (item["input_tokens"] as JsonNumber).toInt(),
          (item["questions"] as List<*>).map { question(it as Map<*, *>) },
        )
      }
    val questions = items.sumOf { it.questions.size }
    require(
      items.size == (asset["records"] as JsonNumber).toInt() &&
        questions == (asset["questions"] as JsonNumber).toInt()
    ) {
      "$ASSET_NAME declares ${asset["records"]} records / ${asset["questions"]} questions, holds ${items.size} / $questions"
    }
    return items
  }

  fun parseProbes(bytes: ByteArray): List<KevTokenizerProbe> =
    ((KevJson.parse(bytes) as Map<*, *>)["cases"] as List<*>).map { entry ->
      val probe = entry as Map<*, *>
      KevTokenizerProbe(
        probe["id"] as String,
        probe["category"] as String,
        probe["text"] as String,
        ints(probe["raw_ids"]),
        ints(probe["user_ids"]),
      )
    }

  /** Max |a − b| over the entries of two equal-length arrays. */
  fun maxAbsDifference(a: DoubleArray, b: DoubleArray): Double {
    require(a.size == b.size) { "${a.size} vs ${b.size} values" }
    return a.indices.maxOfOrNull { abs(a[it] - b[it]) } ?: 0.0
  }

  /** Index where two ID arrays start to differ, or null when they are equal. */
  fun firstDifference(expected: IntArray, actual: IntArray): Int? =
    (0 until maxOf(expected.size, actual.size)).firstOrNull {
      expected.getOrNull(it) != actual.getOrNull(it)
    }

  private fun question(json: Map<*, *>): KevGateQuestion =
    KevGateQuestion(
      json["qid"] as String,
      json["type"] as String,
      (json["keys"] as List<*>).map { it as String },
      ints(json["row_ids"]),
      (json["decide_idx"] as JsonNumber).toInt(),
      ints(json["opt_idx"]),
      (json["probs"] as List<*>).map { (it as JsonNumber).toDouble() }.toDoubleArray(),
      json["answer"] as Map<*, *>,
    )

  private fun ints(value: Any?): IntArray =
    (value as List<*>).map { (it as JsonNumber).toInt() }.toIntArray()
}
