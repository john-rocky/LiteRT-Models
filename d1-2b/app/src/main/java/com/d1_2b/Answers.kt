/*
 * Ports api.py, box.py, calibration.py, json_spans.py and parts of pointer.py of LiquidAI/d1-2B
 * (revision 10e7671d) to Kotlin; that code is under the LFM Open License v1.0 (see LICENSE and
 * NOTICE).
 */

package com.d1_2b

import java.io.File
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

// ------------------------------------------------------------------------------------- calibration

/** Temperature scaling of option scores, and monotone tables for the typed answers (calibration.py). */
class Calibration(
  private val temperatures: Map<String, Double>,
  private val default: Double,
  private val tables: Map<String, List<DoubleArray>>,
) {
  fun temperature(q: Question): Double = temperatures[questionBucket(q)] ?: default

  fun through(name: String, value: Double): Double = through(value, tables[name])

  /** The level whose observed share is [share]: the table read backwards. */
  fun level(name: String, share: Double): Double =
    through(share, tables[name]?.map { doubleArrayOf(it[1], it[0]) })

  companion object {
    @Suppress("UNCHECKED_CAST")
    fun load(file: File): Calibration {
      val doc = D1Json.parse(file.readBytes()) as Map<String, Any?>
      val temperatures = (doc["temperatures"] as Map<String, Any?>).mapValues { (it.value as JsonNumber).toDouble() }
      val default = (doc["default"] as JsonNumber?)?.toDouble() ?: 1.0
      val tables =
        ((doc["tables"] as Map<String, Any?>?) ?: emptyMap()).mapValues { (_, knots) ->
          (knots as List<*>).map { knot ->
            val pair = knot as List<*>
            doubleArrayOf((pair[0] as JsonNumber).toDouble(), (pair[1] as JsonNumber).toDouble())
          }
        }
      return Calibration(temperatures, default, tables)
    }

    fun bucket(kind: String, k: Int): String =
      when {
        kind == "noul" -> "noul"
        kind == "score" -> "score:$k"
        k <= 3 -> "choice:small"
        k <= 8 -> "choice:mid"
        k <= 26 -> "choice:large"
        else -> "choice:inventory"
      }

    fun cardinality(q: Question): Int =
      when (q) {
        is Estimate -> q.grid.size
        is NumberSlot -> q.grid.size
        is Noul -> 2
        is Choice -> q.criteria.size
        is Score -> q.criteria.size
        else -> throw IllegalArgumentException("no cardinality for ${q.type}")
      }

    fun questionBucket(q: Question): String {
      val kind = if (q is Noul) "noul" else if (q is Score) "score" else "choice"
      return bucket(kind, cardinality(q))
    }

    /** [value] through increasing (raw, calibrated) knots, linear between them, flat past the ends. */
    fun through(value: Double, knots: List<DoubleArray>?): Double {
      if (knots.isNullOrEmpty()) return value
      var index = 0
      while (index < knots.size && knots[index][0] < value) index++ // bisect_left
      if (index == 0) return knots[0][1]
      if (index == knots.size) return knots.last()[1]
      val (a, pa) = knots[index - 1].let { it[0] to it[1] }
      val (b, pb) = knots[index].let { it[0] to it[1] }
      return pa + (value - a) * (pb - pa) / (b - a)
    }
  }
}

// ------------------------------------------------------------------------------------- boxes

/** A box: a presence and four edges, each a distribution over the thousandths (box.py). */
object Boxes {
  val EDGES = listOf("top", "bottom", "left", "right")
  val GRID = Grid(0.0, 1.0)
  val THOUSANDTHS = Grid(0.0, 0.999, 1000)
  const val DRAWS = 512
  const val INSTANCES = 16
  const val MAX_INSTANCES = 64

  private fun readAs(distribution: DoubleArray): Grid = if (distribution.size == THOUSANDTHS.size) THOUSANDTHS else GRID

  private fun ordered(a: Double, b: Double, grid: Grid): Pair<Double, Double> {
    if (a < b) return a to b
    val half = grid.step / 2
    val middle = min(1 - half, max(half, (a + b) / 2))
    return (middle - half) to (middle + half)
  }

  private fun rectangle(edges: List<Double>, grid: Grid): Map<String, Double> {
    val (top, bottom) = ordered(edges[0], edges[1], grid)
    val (left, right) = ordered(edges[2], edges[3], grid)
    return linkedMapOf("top" to top, "bottom" to bottom, "left" to left, "right" to right)
  }

  private fun area(box: Map<String, Double>): Double =
    (box.getValue("bottom") - box.getValue("top")) * (box.getValue("right") - box.getValue("left"))

  private fun iou(a: Map<String, Double>, b: Map<String, Double>): Double {
    val overlap =
      max(0.0, min(a.getValue("bottom"), b.getValue("bottom")) - max(a.getValue("top"), b.getValue("top"))) *
        max(0.0, min(a.getValue("right"), b.getValue("right")) - max(a.getValue("left"), b.getValue("left")))
    return overlap / (area(a) + area(b) - overlap)
  }

  /** [DRAWS] points of a 4-D Halton sequence: the same draws every call. */
  private val samples: List<DoubleArray> by lazy {
    fun radicalInverse(index: Int, base: Int): Double {
      var i = index
      var value = 0.0
      var scale = 1.0
      while (i != 0) {
        val digit = i % base
        i /= base
        scale /= base
        value += digit * scale
      }
      return value
    }
    (1..DRAWS).map { i -> doubleArrayOf(radicalInverse(i, 2), radicalInverse(i, 3), radicalInverse(i, 5), radicalInverse(i, 7)) }
  }

  /** The share of boxes drawn from the edge distributions with IoU >= 0.5 against [box]. */
  private fun confidence(edges: List<DoubleArray>, box: Map<String, Double>): Double {
    val cdfs =
      edges.map { row ->
        val total = PyMath.fsum(row)
        var mass = 0.0
        DoubleArray(row.size) { i -> (mass + row[i] / total).also { mass = it } }
      }
    val grid = readAs(edges[0])
    val values = grid.values
    var hits = 0
    for (point in samples) {
      val drawn =
        cdfs.indices.map { e ->
          val cdf = cdfs[e]
          values[min(grid.size - 1, bisectLeft(cdf, point[e]))]
        }
      if (iou(rectangle(drawn, grid), box) >= 0.5) hits++
    }
    return hits.toDouble() / DRAWS
  }

  private fun bisectLeft(sorted: DoubleArray, value: Double): Int {
    var lo = 0
    var hi = sorted.size
    while (lo < hi) {
      val mid = (lo + hi) ushr 1
      if (sorted[mid] < value) lo = mid + 1 else hi = mid
    }
    return lo
  }

  private fun crossing(probabilities: DoubleArray, grid: Grid): Double {
    val total = PyMath.fsum(probabilities)
    var mass = 0.0
    for (i in probabilities.indices) {
      val share = probabilities[i] / total
      if (share > 0 && mass + share >= 0.5) {
        return min(grid.high, max(grid.low, grid.values[i] - grid.step / 2 + (0.5 - mass) / share * grid.step))
      }
      mass += share
    }
    return grid.high
  }

  private fun medians(edges: List<DoubleArray>, read: String): Map<String, Double> {
    val grid = readAs(edges[0])
    return rectangle(edges.map { row -> if (read == "interp") crossing(row, grid) else grid.read(row).getValue("median") }, grid)
  }

  /** The API's box: presence ([yes, no]) then the four edges' distributions. */
  fun answer(probabilities: List<DoubleArray>, through: (String, Double) -> Double, edges: String): LinkedHashMap<String, Any?> {
    val found = through("found:box", probabilities[0][0])
    if (found < 0.5) return linkedMapOf("type" to "box", "box" to null, "found" to found, "confidence" to 0.0)
    val box = medians(probabilities.subList(1, 5), edges)
    return linkedMapOf(
      "type" to "box",
      "box" to box,
      "found" to found,
      "confidence" to through("box", confidence(probabilities.subList(1, 5), box)),
    )
  }

  /** The API's `boxes`: blocks of presence and four edges; the present ones in reading order. */
  fun answerMany(probabilities: List<DoubleArray>, through: (String, Double) -> Double, edges: String): LinkedHashMap<String, Any?> {
    val presence = ArrayList<Double>()
    var below = 1.0
    for (k in probabilities.indices step 5) {
      below = min(below, through("found:boxes", probabilities[k][0]))
      presence.add(below)
    }
    val out = ArrayList<LinkedHashMap<String, Any?>>()
    for ((k, found) in presence.withIndex()) {
      if (found < 0.5) break
      val block = probabilities.subList(5 * k + 1, 5 * k + 5)
      val box = medians(block, edges)
      out.add(linkedMapOf("box" to box, "found" to found, "confidence" to through("boxes", confidence(block, box))))
    }
    out.sortWith(
      compareBy<LinkedHashMap<String, Any?>> { (it["box"] as Map<*, *>)["left"] as Double }
        .thenBy { (it["box"] as Map<*, *>)["top"] as Double }
    )
    return linkedMapOf(
      "type" to "boxes",
      "boxes" to out,
      "count" to out.size,
      "at_least" to (out.isNotEmpty() && out.size == presence.size),
      "presence" to presence,
    )
  }
}

// ------------------------------------------------------------------------------------- pointer

/** Where JSON string values sit in a rendered JSON state (json_spans.py). */
object JsonSpans {
  /** String-value spans in rendered JSON, excluding quotes; paths use RFC 6901 escaping. */
  fun stringPaths(text: String): List<Triple<Int, Int, String>> {
    val ranges = ArrayList<Triple<Int, Int, String>>()
    fun skip(from: Int): Int {
      var at = from
      while (at < text.length && pyIsSpace(text[at])) at++
      return at
    }
    fun value(from: Int, path: String): Int {
      var at = skip(from)
      when (text[at]) {
        '{' -> {
          at = skip(at + 1)
          while (text[at] != '}') {
            val (key, end) = rawString(text, at) ?: throw IllegalArgumentException("invalid JSON object")
            if (text[skip(end)] != ':') throw IllegalArgumentException("invalid JSON object")
            val name = key.replace("~", "~0").replace("/", "~1")
            at = skip(value(skip(end) + 1, "$path/$name"))
            if (text[at] == ',') at = skip(at + 1)
            else if (text[at] != '}') throw IllegalArgumentException("invalid JSON object separator")
          }
          return at + 1
        }
        '[' -> {
          at = skip(at + 1)
          var index = 0
          while (text[at] != ']') {
            at = skip(value(at, "$path/$index"))
            index++
            if (text[at] == ',') at = skip(at + 1)
            else if (text[at] != ']') throw IllegalArgumentException("invalid JSON array separator")
          }
          return at + 1
        }
        '"' -> {
          val (_, end) = rawString(text, at) ?: throw IllegalArgumentException("invalid JSON string")
          ranges.add(Triple(at + 1, end - 1, path))
          return end
        }
        else -> return rawScalarEnd(text, at)
      }
    }
    val end = value(0, "")
    if (skip(end) != text.length) throw IllegalArgumentException("trailing JSON text")
    return ranges
  }

  /** A JSON string literal starting at [at]: its value and the index after its closing quote. */
  private fun rawString(text: String, at: Int): Pair<String, Int>? {
    if (text[at] != '"') return null
    var i = at + 1
    while (i < text.length) {
      when (text[i]) {
        '\\' -> i += 2
        '"' -> return (decode(text.substring(at + 1, i)) ?: return null) to (i + 1)
        else -> i++
      }
    }
    return null
  }

  /** The end of a number, true, false or null at [at] (as json.JSONDecoder.raw_decode reads one). */
  private fun rawScalarEnd(text: String, at: Int): Int {
    for (word in listOf("true", "false", "null", "NaN", "Infinity", "-Infinity")) {
      if (text.startsWith(word, at)) return at + word.length
    }
    val m = NUMBER.matcher(text).region(at, text.length)
    if (m.lookingAt()) return m.end()
    throw IllegalArgumentException("invalid JSON value")
  }

  private val NUMBER = java.util.regex.Pattern.compile("-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?")

  /** A slice of a rendered JSON string as the characters it encodes (unchanged when it cuts an escape). */
  fun jsonText(escaped: String): String = decode(escaped) ?: escaped

  /** `json.loads('"' + s + '"')`, or null where Python raises. */
  private fun decode(s: String): String? {
    val out = StringBuilder(s.length)
    var i = 0
    while (i < s.length) {
      val c = s[i]
      when {
        c == '"' || c < ' ' -> return null
        c != '\\' -> {
          out.append(c)
          i++
        }
        i + 1 >= s.length -> return null
        else -> {
          when (val e = s[i + 1]) {
            '"', '\\', '/' -> out.append(e)
            'b' -> out.append('\b')
            'f' -> out.append('\u000c')
            'n' -> out.append('\n')
            'r' -> out.append('\r')
            't' -> out.append('\t')
            'u' -> {
              if (i + 6 > s.length) return null
              val hex = s.substring(i + 2, i + 6)
              if (!hex.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }) return null
              out.append(hex.toInt(16).toChar())
              i += 4
            }
            else -> return null
          }
          i += 2
        }
      }
    }
    return out.toString()
  }
}

/** Token positions mapped to the caller's rendered state; null = not a text position (pointer.StatePositions). */
class StatePositions(
  val text: String,
  val offsets: List<IntArray?>,
  private val paths: List<Triple<Int, Int, String>>,
  private val isJson: Boolean,
) {
  val pointerMask: BooleanArray by lazy {
    val valid = BooleanArray(offsets.size) { it == 0 || offsets[it] != null }
    val last = valid.indices.last { valid[it] }
    valid.copyOf(last + 1)
  }

  /** The state text between two pointer endpoints (trimmed), with its JSON path when it lies in one string. */
  fun read(first: Int, last: Int, maxTokens: Int): LinkedHashMap<String, Any?> {
    if (first == 0 && last == 0) return linkedMapOf("text" to null, "span" to null)
    require(first in 1..last && last < offsets.size && last - first < maxTokens) { "invalid pointer endpoints" }
    val a = offsets[first] ?: throw IllegalArgumentException("pointer leaves the state text")
    val b = offsets[last] ?: throw IllegalArgumentException("pointer leaves the state text")
    var start = a[0]
    var end = b[1]
    while (start < end && pyIsSpace(text[start])) start++
    while (end > start && pyIsSpace(text[end - 1])) end--
    val held = paths.filter { (s, e, _) -> s < end && start < e }
    if (held.size == 1 && isJson) {
      val (s, e, path) = held[0]
      start = max(start, s)
      end = min(end, e)
      return linkedMapOf("text" to JsonSpans.jsonText(text.substring(start, end)), "span" to intArrayOf(start, end), "path" to path)
    }
    return linkedMapOf("text" to text.substring(start, end), "span" to intArrayOf(start, end))
  }

  companion object {
    /** pointer.state_positions: each token of [fullText] mapped onto the rendered state. */
    fun of(prompt: Prompt, state: Any?, prefix: String, fullText: String, encoded: Encoding): StatePositions {
      require(fullText.startsWith(prefix)) { "the tokenized prompt must start with the state prefix" }
      val boundaries: List<Int>
      val start: Int
      if (state == null) {
        boundaries = listOf(0)
        start = prefix.length
      } else {
        val body = prompt.stateBlock(state)
        val (escaped, bounds) = prompt.plainOffsets(body)
        val tail = escaped + "\n" + prompt.questionHeader(state)
        require(prefix.endsWith(tail)) { "state rendering differs from the prompt prefix" }
        start = prefix.length - tail.length
        boundaries = bounds!!
      }
      val text = if (state != null) prompt.stateText(state) else ""
      // bisect_right(boundaries, len(text)) - 1
      var right = 0
      while (right < boundaries.size && boundaries[right] <= text.length) right++
      val stop = start + right - 1
      val offsets = ArrayList<IntArray?>(encoded.size)
      for (t in 0 until encoded.size) {
        val a = encoded.starts[t]
        val b = encoded.ends[t]
        var span: IntArray? = null
        val lo = max(a, start)
        val hi = min(b, stop)
        if (lo < hi && fullText.substring(a, max(a, lo)).isBlankPy() && fullText.substring(min(hi, b), b).isBlankPy()) {
          val left = boundaries[lo - start]
          val rightAt = boundaries[hi - start]
          if (left < rightAt && rightAt <= text.length && !text.substring(left, rightAt).isBlankPy()) {
            span = intArrayOf(left, rightAt)
          }
        }
        offsets.add(span)
      }
      require(offsets.isNotEmpty() && offsets[0] == null) { "position 0 must be a template token for the nothing readout" }
      val isJson = state != null && state !is String
      val paths = if (isJson) JsonSpans.stringPaths(text) else emptyList()
      return StatePositions(text, offsets, paths, isJson)
    }

    private fun String.isBlankPy(): Boolean = all { pyIsSpace(it) }

    /**
     * pointer.spans on one read's start / end scores ([2][T]) in float64: found when P(found) >=
     * 0.5, then the likeliest span within [maxTokens]; (0, 0) for nothing.
     */
    fun spans(startScores: FloatArray, endScores: FloatArray, maxTokens: Int): Pair<IntArray, Double> {
      val t = startScores.size
      val start = logSoftmax(startScores)
      val end = logSoftmax(endScores)
      val found = -expm1(start[0])
      if (t == 1) return intArrayOf(0, 0) to found
      val cap = min(maxTokens, t - 1)
      var bestScore = 0.0
      var bestIndex = 0
      var bestWithin = 0
      for (i in 0 until t - 1) {
        // the first maximum of end[1 + i .. 1 + i + cap), padded with -inf past the end
        var within = 0
        var best = end[1 + i]
        for (w in 1 until cap) {
          val value = if (1 + i + w < t) end[1 + i + w] else Double.NEGATIVE_INFINITY
          if (value > best) {
            best = value
            within = w
          }
        }
        val score = start[1 + i] + best
        if (i == 0 || score > bestScore) {
          bestScore = score
          bestIndex = i
          bestWithin = within
        }
      }
      val at = if (found < 0.5) intArrayOf(0, 0) else intArrayOf(bestIndex + 1, bestIndex + bestWithin + 1)
      return at to found
    }

    /** x - max - log(sum(exp(x - max))) in float64, as numpy computes it on the float64 copy. */
    private fun logSoftmax(scores: FloatArray): DoubleArray {
      var top = Double.NEGATIVE_INFINITY
      for (s in scores) if (s.toDouble() > top) top = s.toDouble()
      var sum = 0.0
      for (s in scores) sum += exp(s.toDouble() - top)
      val log = ln(sum)
      return DoubleArray(scores.size) { scores[it].toDouble() - top - log }
    }
  }
}

// ------------------------------------------------------------------------------------- questions

/** A caller's question in the API's JSON (`check_question` + `as_question` of prompt.py). */
object Questions {
  val TYPES = listOf("noul", "choice", "score", "text", "number", "date", "box", "boxes")

  fun parse(raw: Any?, mostBlocks: Int = Boxes.INSTANCES): Question {
    require(raw is Map<*, *>) { "a question is an object such as {\"type\": \"noul\", \"instructions\": \"Is it urgent?\"}" }
    val kind = raw["type"]
    val said = raw["instructions"]
    val criteria = raw["criteria"]
    require(kind in TYPES) { "\"type\" is one of ${TYPES.joinToString(", ")}, not ${pyRepr(kind)}" }
    require(said is String && said.pyStrip().isNotEmpty()) { "\"instructions\" is the question as text" }
    if (kind == "choice") {
      require(criteria is Map<*, *> && criteria.isNotEmpty() && criteria.keys.all { it is String }) {
        "a choice's \"criteria\" names the options: {\"billing\": \"Charges and refunds\", \"shipping\": \"Delivery\"}" +
          if (criteria is List<*>) "; a list of levels is a score" else ""
      }
    }
    if (kind == "score") {
      require(criteria is List<*> && criteria.size in 2..10 && criteria.all { it is String }) {
        "a score's \"criteria\" lists 2 to 10 levels, lowest first: [\"Can wait\", \"Today\", \"Now\"]"
      }
    }
    if (kind == "noul") {
      require(criteria == null || criteria is Map<*, *>) {
        "a noul's \"criteria\", when given, is {\"true\": \"what yes means\", \"false\": \"what no means\"}"
      }
    }
    if (truthy(raw["multiple"])) require(kind == "choice") { "\"multiple\" belongs to a choice" }
    if (raw.containsKey("max")) {
      val most = raw["max"]
      require(kind == "boxes" && most is JsonNumber && most.isInteger && most.toInt() in 1..mostBlocks) {
        "\"max\" belongs to boxes, a whole number from 1 to $mostBlocks"
      }
    }
    if (raw.containsKey("step")) require(raw.containsKey("range")) { "\"step\" needs a \"range\"" }
    if (raw.containsKey("range")) {
      val bounds = raw["range"]
      require(kind == "number" && bounds is List<*> && bounds.size == 2 && bounds.all { it is JsonNumber }) {
        "an estimate is {\"type\": \"number\", \"range\": [low, high]}, with an optional \"step\""
      }
    }
    val unit = raw["unit"] as? String
    if (kind == "number" && raw.containsKey("range")) {
      val bounds = raw["range"] as List<*>
      val low = (bounds[0] as JsonNumber).toDouble()
      val high = (bounds[1] as JsonNumber).toDouble()
      val step = (raw["step"] as JsonNumber?)?.toDouble()
      val ratio = if (step != null) (high - low) / step else 0.0
      require(low < high && (step == null || (step > 0 && isClose(ratio, PyMath.roundHalfEven(ratio))))) {
        "an estimate's range runs low to high and its step, above 0, divides it"
      }
      return Estimate(said, low, high, step, unit)
    }
    return when (kind) {
      "text", "number", "date" -> Read(said, kind as String, unit)
      "box" -> BoxQ(said)
      "boxes" -> BoxesQ(said, (raw["max"] as JsonNumber?)?.toInt())
      "noul" -> Noul(said, (criteria as Map<*, *>?)?.entries?.associate { it.key.toString() to it.value })
      "score" -> Score(said, (criteria as List<*>).map { it as String })
      else -> {
        val options = LinkedHashMap<String, String?>()
        (criteria as Map<*, *>).forEach { (name, text) -> options[name as String] = text?.let { pyStr(it) } }
        Choice(said, options, truthy(raw["multiple"]))
      }
    }
  }

  private fun truthy(value: Any?): Boolean =
    when (value) {
      null -> false
      is Boolean -> value
      is JsonNumber -> value.toDouble() != 0.0
      is String -> value.isNotEmpty()
      is Collection<*> -> value.isNotEmpty()
      is Map<*, *> -> value.isNotEmpty()
      else -> true
    }

  private fun isClose(a: Double, b: Double): Boolean = kotlin.math.abs(a - b) <= max(1e-9 * max(kotlin.math.abs(a), kotlin.math.abs(b)), 0.0)

  private fun pyStr(value: Any): String = if (value is JsonNumber) value.pythonString() else value.toString()

  private fun pyRepr(value: Any?): String = if (value is String) "'$value'" else value?.toString() ?: "None"
}

// ------------------------------------------------------------------------------------- answers

/** One question's answer as `system_one` gives it (api.answer). */
object Api {
  const val SCORE_READOUT = "median"

  fun scoreLevel(probs: DoubleArray, readout: String = SCORE_READOUT): Int {
    val total = probs.sum().takeIf { it != 0.0 } ?: 1.0
    when (readout) {
      "mean" -> {
        var mean = 0.0
        for (i in probs.indices) mean += i * probs[i] / total
        return max(0, min(probs.size - 1, PyMath.roundHalfEven(mean).toInt()))
      }
      "median" -> {
        var mass = 0.0
        for (i in probs.indices) {
          mass += probs[i] / total
          if (mass >= 0.5) return i
        }
        return probs.size - 1
      }
      else -> return argmax(probs)
    }
  }

  private fun argmax(values: DoubleArray): Int {
    var best = 0
    for (i in values.indices) if (values[i] > values[best]) best = i
    return best
  }

  /**
   * [probs] as the host gives it: a read's map (text, span, found, path), a box's or a multiple
   * choice's list of rows, any other question's option probabilities. [context] = the state's
   * rendered text for a read's number or date.
   */
  fun answer(q: Question, probs: Any, context: String, calibration: Calibration?, edges: String = "median"): LinkedHashMap<String, Any?> {
    val through: (String, Double) -> Double = { name, value -> calibration?.through(name, value) ?: value }
    @Suppress("UNCHECKED_CAST")
    when (q) {
      is BoxQ -> return Boxes.answer(probs as List<DoubleArray>, through, edges)
      is BoxesQ -> return Boxes.answerMany(probs as List<DoubleArray>, through, edges)
      is Estimate -> {
        val value = q.grid.read(probs as DoubleArray, calibration?.let { c -> { share: Double -> c.level("estimate", share) } })
        return linkedMapOf(
          "type" to "number",
          "number" to value["median"],
          "interval" to listOf(value["p10"], value["p90"]),
          "unit" to q.unit,
        )
      }
      is Read -> {
        val read = probs as Map<String, Any?>
        val found = through("found:${q.type}", read["found"] as Double)
        val present = found >= 0.5
        val text = if (present) read["text"] as String? else null
        val span = if (present) read["span"] else null
        var value: Any? = text
        var unit = q.unit
        if (text != null && q.type == "number") {
          unit = unit?.takeIf { it.isNotEmpty() } ?: Parsing.writtenUnit(text)
          value = Parsing.parseNumber(text, context, q.unit)
        } else if (text != null && q.type == "date") {
          value = Parsing.parseDate(text, context).first
        }
        val result = linkedMapOf<String, Any?>("type" to q.type, q.type to value, "span" to span, "found" to found)
        if (q.type != "text") result["text"] = text
        if (present && read.containsKey("path")) result["path"] = read["path"]
        if (q.type == "number") result["unit"] = unit
        return result
      }
      is Noul -> return linkedMapOf("type" to "noul", "noul" to (probs as DoubleArray)[0])
      is Choice ->
        if (q.multiple) {
          val calibrated = (probs as DoubleArray).map { through("multiple", it) }
          val names = q.criteria.keys.toList()
          return linkedMapOf(
            "type" to "choice",
            "choices" to names.filterIndexed { i, _ -> calibrated[i] >= 0.5 },
            "probabilities" to LinkedHashMap<String, Any?>().apply { names.forEachIndexed { i, n -> put(n, calibrated[i]) } },
          )
        } else {
          val p = probs as DoubleArray
          val names = q.criteria.keys.toList()
          val pick = argmax(p)
          return linkedMapOf(
            "type" to "choice",
            "choice" to names[pick],
            "confidence" to p[pick],
            "probabilities" to LinkedHashMap<String, Any?>().apply { names.forEachIndexed { i, n -> put(n, p[i]) } },
          )
        }
      is Score -> {
        val p = probs as DoubleArray
        return linkedMapOf(
          "type" to "score",
          "score" to scoreLevel(p).toDouble(),
          "confidence" to p[argmax(p)],
          "probabilities" to LinkedHashMap<String, Any?>().apply { p.forEachIndexed { i, v -> put(i.toString(), v) } },
          "legend" to LinkedHashMap<String, Any?>().apply { q.criteria.forEachIndexed { i, t -> put(i.toString(), t) } },
        )
      }
      else -> throw IllegalArgumentException("no answer for ${q.type}")
    }
  }
}
