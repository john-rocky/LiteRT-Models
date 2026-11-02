/*
 * Ports parts of prompt.py, attention.py and grid.py of LiquidAI/d1-2B (revision 10e7671d) to
 * Kotlin; that code is under the LFM Open License v1.0 (see LICENSE and NOTICE).
 */

package com.d1_omni_2b

import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode
import java.util.regex.Pattern
import kotlin.math.abs

/** Values from [low] to [high] in [size] equal steps, read as a distribution (grid.py). */
class Grid(val low: Double, val high: Double, val size: Int = 128) {
  init {
    require(size >= 2 && low < high) { "a grid needs two values or more, low to high" }
  }

  val step: Double
    get() = (high - low) / (size - 1)

  val values: DoubleArray by lazy {
    DoubleArray(size) { if (it == size - 1) high else low + it * step }
  }

  /** The mean, median and central 80 percent interval; [level] maps a quantile to the level it is read at. */
  fun read(probabilities: DoubleArray, level: ((Double) -> Double)? = null): Map<String, Double> {
    val total = PyMath.fsum(probabilities)
    val out = LinkedHashMap<String, Double>()
    out["mean"] = PyMath.fsum(DoubleArray(size) { values[it] * (probabilities[it] / total) })
    val at = listOf("p10" to 0.1, "median" to 0.5, "p90" to 0.9).map { (key, q) -> key to (level?.invoke(q) ?: q) }
    var mass = 0.0
    for (i in 0 until size) {
      mass += probabilities[i] / total
      for ((key, quantile) in at) if (key !in out && mass >= quantile) out[key] = values[i]
    }
    for ((key, _) in at) out.putIfAbsent(key, values[size - 1])
    return out
  }

  override fun equals(other: Any?): Boolean =
    other is Grid && other.low == low && other.high == high && other.size == size

  override fun hashCode(): Int = (low.hashCode() * 31 + high.hashCode()) * 31 + size
}

/** A question as the API reads it (`prompt.as_question`), or one of the readouts a typed answer is read at. */
sealed class Question {
  abstract val instructions: String
  open val type: String
    get() = javaClass.simpleName.lowercase()
}

/** One of the options [criteria] names, or each its own yes or no when [multiple]. */
data class Choice(
  override val instructions: String,
  val criteria: LinkedHashMap<String, String?>,
  val multiple: Boolean = false,
) : Question() {
  override val type = "choice"
}

/** Yes or no; [criteria] optionally `{"true": what yes means, "false": what no means}`. */
data class Noul(override val instructions: String, val criteria: Map<String, Any?>? = null) : Question() {
  override val type = "noul"
}

/** One of 2 to 10 ordered levels, lowest first. */
data class Score(override val instructions: String, val criteria: List<String>) : Question() {
  override val type = "score"
}

/** A span read from the state: text, or a number or a date written there. */
data class Read(override val instructions: String, override val type: String, val unit: String? = null) : Question()

/** A number estimated on a grid over its range, [step] apart (at most 128 values). */
data class Estimate(
  override val instructions: String,
  val low: Double,
  val high: Double,
  val step: Double? = null,
  val unit: String? = null,
) : Question() {
  override val type = "number"
  val grid: Grid by lazy {
    Grid(low, high, if (step == null) 128 else minOf(128, PyMath.roundHalfEven((high - low) / step).toInt() + 1))
  }
}

/** Where one thing is in the picture: a presence and four edges. */
data class BoxQ(override val instructions: String) : Question() {
  override val type = "box"
}

/** Every instance in the picture, [max] blocks read (else [Boxes.INSTANCES]). */
data class BoxesQ(override val instructions: String, val max: Int? = null) : Question() {
  override val type = "boxes"
  val blocks: Int
    get() = max ?: Boxes.INSTANCES
}

/** A pointer slot of a read: its start or its end. */
data class Position(val name: String) : Question() {
  override val instructions = ""
  override val type = "position"
}

/** A slot over a grid's values as the model writes them: value i times [scale] is one token of its own. */
data class NumberSlot(override val instructions: String, val grid: Grid, val scale: Int = 1) : Question() {
  override val type = "grid"
}

/** A readout table: every token once, each option's forms as positions among them. */
class ReadoutTable(val tokens: IntArray, val forms: Array<IntArray>)

/** The prompt of the model's own code (prompt.py), with the checkpoint's settings from config.json. */
class Prompt(
  val tokenizer: Tokenizer,
  /** `d1_codes`: the option-code scheme. */
  private val codes: Int = 2,
  /** `d1_numbers`: grids read as the tokenizer's numbers. */
  private val numbers: Boolean = true,
  /** `d1_prompt`: the prompt version. */
  val version: Int = 1,
) {
  private val aliasCache = HashMap<List<String>, List<Pair<String, Int>>>()
  private val idsCache = HashMap<List<String>, IntArray>()
  private val tableCache = HashMap<Question, ReadoutTable>()
  private val numberIdsCache = HashMap<Pair<Grid, Int>, Array<IntArray>?>()
  private val specials: Pattern? = run {
    val names = tokenizer.specialTokens.filter { it.length > 1 && BREAK !in it }.sortedByDescending { it.length }
    if (names.isEmpty()) null else Pattern.compile(names.joinToString("|") { Pattern.quote(it) })
  }

  val bos: String = "<|startoftext|>"

  /** What the model's template writes between the assistant header and the answer (the LFM3 lead). */
  val lead: String = "<think></think>"

  // ------------------------------------------------------------------------------------------- codes

  /** Native letters when the labels are letters, else A..Z, else the scheme's codes (`option_codes`). */
  fun optionCodes(labels: List<String>): List<String> {
    val labs = labels.map { it.trim() }
    if (labs.isNotEmpty() && labs.all { it.length == 1 && it[0].isLetter() }) return labs
    if (labs.size <= LETTERS.size) return LETTERS.take(labs.size)
    if (codes == 2) return (LETTERS + LETTERS.flatMap { a -> LETTERS.map { b -> a + b } }).take(labs.size)
    return labs.indices.map { String.format("%02d", it) }
  }

  /** Every label's distinct single-token code: [(code, token id)] (`aliases`). */
  fun aliases(labels: List<String>): List<Pair<String, Int>> {
    val codeList = optionCodes(labels)
    return aliasCache.getOrPut(codeList) {
      val used = HashSet<Int>()
      val out = ArrayList<Pair<String, Int>>()
      fun take(raw: String): Boolean {
        val enc = tokenizer.encode(raw)
        if (enc.size != 1 || enc[0] in used) return false
        out.add(raw to enc[0])
        used.add(enc[0])
        return true
      }
      for (code in codeList) {
        if (take(code)) continue
        if (!FALLBACK_POOL.any { take(it) }) error("no single-token alias left for ${codeList.size} options")
      }
      out
    }
  }

  /** The single-token ids among [texts], first occurrence of each (`_ids`). */
  fun ids(texts: List<String>): IntArray =
    idsCache.getOrPut(texts) {
      val out = ArrayList<Int>()
      val seen = HashSet<Int>()
      for (t in texts) {
        val enc = tokenizer.encode(t)
        if (enc.size == 1 && seen.add(enc[0])) out.add(enc[0])
      }
      out.toIntArray()
    }

  // ------------------------------------------------------------------------------------------- slots

  /** Each answer position's readout, in order (`question_slots`). */
  fun questionSlots(q: Question): List<Question> =
    when {
      q is Read -> listOf(Position("start"), Position("end"))
      q is Estimate -> listOf(gridSlot(q.instructions, q.grid))
      q is BoxQ -> {
        val (grid, scale) = edgeGrid()
        listOf(Noul(q.instructions)) + List(4) { gridSlot(q.instructions, grid, scale) }
      }
      q is BoxesQ -> List(q.blocks) { questionSlots(BoxQ(q.instructions)) }.flatten()
      q is Choice && q.multiple -> {
        require(q.criteria.isNotEmpty()) { "choice with no options" }
        List(q.criteria.size) { Noul(q.instructions) }
      }
      else -> listOf(q)
    }

  /** A typed answer, read at its own slots and answered in its own shape (`typed`). */
  fun typed(q: Question): Boolean = q.type in TYPED || q is Choice && q.multiple

  /** The edge grid: thousandths written as numbers where every one is one token, else 128 codes (`box.edge_grid`). */
  fun edgeGrid(): Pair<Grid, Int> =
    if (numbers && numberIds(Boxes.THOUSANDTHS, 1000) != null) Boxes.THOUSANDTHS to 1000 else Boxes.GRID to 1

  /** Each grid value times [scale] as one token, plain and after a space; null when one is not (`number_ids`). */
  fun numberIds(grid: Grid, scale: Int): Array<IntArray>? =
    numberIdsCache.getOrPut(grid to scale) {
      val groups = ArrayList<IntArray>()
      for (value in grid.values) {
        val n = PyMath.roundHalfEven(value * scale)
        if (abs(n - value * scale) > 1e-6) return@getOrPut null
        val written = n.toLong().toString()
        val group = ids(listOf(written, " $written"))
        if (group.isEmpty() || tokenizer.encode(written).size != 1) return@getOrPut null
        groups.add(group)
      }
      groups.toTypedArray()
    }

  /** A grid's slot: the tokenizer's numbers where the checkpoint reads them so, else codes (`grid_slot`). */
  fun gridSlot(instructions: String, grid: Grid, scale: Int = 1): Question =
    if (numbers && numberIds(grid, scale) != null) NumberSlot(instructions, grid, scale)
    else gridQuestion(instructions, grid)

  /** The grid's ordered bins on the digit or option-code readout (`grid_question`). */
  fun gridQuestion(instructions: String, grid: Grid): Question {
    val labels = grid.values.map { PyFormat.g(it) }
    if (grid.size <= 10) return Score(instructions, labels)
    val criteria = LinkedHashMap<String, String?>()
    labels.forEachIndexed { i, v -> criteria[i.toString()] = v }
    return Choice(instructions, criteria)
  }

  // ------------------------------------------------------------------------------------------- readout

  /** Token ids to score, one group per option (`readout_ids`). */
  fun readoutIds(q: Question): List<IntArray> =
    when (q) {
      is Estimate -> readoutIds(questionSlots(q)[0])
      is NumberSlot -> requireNotNull(numberIds(q.grid, q.scale)).toList()
      is Read, is Position -> throw IllegalArgumentException("a pointer reads state positions, not vocabulary codes")
      is Choice ->
        if (q.multiple) {
          throw IllegalArgumentException("a multiple choice reads one yes/no distribution per option")
        } else {
          aliases(q.criteria.keys.toList()).map { (code, tid) ->
            val extra = ids(listOf(" $code"))
            intArrayOf(tid) + extra.filter { it != tid }.toIntArray()
          }
        }
      is Noul -> {
        val yes = ids(YES_FORMS)
        val no = ids(NO_FORMS)
        check(yes.isNotEmpty() && no.isNotEmpty()) { "tokenizer has no single-token yes/no" }
        listOf(yes, no)
      }
      is Score ->
        q.criteria.indices.map { i ->
          ids(listOf(i.toString())).also { check(it.isNotEmpty()) { "score levels need single-token digits" } }
        }
      is BoxQ, is BoxesQ -> throw IllegalArgumentException("a box is read at its own slots")
    }

  /** `readout_ids` as one gather: every token once; a short group repeats its first form (`readout_table`). */
  fun readoutTable(q: Question): ReadoutTable =
    tableCache.getOrPut(q) {
      val groups = readoutIds(q)
      val order = LinkedHashMap<Int, Int>()
      for (g in groups) for (t in g) order.putIfAbsent(t, order.size)
      val width = groups.maxOf { it.size }
      val forms = Array(groups.size) { o -> IntArray(width) { k -> order.getValue(groups[o][if (k < groups[o].size) k else 0]) } }
      ReadoutTable(order.keys.toIntArray(), forms)
    }

  // ------------------------------------------------------------------------------------------- text

  /** The state as the block before `QUESTION:` (`state_block`, the default `json_only` style). */
  fun stateBlock(state: Any?): String =
    if (state is String) "$state\n\n" else D1Json.writeIndented(state, 2) + "\n\n"

  /** The state as rendered, without the blank line after it: what a read's span offsets index (`state_text`). */
  fun stateText(state: Any?): String = stateBlock(state).let { it.substring(0, it.length - 2) }

  /** [text] with every special token broken by a zero-width space after its first character (`plain`). */
  fun plain(text: String): String = plainOffsets(text, false).first

  /** The escaped text and every character boundary mapped to the original (`plain_offsets`). */
  fun plainOffsets(text: String, offsets: Boolean = true): Pair<String, MutableList<Int>?> {
    val boundaries: MutableList<Int>? = if (offsets) MutableList(text.length + 1) { it } else null
    val pattern = specials ?: return text to boundaries
    var current = text
    var n = 1
    while (n > 0) {
      val matcher = pattern.matcher(current)
      val insertions = ArrayList<Int>()
      val out = StringBuilder()
      var last = 0
      n = 0
      while (matcher.find()) {
        insertions.add(matcher.start() + 1)
        out.append(current, last, matcher.start())
        val m = matcher.group()
        out.append(m[0]).append(BREAK).append(m, 1, m.length)
        last = matcher.end()
        n++
      }
      out.append(current, last, current.length)
      if (boundaries != null) for (at in insertions.asReversed()) boundaries.add(at, boundaries[at])
      current = out.toString()
    }
    return current to boundaries
  }

  /** What opens a question after the state, repeated before each question packed after it (`question_header`). */
  fun questionHeader(state: Any?): String = if (state == null) "" else "QUESTION:\n"

  /** Everything before the first question: the media's markup, then the state (`prefix_text`). */
  fun prefixText(state: Any?, media: String = ""): String {
    val body = if (state == null) "" else plain(stateBlock(state)) + "\n" + questionHeader(state)
    return "$bos$IM_START" + "user\n" + media + body
  }

  /** The legend of a box's edges (`edge_legend`). */
  private fun edgeLegend(): String {
    val (grid, scale) = edgeGrid()
    if (scale > 1) {
      return "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, " +
        "left and right from its left edge."
    }
    val codesOf = aliases((gridQuestion("", grid) as Choice).criteria.keys.toList()).map { it.first }
    return "Each edge is a code on a ${grid.size}-value scale: ${codesOf.first()} is 0 (top or left), " +
      "${codesOf.last()} is 1 (bottom or right). Values are equally spaced."
  }

  /** The question as the prompt writes it (`question_block`, option style `desc`). */
  fun questionBlock(q: Question): String {
    val said = q.instructions
    return when {
      q is BoxQ -> {
        val one = if (questionSlots(q).last() is NumberSlot) "one number" else "one code"
        "$said\n\nPresence is yes or no. ${edgeLegend()} Reply with yes or no at present, " +
          "then $one at top, bottom, left and right."
      }
      q is BoxesQ -> {
        val most = if (version == 1) "${Boxes.INSTANCES} at most, " else ""
        "$said\n\nEvery one of them, ${most}in reading order (left edge, then top). " +
          "Present k is yes when there are at least k. ${edgeLegend()} Reply with yes or no at each " +
          "present, then its top, bottom, left and right."
      }
      q is Estimate && questionSlots(q)[0] is NumberSlot -> {
        val unit = if (q.unit != null) " ${q.unit}" else ""
        "$said\n\nEstimate from ${PyFormat.g(q.grid.low)} to ${PyFormat.g(q.grid.high)}$unit, in steps of " +
          "${PyFormat.g(q.grid.step)}. Reply with the number only."
      }
      q is Estimate -> {
        val bins = gridQuestion("", q.grid)
        val codesOf =
          if (bins is Score) List(q.grid.size) { it.toString() }
          else aliases((bins as Choice).criteria.keys.toList()).map { it.first }
        val unit = if (q.unit != null) " ${q.unit}" else ""
        "$said\n\nEstimate from ${PyFormat.g(q.grid.low)} to ${PyFormat.g(q.grid.high)}$unit on " +
          "${q.grid.size} equally spaced values, step ${PyFormat.g(q.grid.step)}: " +
          "${codesOf.first()} is ${PyFormat.g(q.grid.low)}, ${codesOf.last()} is ${PyFormat.g(q.grid.high)}. " +
          "Reply with the code only."
      }
      q is Read -> {
        val unit = if (q.type == "number" && !q.unit.isNullOrEmpty()) " (unit: ${q.unit})" else ""
        "$said\n\nRead the ${q.type}$unit from the state, or nothing if it is absent."
      }
      q is Choice && q.multiple -> {
        val lines = q.criteria.entries.joinToString("\n") { (name, desc) ->
          "${D1Json.dumps(name)}: ${if (desc.isNullOrEmpty()) name else desc}"
        }
        "$said\n\nOptions:\n$lines\n\nReply with yes or no for each option."
      }
      q is Choice -> {
        val labels = q.criteria.keys.toList()
        val lines = aliases(labels).zip(labels).joinToString("\n") { (alias, label) ->
          "${alias.first} ${q.criteria[label].takeUnless { it.isNullOrEmpty() } ?: label.replace("_", " ")}"
        }
        "$said\n\nOptions:\n$lines\n\nReply with the option code only."
      }
      q is Noul -> {
        val extra = if (!q.criteria.isNullOrEmpty()) {
          "\nYes: ${pyStr(q.criteria["true"])}\nNo: ${pyStr(q.criteria["false"])}"
        } else ""
        "$said$extra\n\nReply with yes or no only."
      }
      q is Score -> {
        val legend = q.criteria.withIndex().joinToString("\n") { (i, name) -> "$i $name" }
        "$said\n\n$legend\n\nReply with a single digit 0-${q.criteria.size - 1} only."
      }
      else -> throw IllegalArgumentException("unknown question type ${q.javaClass.simpleName}")
    }
  }

  /** The question and the assistant header up to the answer slot, and each slot's token offset (`suffix_layout`). */
  fun suffixLayout(q: Question): Pair<String, IntArray> {
    val body = questionBlock(q)
    return when {
      q is Read -> markedSuffix(body, listOf("start", "end"))
      q is BoxQ -> markedSuffix(body, listOf("present") + Boxes.EDGES)
      q is BoxesQ -> markedSuffix(
        body,
        (1..q.blocks).flatMap { k -> (listOf("present") + Boxes.EDGES).map { "$it $k" } },
        if (q.blocks < Boxes.INSTANCES || version > 1) "\n" else "",
      )
      q is Choice && q.multiple -> markedSuffix(body, q.criteria.keys.map { D1Json.dumps(it) })
      else -> "${plain(body)}$IM_END\n${IM_START}assistant\n$lead" to intArrayOf(0)
    }
  }

  /** A question's named answer markers and their token offsets from its last token (`marked_suffix`). */
  private fun markedSuffix(body: String, names: List<String>, after: String = ""): Pair<String, IntArray> {
    require(names.isNotEmpty() && names.toSet().size == names.size && names.all { it.isNotEmpty() && '\n' !in it }) {
      "slot names must be distinct nonempty single lines"
    }
    var text = "${plain(body)}$IM_END\n${IM_START}assistant\n$lead"
    val ends = ArrayList<Int>()
    for ((i, name) in names.withIndex()) {
      text += (if (i > 0) "\n" else "") + plain(name) + ":"
      ends.add(text.length)
    }
    if (after.isNotEmpty()) {
      val tail = tokenizer.encode(text + after)
      if (tokenizer.decode(intArrayOf(tail.last())).trimEnd().endsWith(":")) text += after
    }
    val encoded = tokenizer.encodeWithOffsets(text)
    val slots = IntArray(ends.size)
    for ((k, chars) in ends.withIndex()) {
      val matching = (0 until encoded.size).filter { encoded.starts[it] <= chars - 1 && chars - 1 < encoded.ends[it] }
      require(matching.size == 1) { "a slot marker must have an exact colon token position" }
      val end = matching[0]
      val stop = encoded.ends[end]
      require(text.substring(chars, stop).isBlank() && tokenizer.decode(intArrayOf(encoded.ids[end])).trimEnd().endsWith(":")) {
        "a slot marker token must end with its colon and optional whitespace"
      }
      slots[k] = end - encoded.size + 1
    }
    return text to slots
  }

  private fun pyStr(value: Any?): String =
    when (value) {
      null -> "None"
      is Boolean -> if (value) "True" else "False"
      is JsonNumber -> value.pythonString()
      is Double -> PythonFloat.repr(value)
      else -> value.toString()
    }

  companion object {
    const val IM_START = "<|im_start|>"
    const val IM_END = "<|im_end|>"
    const val BREAK = "​"
    /** A spoken question's place in its suffix until its clip's markup replaces it (media.SPOKEN). */
    const val SPOKEN = "\uE000"
    val TYPED = setOf("text", "number", "date", "box", "boxes")
    val YES_FORMS = listOf("yes", "Yes", "YES")
    val NO_FORMS = listOf("no", "No", "NO")
    val LETTERS = ('A'..'Z').map { it.toString() }
    val FALLBACK_POOL: List<String> =
      ('A'..'Z').map { it.toString() } + (0 until 100).map { String.format("%02d", it) } +
        ('a'..'z').map { it.toString() } + (0 until 200).map { "#$it" } +
        ('A'..'Z').flatMap { a -> ('A'..'Z').map { b -> "$a$b" } }
  }
}

/** Python number helpers the ported code needs. */
object PyMath {
  /** Python's `round()` of a float to an integer value: halves to even. */
  fun roundHalfEven(x: Double): Double = Math.rint(x)

  /** `math.fsum`: the correctly rounded sum (Shewchuk's partials, as CPython). */
  fun fsum(values: DoubleArray): Double {
    val partials = ArrayList<Double>()
    for (value in values) {
      var x = value
      var i = 0
      for (yOriginal in partials.toList()) {
        var y = yOriginal
        if (abs(x) < abs(y)) {
          val t = x
          x = y
          y = t
        }
        val hi = x + y
        val lo = y - (hi - x)
        if (lo != 0.0) {
          partials[i] = lo
          i++
        }
        x = hi
      }
      while (partials.size > i) partials.removeAt(partials.size - 1)
      if (x != 0.0) partials.add(x)
    }
    // The CPython ending: sum the partials from the top, with the half-way correction.
    if (partials.isEmpty()) return 0.0
    var n = partials.size - 1
    var hi = partials[n]
    var lo = 0.0
    while (n > 0) {
      val x = hi
      n--
      val y = partials[n]
      hi = x + y
      val yr = hi - x
      lo = y - yr
      if (lo != 0.0) break
    }
    if (n > 0 && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0))) {
      val y = lo * 2.0
      val x = hi + y
      val yr = x - hi
      if (y == yr) hi = x
    }
    return hi
  }
}

/** Python's `format(x, "g")` (6 significant digits, trailing zeros removed). */
object PyFormat {
  fun g(x: Double): String {
    if (x.isNaN()) return "nan"
    if (x.isInfinite()) return if (x > 0) "inf" else "-inf"
    if (x == 0.0) return if (1.0 / x < 0) "-0" else "0"
    val rounded = BigDecimal(x).round(MathContext(6, RoundingMode.HALF_EVEN))
    val exponent = rounded.precision() - rounded.scale() - 1
    return if (exponent < -4 || exponent >= 6) {
      val digits = rounded.unscaledValue().abs().toString().trimEnd('0').ifEmpty { "0" }
      val mantissa = if (digits.length == 1) digits else "${digits[0]}.${digits.substring(1)}"
      (if (x < 0) "-" else "") + mantissa + "e" + (if (exponent < 0) "-" else "+") +
        abs(exponent).toString().padStart(2, '0')
    } else {
      val plain = rounded.toPlainString()
      if ('.' in plain) plain.trimEnd('0').trimEnd('.') else plain
    }
  }
}
