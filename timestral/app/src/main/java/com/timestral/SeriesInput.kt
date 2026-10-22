package com.timestral

import java.text.Normalizer
import java.util.Locale

/**
 * A series read from pasted text or a file: [values] in order (NaN = missing), and the notes the
 * screen shows next to the forecast.
 */
class ParsedSeries(val values: FloatArray, val notes: List<String>) {
  /** Values that are numbers (finite). */
  val observedCount: Int
    get() = values.count { it.isFinite() }
}

/** Why a series or horizon cannot be forecast; [message] is the sentence the screen shows. */
class InputException(message: String) : Exception(message)

/**
 * Reads the user's series and checks it against the sample's limits. Android-free.
 *
 * Two layouts are read:
 * - a table: two or more lines with the same number of fields (separated by commas, semicolons or
 *   tabs), either two fields per line or a header line or a first column of labels (dates); the
 *   last column is the series and a first line whose last field is not a number is a header;
 * - a list: anything else; every field separated by commas, semicolons, tabs, spaces or new lines
 *   is one value.
 *
 * An empty field, `NaN`, `NA`, `N/A`, `null`, `none`, `-`, `?` or an infinite value is a missing
 * step; the model fills it the official way (it attends only to observed steps).
 */
object SeriesInput {
  /** Fewest values, and fewest observed values, the sample forecasts from. */
  const val MIN_VALUES = 16

  private val MISSING = setOf("", "nan", "na", "n/a", "null", "none", "-", "?", "inf", "-inf", "+inf")
  private val TABLE_SEPARATORS = Regex("[,;\t]")
  private val LIST_SEPARATORS = Regex("[,;\\s]+")

  /**
   * The series in [text], or [InputException] with the sentence to show: no values, a value that
   * is not a number, fewer than [MIN_VALUES] values, or fewer than [MIN_VALUES] observed values in
   * the context the model would read (the last [TimestralHost.MAX_CONTEXT] steps).
   */
  fun parse(text: String): ParsedSeries {
    val fields = fields(normalize(text))
    if (fields.isEmpty()) {
      throw InputException("Paste a series first: numbers separated by commas, spaces or new lines.")
    }
    val values = FloatArray(fields.size)
    for ((index, field) in fields.withIndex()) {
      values[index] =
        value(field)
          ?: throw InputException(
            "Value ${index + 1} (\"${field.take(24)}\") is not a number. Use numbers, or leave a " +
              "field empty or write NaN for a missing value."
          )
    }
    if (values.size < MIN_VALUES) {
      throw InputException(
        "The series has ${values.size} values; this sample needs at least $MIN_VALUES."
      )
    }
    val notes = ArrayList<String>()
    val context =
      if (values.size > TimestralHost.MAX_CONTEXT) {
        notes.add(
          "Only the last ${format(TimestralHost.MAX_CONTEXT)} of ${format(values.size)} values " +
            "are the context (the largest graph of this sample)."
        )
        values.copyOfRange(values.size - TimestralHost.MAX_CONTEXT, values.size)
      } else {
        values
      }
    val observed = context.count { it.isFinite() }
    if (observed < MIN_VALUES) {
      throw InputException(
        "Only $observed of ${format(context.size)} values in the context are numbers; this " +
          "sample needs at least $MIN_VALUES observed values (empty, NaN and infinite values are " +
          "missing)."
      )
    }
    if (observed < context.size) {
      notes.add("${format(context.size - observed)} missing values: the model reads only the observed ones.")
    }
    constantValue(context)?.let { level ->
      notes.add(
        "Every observed value is ${number(level)}: the forecast stays at that level with almost " +
          "no spread."
      )
    }
    return ParsedSeries(values, notes)
  }

  /**
   * The horizon in [text] (a whole number of steps), or [InputException] when it is not one from
   * 1 to [TimestralHost.MAX_HORIZON].
   */
  fun horizon(text: String): Int {
    val steps = normalize(text).trim().replace(",", "").toIntOrNull()
    if (steps == null || steps < 1 || steps > TimestralHost.MAX_HORIZON) {
      throw InputException(
        "The horizon must be a whole number of steps from 1 to " +
          "${format(TimestralHost.MAX_HORIZON)} (one decoder block of 32 patches); longer " +
          "horizons are not part of this sample."
      )
    }
    return steps
  }

  /**
   * The sentence when none of the [graphs] that [backend] runs holds a context of [contextSize]
   * values, else null. With the shipped graphs that is the NPU past 2,048 steps. The 8,192-step
   * graphs run on the GPU and the CPU only: on a Galaxy S26, 8,192-step builds of this model ran 82
   * to 97 ms per forecast on the NPU against 69 to 73 ms for these graphs on the GPU. The sample
   * does not move a forecast to another accelerator by itself.
   */
  fun contextTooLong(graphs: List<GraphSpec>, backend: Backend, contextSize: Int): InputException? {
    val longest =
      graphs.filter { backend in it.backends }.maxOfOrNull { it.contextPatches } ?: return null
    val steps = longest * TimestralHost.PATCH
    if (contextSize <= steps) return null
    val reason =
      if (backend == Backend.NPU) {
        " The 8,192-step graphs run on the GPU and the CPU only: on a Galaxy S26, 8,192-step " +
          "builds of this model ran 82 to 97 ms per forecast on the NPU against 69 to 73 ms " +
          "for these graphs on the GPU."
      } else {
        ""
      }
    return InputException(
      "The ${backend.label} takes a context of up to ${format(steps)} steps, and this one has " +
        "${format(contextSize)}.$reason Choose another accelerator, or keep the last " +
        "${format(steps)} values of the series."
    )
  }

  /** The value when every observed value of [values] is the same, else null. */
  fun constantValue(values: FloatArray): Float? {
    var level = Float.NaN
    for (value in values) {
      if (!value.isFinite()) continue
      if (level.isNaN()) {
        level = value
      } else if (value != level) {
        return null
      }
    }
    return if (level.isNaN()) null else level
  }

  /**
   * NFKC: the full-width digits, commas, periods and minus signs a CJK keyboard types (or a pasted
   * spreadsheet holds) become their ASCII forms; a Japanese keyboard in full-width mode typed
   * U+FF12 U+FF11 U+FF12 U+FF0C ... for "212,..." on a Pixel 8a.
   */
  private fun normalize(text: String): String = Normalizer.normalize(text, Normalizer.Form.NFKC)

  private fun fields(text: String): List<String> {
    val lines = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
    if (lines.isEmpty()) return emptyList()
    val split = lines.map { line -> line.split(TABLE_SEPARATORS).map { it.trim() } }
    val columns = split.first().size
    val header = value(split.first().last()) == null
    val labels = split.any { value(it.first()) == null }
    val table =
      lines.size >= 2 &&
        columns >= 2 &&
        split.all { it.size == columns } &&
        (columns == 2 || header || labels)
    if (table) {
      val column = split.map { it.last() }
      return if (header) column.drop(1) else column
    }
    return text.trim().split(LIST_SEPARATORS).filter { it.isNotEmpty() }
  }

  /** The float of [field]; NaN for a missing marker or an infinite value; null for anything else. */
  private fun value(field: String): Float? {
    val trimmed = field.trim()
    if (trimmed.lowercase(Locale.ROOT) in MISSING) return Float.NaN
    val number = trimmed.toDoubleOrNull() ?: return null
    if (trimmed.contains("Infinity", ignoreCase = true)) return Float.NaN
    val single = number.toFloat()
    return if (single.isFinite()) single else Float.NaN
  }

  private fun format(count: Int): String = String.format(Locale.US, "%,d", count)

  private fun number(value: Float): String = value.toString().removeSuffix(".0")
}
