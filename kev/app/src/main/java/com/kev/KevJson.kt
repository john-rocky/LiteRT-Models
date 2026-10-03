package com.kev

import java.math.BigDecimal
import java.math.BigInteger
import java.math.MathContext
import java.math.RoundingMode
import kotlin.math.abs

/**
 * A JSON number kept as its literal. Python's `json` parses `1` as an int and `1.0` as a float, and
 * the author's `render` prints them differently (`1` / `1.0`), so the difference has to survive
 * parsing.
 */
class JsonNumber(val literal: String) {
  /** True when Python's `json` parses the literal as an int (no fraction, no exponent). */
  val isInteger: Boolean = literal.isNotEmpty() && literal.all { it == '-' || it in '0'..'9' }

  /** The value as a double, correctly rounded like Python's `float()`. */
  fun toDouble(): Double = literal.toDouble()

  /** The value of an integer literal; fails for floats and for values outside Int. */
  fun toInt(): Int {
    require(isInteger) { "Not an integer: $literal" }
    // BigInteger.intValueExact() needs API 31; an Int holds at most 31 value bits.
    val value = BigInteger(literal)
    require(value.bitLength() < Int.SIZE_BITS) { "Outside Int: $literal" }
    return value.toInt()
  }

  /** Python's `str()` of the value `json.loads` makes of this literal. */
  fun pythonString(): String =
    if (isInteger) BigInteger(literal).toString() else PythonFloat.repr(toDouble())

  override fun equals(other: Any?): Boolean = other is JsonNumber && other.literal == literal

  override fun hashCode(): Int = literal.hashCode()

  override fun toString(): String = literal
}

/** Python's `repr(float)` (`str()` is the same): the shortest digits that read back exactly. */
object PythonFloat {
  /** Most significant digits a double needs to read back exactly. */
  private const val MAX_DIGITS = 17

  /** `repr` switches to an exponent below 1e-4 and from 1e16 on. */
  private const val SMALLEST_FIXED_POINT = -4
  private const val LARGEST_FIXED_POINT = 16

  fun repr(value: Double): String {
    if (value.isNaN()) return "nan"
    if (value.isInfinite()) return if (value > 0) "inf" else "-inf"
    if (value == 0.0) return if (1.0 / value < 0) "-0.0" else "0.0"
    val sign = if (value < 0) "-" else ""
    val shortest = shortest(abs(value))
    val digits = shortest.unscaledValue().toString()
    // The decimal point sits after `point` digits (dtoa's decpt).
    val point = digits.length - shortest.scale()
    val body =
      if (point <= SMALLEST_FIXED_POINT || point > LARGEST_FIXED_POINT) {
        val exponent = point - 1
        val mantissa = if (digits.length == 1) digits else "${digits[0]}.${digits.substring(1)}"
        val exponentSign = if (exponent < 0) "-" else "+"
        "${mantissa}e$exponentSign${abs(exponent).toString().padStart(2, '0')}"
      } else if (point <= 0) {
        "0." + "0".repeat(-point) + digits
      } else if (point >= digits.length) {
        digits + "0".repeat(point - digits.length) + ".0"
      } else {
        digits.substring(0, point) + "." + digits.substring(point)
      }
    return sign + body
  }

  /**
   * The shortest decimal that reads back as [value] (positive, finite), and of those the nearest
   * one, as David Gay's dtoa mode 0 that CPython uses. At each length only the two neighbours of
   * the exact binary value can read back, so both are tried.
   */
  private fun shortest(value: Double): BigDecimal {
    val exact = BigDecimal(value)
    for (precision in 1..MAX_DIGITS) {
      val down = exact.round(MathContext(precision, RoundingMode.DOWN))
      val up = exact.round(MathContext(precision, RoundingMode.UP))
      val downReadsBack = down.toDouble() == value
      val upReadsBack = up.toDouble() == value
      if (downReadsBack || upReadsBack) {
        val chosen =
          if (downReadsBack && upReadsBack) {
            val below = exact.subtract(down)
            val above = up.subtract(exact)
            when {
              below < above -> down
              above < below -> up
              else -> exact.round(MathContext(precision, RoundingMode.HALF_EVEN))
            }
          } else if (downReadsBack) {
            down
          } else {
            up
          }
        return chosen.stripTrailingZeros()
      }
    }
    error("No $MAX_DIGITS-digit decimal reads back as $value")
  }
}

/**
 * Pull reader over UTF-8 JSON with the semantics of Python's `json.loads`: objects keep their key
 * order (a repeated key keeps its first position and its last value), numbers keep their literal,
 * `NaN`, `Infinity` and `-Infinity` are accepted, and raw control characters inside strings are
 * rejected. The tokenizer streams its 248k-entry vocabulary through it without building a tree.
 */
class KevJsonReader(private val data: ByteArray) {
  /** The kind of the next value. */
  enum class Kind {
    OBJECT,
    ARRAY,
    STRING,
    NUMBER,
    BOOLEAN,
    NULL,
  }

  private var position = 0
  // One entry per open container: true until its first element has been announced by hasNext().
  private val firstElement = ArrayList<Boolean>()
  // hasNext() returned true and consumed the separating comma; the next read takes the element.
  private var elementAnnounced = false

  /** Kind of the next value. */
  fun peek(): Kind {
    skipWhitespace()
    if (position >= data.size) fail("Expecting value")
    return when (data[position].toInt().toChar()) {
      '{' -> Kind.OBJECT
      '[' -> Kind.ARRAY
      '"' -> Kind.STRING
      't',
      'f' -> Kind.BOOLEAN
      'n' -> Kind.NULL
      else -> Kind.NUMBER
    }
  }

  fun beginObject() {
    startValue()
    expect('{')
    firstElement.add(true)
  }

  fun endObject() {
    skipWhitespace()
    expect('}')
    firstElement.removeAt(firstElement.lastIndex)
  }

  fun beginArray() {
    startValue()
    expect('[')
    firstElement.add(true)
  }

  fun endArray() {
    skipWhitespace()
    expect(']')
    firstElement.removeAt(firstElement.lastIndex)
  }

  /** True when the open object or array has another element; consumes the comma before it. */
  fun hasNext(): Boolean {
    if (elementAnnounced) return true
    check(firstElement.isNotEmpty()) { "hasNext() outside an object or array" }
    skipWhitespace()
    if (position < data.size && (data[position] == CLOSE_OBJECT || data[position] == CLOSE_ARRAY)) {
      return false
    }
    if (!firstElement.last()) {
      expect(',')
      skipWhitespace()
    }
    firstElement[firstElement.lastIndex] = false
    elementAnnounced = true
    return true
  }

  /** The next member name of the open object (after hasNext() returned true). */
  fun nextName(): String {
    check(elementAnnounced) { "nextName() without hasNext()" }
    elementAnnounced = false
    skipWhitespace()
    val name = readString()
    skipWhitespace()
    expect(':')
    return name
  }

  fun nextString(): String {
    startValue()
    return readString()
  }

  fun nextNumber(): JsonNumber {
    startValue()
    val start = position
    for (special in SPECIAL_NUMBERS) {
      if (matches(special)) {
        position += special.length
        return JsonNumber(special)
      }
    }
    if (current() == '-') position++
    when (current()) {
      '0' -> position++
      in '1'..'9' -> skipDigits()
      else -> fail("Expecting value")
    }
    if (current() == '.') {
      position++
      if (current() !in '0'..'9') fail("Expecting digits after the decimal point")
      skipDigits()
    }
    if (current() == 'e' || current() == 'E') {
      position++
      if (current() == '+' || current() == '-') position++
      if (current() !in '0'..'9') fail("Expecting exponent digits")
      skipDigits()
    }
    return JsonNumber(String(data, start, position - start, Charsets.ISO_8859_1))
  }

  /** The next number as an Int (an integer literal). */
  fun nextInt(): Int = nextNumber().toInt()

  fun nextBoolean(): Boolean {
    startValue()
    return when {
      matches("true") -> {
        position += 4
        true
      }
      matches("false") -> {
        position += 5
        false
      }
      else -> fail("Expecting value")
    }
  }

  fun nextNull() {
    startValue()
    if (!matches("null")) fail("Expecting value")
    position += 4
  }

  /**
   * The next value as a tree: [LinkedHashMap] for objects, [ArrayList] for arrays, [String],
   * [JsonNumber], [Boolean] or null.
   */
  fun readValue(): Any? =
    when (peek()) {
      Kind.OBJECT -> {
        val map = LinkedHashMap<String, Any?>()
        beginObject()
        while (hasNext()) {
          val name = nextName()
          map[name] = readValue()
        }
        endObject()
        map
      }
      Kind.ARRAY -> {
        val list = ArrayList<Any?>()
        beginArray()
        while (hasNext()) {
          list.add(readValue())
        }
        endArray()
        list
      }
      Kind.STRING -> nextString()
      Kind.NUMBER -> nextNumber()
      Kind.BOOLEAN -> nextBoolean()
      Kind.NULL -> {
        nextNull()
        null
      }
    }

  /** Requires that only whitespace follows the top-level value. */
  fun endDocument() {
    skipWhitespace()
    if (position != data.size) fail("Extra data")
  }

  private fun startValue() {
    elementAnnounced = false
    skipWhitespace()
  }

  private fun readString(): String {
    expect('"')
    var builder: StringBuilder? = null
    var runStart = position
    while (true) {
      if (position >= data.size) fail("Unterminated string")
      val byte = data[position].toInt() and BYTE_MASK
      when {
        byte == '"'.code -> {
          val run = String(data, runStart, position - runStart, Charsets.UTF_8)
          position++
          return builder?.append(run)?.toString() ?: run
        }
        byte == '\\'.code -> {
          val text = builder ?: StringBuilder().also { builder = it }
          text.append(String(data, runStart, position - runStart, Charsets.UTF_8))
          position++
          appendEscape(text)
          runStart = position
        }
        byte < FIRST_PRINTABLE -> fail("Invalid control character")
        else -> position++
      }
    }
  }

  private fun appendEscape(text: StringBuilder) {
    if (position >= data.size) fail("Unterminated string")
    val escape = data[position].toInt().toChar()
    position++
    when (escape) {
      '"' -> text.append('"')
      '\\' -> text.append('\\')
      '/' -> text.append('/')
      'b' -> text.append('\b')
      'f' -> text.append('\u000c')
      'n' -> text.append('\n')
      'r' -> text.append('\r')
      't' -> text.append('\t')
      'u' -> {
        if (position + HEX_DIGITS > data.size) fail("Invalid \\uXXXX escape")
        val hex = String(data, position, HEX_DIGITS, Charsets.ISO_8859_1)
        val code = hex.toIntOrNull(HEX_RADIX) ?: fail("Invalid \\uXXXX escape")
        // A surrogate pair arrives as two escapes; as UTF-16 code units they join by themselves.
        text.append(code.toChar())
        position += HEX_DIGITS
      }
      else -> fail("Invalid \\escape")
    }
  }

  private fun skipDigits() {
    while (current() in '0'..'9') position++
  }

  private fun skipWhitespace() {
    while (position < data.size) {
      when (data[position].toInt().toChar()) {
        ' ',
        '\t',
        '\n',
        '\r' -> position++
        else -> return
      }
    }
  }

  private fun current(): Char =
    if (position < data.size) (data[position].toInt() and BYTE_MASK).toChar() else END

  private fun matches(word: String): Boolean {
    if (position + word.length > data.size) return false
    for (index in word.indices) {
      if (data[position + index].toInt() != word[index].code) return false
    }
    return true
  }

  private fun expect(character: Char) {
    if (current() != character) fail("Expecting '$character'")
    position++
  }

  private fun fail(message: String): Nothing =
    throw IllegalArgumentException("$message at byte $position")

  private companion object {
    const val BYTE_MASK = 0xff
    const val FIRST_PRINTABLE = 0x20
    const val HEX_DIGITS = 4
    const val HEX_RADIX = 16
    const val END = '\u0000'
    const val CLOSE_OBJECT = '}'.code.toByte()
    const val CLOSE_ARRAY = ']'.code.toByte()
    // Longest first: "-Infinity" must win over a plain minus sign.
    val SPECIAL_NUMBERS = listOf("-Infinity", "Infinity", "NaN")
  }
}

/** Parsing and writing with Python `json` semantics (see [KevJsonReader]). */
object KevJson {
  fun parse(text: String): Any? = parse(text.toByteArray(Charsets.UTF_8))

  fun parse(data: ByteArray): Any? {
    val reader = KevJsonReader(data)
    val value = reader.readValue()
    reader.endDocument()
    return value
  }

  /**
   * Compact JSON in insertion order, like Python's `json.dumps(value, ensure_ascii=False,
   * separators=(",", ":"))`: floats print as `repr`, non-ASCII characters stay as they are.
   */
  fun write(value: Any?): String = StringBuilder().also { append(it, value) }.toString()

  /**
   * [write] laid out like Python's `json.dumps(value, indent=indent, ensure_ascii=False)`: one
   * member per line, `": "` after keys, empty objects and arrays as `{}` and `[]`.
   */
  fun writeIndented(value: Any?, indent: Int = 2): String =
    StringBuilder().also { appendIndented(it, value, indent, 0) }.toString()

  private fun appendIndented(out: StringBuilder, value: Any?, indent: Int, level: Int) {
    val items: List<*>? =
      when (value) {
        is IntArray -> value.asList()
        is FloatArray -> value.asList()
        is DoubleArray -> value.asList()
        is Iterable<*> -> value.toList()
        else -> null
      }
    when {
      value is Map<*, *> && value.isNotEmpty() -> {
        out.append('{')
        var first = true
        for ((key, member) in value) {
          out.append(if (first) "\n" else ",\n").append(" ".repeat(indent * (level + 1)))
          first = false
          appendString(out, key as String)
          out.append(": ")
          appendIndented(out, member, indent, level + 1)
        }
        out.append('\n').append(" ".repeat(indent * level)).append('}')
      }
      items != null && items.isNotEmpty() -> {
        out.append('[')
        var first = true
        for (item in items) {
          out.append(if (first) "\n" else ",\n").append(" ".repeat(indent * (level + 1)))
          first = false
          appendIndented(out, item, indent, level + 1)
        }
        out.append('\n').append(" ".repeat(indent * level)).append(']')
      }
      else -> append(out, value)
    }
  }

  private fun append(out: StringBuilder, value: Any?) {
    when (value) {
      null -> out.append("null")
      is Boolean -> out.append(if (value) "true" else "false")
      is String -> appendString(out, value)
      is JsonNumber -> out.append(if (value.isInteger) value.pythonString() else number(value.toDouble()))
      is Int,
      is Long -> out.append(value.toString())
      is Double -> out.append(number(value))
      is Float -> out.append(number(value.toDouble()))
      is Map<*, *> -> {
        out.append('{')
        var first = true
        for ((key, member) in value) {
          if (!first) out.append(',')
          first = false
          appendString(out, key as String)
          out.append(':')
          append(out, member)
        }
        out.append('}')
      }
      is Iterable<*> -> appendArray(out, value.iterator())
      is IntArray -> appendArray(out, value.iterator())
      is FloatArray -> appendArray(out, value.iterator())
      is DoubleArray -> appendArray(out, value.iterator())
      else -> throw IllegalArgumentException("Not JSON: ${value::class.java.name}")
    }
  }

  private fun appendArray(out: StringBuilder, items: Iterator<*>) {
    out.append('[')
    var first = true
    for (item in items) {
      if (!first) out.append(',')
      first = false
      append(out, item)
    }
    out.append(']')
  }

  /** Python `json` writes non-finite floats as `NaN` / `Infinity` / `-Infinity`. */
  private fun number(value: Double): String =
    when {
      value.isNaN() -> "NaN"
      value == Double.POSITIVE_INFINITY -> "Infinity"
      value == Double.NEGATIVE_INFINITY -> "-Infinity"
      else -> PythonFloat.repr(value)
    }

  private fun appendString(out: StringBuilder, text: String) {
    out.append('"')
    for (character in text) {
      when (character) {
        '"' -> out.append("\\\"")
        '\\' -> out.append("\\\\")
        '\n' -> out.append("\\n")
        '\r' -> out.append("\\r")
        '\t' -> out.append("\\t")
        '\b' -> out.append("\\b")
        '\u000c' -> out.append("\\f")
        else ->
          if (character.code < CONTROL_LIMIT) {
            out.append("\\u").append(character.code.toString(HEX_RADIX).padStart(HEX_DIGITS, '0'))
          } else {
            out.append(character)
          }
      }
    }
    out.append('"')
  }

  private const val CONTROL_LIMIT = 0x20
  private const val HEX_RADIX = 16
  private const val HEX_DIGITS = 4
}
