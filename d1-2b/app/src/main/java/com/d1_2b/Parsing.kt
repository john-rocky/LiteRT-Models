/*
 * Ports parsing.py of LiquidAI/d1-2B (revision 10e7671d) to Kotlin; that code is under the
 * LFM Open License v1.0 (see LICENSE and NOTICE).
 */

package com.d1_2b

import java.math.BigDecimal
import java.text.Normalizer
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.util.regex.Matcher
import java.util.regex.Pattern

/** Strict readings of written values: no arithmetic, unit conversion or invented date components. */
object Parsing {
  // Python's `re` on str patterns is Unicode-aware: \d, \w, \s and \b, and case folding under re.I. The JVM
  // needs UNICODE_CHARACTER_CLASS and the inline flag u for that; Android's engine (ICU) is Unicode-aware by default
  // and refuses both (IllegalArgumentException, PatternSyntaxException), so they are used only where they compile.
  private val U: Int =
    try {
      Pattern.compile("\\w", Pattern.UNICODE_CHARACTER_CLASS)
      Pattern.UNICODE_CHARACTER_CLASS
    } catch (unsupported: IllegalArgumentException) {
      0
    }
  private val IGNORE_CASE_GROUP = if (U != 0) "(?iu:" else "(?i:"
  private const val I = Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE

  private val MONTH_NAMES =
    listOf(
      "january janvier januar enero gennaio janeiro januari stycznia styczen января",
      "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
      "march mars marz marzo marco maart marca марта",
      "april avril abril aprile kwietnia kwiecien апреля",
      "may mai mayo maggio maio mei maj maja мая",
      "june juin juni junio giugno junho czerwca czerwiec июня",
      "july juillet juli julio luglio julho lipca lipiec июля",
      "august aout agosto augustus augusti sierpnia sierpien августа",
      "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
      "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
      "november novembre noviembre novembro listopada listopad ноября",
      "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
    )

  val MONTHS: LinkedHashMap<String, Int> =
    LinkedHashMap<String, Int>().apply {
      MONTH_NAMES.forEachIndexed { index, names -> names.split(" ").forEach { put(it, index + 1) } }
      val abbreviations = LinkedHashMap<String, MutableSet<Int>>()
      for ((name, month) in this) abbreviations.getOrPut(name.take(3)) { LinkedHashSet() }.add(month)
      for ((name, months) in abbreviations) if (months.size == 1) put(name, months.first())
      put("janv", 1)
      put("fevr", 2)
      put("juil", 7)
      put("sept", 9)
    }

  private val WEEKDAYS =
    ("monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun " +
        "lundi mardi mercredi jeudi vendredi samedi dimanche montag dienstag mittwoch donnerstag freitag samstag " +
        "sonntag lunes martes miercoles jueves viernes sabado domingo lunedi martedi mercoledi giovedi venerdi " +
        "sabato domenica segunda terca quarta quinta sexta")
      .split(" ")
      .toSet()

  private val CJK_DATE = Pattern.compile("(\\d{4})\\s*[年년]\\s*(\\d{1,2})\\s*[月월]\\s*(\\d{1,2})\\s*[日일]", U)
  private val NUMERIC_DATE = Pattern.compile("(?<!\\d)(\\d{1,4})([/.-])(\\d{1,2})\\2(\\d{2,4})(?!\\d)", U)
  private val MONTH_NAME = MONTHS.keys.sortedByDescending { it.length }.joinToString("|")
  private val WORD_DATE =
    Pattern.compile(
      "\\b(?:\\d{1,2}\\w*[ .]+(?:de\\s+)?(?:$MONTH_NAME)[ ,.]+(?:de\\s+)?\\d{4}" +
        "|(?:$MONTH_NAME)[ .]+\\d{1,2}\\w*[ ,.]+\\d{4})\\b",
      U,
    )

  /** Currencies by the decimal mark their amounts are written with: "" for none, null for both in use. */
  private val CURRENCIES: List<Pair<String?, List<String>>> =
    listOf(
      "." to listOf("$", "£", "₹", "USD", "GBP", "CAD", "AUD", "NZD", "INR", "Rs.", "Rs", "MXN", "CNY", "RMB", "元", "HKD", "SGD"),
      "," to listOf("R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY"),
      "" to listOf("¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP"),
      null to listOf("€", "EUR", "CHF"),
    )
  private val NAMES = LinkedHashMap<String, String>().apply { CURRENCIES.forEach { (_, cs) -> cs.forEach { put(fold(it, false), it) } } }
  private val MARK = HashMap<String, String?>().apply { CURRENCIES.forEach { (mark, cs) -> cs.forEach { put(fold(it, false), mark) } } }
  private val CURRENCY: String =
    IGNORE_CASE_GROUP +
      NAMES.values.sortedByDescending { it.length }.joinToString("|") { c ->
        if (c[0].code < 128 && c[0].isLetter()) "\\b" + Pattern.quote(c) + "(?![a-z])" else Pattern.quote(c)
      } +
      ")"
  private val CURRENCY_PATTERN = Pattern.compile(CURRENCY, U)

  private val SCALES: Map<String, Long> =
    LinkedHashMap<String, Long>().apply {
      listOf("thousand", "mil", "tausend", "тыс", "千").forEach { put(it, 1_000L) }
      listOf("万", "萬").forEach { put(it, 10_000L) }
      listOf("lakh", "lakhs").forEach { put(it, 100_000L) }
      listOf("crore", "crores").forEach { put(it, 10_000_000L) }
      listOf(
          "million", "millions", "mio", "mln", "millón", "millones", "milhão", "milhões", "milione", "milioni", "млн",
        )
        .forEach { put(it, 1_000_000L) }
      listOf("亿", "億").forEach { put(it, 100_000_000L) }
      listOf(
          "billion", "billions", "milliard", "milliards", "milliarde", "milliarden", "mrd", "miliardo", "miliardi", "млрд",
        )
        .forEach { put(it, 1_000_000_000L) }
      put("trillion", 1_000_000_000_000L)
    }
  private val SCALE =
    Pattern.compile(
      "\\s*(" + SCALES.keys.map { Pattern.quote(it) }.sortedByDescending { it.length }.joinToString("|") + ")\\.?$",
      U or I,
    )
  private val SUFFIX = Pattern.compile("(?<=\\d)(k|m|bn|b)$", U or I)
  private val COUNTED =
    Pattern.compile(
      "\\b(?:money|price|invoice|revenue|cost|count|quantity|population|people|items|units|users|downloads)\\b",
      U or I,
    )
  private val GROUPED =
    Pattern.compile("(?<![\\d.,])\\d{1,3}(?:([,. \u00a0\u202f'’])\\d{3})+(?!\\1)([.,])\\d+(?![\\d.,])", U)
  private val SHORT = Pattern.compile("(?<![\\d.,])\\d+([.,])\\d{1,2}(?![\\d.,])", U)
  private val REPEATED = Pattern.compile("(?<![\\d.,])\\d{1,3}(?:([.,])\\d{3}){2,}(?![\\d.,])", U)
  private val LOCALE =
    Pattern.compile(
      "\\b(?:locale|language|lang)[\"']?\\s*[=:]\\s*[\"']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\\b",
      U,
    )
  private val DECLARATION = Pattern.compile("<\\?[^>]*\\?>", U)

  // a declared numeric date format, in English, French, German, Spanish or Italian
  private const val DAY = "(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)"
  private const val MONTH = "(?:mm?|month|mois|monat|mes|mese)"
  private const val YEAR = "(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)"
  private val DATE_FORMATS =
    listOf(
      Pattern.compile("\\b$DAY\\s*[/.-]\\s*$MONTH\\s*[/.-]\\s*$YEAR\\b", U) to "dmy",
      Pattern.compile("\\b$MONTH\\s*[/.-]\\s*$DAY\\s*[/.-]\\s*$YEAR\\b", U) to "mdy",
    )
  private val LOCALES =
    Pattern.compile(
      "\\b(?:locale|language|lang|langue|sprache|idioma|lingua)[\"']?\\s*[=:]\\s*[\"']?" +
        "(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\\b",
      U,
    )
  private val TIMED =
    Pattern.compile("(.+?)[T\\s]+(\\d{1,2}:\\d{2}(?::\\d{2}(?:\\.\\d+)?)?(?:Z|[+-]\\d{2}:\\d{2})?)", U)
  private val ORDINAL = Pattern.compile("(?<=\\d)(?:st|nd|rd|th|er)\\b", U)
  private val ORDINAL_MARK = Pattern.compile("(?<=\\d)\\.?(?:o|a|°)\\b", U)
  private val FILLER = Pattern.compile("\\b(?:de|of|the|le|el|il|den)\\b", U)
  private val PUNCTUATION = Pattern.compile("[,.]", U)
  private val WHITESPACE = Pattern.compile("\\s+", U)

  // ----------------------------------------------------------------------------------- numbers

  /** Only an explicitly written percent or currency marker; a currency sign stays a sign. */
  fun writtenUnit(text: String): String? {
    if (text.pyRstrip().endsWith("%")) return "%"
    val markers = findAll(CURRENCY_PATTERN, text)
    return if (markers.size == 1) NAMES[fold(markers[0], false)] else null
  }

  private fun currencyMarks(text: String): Set<String> =
    findAll(CURRENCY_PATTERN, text).map { MARK[fold(it, false)] }.filterNotNull().toSet()

  /** The decimal mark ("" when amounts have none) from the first evidence that speaks, else null. */
  private fun decimalMark(written: String, contextIn: String): String? {
    val context = DECLARATION.matcher(contextIn).replaceAll("")
    val numbers = LinkedHashSet<String>()
    GROUPED.matcher(context).let { m -> while (m.find()) numbers.add(m.group(2)!!) }
    SHORT.matcher(context).let { m -> while (m.find()) numbers.add(m.group(1)!!) }
    REPEATED.matcher(context).let { m -> while (m.find()) numbers.add(if (m.group(1) == ".") "," else ".") }
    val locales = LinkedHashSet<String>()
    LOCALE.matcher(fold(context, false)).let { m ->
      while (m.find()) locales.add(if (m.group(1)!! in setOf("en", "ja", "zh", "ko")) "." else ",")
    }
    val evidence =
      listOf(currencyMarks(written), numbers, currencyMarks(context), locales).firstOrNull { it.isNotEmpty() }
        ?: emptySet()
    return one(evidence)
  }

  /**
   * The written magnitude, percent points included: its currency and percent sign set apart, its
   * scale word applied; no implicit unit and no conversion. Null when the writing does not settle
   * one value. Whole values come back as [Long] (or [java.math.BigInteger]), others as [Double].
   */
  fun parseNumber(textIn: String, contextIn: String = "", unit: String? = null): Number? {
    var context = contextIn
    if (!unit.isNullOrEmpty() && CURRENCY_PATTERN.matcher(unit.pyStrip()).matches()) context += "\n$unit"
    val written = textIn
    var text =
      textIn.pyStrip().replace("\u2212", "-").replace("\u00a0", " ").replace("\u202f", " ").removeSuffix("%").pyRstrip()
    val negative = text.startsWith("(") && text.endsWith(")")
    if (negative) text = text.substring(1, text.length - 1).pyStrip()
    var sign: String? = null
    var currency: String? = null
    val split = Pattern.compile("($CURRENCY)\\s*(.*)|(.*?)\\s*($CURRENCY)", U)
    repeat(2) {
      if (currency == null) {
        val m = split.matcher(text)
        if (m.matches()) {
          currency = m.group(1) ?: m.group(4)
          text = (if (m.group(1) != null) m.group(2)!! else m.group(3)!!).pyStrip()
        }
      }
      if (sign == null && (text.startsWith("-") || text.startsWith("+"))) {
        sign = text.substring(0, 1)
        text = text.substring(1).pyStrip()
      }
    }
    if (negative && sign != null) return null
    var scale = 1L
    val word = SCALE.matcher(text)
    if (word.find()) {
      scale = if (word.group(1) != "Billion") SCALES.getValue(word.group(1)!!.lowercase()) else 0L
      text = text.substring(0, word.start())
    } else {
      val suffix = SUFFIX.matcher(text)
      if (suffix.find()) {
        if (!(CURRENCY_PATTERN.matcher(written + "\n" + context).find() || COUNTED.matcher(context).find())) return null
        scale =
          when (suffix.group(1)!!.lowercase()) {
            "k" -> 1_000L
            "m" -> 1_000_000L
            else -> 1_000_000_000L
          }
        text = text.substring(0, suffix.start())
      }
    }
    val plain = digits(text, written, context) ?: return null
    if (scale == 0L) return null
    val value = BigDecimal((if (negative) "-" else sign ?: "") + asciiDigits(plain)).multiply(BigDecimal.valueOf(scale))
    val stripped = value.stripTrailingZeros()
    if (stripped.scale() <= 0) {
      val whole = stripped.toBigIntegerExact()
      return if (whole.bitLength() < 64) whole.toLong() else whole
    }
    val result = value.toPlainString().toDouble()
    return if (result.isFinite()) result else null
  }

  /** The number as plain digits and a point: the decimal mark from the evidence, the other mark a complete grouping. */
  private fun digits(text: String, written: String, context: String): String? {
    if (!Pattern.compile("\\d[\\d., '’]*", U).matcher(text).matches()) return null
    if (Pattern.compile("[ '’]", U).matcher(text).find()) {
      val grouping = Pattern.compile("(\\d{1,3}(?:[ '’]\\d{3})+)([.,]\\d+)?", U).matcher(text)
      if (!grouping.matches()) return null
      return digits(grouping.group(1)!!.replace(Regex("[ '’]"), "") + (grouping.group(2) ?: ""), written, context)
    }
    if (Pattern.compile("\\d{1,2}(?:,\\d{2})+,\\d{3}(?:\\.\\d+)?", U).matcher(text).matches() &&
      Pattern.compile("₹|\\bINR\\b|\\bRs\\b|lakh|crore", U or I).matcher(written + "\n" + context).find()
    ) {
      return text.replace(",", "")
    }
    if ('.' in text && ',' in text) {
      val (point, group) = if (text.lastIndexOf('.') > text.lastIndexOf(',')) "." to "," else "," to "."
      val shape = "\\d{1,3}(?:${Pattern.quote(group)}\\d{3})+${Pattern.quote(point)}\\d+"
      if (!Pattern.compile(shape, U).matcher(text).matches()) return null
      return text.replace(group, "").replace(point, ".")
    }
    if ('.' !in text && ',' !in text) return text
    val mark = if ('.' in text) "." else ","
    val parts = text.split(mark)
    if (parts.any { it.isEmpty() }) return null
    val point = decimalMark(written, context)
    if (parts.size > 2 || (point != null && point != mark)) {
      return if (parts[0].length in 1..3 && parts.drop(1).all { it.length == 3 }) parts.joinToString("") else null
    }
    if (point == null && parts[1].length == 3 && parts[0].length <= 3) return null
    return parts.joinToString(".")
  }

  // ----------------------------------------------------------------------------------- dates

  /** Declared format or locale first, then the state's other unambiguous numeric dates; conflicts stay unresolved. */
  fun dateOrder(context: String, foldedIn: String? = null): String? {
    val folded = foldedIn ?: fold(context)
    val explicit = LinkedHashSet<String>()
    for ((pattern, order) in DATE_FORMATS) if (pattern.matcher(folded).find()) explicit.add(order)
    if (explicit.isNotEmpty()) return one(explicit)
    val locales = findAllGroup(LOCALES, folded, 1)
    if (locales.isNotEmpty()) {
      return one(locales.map { if (it.replace("_", "-") in setOf("en", "en-us")) "mdy" else "dmy" }.toSet())
    }
    val orders = LinkedHashSet<String>()
    val m = NUMERIC_DATE.matcher(context)
    while (m.find()) {
      val a = m.group(1)!!.toInt()
      val b = m.group(3)!!.toInt()
      val year = m.group(4)!!.toInt()
      if (m.group(4)!!.length != 4 || a > 31 || b > 31 || a < 1 || b < 1) continue
      for ((day, month, order) in listOf(Triple(a, b, "dmy"), Triple(b, a, "mdy"))) {
        if (date(year, month, day) == null) continue
        if ((a > 12) != (b > 12)) orders.add(order)
      }
    }
    return one(orders)
  }

  /** The ISO value and whether an ambiguity was left unresolved; an invalid or unrecognised span is (null, false). */
  fun parseDate(text: String, context: String = ""): Pair<String?, Boolean> {
    var folded: String? = null
    val foldedOf = { folded ?: fold(context).also { folded = it } }
    var orderComputed = false
    var order: String? = null
    val orderOf = {
      if (!orderComputed) {
        order = dateOrder(context, foldedOf())
        orderComputed = true
      }
      order
    }
    return parseDateWith(text, context, foldedOf, orderOf)
  }

  private fun parseDateWith(
    textIn: String,
    context: String,
    folded: () -> String,
    order: () -> String?,
  ): Pair<String?, Boolean> {
    val text = textIn.pyStrip()
    if (text.isEmpty()) return null to false
    val timed = TIMED.matcher(text)
    if (timed.matches()) {
      val (day, ambiguous) = parseDateWith(timed.group(1)!!, context, folded, order)
      if (day == null) return null to ambiguous
      var clock = timed.group(2)!!
      if (clock[1] == ':') clock = "0$clock"
      return (isoDateTime(day, clock) to false)
    }
    val numeric = NUMERIC_DATE.matcher(text)
    if (numeric.matches()) {
      val a = numeric.group(1)!!.toInt()
      val b = numeric.group(3)!!.toInt()
      val c = numeric.group(4)!!.toInt()
      val candidates: List<Triple<Int, Int, Int>>
      if (numeric.group(1)!!.length == 4) {
        candidates = listOf(Triple(a, b, c))
      } else if (numeric.group(4)!!.length == 4) {
        candidates = orderCandidates(c, a, b, order())
      } else {
        // A century is never invented: another unambiguous complete date must establish it.
        if (numeric.group(4)!!.length != 2) return null to false
        val centuries = LinkedHashSet<Int>()
        val complete = ArrayList<String>()
        val inContext = NUMERIC_DATE.matcher(context)
        while (inContext.find()) {
          if (inContext.group(1)!!.length == 4 || inContext.group(4)!!.length == 4) complete.add(inContext.group())
        }
        val words = WORD_DATE.matcher(folded())
        while (words.find()) complete.add(words.group())
        for (candidate in complete) {
          val (value, ambiguous) = parseDateWith(candidate, context, folded, order)
          if (value != null && !ambiguous) centuries.add(value.substring(0, 4).toInt() / 100 * 100)
        }
        if (centuries.size != 1) return null to false
        candidates = orderCandidates(centuries.first() + c, a, b, order())
      }
      val values = LinkedHashSet<String>()
      for ((year, month, day) in candidates) date(year, month, day)?.let { values.add(it) }
      if (values.size > 1) return null to true
      return values.firstOrNull() to false
    }
    val cjk = CJK_DATE.matcher(text)
    if (cjk.matches()) return date(cjk.group(1)!!.toInt(), cjk.group(2)!!.toInt(), cjk.group(3)!!.toInt()) to false
    var words = fold(text)
    words = ORDINAL.matcher(words).replaceAll("")
    words = ORDINAL_MARK.matcher(words).replaceAll("")
    words = FILLER.matcher(words).replaceAll(" ")
    var parts = WHITESPACE.split(PUNCTUATION.matcher(words).replaceAll(" ").pyStrip()).filter { it.isNotEmpty() }
    if (parts.isNotEmpty() && parts[0].removeSuffix("-feira") in WEEKDAYS) parts = parts.drop(1)
    if (parts.size != 3) return null to false
    val month: Int
    val day: String
    val year: String
    if (parts[0] in MONTHS) {
      month = MONTHS.getValue(parts[0])
      day = parts[1]
      year = parts[2]
    } else if (parts[1] in MONTHS) {
      day = parts[0]
      month = MONTHS.getValue(parts[1])
      year = parts[2]
    } else {
      return null to false
    }
    if (!pyIsDigit(day) || !pyIsDigit(year) || year.length != 4) return null to false
    return date(asciiDigits(year).toInt(), month, asciiDigits(day).toIntOrNull() ?: return null to false) to false
  }

  private fun orderCandidates(year: Int, a: Int, b: Int, order: String?): List<Triple<Int, Int, Int>> {
    val dmy = Triple(year, b, a)
    val mdy = Triple(year, a, b)
    return if (order == null) listOf(dmy, mdy) else listOf(if (order == "dmy") dmy else mdy)
  }

  /** `datetime.date(year, month, day).isoformat()`, or null for a date Python refuses. */
  private fun date(year: Int, month: Int, day: Int): String? {
    if (year !in 1..9999) return null
    return try {
      LocalDate.of(year, month, day).let { String.format("%04d-%02d-%02d", it.year, it.monthValue, it.dayOfMonth) }
    } catch (_: Exception) {
      null
    }
  }

  /** `datetime.fromisoformat(day + "T" + clock).isoformat()` for the clocks [TIMED] accepts, else null. */
  private fun isoDateTime(day: String, clock: String): String? {
    val m =
      Pattern.compile("(\\d{2}):(\\d{2})(?::(\\d{2})(?:\\.(\\d+))?)?(Z|[+-]\\d{2}:\\d{2})?").matcher(asciiDigits(clock))
    if (!m.matches()) return null
    return try {
      val hour = m.group(1)!!.toInt()
      val minute = m.group(2)!!.toInt()
      val second = m.group(3)?.toInt() ?: 0
      val fraction = m.group(4)?.take(6)?.padEnd(6, '0')?.toInt() ?: 0
      val base = LocalDate.parse(day)
      val stamp =
        if (hour == 24 && minute == 0 && second == 0 && fraction == 0) {
          LocalDateTime.of(base.plusDays(1), LocalTime.MIDNIGHT)
        } else {
          LocalDateTime.of(base, LocalTime.of(hour, minute, second, fraction * 1000))
        }
      if (stamp.year !in 1..9999) return null
      val text =
        stamp.format(DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss")) +
          (if (stamp.nano != 0) "." + String.format("%06d", stamp.nano / 1000) else "")
      val zone = m.group(5)
      when {
        zone == null -> text
        zone == "Z" -> "$text+00:00"
        else -> {
          val zh = zone.substring(1, 3).toInt()
          val zm = zone.substring(4, 6).toInt()
          if (zh > 23 || zm > 59) null else text + zone
        }
      }
    } catch (_: Exception) {
      null
    }
  }

  // ----------------------------------------------------------------------------------- helpers

  /** Python's `casefold()` (with NFKD and combining marks dropped when [strip]) as the provider folds text. */
  fun fold(text: String, strip: Boolean = true): String {
    val lower = text.lowercase().replace("ß", "ss").replace("ẞ", "ss").replace("ς", "σ")
    if (!strip) return lower
    val decomposed = Normalizer.normalize(lower, Normalizer.Form.NFKD)
    val out = StringBuilder(decomposed.length)
    for (c in decomposed) if (Character.getType(c) != Character.NON_SPACING_MARK.toInt()) out.append(c)
    return out.toString()
  }

  private fun <T> one(values: Set<T>): T? = if (values.size == 1) values.first() else null

  private fun findAll(pattern: Pattern, text: String): List<String> {
    val out = ArrayList<String>()
    val m: Matcher = pattern.matcher(text)
    while (m.find()) out.add(m.group())
    return out
  }

  private fun findAllGroup(pattern: Pattern, text: String, group: Int): List<String> {
    val out = ArrayList<String>()
    val m = pattern.matcher(text)
    while (m.find()) out.add(m.group(group) ?: "")
    return out
  }

  /** Every decimal digit as its ASCII form (Python's `int` and `Decimal` read any Unicode digit). */
  private fun asciiDigits(text: String): String {
    val out = StringBuilder(text.length)
    for (c in text) out.append(if (Character.isDigit(c)) ('0' + Character.digit(c, 10)) else c)
    return out.toString()
  }

  private fun pyIsDigit(text: String): Boolean = text.isNotEmpty() && text.all { Character.isDigit(it) }
}

/** Python's `str.strip()`: Unicode whitespace from both ends. */
fun String.pyStrip(): String = pyLstrip().pyRstrip()

fun String.pyLstrip(): String {
  var start = 0
  while (start < length && pyIsSpace(this[start])) start++
  return substring(start)
}

fun String.pyRstrip(): String {
  var end = length
  while (end > 0 && pyIsSpace(this[end - 1])) end--
  return substring(0, end)
}

/** Python's `str.isspace()` for one character. */
fun pyIsSpace(c: Char): Boolean =
  c in "\t\n\u000b\u000c\r\u001c\u001d\u001e\u001f \u0085\u00a0\u1680\u2028\u2029\u202f\u205f\u3000" ||
    c in '\u2000'..'\u200a'
