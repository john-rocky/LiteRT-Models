package com.kokoro

/**
 * Lightweight English text normalization: turns numbers, currency and common symbols
 * into spoken words BEFORE phonemization, so they don't get dropped as OOV.
 *
 *   "I have 42 cats"   -> "I have forty two cats"
 *   "$5.99"            -> "five dollars and ninety nine cents"
 *   "covid19"          -> "covid nineteen"
 *   "100%"             -> "one hundred percent"
 *   "A & B"            -> "A and B"
 *
 * This is intentionally simple (no NLP); it covers the cases that actually break
 * free-text input. Anything it leaves unexpanded falls through to the CMU dictionary
 * and then the neural OOV fallback.
 */
object EnglishTextNormalizer {

    fun normalize(text: String): String {
        // Fold accents to ASCII so accented names keep all their letters (José -> Jose),
        // instead of the diacritic chars being dropped downstream.
        var t = asciiFold(text)
        // Split letter/digit boundaries: "covid19" -> "covid 19", "3rd" handled below.
        t = Regex("(?<=[A-Za-z])(?=\\d)|(?<=\\d)(?=[A-Za-z])").replace(t, " ")
        // Currency: $<number> -> "<words> dollars [and <cc> cents]"
        t = Regex("\\$\\s?(\\d+(?:,\\d{3})*)(?:\\.(\\d{1,2}))?").replace(t) { m ->
            val dollars = cardinal(m.groupValues[1].replace(",", "").toLong())
            val cents = m.groupValues[2]
            if (cents.isNotEmpty()) {
                val c = cents.padEnd(2, '0').toLong()
                "$dollars dollars and ${cardinal(c)} cents"
            } else "$dollars dollars"
        }
        // Percent: <number>% -> "<words> percent"
        t = Regex("(\\d+(?:\\.\\d+)?)\\s?%").replace(t) { m -> "${spokenNumber(m.groupValues[1])} percent" }
        // Remaining standalone numbers (integers, optional thousands separators, decimals).
        t = Regex("\\d+(?:,\\d{3})*(?:\\.\\d+)?").replace(t) { m -> spokenNumber(m.value) }
        // Common symbols.
        for ((sym, word) in SYMBOLS) t = t.replace(sym, word)
        // Collapse whitespace.
        return t.replace(Regex("\\s+"), " ").trim()
    }

    /** Strip diacritics: NFD-decompose then drop combining marks (é -> e, ñ -> n, ü -> u). */
    private fun asciiFold(s: String): String =
        java.text.Normalizer.normalize(s, java.text.Normalizer.Form.NFD)
            .replace(Regex("\\p{Mn}+"), "")

    /** A number token (possibly with a decimal part) -> words. */
    private fun spokenNumber(s: String): String {
        val clean = s.replace(",", "")
        val dot = clean.indexOf('.')
        if (dot < 0) return cardinal(clean.toLong())
        val intPart = clean.substring(0, dot).ifEmpty { "0" }.toLong()
        val frac = clean.substring(dot + 1)
        val fracWords = frac.map { ONES[it - '0'] }.joinToString(" ")
        return "${cardinal(intPart)} point $fracWords"
    }

    /** Cardinal words for 0..999,999,999,999. */
    private fun cardinal(n: Long): String {
        if (n == 0L) return "zero"
        if (n < 0) return "minus ${cardinal(-n)}"
        val sb = StringBuilder()
        var rem = n
        for ((value, name) in SCALES) {
            if (rem >= value) {
                sb.append(threeDigits((rem / value).toInt())).append(' ').append(name).append(' ')
                rem %= value
            }
        }
        if (rem > 0) sb.append(threeDigits(rem.toInt()))
        return sb.toString().trim().replace(Regex("\\s+"), " ")
    }

    /** Words for 0..999 (no leading "zero"). */
    private fun threeDigits(n: Int): String {
        val sb = StringBuilder()
        val h = n / 100
        val t = n % 100
        if (h > 0) { sb.append(ONES[h]).append(" hundred"); if (t > 0) sb.append(' ') }
        if (t in 1..19) sb.append(ONES[t])
        else if (t >= 20) {
            sb.append(TENS[t / 10])
            if (t % 10 != 0) sb.append(' ').append(ONES[t % 10])
        }
        return sb.toString()
    }

    private val ONES = arrayOf(
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    )
    private val TENS = arrayOf("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    // Largest first so the greedy loop in cardinal() consumes correctly.
    private val SCALES = listOf(
        1_000_000_000L to "billion",
        1_000_000L to "million",
        1_000L to "thousand",
    )
    private val SYMBOLS = listOf(
        "&" to " and ", "%" to " percent ", "@" to " at ", "#" to " number ",
        "+" to " plus ", "=" to " equals ", "°" to " degrees ",
    )
}
