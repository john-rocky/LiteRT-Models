package com.ming3tts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Unicode character classes exactly as the Python reference sees them.
 *
 * `char_classes.bin` (written by scripts/make_app_assets.py) holds runs of class bits over every
 * code point, taken from Python's `regex` module (`\p{L}`, `\p{N}`, `\s` of the Qwen
 * pre-tokenizer) and from Python's `str` / `re` (`isspace`, `isalnum`, `\d` of the upstream text
 * frontend). Java and Android ICU disagree with Python on a few of these (`\s` vs U+001C..U+001F,
 * NEL, non-breaking spaces; newer Unicode versions), so the device uses this table instead of
 * its own regex classes and tokenizes the same as the reference on every Android version.
 */
class CharClasses private constructor(private val starts: IntArray, private val bits: IntArray) {

    companion object {
        const val L = 1
        const val N = 2
        const val WS = 4
        const val PY_SPACE = 8
        const val PY_ALNUM = 16
        const val PY_DIGIT = 32

        fun load(file: File): CharClasses = parse(file.readBytes())

        fun parse(bytes: ByteArray): CharClasses {
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val n = bb.int
            val starts = IntArray(n)
            val bits = IntArray(n)
            for (i in 0 until n) {
                starts[i] = bb.int
                bits[i] = bb.int
            }
            require(n > 0 && starts[0] == 0) { "char_classes.bin: bad header" }
            return CharClasses(starts, bits)
        }
    }

    fun bitsOf(cp: Int): Int {
        var lo = 0
        var hi = starts.size - 1
        while (lo < hi) {
            val mid = (lo + hi + 1) ushr 1
            if (starts[mid] <= cp) lo = mid else hi = mid - 1
        }
        return bits[lo]
    }

    fun isLetter(cp: Int) = bitsOf(cp) and L != 0
    fun isNumber(cp: Int) = bitsOf(cp) and N != 0
    fun isRegexSpace(cp: Int) = bitsOf(cp) and WS != 0
    fun isPySpace(cp: Int) = bitsOf(cp) and PY_SPACE != 0
    fun isPyAlnum(cp: Int) = bitsOf(cp) and PY_ALNUM != 0
    fun isPyDigit(cp: Int) = bitsOf(cp) and PY_DIGIT != 0

    /** Python `str.strip()` (no argument). */
    fun pyStrip(s: String): String = pyRstrip(pyLstrip(s))

    fun pyLstrip(s: String): String {
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            if (!isPySpace(cp)) break
            i += Character.charCount(cp)
        }
        return s.substring(i)
    }

    fun pyRstrip(s: String): String {
        var j = s.length
        while (j > 0) {
            val cp = s.codePointBefore(j)
            if (!isPySpace(cp)) break
            j -= Character.charCount(cp)
        }
        return s.substring(0, j)
    }
}
