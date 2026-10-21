package com.iris3b

import java.io.File
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * Byte-level BPE tokenizer of the Qwen3-VL-4B-Instruct text encoder (its `tokenizer.json`).
 *
 * A port of the Hugging Face fast tokenizer for this checkpoint, in the same order of stages:
 *  1. **Added tokens** (26, all `normalized: false`) are split out of the raw text first, leftmost
 *     then longest, so a typed `<|im_end|>` becomes one id exactly as in Python.
 *  2. **NFC** on the text between them (`"normalizer": {"type": "NFC"}`).
 *  3. **Pre-tokenize** with the checkpoint's Split regex. It is spelled with explicit classes so the
 *     desktop JVM, Android's ICU regex and the Rust tokenizer (Oniguruma) agree: `\s` is written out
 *     as the Unicode White_Space set (the JVM's `\s` is ASCII-only, ICU's and Oniguruma's are
 *     Unicode), and the case-insensitive contractions are written out (`(?i)` is ASCII-only on the
 *     JVM; Oniguruma folds `'ſ` to `'s`).
 *  4. **Byte-level encode** (GPT-2 `bytes_to_unicode`) and **merge** adjacent pairs by rank.
 *
 * Checked against the Hugging Face tokenizer on 15 test captions and 2,000 random strings during
 * the conversion.
 */
class QwenTokenizer(assetsDir: File) {

    private val tokens: Array<String>
    private val tokenToId = HashMap<String, Int>(VOCAB_HINT)
    private val mergeRank = HashMap<Long, Int>(MERGE_HINT)
    private val addedByFirstChar: Map<Char, List<Pair<String, Int>>>
    private val byteToUnicode: IntArray = buildByteEncoder()

    init {
        val list = ArrayList<String>(VOCAB_HINT)
        File(assetsDir, "qwen_vocab.txt").forEachLine { list.add(it) }
        tokens = list.toTypedArray()
        tokens.forEachIndexed { id, token -> tokenToId[token] = id }
        var rank = 0
        File(assetsDir, "qwen_merges.txt").forEachLine { line ->
            val space = line.indexOf(' ')
            val left = tokenToId[line.substring(0, space)]
            val right = tokenToId[line.substring(space + 1)]
            if (left != null && right != null) {
                mergeRank[pairKey(left, right)] = rank
            }
            rank++
        }
        val added = ArrayList<Pair<String, Int>>()
        File(assetsDir, "qwen_added.txt").forEachLine { line ->
            val tab = line.lastIndexOf('\t')
            added.add(line.substring(0, tab) to line.substring(tab + 1).toInt())
        }
        addedByFirstChar = added.sortedByDescending { it.first.length }.groupBy { it.first[0] }
    }

    /** Tokenizes raw text with no template and no special tokens added (`add_special_tokens=False`). */
    fun encode(text: String): List<Int> {
        val ids = ArrayList<Int>(text.length / 2 + 1)
        var segmentStart = 0
        var i = 0
        while (i < text.length) {
            val hit = addedTokenAt(text, i)
            if (hit == null) {
                i++
                continue
            }
            if (segmentStart < i) {
                encodeSegment(text.substring(segmentStart, i), ids)
            }
            ids.add(hit.second)
            i += hit.first.length
            segmentStart = i
        }
        if (segmentStart < text.length) {
            encodeSegment(text.substring(segmentStart), ids)
        }
        return ids
    }

    /** The longest added token that starts at [index], or null. */
    private fun addedTokenAt(text: String, index: Int): Pair<String, Int>? {
        val candidates = addedByFirstChar[text[index]] ?: return null
        return candidates.firstOrNull { text.startsWith(it.first, index) }
    }

    private fun encodeSegment(segment: String, out: MutableList<Int>) {
        val normalized = Normalizer.normalize(segment, Normalizer.Form.NFC)
        val matcher = PRE_TOKENIZER.matcher(normalized)
        var position = 0
        while (position < normalized.length) {
            if (!matcher.find(position)) {
                out.addAll(merge(normalized.substring(position)))
                break
            }
            if (matcher.start() > position) {
                out.addAll(merge(normalized.substring(position, matcher.start())))
            }
            out.addAll(merge(matcher.group()))
            position = matcher.end()
        }
    }

    /** Byte-level encodes one piece and repeatedly merges the lowest-ranked adjacent pair. */
    private fun merge(piece: String): List<Int> {
        val symbols = ArrayList<Int>(piece.length * 2)
        for (byte in piece.toByteArray(Charsets.UTF_8)) {
            val symbol = String(Character.toChars(byteToUnicode[byte.toInt() and BYTE_MASK]))
            symbols.add(tokenToId.getValue(symbol))
        }
        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIndex = -1
            for (i in 0 until symbols.size - 1) {
                val rank = mergeRank[pairKey(symbols[i], symbols[i + 1])] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if (bestIndex < 0) {
                break
            }
            symbols[bestIndex] = tokenToId.getValue(tokens[symbols[bestIndex]] + tokens[symbols[bestIndex + 1]])
            symbols.removeAt(bestIndex + 1)
        }
        return symbols
    }

    private companion object {
        /** The Unicode White_Space set = Oniguruma's Unicode `\s` (written out for the JVM). */
        const val WS = """\t\n\u000B\f\r \u0085   -     　"""

        /** tokenizer.json's Split pattern with `\s`, `\S` and `(?i:...)` written out. */
        val PRE_TOKENIZER: Pattern = Pattern.compile(
            """'(?:[sSſ]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])""" +
                """|[^\r\n\p{L}\p{N}]?\p{L}+""" +
                """|\p{N}""" +
                """| ?[^$WS\p{L}\p{N}]+[\r\n]*""" +
                """|[$WS]*[\r\n]+""" +
                """|[$WS]+(?![^$WS])""" +
                """|[$WS]+""")

        const val VOCAB_HINT = 1 shl 18
        const val MERGE_HINT = 1 shl 18
        const val BYTE_MASK = 0xff

        fun pairKey(left: Int, right: Int): Long = (left.toLong() shl 32) or right.toLong()

        /** GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest from U+0100 on. */
        fun buildByteEncoder(): IntArray {
            val printable = ('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)
            val encoder = IntArray(256)
            var next = 0
            for (byte in 0 until 256) {
                encoder[byte] = if (byte in printable) byte else 256 + next++
            }
            return encoder
        }
    }
}
