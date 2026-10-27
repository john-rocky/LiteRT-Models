package com.ming3tts

import java.io.File
import java.text.Normalizer

/**
 * The Qwen2 byte-level BPE of Ming-3.0-tts (`tokenizer.json`), ported from the host spec
 * scripts/host_mirror.py (`QwenBPE`):
 *  1. added tokens (`<|im_start|>`, `<audio>`, ...) are split out first, leftmost-longest;
 *  2. every other piece is NFC-normalized and split by the pre-tokenizer regex
 *     `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|
 *      \s*[\r\n]+|\s+(?!\S)|\s+` — matched by hand over code points with [CharClasses], because
 *     java.util.regex (and Android's ICU engine) classify some characters differently;
 *  3. each split piece is UTF-8 encoded, mapped byte -> printable char (GPT-2 table) and merged
 *     by BPE rank (lowest rank first, leftmost on ties).
 *
 * Files (scripts/make_app_assets.py): tokenizer_vocab.txt (token per line in id order),
 * tokenizer_merges.txt ("a b" per line in rank order), tokenizer_added.txt ("id<TAB>token").
 */
class QwenTokenizer(
    vocabFile: File,
    mergesFile: File,
    addedFile: File,
    private val cc: CharClasses,
) {
    private val vocab = HashMap<String, Int>(200_000)

    // (left id << 32 | right id) -> (rank << 32 | merged id)
    private val merges = HashMap<Long, Long>(200_000)
    private val added = ArrayList<Pair<String, Int>>()
    private val byteChar = IntArray(256)
    private val singleByteId = IntArray(256)
    private val cache = HashMap<String, IntArray>()

    init {
        vocabFile.bufferedReader(Charsets.UTF_8).useLines { lines ->
            var id = 0
            for (line in lines) {
                if (line.isEmpty()) continue
                vocab[line] = id++
            }
        }
        mergesFile.bufferedReader(Charsets.UTF_8).useLines { lines ->
            var rank = 0
            for (line in lines) {
                if (line.isEmpty()) continue
                val sp = line.indexOf(' ')
                val a = line.substring(0, sp)
                val b = line.substring(sp + 1)
                val ia = requireNotNull(vocab[a]) { "merge left not in vocab: $a" }
                val ib = requireNotNull(vocab[b]) { "merge right not in vocab: $b" }
                val im = requireNotNull(vocab[a + b]) { "merge result not in vocab: $a$b" }
                val key = (ia.toLong() shl 32) or (ib.toLong() and 0xffffffffL)
                // A repeated pair keeps its last rank, as the host_mirror dict does.
                merges[key] = (rank.toLong() shl 32) or (im.toLong() and 0xffffffffL)
                rank++
            }
        }
        addedFile.bufferedReader(Charsets.UTF_8).useLines { lines ->
            for (line in lines) {
                if (line.isEmpty()) continue
                val tab = line.indexOf('\t')
                added.add(line.substring(tab + 1) to line.substring(0, tab).toInt())
            }
        }
        added.sortByDescending { it.first.length }
        // GPT-2 bytes_to_unicode
        val bs = ArrayList<Int>()
        for (b in '!'.code..'~'.code) bs.add(b)
        for (b in 0xA1..0xAC) bs.add(b)
        for (b in 0xAE..0xFF) bs.add(b)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0 until 256) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        for (i in bs.indices) byteChar[bs[i]] = cs[i]
        for (b in 0 until 256) {
            singleByteId[b] = requireNotNull(vocab[String(Character.toChars(byteChar[b]))]) {
                "byte $b has no vocab entry"
            }
        }
    }

    /** Id of a vocab or added token, e.g. `<|vision_start|>`. */
    fun tokenId(token: String): Int =
        added.firstOrNull { it.first == token }?.second
            ?: requireNotNull(vocab[token]) { "unknown token $token" }

    fun encode(text: String): IntArray {
        val out = IntList()
        var pos = 0
        var i = 0
        while (i < text.length) {
            val hit = added.firstOrNull { text.startsWith(it.first, i) }
            if (hit == null) {
                i++
                continue
            }
            if (i > pos) encodePlain(text.substring(pos, i), out)
            out.add(hit.second)
            i += hit.first.length
            pos = i
        }
        if (pos < text.length) encodePlain(text.substring(pos), out)
        return out.toArray()
    }

    private fun encodePlain(raw: String, out: IntList) {
        val text = Normalizer.normalize(raw, Normalizer.Form.NFC)
        val cps = text.codePoints().toArray()
        var p = 0
        while (p < cps.size) {
            val e = matchAt(cps, p)
            val piece = String(cps, p, e - p)
            out.addAll(bpe(piece))
            p = e
        }
    }

    private fun isCr(c: Int) = c == 0x0D || c == 0x0A

    private fun isOther(c: Int) = !cc.isRegexSpace(c) && !cc.isLetter(c) && !cc.isNumber(c)

    private fun ci(c: Int, lower: Char): Boolean =
        c == lower.code || c == lower.uppercaseChar().code || (lower == 's' && c == 0x017F)

    /** End (exclusive) of the pre-tokenizer match at [p]; the alternatives in regex order. */
    internal fun matchAt(cps: IntArray, p: Int): Int {
        val n = cps.size
        val c = cps[p]
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\''.code && p + 1 < n) {
            val c1 = cps[p + 1]
            val c2 = if (p + 2 < n) cps[p + 2] else -1
            when {
                ci(c1, 's') || ci(c1, 't') || ci(c1, 'm') || ci(c1, 'd') -> return p + 2
                ci(c1, 'r') && ci(c2, 'e') -> return p + 3
                ci(c1, 'v') && ci(c2, 'e') -> return p + 3
                ci(c1, 'l') && ci(c2, 'l') -> return p + 3
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!isCr(c) && !cc.isLetter(c) && !cc.isNumber(c) && p + 1 < n && cc.isLetter(cps[p + 1])) {
            var e = p + 2
            while (e < n && cc.isLetter(cps[e])) e++
            return e
        }
        if (cc.isLetter(c)) {
            var e = p + 1
            while (e < n && cc.isLetter(cps[e])) e++
            return e
        }
        // \p{N}
        if (cc.isNumber(c)) return p + 1
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        val otherStart = when {
            c == ' '.code && p + 1 < n && isOther(cps[p + 1]) -> p + 1
            isOther(c) -> p
            else -> -1
        }
        if (otherStart >= 0) {
            var e = otherStart + 1
            while (e < n && isOther(cps[e])) e++
            while (e < n && isCr(cps[e])) e++
            return e
        }
        // c is whitespace from here on: the run [p, q)
        var q = p
        while (q < n && cc.isRegexSpace(cps[q])) q++
        check(q > p) { "unclassified code point ${Integer.toHexString(c)}" }
        // \s*[\r\n]+ : up to the last CR/LF of the run
        for (k in q - 1 downTo p) {
            if (isCr(cps[k])) return k + 1
        }
        // \s+(?!\S)
        if (q == n) return q
        if (q - p >= 2) return q - 1
        // \s+
        return q
    }

    private fun bpe(piece: String): IntArray {
        cache[piece]?.let { return it }
        val bytes = piece.toByteArray(Charsets.UTF_8)
        var ids = IntArray(bytes.size) { singleByteId[bytes[it].toInt() and 0xFF] }
        var len = ids.size
        while (len > 1) {
            var bestRank = Long.MAX_VALUE
            var bestIdx = -1
            var bestMerged = 0
            for (i in 0 until len - 1) {
                val v = merges[(ids[i].toLong() shl 32) or (ids[i + 1].toLong() and 0xffffffffL)] ?: continue
                val rank = v ushr 32
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                    bestMerged = (v and 0xffffffffL).toInt()
                }
            }
            if (bestIdx < 0) break
            ids[bestIdx] = bestMerged
            System.arraycopy(ids, bestIdx + 2, ids, bestIdx + 1, len - bestIdx - 2)
            len--
        }
        ids = ids.copyOf(len)
        if (cache.size > 50_000) cache.clear()
        cache[piece] = ids
        return ids
    }
}

/** Growable int list without boxing. */
class IntList(capacity: Int = 64) {
    private var a = IntArray(capacity)
    var size = 0
        private set

    fun add(v: Int) {
        if (size == a.size) a = a.copyOf(a.size * 2)
        a[size++] = v
    }

    fun addAll(v: IntArray) {
        for (x in v) add(x)
    }

    operator fun get(i: Int) = a[i]

    fun toArray(): IntArray = a.copyOf(size)
}
