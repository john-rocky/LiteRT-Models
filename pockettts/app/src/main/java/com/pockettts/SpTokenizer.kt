package com.pockettts

import java.io.File

/**
 * SentencePiece **unigram** encoder/decoder for the Pocket TTS 4000-piece
 * tokenizer, ported for on-device use (no JNI).
 *
 * The tokenizer model uses the identity normalizer (no NFKC, no charsmap),
 * `add_dummy_prefix=true`, `escape_whitespaces=true`,
 * `remove_extra_whitespaces=false` and `byte_fallback=true`, so encoding is:
 *
 *   1. Prepend "▁" and replace every space with "▁" (U+2581).
 *   2. Viterbi over the UTF-8 bytes: pick the segmentation with the highest
 *      summed piece score; any byte not covered by a piece falls back to its
 *      `<0xNN>` byte piece at score `min(normal scores) - 10` (SentencePiece's
 *      unknown penalty — byte pieces carry score 0 in the model proto).
 *
 * Verified against `sentencepiece.SentencePieceProcessor.encode` on 613
 * mixed-content strings (all exact) by scripts/build_pockettts.py's mirror.
 *
 * The piece table is `pt_tokenizer.tsv`: `id \t type \t score \t piece` with
 * `\t`, `\n` and `\\` escaped in the piece column.
 */
class SpTokenizer(file: File) {

    private class Piece(val id: Int, val score: Float)

    /** UTF-8 byte strings (ISO-8859-1-decoded so each char is one byte) -> piece. */
    private val pieces = HashMap<String, Piece>(8192)
    private val byteIds = IntArray(256) { -1 }
    private val idToPiece = arrayOfNulls<String>(4000)
    private val idIsByte = BooleanArray(4000)
    private var maxLen = 1
    private val byteScore: Float

    init {
        var minScore = Float.MAX_VALUE
        file.forEachLine { line ->
            if (line.isEmpty()) return@forEachLine
            val cols = line.split('\t', limit = 4)
            val id = cols[0].toInt()
            val type = cols[1].toInt()
            val score = cols[2].toFloat()
            val piece = unescape(cols[3])
            idToPiece[id] = piece
            when (type) {
                6 -> {                                     // <0xNN> byte piece
                    byteIds[piece.substring(3, 5).toInt(16)] = id
                    idIsByte[id] = true
                }
                1 -> {                                     // normal piece
                    val key = String(piece.toByteArray(Charsets.UTF_8), Charsets.ISO_8859_1)
                    pieces[key] = Piece(id, score)
                    if (key.length > maxLen) maxLen = key.length
                    if (score < minScore) minScore = score
                }
                // 2 (unk) and 3 (control) never match raw text
            }
        }
        byteScore = minScore - 10f
    }

    private fun unescape(s: String): String {
        if ('\\' !in s) return s
        val sb = StringBuilder(s.length)
        var i = 0
        while (i < s.length) {
            val c = s[i]
            if (c == '\\' && i + 1 < s.length) {
                when (s[i + 1]) {
                    't' -> { sb.append('\t'); i += 2 }
                    'n' -> { sb.append('\n'); i += 2 }
                    '\\' -> { sb.append('\\'); i += 2 }
                    else -> { sb.append(c); i++ }
                }
            } else { sb.append(c); i++ }
        }
        return sb.toString()
    }

    /** Encode `text` exactly as SentencePiece would (see class doc). */
    fun encode(text: String): IntArray {
        val norm = String(
            ("▁" + text.replace(' ', '▁')).toByteArray(Charsets.UTF_8),
            Charsets.ISO_8859_1,
        )
        val n = norm.length
        val best = FloatArray(n + 1) { Float.NEGATIVE_INFINITY }
        val backPos = IntArray(n + 1)
        val backId = IntArray(n + 1)
        best[0] = 0f
        for (i in 0 until n) {
            if (best[i] == Float.NEGATIVE_INFINITY) continue
            val lim = minOf(maxLen, n - i)
            for (len in 1..lim) {
                val hit = pieces[norm.substring(i, i + len)] ?: continue
                val s = best[i] + hit.score
                if (s > best[i + len]) {
                    best[i + len] = s; backPos[i + len] = i; backId[i + len] = hit.id
                }
            }
            val s = best[i] + byteScore                    // byte fallback
            if (s > best[i + 1]) {
                best[i + 1] = s; backPos[i + 1] = i
                backId[i + 1] = byteIds[norm[i].code and 0xFF]
            }
        }
        val out = ArrayList<Int>(n / 3 + 1)
        var j = n
        while (j > 0) { out.add(backId[j]); j = backPos[j] }
        out.reverse()
        return out.toIntArray()
    }

    /** Inverse of [encode]: pieces joined, "▁" back to space, dummy prefix dropped. */
    fun decode(ids: List<Int>): String {
        val bytes = java.io.ByteArrayOutputStream()
        for (id in ids) {
            val piece = idToPiece[id] ?: continue
            if (idIsByte[id]) bytes.write(piece.substring(3, 5).toInt(16))
            else bytes.write(piece.toByteArray(Charsets.UTF_8))
        }
        return String(bytes.toByteArray(), Charsets.UTF_8)
            .replace('▁', ' ')
            .removePrefix(" ")
    }
}
