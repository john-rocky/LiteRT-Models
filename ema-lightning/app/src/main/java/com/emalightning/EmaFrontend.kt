package com.emalightning

import java.text.Normalizer

/**
 * Text handling of the ema-lightning package, ported: the alphabet step (Turkish lowercase, accents stripped except
 * Turkish letters, anything outside the model's 49 symbols becomes a space), the chunker (pieces of at most ~10 s cut at
 * sentence / clause / word boundaries, each ending in terminal punctuation) and the per-piece bookkeeping (letter ids,
 * the word of each letter, the first letter of that word).
 *
 * Not ported: normalizer-tr (Rust) — numbers, dates, amounts and abbreviations are read aloud only when written out in
 * words ("14:30" → "on dört otuz"). The Hub repo's ema_host.py runs normalizer-tr on a desktop host.
 */
class EmaFrontend(vocab: List<String>) {
    private val vocab: Set<String> = vocab.toSet()
    private val stoi: Map<String, Int> = vocab.withIndex().associate { it.value to it.index }

    companion object {
        private const val TURKISH = "çğıöşüÇĞİÖŞÜ"
        private val TYPOGRAPHY = mapOf(
            '’' to "'", '‘' to "'", 'ʼ' to "'", '´' to "'", '`' to "'", '“' to "\"", '”' to "\"", '„' to "\"",
            '«' to "\"", '»' to "\"", '–' to "-", '—' to "-", '−' to "-", '…' to "...",
        )
        private val UNSAFE = Regex("[\\x00-\\x08\\x0b-\\x1f\\x7f-\\x9f\\u061c\\u200e\\u200f\\u202a-\\u202e\\u2066-\\u2069]")
        const val LETTERS_PER_SECOND = 18.0
        const val MAX_SECONDS = 10.0
        const val MAX_LETTERS = 250
        const val SENTENCE_PAUSE = 0.25
        const val CLAUSE_PAUSE = 0.12
        private val CUTS = listOf(
            Regex("[.!?]+[\"')]*(?= )") to SENTENCE_PAUSE,
            Regex("[,;:](?= )") to CLAUSE_PAUSE,
            Regex("\\S(?= )") to CLAUSE_PAUSE,
        )
        private val LETTER = Regex("[^\\W\\d_]")
    }

    /** Any written Turkish -> text in the model's alphabet (the package's Frontend.alphabet without normalizer-tr). */
    fun alphabet(raw: String): String {
        var text = UNSAFE.replace(raw, " ")
        if (text.isBlank()) return ""
        text = buildString { for (ch in text) append(TYPOGRAPHY[ch] ?: ch.toString()) }
            .replace("İ", "i").replace("I", "ı").lowercase()
        val out = StringBuilder()
        for (ch in text) {
            var s = ch.toString()
            if (ch !in TURKISH) {
                s = Normalizer.normalize(s, Normalizer.Form.NFKD).filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
            }
            out.append(if (s.isNotEmpty() && s.all { it.toString() in vocab }) s else " ")
        }
        return out.toString().replace(Regex("\\s+"), " ").trim()
    }

    /** [(piece, seconds of silence after it)], each piece ending in terminal punctuation. */
    fun chunk(text: String, speed: Double): List<Pair<String, Double>> {
        val limit = minOf(MAX_LETTERS.toDouble(), LETTERS_PER_SECOND * MAX_SECONDS * speed).toInt()
        val pieces = ArrayList<Pair<String, Double>>()
        var rest = text.trim()
        while (rest.isNotEmpty()) {
            var cut = rest.length
            var pause = 0.0
            if (rest.length > limit) {
                cut = limit
                for ((pattern, gap) in CUTS) {
                    val ends = pattern.findAll(rest.substring(0, minOf(limit + 1, rest.length))).map { it.range.last + 1 }.toList()
                    if (ends.isNotEmpty()) { cut = ends.last(); pause = gap; break }
                }
            }
            val piece = rest.substring(0, cut).trim()
            rest = rest.substring(cut).trim()
            if (LETTER.containsMatchIn(piece)) pieces.add(finish(piece) to pause)
        }
        if (pieces.isNotEmpty()) pieces[pieces.size - 1] = pieces.last().first to 0.0
        return pieces
    }

    private fun finish(piece: String): String {
        val last = piece.trimEnd('"', '\'', ')').takeLast(1)
        return if (last in listOf(".", "!", "?")) piece else piece.trimEnd(',', ';', ':', '-', ' ') + "."
    }

    class Piece(val text: String, val ids: IntArray, val cw: IntArray, val wstart: IntArray)

    /** engine.Engine.piece: letters -> ids, the word of each letter, the first letter of that word. */
    fun piece(text: String): Piece {
        val ids = IntArray(text.length) { stoi[text[it].toString()] ?: 1 }
        val starts = ArrayList<Int>()
        for (i in text.indices) if (text[i] != ' ' && (i == 0 || text[i - 1] == ' ')) starts.add(i)
        if (starts.isEmpty()) starts.add(0)
        val bounds = ArrayList<Int>().apply { add(0); addAll(starts.drop(1)); add(text.length) }
        val cw = IntArray(text.length)
        val wstart = IntArray(text.length)
        for (w in 0 until bounds.size - 1) for (i in bounds[w] until bounds[w + 1]) { cw[i] = w; wstart[i] = bounds[w] }
        return Piece(text, ids, cw, wstart)
    }
}
