package com.sam3

import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * CLIP byte-level BPE tokenizer (vocab.json + merges.txt from facebook/sam3.1), matching
 * sam3's open_clip SimpleTokenizer for typical prompts: lowercase, contraction split
 * ('s 't 're 've 'm 'll 'd), letters / digit / punctuation runs, byte-to-unicode mapping,
 * BPE merges, "</w>" word endings. SAM3 convention: context 32, ZERO padding (id 0),
 * mask = (id == 0). Verified against the python tokenizer:
 *   "wheel" -> [49406, 6744, 49407], "paper bag" -> [49406, 2802, 3365, 49407],
 *   "person's shoe" -> [49406, 2533, 568, 7342, 49407].
 */
class BpeTokenizer(ctx: Context) {

    companion object {
        const val BOS = 49406
        const val EOT = 49407
        const val MAX_LEN = 32
    }

    private val encoder = HashMap<String, Int>()
    private val ranks = HashMap<Pair<String, String>, Int>()
    private val byteToUnicode = HashMap<Int, Char>()
    private val pieceRegex =
        Regex("""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""")

    init {
        val vocab = JSONObject(File(ctx.filesDir, "vocab.json").readText())
        for (k in vocab.keys()) encoder[k] = vocab.getInt(k)
        File(ctx.filesDir, "merges.txt").readLines().drop(1).forEachIndexed { i, line ->
            val p = line.trim().split(" ")
            if (p.size == 2) ranks[Pair(p[0], p[1])] = i
        }
        // GPT-2 byte-to-unicode table
        val bs = ArrayList<Int>()
        for (c in '!'.code..'~'.code) bs.add(c)
        for (c in 0xA1..0xAC) bs.add(c)
        for (c in 0xAE..0xFF) bs.add(c)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0..255) {
            if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
        }
        for (i in bs.indices) byteToUnicode[bs[i]] = cs[i].toChar()
    }

    private fun bpe(tokenIn: String): List<String> {
        var word = tokenIn.dropLast(4).map { it.toString() }.toMutableList()
        if (word.isEmpty()) return listOf(tokenIn)
        word[word.size - 1] = word.last() + "</w>"
        while (word.size > 1) {
            var best: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.size - 1) {
                val r = ranks[Pair(word[i], word[i + 1])] ?: continue
                if (r < bestRank) { bestRank = r; best = Pair(word[i], word[i + 1]) }
            }
            val b = best ?: break
            val merged = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == b.first && word[i + 1] == b.second) {
                    merged.add(b.first + b.second); i += 2
                } else { merged.add(word[i]); i++ }
            }
            word = merged
        }
        return word
    }

    /** prompt -> 32 ids: [BOS, tokens..., EOT, 0-pad...]. Pad mask = (id == 0). */
    fun encode(text: String): IntArray {
        val clean = text.lowercase().trim().replace(Regex("\\s+"), " ")
        val ids = ArrayList<Int>()
        for (m in pieceRegex.findAll(clean)) {
            val mapped = m.value.toByteArray(Charsets.UTF_8)
                .map { byteToUnicode[it.toInt() and 0xFF]!! }.joinToString("") + "</w>"
            for (t in bpe(mapped)) encoder[t]?.let { ids.add(it) } ?: run {
                for (ch in t) encoder[ch.toString()]?.let { ids.add(it) }
            }
        }
        val out = IntArray(MAX_LEN)                 // zero padded
        out[0] = BOS
        val n = minOf(ids.size, MAX_LEN - 2)
        for (i in 0 until n) out[i + 1] = ids[i]
        out[n + 1] = EOT
        return out
    }
}
