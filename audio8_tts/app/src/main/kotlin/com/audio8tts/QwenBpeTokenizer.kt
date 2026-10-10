package com.audio8tts

import android.util.JsonReader
import java.io.File
import java.text.Normalizer

/**
 * Byte-level BPE tokenizer for the Qwen2 vocabulary, read from a Hugging Face `tokenizer.json`.
 *
 * Adapted from the Qwen3-TTS LiteRT sample's QwenBpeTokenizer (vocab.json + merges.txt), here reading the single
 * `tokenizer.json` Audio8 ships: `model.vocab` (piece -> id), `model.merges` (a list of [left, right] pairs, rank =
 * list index) and `added_tokens` (the chat/control/semantic tokens, all `normalized: false`, no strip flags).
 * Pipeline of the reference `tokenizers` library with add_special_tokens=False: split the raw text on added-token
 * strings (each becomes its own id), NFC-normalize the rest, pre-tokenize with the Qwen2 regex, map UTF-8 bytes to the
 * byte-level alphabet, apply ranked merges. The file is parsed with a streaming reader (12 MB).
 */
class QwenBpeTokenizer(file: File) {

    private val vocab = HashMap<String, Int>(160_000)
    private val ranks = HashMap<Long, Int>(160_000)
    private val added = HashMap<String, Int>()
    private val addedLengths: IntArray
    private val byteToChar = CharArray(256)

    // Qwen2 pre-tokenization regex (tokenizer.json pre_tokenizer.Split). The reference is Rust with Unicode-aware
    // \s; Android's ICU-backed regex is Unicode-aware by default. Verified by the startup self-test (incl. U+3000).
    private val pretokenize = Regex(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" +
            " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")

    val vocabSize: Int get() = vocab.size
    val mergeCount: Int get() = ranks.size
    val addedCount: Int get() = added.size

    init {
        // GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to U+0100.. in order.
        val direct = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)).toHashSet()
        var next = 256
        for (b in 0 until 256) {
            byteToChar[b] = if (b in direct) b.toChar() else (next++).toChar()
        }
        val pendingMerges = ArrayList<Pair<String, String>>(160_000)
        JsonReader(file.bufferedReader(Charsets.UTF_8, 1 shl 16)).use { r ->
            r.beginObject()
            while (r.hasNext()) {
                when (r.nextName()) {
                    "added_tokens" -> {
                        r.beginArray()
                        while (r.hasNext()) {
                            var id = -1
                            var content: String? = null
                            r.beginObject()
                            while (r.hasNext()) {
                                when (r.nextName()) {
                                    "id" -> id = r.nextInt()
                                    "content" -> content = r.nextString()
                                    else -> r.skipValue()
                                }
                            }
                            r.endObject()
                            if (content != null && id >= 0) added[content] = id
                        }
                        r.endArray()
                    }
                    "model" -> {
                        r.beginObject()
                        while (r.hasNext()) {
                            when (r.nextName()) {
                                "vocab" -> {
                                    r.beginObject()
                                    while (r.hasNext()) {
                                        val k = r.nextName()
                                        vocab[k] = r.nextInt()
                                    }
                                    r.endObject()
                                }
                                "merges" -> {
                                    r.beginArray()
                                    while (r.hasNext()) {
                                        // Newer tokenizers write [left, right]; older ones "left right".
                                        if (r.peek() == android.util.JsonToken.BEGIN_ARRAY) {
                                            r.beginArray()
                                            val a = r.nextString()
                                            val b = r.nextString()
                                            r.endArray()
                                            pendingMerges.add(a to b)
                                        } else {
                                            val line = r.nextString()
                                            val sp = line.indexOf(' ')
                                            pendingMerges.add(line.substring(0, sp) to line.substring(sp + 1))
                                        }
                                    }
                                    r.endArray()
                                }
                                else -> r.skipValue()
                            }
                        }
                        r.endObject()
                    }
                    else -> r.skipValue()
                }
            }
            r.endObject()
        }
        for ((rank, m) in pendingMerges.withIndex()) {
            val a = vocab[m.first] ?: error("merge piece not in vocab: ${m.first}")
            val b = vocab[m.second] ?: error("merge piece not in vocab: ${m.second}")
            ranks.putIfAbsent(pairKey(a, b), rank)
        }
        addedLengths = added.keys.map { it.length }.distinct().sortedDescending().toIntArray()
    }

    private fun pairKey(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

    /** Encodes text to ids (add_special_tokens=False: no BOS/EOS, but added-token strings map to their ids). */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length / 2 + 8)
        var start = 0
        var i = 0
        while (i < text.length) {
            val hit = if (text[i] == '<') matchAdded(text, i) else null
            if (hit != null) {
                if (i > start) encodePlain(text.substring(start, i), out)
                out.add(hit.second)
                i += hit.first
                start = i
            } else {
                i++
            }
        }
        if (start < text.length) encodePlain(text.substring(start), out)
        return out.toIntArray()
    }

    /** Longest added token starting at [at] (all Audio8 added tokens start with '<'). */
    private fun matchAdded(text: String, at: Int): Pair<Int, Int>? {
        for (len in addedLengths) {
            if (at + len > text.length) continue
            val id = added[text.substring(at, at + len)] ?: continue
            return len to id
        }
        return null
    }

    private fun encodePlain(text: String, out: ArrayList<Int>) {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
        for (match in pretokenize.findAll(normalized)) {
            val bytes = match.value.toByteArray(Charsets.UTF_8)
            val mapped = StringBuilder(bytes.size)
            for (b in bytes) mapped.append(byteToChar[b.toInt() and 0xFF])
            bpe(mapped.toString(), out)
        }
    }

    private fun bpe(token: String, out: ArrayList<Int>) {
        // Word as a list of piece ids, merged greedily by rank (leftmost occurrence of the best pair first, as the
        // reference's priority queue does).
        val ids = ArrayList<Int>(token.length)
        val pieces = ArrayList<String>(token.length)
        for (c in token) {
            val s = c.toString()
            pieces.add(s)
            ids.add(vocab[s] ?: error("byte piece missing: $s"))
        }
        while (ids.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (k in 0 until ids.size - 1) {
                val r = ranks[pairKey(ids[k], ids[k + 1])] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestIdx = k
                }
            }
            if (bestIdx < 0) break
            val merged = pieces[bestIdx] + pieces[bestIdx + 1]
            val mergedId = vocab[merged] ?: error("merged piece not in vocab: $merged")
            pieces[bestIdx] = merged
            ids[bestIdx] = mergedId
            pieces.removeAt(bestIdx + 1)
            ids.removeAt(bestIdx + 1)
        }
        out.addAll(ids)
    }
}
