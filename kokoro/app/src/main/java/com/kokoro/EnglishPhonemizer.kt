package com.kokoro

import android.content.Context
import android.util.Log

/**
 * English text → Kokoro vocab IDs via CMU Pronouncing Dictionary + ARPABET → Misaki IPA mapping.
 *
 * Quality note: this is a simple lookup-only G2P with NO NLP context (no
 * function-word reduction, no part-of-speech disambiguation). The resulting
 * phoneme sequence will differ from misaki's output for the same text but
 * will still be intelligible since Kokoro generalizes from training data.
 *
 * OOV (out-of-vocabulary) words are dropped silently and logged.
 */
class EnglishPhonemizer private constructor(
    private val cmu: Map<String, String>,
    private val vocab: Map<String, Int>,
) : Phonemizer {

    override fun phonemize(text: String): IntArray {
        val result = ArrayList<Int>(text.length * 2)
        val tokens = tokenize(text)
        var prevWasWord = false

        for (tok in tokens) {
            if (tok.isEmpty()) continue

            // Punctuation directly maps to vocab
            if (tok.length == 1 && tok in PUNCT_VOCAB) {
                vocab[tok]?.let { result.add(it) }
                prevWasWord = false
                continue
            }

            // Insert space between consecutive words
            if (prevWasWord) {
                vocab[" "]?.let { result.add(it) }
            }
            prevWasWord = true

            val arpabet = cmu[tok.lowercase()]
            if (arpabet == null) {
                Log.w(TAG, "OOV: $tok")
                continue
            }
            arpabetToTokens(arpabet, result)
        }
        return result.toIntArray()
    }

    /** Split text into words and punctuation, preserving order. */
    private fun tokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val sb = StringBuilder()
        for (c in text) {
            if (c.isLetterOrDigit() || c == '\'' || c == '-') {
                sb.append(c)
            } else {
                if (sb.isNotEmpty()) {
                    tokens.add(sb.toString())
                    sb.clear()
                }
                if (c.toString() in PUNCT_VOCAB) tokens.add(c.toString())
                // Whitespace is dropped — word boundaries handled separately
            }
        }
        if (sb.isNotEmpty()) tokens.add(sb.toString())
        return tokens
    }

    /** Convert one ARPABET sequence ("HH AH0 L OW1") into Kokoro vocab IDs. */
    private fun arpabetToTokens(arpabet: String, out: MutableList<Int>) {
        for (sym in arpabet.split(" ")) {
            if (sym.isEmpty()) continue

            // Strip stress digit if present
            val stressDigit = sym.lastOrNull()?.takeIf { it.isDigit() }?.digitToInt() ?: -1
            val baseSym = if (stressDigit >= 0) sym.dropLast(1) else sym

            val ipa = mapArpabet(baseSym, stressDigit) ?: continue

            // Stress marker comes BEFORE the IPA character (Misaki convention)
            when (stressDigit) {
                1 -> vocab[STRESS_PRIMARY]?.let { out.add(it) }
                2 -> vocab[STRESS_SECONDARY]?.let { out.add(it) }
            }

            for (c in ipa) {
                val id = vocab[c.toString()]
                if (id != null) out.add(id)
                else Log.w(TAG, "Vocab missing IPA char '$c' (U+${c.code.toString(16)})")
            }
        }
    }

    /** Map ARPABET symbol to Misaki/Kokoro IPA. Some symbols depend on stress. */
    private fun mapArpabet(sym: String, stress: Int): String? = when (sym) {
        "AA" -> "ɑ"
        "AE" -> "æ"
        "AH" -> if (stress >= 1) "ʌ" else "ə"
        "AO" -> "ɔ"
        "AW" -> "W"   // Misaki diphthong /aʊ/
        "AY" -> "I"   // Misaki diphthong /aɪ/
        "B"  -> "b"
        "CH" -> "ʧ"
        "D"  -> "d"
        "DH" -> "ð"
        "EH" -> "ɛ"
        "ER" -> if (stress >= 1) "ɜɹ" else "ɚ"
        "EY" -> "A"   // Misaki diphthong /eɪ/
        "F"  -> "f"
        "G"  -> "ɡ"   // Latin small letter script g (U+0261)
        "HH" -> "h"
        "IH" -> "ɪ"
        "IY" -> "i"
        "JH" -> "ʤ"
        "K"  -> "k"
        "L"  -> "l"
        "M"  -> "m"
        "N"  -> "n"
        "NG" -> "ŋ"
        "OW" -> "O"   // Misaki diphthong /oʊ/
        "OY" -> "Y"   // Misaki diphthong /ɔɪ/ (note: distinct from "y" = /j/)
        "P"  -> "p"
        "R"  -> "ɹ"
        "S"  -> "s"
        "SH" -> "ʃ"
        "T"  -> "t"
        "TH" -> "θ"
        "UH" -> "ʊ"
        "UW" -> "u"
        "V"  -> "v"
        "W"  -> "w"   // consonant /w/
        "Y"  -> "j"   // consonant /j/ (ARPABET Y, lowercase j in IPA)
        "Z"  -> "z"
        "ZH" -> "ʒ"
        else -> {
            Log.w(TAG, "Unknown ARPABET symbol: $sym")
            null
        }
    }

    companion object {
        private const val TAG = "EnglishPhonemizer"
        private const val DICT_ASSET = "cmudict.txt"
        private const val STRESS_PRIMARY = "ˈ"
        private const val STRESS_SECONDARY = "ˌ"
        private val PUNCT_VOCAB = setOf(".", ",", "!", "?", ";", ":")

        fun load(context: Context, vocab: Map<String, Int>): EnglishPhonemizer {
            val cmu = HashMap<String, String>(140_000)
            context.assets.open(DICT_ASSET).bufferedReader().useLines { lines ->
                for (line in lines) {
                    val space = line.indexOf(' ')
                    if (space <= 0) continue
                    val word = line.substring(0, space)
                    val phones = line.substring(space + 1)
                    cmu[word] = phones
                }
            }
            Log.i(TAG, "Loaded CMU dict: ${cmu.size} entries")
            return EnglishPhonemizer(cmu, vocab)
        }
    }
}
