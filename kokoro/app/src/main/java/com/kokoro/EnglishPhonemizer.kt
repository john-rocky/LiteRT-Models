package com.kokoro

import android.content.Context
import android.util.Log

/**
 * English text → Kokoro vocab IDs via CMU Pronouncing Dictionary + ARPABET → Misaki IPA mapping.
 *
 * Pipeline for robust free-text input:
 *   1. [EnglishTextNormalizer] expands numbers / currency / symbols to words.
 *   2. In-dictionary words use CMU (exact, with stress).
 *   3. Out-of-dictionary words fall back to the neural [NeuralG2p] (DeepPhonemizer,
 *      MIT) instead of being dropped — so names, brands and new words still speak.
 *
 * Quality note: lookup has NO NLP context (no function-word reduction or POS
 * disambiguation), and the neural fallback is stress-less, so output differs from
 * misaki's, but is intelligible since Kokoro generalizes from training data.
 */
class EnglishPhonemizer private constructor(
    private val cmu: Map<String, String>,
    private val vocab: Map<String, Int>,
    private val neural: NeuralG2p?,
) : Phonemizer {

    override fun phonemize(text: String): IntArray {
        val result = ArrayList<Int>(text.length * 2)
        // Expand numbers/currency/symbols to words first so they aren't dropped.
        val tokens = tokenize(EnglishTextNormalizer.normalize(text))
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
            if (arpabet != null) {
                arpabetToTokens(arpabet, result)
            } else {
                // Out-of-dictionary: fall back to the neural G2P (stress-less ARPABET).
                val arpa = neural?.predict(tok)
                if (!arpa.isNullOrEmpty()) {
                    arpabetToTokens(arpa.joinToString(" "), result)
                } else {
                    // Last resort: never drop a word — spell it out letter by letter.
                    spellOut(tok, result)
                }
            }
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

    /**
     * Absolute last resort so a word is never silently dropped: pronounce each letter
     * by its name ("NXT" -> "en ex tee"). Only reached if both CMU and the neural G2P
     * produced nothing (e.g. the neural model is missing, or a token with no a-z letters).
     */
    private fun spellOut(word: String, out: MutableList<Int>) {
        var first = true
        for (c in word.lowercase()) {
            val arpa = LETTER_ARPABET[c] ?: continue
            if (!first) vocab[" "]?.let { out.add(it) }
            first = false
            arpabetToTokens(arpa, out)
        }
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

        // Spoken letter names (ARPABET) for the spell-out last resort.
        private val LETTER_ARPABET = mapOf(
            'a' to "EY1", 'b' to "B IY1", 'c' to "S IY1", 'd' to "D IY1", 'e' to "IY1",
            'f' to "EH1 F", 'g' to "JH IY1", 'h' to "EY1 CH", 'i' to "AY1", 'j' to "JH EY1",
            'k' to "K EY1", 'l' to "EH1 L", 'm' to "EH1 M", 'n' to "EH1 N", 'o' to "OW1",
            'p' to "P IY1", 'q' to "K Y UW1", 'r' to "AA1 R", 's' to "EH1 S", 't' to "T IY1",
            'u' to "Y UW1", 'v' to "V IY1", 'w' to "D AH1 B AH0 L Y UW0", 'x' to "EH1 K S",
            'y' to "W AY1", 'z' to "Z IY1",
        )

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
            val neural = NeuralG2p.load(context)
            Log.i(TAG, "Loaded CMU dict: ${cmu.size} entries (neural OOV G2P: ${neural != null})")
            return EnglishPhonemizer(cmu, vocab, neural)
        }
    }
}
