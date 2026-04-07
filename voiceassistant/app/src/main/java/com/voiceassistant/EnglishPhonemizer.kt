package com.voiceassistant

import android.content.Context
import android.util.Log

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
            if (tok.length == 1 && tok in PUNCT_VOCAB) {
                vocab[tok]?.let { result.add(it) }
                prevWasWord = false
                continue
            }
            if (prevWasWord) vocab[" "]?.let { result.add(it) }
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

    private fun tokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val sb = StringBuilder()
        for (c in text) {
            if (c.isLetterOrDigit() || c == '\'' || c == '-') {
                sb.append(c)
            } else {
                if (sb.isNotEmpty()) { tokens.add(sb.toString()); sb.clear() }
                if (c.toString() in PUNCT_VOCAB) tokens.add(c.toString())
            }
        }
        if (sb.isNotEmpty()) tokens.add(sb.toString())
        return tokens
    }

    private fun arpabetToTokens(arpabet: String, out: MutableList<Int>) {
        for (sym in arpabet.split(" ")) {
            if (sym.isEmpty()) continue
            val stressDigit = sym.lastOrNull()?.takeIf { it.isDigit() }?.digitToInt() ?: -1
            val baseSym = if (stressDigit >= 0) sym.dropLast(1) else sym
            val ipa = mapArpabet(baseSym, stressDigit) ?: continue
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

    private fun mapArpabet(sym: String, stress: Int): String? = when (sym) {
        "AA" -> "ɑ"; "AE" -> "æ"
        "AH" -> if (stress >= 1) "ʌ" else "ə"
        "AO" -> "ɔ"; "AW" -> "W"; "AY" -> "I"
        "B" -> "b"; "CH" -> "ʧ"; "D" -> "d"; "DH" -> "ð"; "EH" -> "ɛ"
        "ER" -> if (stress >= 1) "ɜɹ" else "ɚ"
        "EY" -> "A"; "F" -> "f"; "G" -> "ɡ"; "HH" -> "h"; "IH" -> "ɪ"
        "IY" -> "i"; "JH" -> "ʤ"; "K" -> "k"; "L" -> "l"; "M" -> "m"
        "N" -> "n"; "NG" -> "ŋ"; "OW" -> "O"; "OY" -> "Y"; "P" -> "p"
        "R" -> "ɹ"; "S" -> "s"; "SH" -> "ʃ"; "T" -> "t"; "TH" -> "θ"
        "UH" -> "ʊ"; "UW" -> "u"; "V" -> "v"; "W" -> "w"; "Y" -> "j"
        "Z" -> "z"; "ZH" -> "ʒ"
        else -> { Log.w(TAG, "Unknown ARPABET: $sym"); null }
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
                    cmu[line.substring(0, space)] = line.substring(space + 1)
                }
            }
            Log.i(TAG, "Loaded CMU dict: ${cmu.size} entries")
            return EnglishPhonemizer(cmu, vocab)
        }
    }
}
