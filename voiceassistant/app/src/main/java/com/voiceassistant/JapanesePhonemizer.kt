package com.voiceassistant

import android.content.Context
import android.util.Log
import com.atilika.kuromoji.ipadic.Tokenizer

class JapanesePhonemizer private constructor(
    private val tokenizer: Tokenizer,
    private val vocab: Map<String, Int>,
) : Phonemizer {

    override fun phonemize(text: String): IntArray {
        val result = ArrayList<Int>(text.length * 2)
        val tokens = tokenizer.tokenize(text)
        var prevWasWord = false

        for (tok in tokens) {
            val surface = tok.surface
            if (surface.length == 1 && surface in PUNCT_VOCAB) {
                vocab[surface]?.let { result.add(it) }
                prevWasWord = false
                continue
            }
            if (surface in JA_PUNCT_TO_VOCAB) {
                vocab[JA_PUNCT_TO_VOCAB[surface]!!]?.let { result.add(it) }
                prevWasWord = false
                continue
            }
            val kana = tok.pronunciation?.takeIf { it != "*" }
                ?: tok.reading?.takeIf { it != "*" }
                ?: surface
            if (kana.isEmpty()) continue
            if (prevWasWord) vocab[" "]?.let { result.add(it) }
            prevWasWord = true
            kanaToTokens(kana, result)
        }
        return result.toIntArray()
    }

    private fun kanaToTokens(kana: String, out: MutableList<Int>) {
        var i = 0
        while (i < kana.length) {
            val c = kana[i]
            if (c == 'ー') { vocab[LONG_VOWEL]?.let { out.add(it) }; i++; continue }
            if (c == 'ッ') { i++; continue }
            if (c == 'ン') { vocab[MORA_N]?.let { out.add(it) }; i++; continue }
            if (i + 1 < kana.length) {
                val next = kana[i + 1]
                if (next == 'ャ' || next == 'ュ' || next == 'ョ') {
                    val yoon = YOON_TABLE["${c}${next}"]
                    if (yoon != null) { appendIpa(yoon, out); i += 2; continue }
                }
            }
            val ipa = KANA_TABLE[c.toString()]
            if (ipa != null) appendIpa(ipa, out)
            else {
                val kataC = hiraganaToKatakana(c)
                val ipa2 = KANA_TABLE[kataC.toString()]
                if (ipa2 != null) appendIpa(ipa2, out)
                else Log.w(TAG, "Unknown kana: '$c' (U+${c.code.toString(16)})")
            }
            i++
        }
    }

    private fun appendIpa(ipa: String, out: MutableList<Int>) {
        for (c in ipa) {
            val id = vocab[c.toString()]
            if (id != null) out.add(id)
            else Log.w(TAG, "Vocab missing IPA char '$c'")
        }
    }

    private fun hiraganaToKatakana(c: Char): Char =
        if (c in 'ぁ'..'ゖ') (c.code + 0x60).toChar() else c

    companion object {
        private const val TAG = "JapanesePhonemizer"
        private const val LONG_VOWEL = "ː"
        private const val MORA_N = "ɴ"
        private val PUNCT_VOCAB = setOf(".", ",", "!", "?", ";", ":")
        private val JA_PUNCT_TO_VOCAB = mapOf("。" to ".", "、" to ",", "！" to "!", "？" to "?")

        fun load(context: Context, vocab: Map<String, Int>): JapanesePhonemizer {
            return JapanesePhonemizer(Tokenizer(), vocab)
        }

        private val KANA_TABLE = mapOf(
            "ア" to "a", "イ" to "i", "ウ" to "ɨ", "エ" to "e", "オ" to "o",
            "カ" to "ka", "キ" to "kʲi", "ク" to "kɨ", "ケ" to "ke", "コ" to "ko",
            "ガ" to "ɡa", "ギ" to "ɡʲi", "グ" to "ɡɨ", "ゲ" to "ɡe", "ゴ" to "ɡo",
            "サ" to "sa", "シ" to "ɕi", "ス" to "sɨ", "セ" to "se", "ソ" to "so",
            "ザ" to "za", "ジ" to "ʥi", "ズ" to "zɨ", "ゼ" to "ze", "ゾ" to "zo",
            "タ" to "ta", "チ" to "ʨi", "ツ" to "ʦɨ", "テ" to "te", "ト" to "to",
            "ダ" to "da", "ヂ" to "ʥi", "ヅ" to "zɨ", "デ" to "de", "ド" to "do",
            "ナ" to "na", "ニ" to "ɲi", "ヌ" to "nɨ", "ネ" to "ne", "ノ" to "no",
            "ハ" to "ha", "ヒ" to "çi", "フ" to "ɸɨ", "ヘ" to "he", "ホ" to "ho",
            "バ" to "ba", "ビ" to "bʲi", "ブ" to "bɨ", "ベ" to "be", "ボ" to "bo",
            "パ" to "pa", "ピ" to "pʲi", "プ" to "pɨ", "ペ" to "pe", "ポ" to "po",
            "マ" to "ma", "ミ" to "mʲi", "ム" to "mɨ", "メ" to "me", "モ" to "mo",
            "ヤ" to "ja", "ユ" to "jɨ", "ヨ" to "jo",
            "ラ" to "ɾa", "リ" to "ɾʲi", "ル" to "ɾɨ", "レ" to "ɾe", "ロ" to "ɾo",
            "ワ" to "βa", "ヲ" to "o",
        )

        private val YOON_TABLE = mapOf(
            "キャ" to "kʲa", "キュ" to "kʲɨ", "キョ" to "kʲo",
            "ギャ" to "ɡʲa", "ギュ" to "ɡʲɨ", "ギョ" to "ɡʲo",
            "シャ" to "ɕa", "シュ" to "ɕɨ", "ショ" to "ɕo",
            "ジャ" to "ʥa", "ジュ" to "ʥɨ", "ジョ" to "ʥo",
            "チャ" to "ʨa", "チュ" to "ʨɨ", "チョ" to "ʨo",
            "ニャ" to "ɲa", "ニュ" to "ɲɨ", "ニョ" to "ɲo",
            "ヒャ" to "ça", "ヒュ" to "çɨ", "ヒョ" to "ço",
            "ビャ" to "bʲa", "ビュ" to "bʲɨ", "ビョ" to "bʲo",
            "ピャ" to "pʲa", "ピュ" to "pʲɨ", "ピョ" to "pʲo",
            "ミャ" to "mʲa", "ミュ" to "mʲɨ", "ミョ" to "mʲo",
            "リャ" to "ɾʲa", "リュ" to "ɾʲɨ", "リョ" to "ɾʲo",
        )
    }
}
