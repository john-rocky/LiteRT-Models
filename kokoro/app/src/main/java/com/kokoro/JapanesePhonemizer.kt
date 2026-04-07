package com.kokoro

import android.content.Context
import android.util.Log
import com.atilika.kuromoji.ipadic.Tokenizer

/**
 * Japanese text → Kokoro vocab IDs via kuromoji morphological analyzer
 * + katakana → Misaki IPA table.
 *
 * Quality note: this approximates misaki's pyopenjtalk-based phonemization
 * but does NOT reproduce pitch accent or sandhi rules. Output is intelligible
 * but slightly less natural in prosody.
 */
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
            // Check punctuation first
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

            // Prefer `pronunciation` (closer to actual sound, e.g. は particle → ワ)
            // over `reading` (literal kana, e.g. は → ハ). Fall back to surface for
            // unknown words that may already be all-katakana.
            val kana = tok.pronunciation?.takeIf { it != "*" }
                ?: tok.reading?.takeIf { it != "*" }
                ?: surface
            if (kana.isEmpty()) continue

            // Insert space between morphemes for better Kokoro pacing
            if (prevWasWord) vocab[" "]?.let { result.add(it) }
            prevWasWord = true

            kanaToTokens(kana, result)
        }
        return result.toIntArray()
    }

    /** Convert a katakana reading string into Kokoro vocab IDs. */
    private fun kanaToTokens(kana: String, out: MutableList<Int>) {
        var i = 0
        while (i < kana.length) {
            val c = kana[i]

            // Long vowel marker — Kokoro vocab uses ː (U+02D0)
            if (c == 'ー') {
                vocab[LONG_VOWEL]?.let { out.add(it) }
                i++
                continue
            }

            // Sokuon ッ — duplicate the next consonant. Simplest approximation:
            // skip; the duration predictor in StyleTTS2 will handle most cases.
            if (c == 'ッ') {
                i++
                continue
            }

            // Mora-final N
            if (c == 'ン') {
                vocab[MORA_N]?.let { out.add(it) }
                i++
                continue
            }

            // Try yōon (palatalized): C + ャ/ュ/ョ
            if (i + 1 < kana.length) {
                val next = kana[i + 1]
                if (next == 'ャ' || next == 'ュ' || next == 'ョ') {
                    val yoon = YOON_TABLE["${c}${next}"]
                    if (yoon != null) {
                        appendIpa(yoon, out)
                        i += 2
                        continue
                    }
                }
            }

            val ipa = KANA_TABLE[c.toString()]
            if (ipa != null) {
                appendIpa(ipa, out)
            } else {
                // Hiragana fallback: try converting to katakana
                val kataC = hiraganaToKatakana(c)
                val ipa2 = KANA_TABLE[kataC.toString()]
                if (ipa2 != null) appendIpa(ipa2, out)
                else Log.w(TAG, "Unknown kana char: '$c' (U+${c.code.toString(16)})")
            }
            i++
        }
    }

    private fun appendIpa(ipa: String, out: MutableList<Int>) {
        for (c in ipa) {
            val id = vocab[c.toString()]
            if (id != null) out.add(id)
            else Log.w(TAG, "Vocab missing IPA char '$c' (U+${c.code.toString(16)})")
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
            val t = Tokenizer()
            return JapanesePhonemizer(t, vocab)
        }

        // Katakana → Misaki Kokoro IPA. Vowels use ɨ (not ɯ), w → β,
        // /j/-glide → "j" lowercase, palatalized consonants get ʲ marker.
        private val KANA_TABLE = mapOf(
            // Vowels
            "ア" to "a", "イ" to "i", "ウ" to "ɨ", "エ" to "e", "オ" to "o",
            // K
            "カ" to "ka", "キ" to "kʲi", "ク" to "kɨ", "ケ" to "ke", "コ" to "ko",
            // G
            "ガ" to "ɡa", "ギ" to "ɡʲi", "グ" to "ɡɨ", "ゲ" to "ɡe", "ゴ" to "ɡo",
            // S
            "サ" to "sa", "シ" to "ɕi", "ス" to "sɨ", "セ" to "se", "ソ" to "so",
            // Z
            "ザ" to "za", "ジ" to "ʥi", "ズ" to "zɨ", "ゼ" to "ze", "ゾ" to "zo",
            // T
            "タ" to "ta", "チ" to "ʨi", "ツ" to "ʦɨ", "テ" to "te", "ト" to "to",
            // D
            "ダ" to "da", "ヂ" to "ʥi", "ヅ" to "zɨ", "デ" to "de", "ド" to "do",
            // N
            "ナ" to "na", "ニ" to "ɲi", "ヌ" to "nɨ", "ネ" to "ne", "ノ" to "no",
            // H
            "ハ" to "ha", "ヒ" to "çi", "フ" to "ɸɨ", "ヘ" to "he", "ホ" to "ho",
            // B
            "バ" to "ba", "ビ" to "bʲi", "ブ" to "bɨ", "ベ" to "be", "ボ" to "bo",
            // P
            "パ" to "pa", "ピ" to "pʲi", "プ" to "pɨ", "ペ" to "pe", "ポ" to "po",
            // M
            "マ" to "ma", "ミ" to "mʲi", "ム" to "mɨ", "メ" to "me", "モ" to "mo",
            // Y
            "ヤ" to "ja", "ユ" to "jɨ", "ヨ" to "jo",
            // R
            "ラ" to "ɾa", "リ" to "ɾʲi", "ル" to "ɾɨ", "レ" to "ɾe", "ロ" to "ɾo",
            // W
            "ワ" to "βa", "ヲ" to "o",
            // Particle wa is written ハ but read βa — kuromoji returns "ハ" for は.
            // Override: when ハ is the topic marker the surface comes from the morpheme,
            // not from KANA_TABLE — but we still need ハ → ha for normal use.
            // The rare reading mismatch is acceptable for MVP.
        )

        // Yōon (palatalized) two-char combinations
        private val YOON_TABLE = mapOf(
            // K
            "キャ" to "kʲa", "キュ" to "kʲɨ", "キョ" to "kʲo",
            // G
            "ギャ" to "ɡʲa", "ギュ" to "ɡʲɨ", "ギョ" to "ɡʲo",
            // S
            "シャ" to "ɕa", "シュ" to "ɕɨ", "ショ" to "ɕo",
            // J
            "ジャ" to "ʥa", "ジュ" to "ʥɨ", "ジョ" to "ʥo",
            // T
            "チャ" to "ʨa", "チュ" to "ʨɨ", "チョ" to "ʨo",
            // N
            "ニャ" to "ɲa", "ニュ" to "ɲɨ", "ニョ" to "ɲo",
            // H
            "ヒャ" to "ça", "ヒュ" to "çɨ", "ヒョ" to "ço",
            // B
            "ビャ" to "bʲa", "ビュ" to "bʲɨ", "ビョ" to "bʲo",
            // P
            "ピャ" to "pʲa", "ピュ" to "pʲɨ", "ピョ" to "pʲo",
            // M
            "ミャ" to "mʲa", "ミュ" to "mʲɨ", "ミョ" to "mʲo",
            // R
            "リャ" to "ɾʲa", "リュ" to "ɾʲɨ", "リョ" to "ɾʲo",
        )
    }
}
