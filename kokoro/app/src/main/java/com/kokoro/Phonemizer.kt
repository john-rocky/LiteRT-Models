package com.kokoro

import android.content.Context

/**
 * Converts text to Kokoro vocab token IDs.
 *
 * Implementations:
 *   - [EnglishPhonemizer]  — CMU Pronouncing Dictionary + ARPABET → Misaki IPA
 *   - [JapanesePhonemizer] — kuromoji morphological analyzer + kana → IPA table
 *
 * Both implementations skip out-of-vocabulary words/characters silently and
 * return only the IDs they can produce. The caller can detect total failure
 * by checking for an empty result.
 */
interface Phonemizer {
    fun phonemize(text: String): IntArray

    companion object {
        /** Pick a phonemizer for the given language code. */
        fun forLanguage(context: Context, language: String, vocab: Map<String, Int>): Phonemizer =
            when (language) {
                "en" -> EnglishPhonemizer.load(context, vocab)
                "ja" -> JapanesePhonemizer.load(context, vocab)
                else -> throw IllegalArgumentException("Unsupported language: $language")
            }

        /** Load the Kokoro vocab from assets/vocab.json. */
        fun loadVocab(context: Context): Map<String, Int> {
            val json = context.assets.open("vocab.json").bufferedReader().use { it.readText() }
            val obj = org.json.JSONObject(json)
            val out = HashMap<String, Int>(obj.length())
            obj.keys().forEach { key -> out[key] = obj.getInt(key) }
            return out
        }
    }
}
