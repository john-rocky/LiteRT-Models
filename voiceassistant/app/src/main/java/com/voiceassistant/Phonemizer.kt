package com.voiceassistant

import android.content.Context

interface Phonemizer {
    fun phonemize(text: String): IntArray

    companion object {
        fun forLanguage(context: Context, language: String, vocab: Map<String, Int>): Phonemizer =
            when (language) {
                "en" -> EnglishPhonemizer.load(context, vocab)
                "ja" -> JapanesePhonemizer.load(context, vocab)
                else -> throw IllegalArgumentException("Unsupported language: $language")
            }

        /** Load Kokoro vocab from assets/kokoro_vocab.json. */
        fun loadVocab(context: Context): Map<String, Int> {
            val json = context.assets.open("kokoro_vocab.json").bufferedReader().use { it.readText() }
            val obj = org.json.JSONObject(json)
            val out = HashMap<String, Int>(obj.length())
            obj.keys().forEach { key -> out[key] = obj.getInt(key) }
            return out
        }
    }
}
