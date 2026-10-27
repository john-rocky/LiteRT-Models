package com.ming3tts

/**
 * Ming-3.0-tts prompt contract (upstream tts_tasks.py + prompt_contract.build_input_ids, via the
 * host spec scripts/host_mirror.py). The strings and the JSON key order are model input: the
 * checkpoint was trained on exactly these captions.
 */
object PromptContract {
    const val DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"

    private val LANG_NAMES = mapOf(
        "zh" to "Chinese", "zh-cn" to "Chinese", "chinese" to "Chinese", "en" to "English",
        "english" to "English", "ja" to "Japanese", "japanese" to "Japanese", "ko" to "Korean",
        "korean" to "Korean", "de" to "German", "german" to "German", "fr" to "French",
        "french" to "French", "es" to "Spanish", "spanish" to "Spanish", "yue" to "Chinese",
        "zh-yue" to "Chinese", "cantonese" to "Chinese",
    )

    /** tts_tasks.build_tts_prompt: the user prompt with the optional target-language line. */
    fun ttsPrompt(languageId: String?): String {
        if (languageId == null || languageId.isBlank()) return DEFAULT_TTS_PROMPT
        val lang = languageId.trim()
        return DEFAULT_TTS_PROMPT + "Target language: ${LANG_NAMES[lang.lowercase()] ?: lang}.\n"
    }

    /** tts_tasks.preserve_multilingual_text: non-zh/en languages skip the zh/en text path. */
    fun preserveMultilingualText(languageId: String?): Boolean {
        if (languageId == null || languageId.isBlank()) return false
        val l = languageId.trim().lowercase().replace('_', '-')
        return l !in setOf("zh", "zh-cn", "chinese", "en", "english")
    }

    /**
     * build_structured_instruction(schema="zh", ip=...) as `json.dumps(..., ensure_ascii=False)`
     * prints it (default separators ", " and ": ").
     */
    fun ipInstruction(ip: String): String =
        "{\"audio_sequence\": [{\"序号\": 1, \"说话人\": \"speaker_1\", \"方言\": null, " +
            "\"风格\": null, \"语速\": null, \"基频\": null, \"音量\": null, \"情感\": null, " +
            "\"BGM\": {\"Genre\": null, \"Mood\": null, \"Instrument\": null, \"Theme\": null, " +
            "\"ENV\": null, \"SNR\": null}, \"IP\": ${jsonString(ip)}}]}"

    private fun jsonString(s: String): String {
        val sb = StringBuilder("\"")
        for (ch in s) {
            when {
                ch == '"' -> sb.append("\\\"")
                ch == '\\' -> sb.append("\\\\")
                ch == '\n' -> sb.append("\\n")
                ch == '\r' -> sb.append("\\r")
                ch == '\t' -> sb.append("\\t")
                ch == '\b' -> sb.append("\\b")
                ch == '\u000c' -> sb.append("\\f")
                ch < ' ' -> sb.append(String.format("\\u%04x", ch.code))
                else -> sb.append(ch)
            }
        }
        return sb.append('"').toString()
    }

    class Ids(val ids: IntArray, val speakerPositions: IntArray, val latentStart: Int)

    /** prompt_contract.build_input_ids (dense layout), one speaker. */
    fun buildInputIds(
        tok: QwenTokenizer,
        prompt: String,
        text: String,
        instruction: String?,
        promptText: String? = null,
        promptLatentSteps: Int = 0,
        numSpeakers: Int = 1,
    ): Ids {
        val out = IntList(256)
        fun enc(s: String) = out.addAll(tok.encode(s))
        enc("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        enc("<|im_start|>user\n")
        enc(prompt)
        for (i in 0 until numSpeakers) {
            enc("  speaker_${i + 1}:")
            enc("<|vision_start|>")
            enc("<|vision_pad|>")
            enc("<|vision_end|>\n")
        }
        val bgm = listOf("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")
        if (!bgm.all { text.contains(it) }) enc(" Text input:\n")
        val withLatent = promptLatentSteps > 0 && promptText != null
        if (withLatent) enc(promptText!!)
        enc(text)
        enc("<|im_end|>\n")
        enc("<|im_start|>assistant\n")
        if (instruction != null) {
            enc(instruction)
            enc("<|im_end|>")
        }
        enc("<audio>")
        val latentStart = if (withLatent) out.size else -1
        if (withLatent) {
            val patch = tok.tokenId("<audioPatch>")
            repeat(promptLatentSteps) { out.add(patch) }
        }
        val ids = out.toArray()
        val start = tok.tokenId("<|vision_start|>")
        val spk = IntList()
        for (i in ids.indices) if (ids[i] == start) spk.add(i + 1)
        check(spk.size == numSpeakers)
        return Ids(ids, spk.toArray(), latentStart)
    }
}
