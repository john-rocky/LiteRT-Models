package com.audio8tts

/**
 * Which typed texts one Speak takes. The text is cleaned as the host loop cleans it (whitespace runs, including the
 * ideographic space U+3000, become one space; ends trimmed), then counted with the model's tokenizer.
 *
 * The limit comes from the generation cap: one call makes at most 512 frames (23.8 s), and the measured sentences need
 * 4.5 (Japanese) to 5.7 (English) frames per token, so 80 tokens end within the cap with room to spare. A longer text
 * would be cut off mid-sentence, so it is refused with a line that says so.
 */
object TextRules {
    const val MAX_TEXT_TOKENS = 80

    /** Only this many characters are tokenized for the count, so a pasted page cannot stall the screen. */
    const val MAX_CHARS = 2000

    fun check(text: String, countTokens: (String) -> Int): Audio8Tts.TextCheck {
        val cleaned = Audio8Engine.clean(text)
        if (cleaned.isEmpty()) return Audio8Tts.TextCheck.Empty
        val truncated = cleaned.length > MAX_CHARS
        val tokens = countTokens(if (truncated) cleaned.substring(0, MAX_CHARS) else cleaned)
        return if (truncated || tokens > MAX_TEXT_TOKENS) Audio8Tts.TextCheck.TooLong(tokens, truncated)
        else Audio8Tts.TextCheck.Ok(tokens)
    }
}
