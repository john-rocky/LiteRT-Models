package com.audio8tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The text rules on the JVM, with a stand-in token count (one token per character). */
class TextRulesTest {
    private val perChar: (String) -> Int = { it.length }

    @Test
    fun emptyAndWhitespaceOnlyAreRefused() {
        for (text in listOf("", "   ", "\n\t", "　　", "   ")) {
            val c = TextRules.check(text, perChar)
            assertTrue("'$text' -> $c", c is Audio8Tts.TextCheck.Empty)
            assertEquals("Type a sentence first.", c.message)
        }
    }

    @Test
    fun countIsTakenOnTheCleanedText() {
        val c = TextRules.check("  a　 b  ", perChar)
        assertTrue(c is Audio8Tts.TextCheck.Ok)
        assertEquals(3, (c as Audio8Tts.TextCheck.Ok).tokens)   // "a b"
    }

    @Test
    fun theLimitItselfIsAccepted() {
        val c = TextRules.check("x".repeat(TextRules.MAX_TEXT_TOKENS), perChar)
        assertTrue(c is Audio8Tts.TextCheck.Ok)
    }

    @Test
    fun overTheLimitIsRefusedWithTheCount() {
        val c = TextRules.check("x".repeat(TextRules.MAX_TEXT_TOKENS + 1), perChar)
        assertTrue(c is Audio8Tts.TextCheck.TooLong)
        assertEquals(TextRules.MAX_TEXT_TOKENS + 1, (c as Audio8Tts.TextCheck.TooLong).tokens)
        assertTrue(c.message.startsWith("Too long: 81 tokens."))
    }

    @Test
    fun aPastedPageIsCountedOnItsFirstCharactersOnly() {
        var counted = 0
        val c = TextRules.check("y".repeat(50_000)) { counted = it.length; 1 }
        assertEquals(TextRules.MAX_CHARS, counted)
        assertTrue(c is Audio8Tts.TextCheck.TooLong && c.truncated)
        assertTrue(c.message!!.startsWith("Too long: more than "))
    }
}
