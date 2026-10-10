// Copyright 2026 Daisuke Majima. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// The prompt limit the screen shows is the point where encodePrompt starts to
// cut: a prompt of exactly the shown maximum keeps every token, one more loses
// one. Runs against the tokenizer tables prep_assets.sh copies into
// src/main/assets; skipped when they are absent.

package com.bonsai.imagegen

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import java.io.File

class PromptRulesTest {
    private val tokDir = File("src/main/assets")
    private lateinit var tok: QwenTokenizer

    @Before
    fun loadTokenizer() {
        assumeTrue(File(tokDir, "vocab.json").exists() && File(tokDir, "merges.txt").exists())
        tok = QwenTokenizer(File(tokDir, "vocab.json").inputStream(), File(tokDir, "merges.txt").inputStream())
    }

    /** "a" then " a" repeated: one token each, so the prompt has exactly [n] tokens. */
    private fun prompt(n: Int) = "a" + " a".repeat(n - 1)

    @Test
    fun emptyAndBlankAreRefused() {
        assertEquals(PromptRules.EMPTY_TEXT, PromptRules.refusal("", tok))
        assertEquals(PromptRules.EMPTY_TEXT, PromptRules.refusal(" \n\t ", tok))
        assertEquals(0, PromptRules.count("   ", tok).tokens)
    }

    @Test
    fun theShownMaximumIsWhereEncodePromptStartsToCut() {
        assertEquals(246, PromptRules.MAX_BODY_TOKENS)
        val max = PromptRules.count("x", tok).max
        assertEquals(244, max)   // "user\n" is 2 tokens

        val atMax = PromptRules.count(prompt(max), tok)
        assertEquals(max, atMax.tokens)
        assertTrue(atMax.fits)
        assertNull(PromptRules.refusal(prompt(max), tok))
        // nothing cut: the body is the whole window left by <|im_start|> and the suffix
        assertEquals(PromptRules.MAX_BODY_TOKENS, tok.encodePrompt(prompt(max)).promptTokenCount)
        assertEquals(PromptRules.MAX_BODY_TOKENS, tok.encode("user\n" + prompt(max)).size)

        val over = PromptRules.count(prompt(max + 1), tok)
        assertEquals(max + 1, over.tokens)
        assertFalse(over.fits)
        assertEquals(PromptRules.tooLongText(over), PromptRules.refusal(prompt(max + 1), tok))
        // encodePrompt would have dropped the last token
        assertEquals(PromptRules.MAX_BODY_TOKENS + 1, tok.encode("user\n" + prompt(max + 1)).size)
        assertEquals(PromptRules.MAX_BODY_TOKENS, tok.encodePrompt(prompt(max + 1)).promptTokenCount)
    }

    @Test
    fun theCounterReadsTokensOfTheMaximum() {
        assertEquals("4 / 244 tokens", PromptRules.counterText(PromptRules.count("a red fox sitting", tok)))
        assertTrue(PromptRules.counterText(PromptRules.count(prompt(300), tok)).endsWith("too long, shorten the prompt"))
    }
}
