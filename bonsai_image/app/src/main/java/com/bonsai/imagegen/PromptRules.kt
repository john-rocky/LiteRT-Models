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

// What the prompt box accepts, shared by the screen and the device check. The
// text encoder reads one 256-token window, [<|im_start|>] + BPE("user\n" +
// prompt) + the assistant suffix (QwenTokenizer.encodePrompt), so the prompt
// gets what the rest leaves; encodePrompt would cut a longer one. The screen
// counts the prompt's tokens as it is typed and refuses Generate past the
// limit, so the end of a prompt is never dropped unseen.

package com.bonsai.imagegen

object PromptRules {

    /** Shown when Generate is pressed with nothing but whitespace in the box. */
    const val EMPTY_TEXT = "Type a prompt first."

    /** Tokens of BPE("user\n" + prompt) the window holds beside <|im_start|> and the suffix. */
    val MAX_BODY_TOKENS = QwenTokenizer.SEQ_LEN - 1 - QwenTokenizer.SUFFIX_IDS.size

    private const val USER_PREFIX = "user\n"

    /** The prompt's tokens and the most it may have: the window less "user\n". [tokens] <= [max] exactly when
     *  BPE("user\n" + prompt) fits [MAX_BODY_TOKENS], merges across the prefix included. */
    class Count(val tokens: Int, val max: Int) {
        val fits: Boolean get() = tokens <= max
    }

    /** Counts the trimmed prompt, as the run sees it. */
    fun count(prompt: String, tokenizer: QwenTokenizer): Count {
        val text = prompt.trim()
        val prefix = tokenizer.encode(USER_PREFIX).size
        val body = if (text.isEmpty()) prefix else tokenizer.encode(USER_PREFIX + text).size
        return Count(body - prefix, MAX_BODY_TOKENS - prefix)
    }

    /** Why Generate cannot start with [prompt], or null when it can. */
    fun refusal(prompt: String, tokenizer: QwenTokenizer): String? {
        if (prompt.isBlank()) return EMPTY_TEXT
        val c = count(prompt, tokenizer)
        return if (c.fits) null else tooLongText(c)
    }

    /** The line under the prompt box: the token count, or the too-long warning. */
    fun counterText(c: Count): String = if (c.fits) "${c.tokens} / ${c.max} tokens" else tooLongText(c)

    fun tooLongText(c: Count): String = "${c.tokens} / ${c.max} tokens: too long, shorten the prompt"
}
