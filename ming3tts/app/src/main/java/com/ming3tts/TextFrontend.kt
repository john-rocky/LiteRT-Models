package com.ming3tts

import java.text.Normalizer

/**
 * Upstream text preparation (`MingTTS.prepare_text` with `text_normalization=False`): markdown
 * code marks removed, a final period added for zh/en, then `cut_text_by_semantic_length`
 * (frontend/text_segment_cut.py, max 24 semantic units) splits the text into fragments that are
 * synthesized one after another, each from a fresh prefill. The FST text normalizer (pynini)
 * does not run on the device; the published benchmarks and this conversion's test sentences use raw text too.
 *
 * Python `str` semantics (isspace, isalnum, `\d`) come from [CharClasses]; regexes here use
 * explicit character ranges only, so the JVM and Android's ICU engine behave the same.
 */
class TextFrontend(private val cc: CharClasses) {

    /** Fragments to synthesize, or empty when the text has nothing to say. */
    fun prepare(text: String, languageId: String?, maxLength: Int = 24): List<String> {
        val t = cc.pyStrip(text)
        if (t.isEmpty()) return emptyList()
        val src =
            if (PromptContract.preserveMultilingualText(languageId)) cc.pyStrip(stripMarkdownCode(t))
            else addPeriod(stripMarkdownCode(t))
        return cut(src, maxLength).filter { cc.pyStrip(it).isNotEmpty() }
    }

    // ------------------------------------------------------------------ toolkit.py

    fun stripMarkdownCode(text: String): String {
        var s = Regex("``([^\\n]+?)``").replace(text, "$1")
        s = Regex("`([^\\n]+?)`").replace(s, "$1")
        return s.replace("`", "")
    }

    fun addPeriod(text: String): String {
        val stripped = cc.pyRstrip(text)
        if (stripped.isEmpty()) return "."
        val last = stripped.codePointBefore(stripped.length)
        if ("，,、;；".indexOf(String(Character.toChars(last))) >= 0) {
            val hasChinese = stripped.any { it in '一'..'鿿' }
            return stripped.substring(0, stripped.length - Character.charCount(last)) +
                (if (hasChinese) "。" else ".")
        }
        if (!cc.isPyAlnum(last)) return stripped
        val isChinese = last in 0x4e00..0x9fff || last in 0x3400..0x4dbf
        return stripped + if (isChinese) "。" else "."
    }

    // ------------------------------------------------------------------ text_segment_cut.py

    private val asciiPunctWs = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\u000b\u000c"

    fun hasValidContent(text: String): Boolean = text.any { asciiPunctWs.indexOf(it) < 0 }

    private val cjk = Regex("[\\u4e00-\\u9fff\\u3041-\\u3096\\u309d-\\u309f\\u30a1-\\u30fa\\u30fc-\\u30ff\\uac00-\\ud7a3]")
    private val latin = "A-Za-z\\u00c0-\\u00d6\\u00d8-\\u00f6\\u00f8-\\u024f\\u1e00-\\u1eff"
    private val latinWord = Regex("[$latin]+(?:['’][$latin]+)*")

    fun semanticLength(text: String): Int {
        val t = Normalizer.normalize(text, Normalizer.Form.NFC)
        return cjk.findAll(t).count() + latinWord.findAll(t).count()
    }

    /** `re.sub(r'(\d)\.(\d)', ...)` with Python's Unicode `\d`. */
    private fun protectDecimalPoints(text: String): String {
        val sb = StringBuilder()
        var i = 0
        val cps = text.codePoints().toArray()
        while (i < cps.size) {
            if (i + 2 < cps.size && cc.isPyDigit(cps[i]) && cps[i + 1] == '.'.code && cc.isPyDigit(cps[i + 2])) {
                sb.appendCodePoint(cps[i]).append(DOT).appendCodePoint(cps[i + 2])
                i += 3
            } else {
                sb.appendCodePoint(cps[i])
                i++
            }
        }
        return sb.toString()
    }

    private val upperDot = Regex("([A-Z])\\.([A-Z])")

    fun cut(text: String, maxLen: Int = 24, minTail: Int = 5): List<String> {
        if (!hasValidContent(text)) return emptyList()
        var processed = protectDecimalPoints(text)
        repeat(3) { processed = upperDot.replace(processed, "$1$DOT$2") }
        processed = processed.replace("\n", " ").replace("。，", "。")
        if (semanticLength(processed) <= maxLen) return listOf(processed.replace(DOT, "."))
        val normalized = processed.replace(".", "。").replace("!", "！").replace("?", "？").replace(",", "，")

        val sentences = ArrayList<String>()
        var current = StringBuilder()
        for (ch in normalized) {
            current.append(ch)
            if (ch == '。' || ch == '！' || ch == '？') {
                val s = cc.pyStrip(current.toString())
                if (s.isNotEmpty()) sentences.add(s)
                current = StringBuilder()
            }
        }
        if (current.isNotEmpty()) {
            sentences.add(cc.pyStrip(current.toString()))
            val last = sentences.last()
            if (!(last.endsWith("。") || last.endsWith("！") || last.endsWith("？"))) {
                sentences[sentences.size - 1] = last + "。"
            }
        }

        var result = ArrayList<String>()
        for (sentence in sentences) {
            val clauses = ArrayList<String>()
            var cur = StringBuilder()
            fun close() {
                val clause = cc.pyStrip(cur.toString())
                if (clause.isNotEmpty() && hasValidContent(clause)) clauses.add(clause)
                else if (clause.isNotEmpty() && clauses.isNotEmpty()) clauses[clauses.size - 1] = clauses.last() + clause
            }
            for (ch in sentence) {
                cur.append(ch)
                if (ch == '，' || ch == ';' || ch == '；') {
                    close()
                    cur = StringBuilder()
                }
            }
            if (cur.isNotEmpty()) close()

            var i = 0
            while (i < clauses.size) {
                val clause = clauses[i]
                val len = semanticLength(clause)
                if (len < minTail && i + 1 < clauses.size) {
                    val combined = clause + clauses[i + 1]
                    if (semanticLength(combined) <= maxLen) {
                        result = appendFragment(result, combined, maxLen, minTail)
                        i += 2
                        continue
                    }
                }
                if (len > maxLen) {
                    for (frag in splitLongFragment(clause, maxLen)) {
                        result = appendFragment(result, frag, maxLen, minTail)
                    }
                } else {
                    result = appendFragment(result, clause, maxLen, minTail)
                }
                i++
            }
        }
        return result.map { it.replace(DOT, ".") }
    }

    private val leadingStrip = "，,:;" + " \t\n\r\u000b\u000c"
    private val alnumStart = Regex("^[a-zA-Z0-9]")

    private fun appendFragment(
        fragments: ArrayList<String>,
        raw: String,
        maxLen: Int,
        minTail: Int,
    ): ArrayList<String> {
        val newText = raw.trimStart { leadingStrip.indexOf(it) >= 0 }
        if (!hasValidContent(newText)) return fragments
        if (fragments.isEmpty()) {
            fragments.add(newText)
            return fragments
        }
        val last = fragments.last()
        val newLen = semanticLength(newText)
        if (semanticLength(last) + newLen <= maxLen) {
            if ((last.endsWith("。") || last.endsWith("！") || last.endsWith("？")) && newLen < minTail) {
                fragments.add(newText)
            } else {
                val sep = if (!last.endsWith(" ") && alnumStart.containsMatchIn(newText)) " " else ""
                fragments[fragments.size - 1] = last + sep + newText
            }
        } else {
            fragments.add(newText)
        }
        return fragments
    }

    // `([，、；。！？,;!?]|\s+[-•·])` with Python's `\s`
    private val clausePattern = Regex("([，、；。！？,;!?]|[$PY_SPACE_CLASS]+[-•·])")

    fun splitLongFragment(text: String, maxLen: Int): List<String> {
        if (semanticLength(text) <= maxLen) return listOf(text)
        // re.split with a capturing group: text pieces and separators, in order
        val parts = ArrayList<String>()
        var prev = 0
        for (m in clausePattern.findAll(text)) {
            parts.add(text.substring(prev, m.range.first))
            parts.add(m.value)
            prev = m.range.last + 1
        }
        parts.add(text.substring(prev))
        val merged = ArrayList<String>()
        for (part in parts) {
            if (part.isEmpty()) continue
            if (merged.isNotEmpty() && clausePattern.matchEntire(part) != null) {
                merged[merged.size - 1] = merged.last() + part
            } else {
                merged.add(part)
            }
        }
        val out = ArrayList<String>()
        var current = ""
        for (clause in merged) {
            val len = semanticLength(clause)
            if (len > maxLen) {
                if (current.isNotEmpty()) {
                    out.add(current)
                    current = ""
                }
                out.add(clause)
                continue
            }
            if (semanticLength(current) + len <= maxLen) {
                current += clause
            } else {
                if (current.isNotEmpty()) out.add(current)
                current = clause
            }
        }
        if (current.isNotEmpty()) out.add(current)
        return out
    }

    companion object {
        private const val DOT = "##DOT##"

        // Python str.isspace() code points (generated from Python over every code point)
        private const val PY_SPACE_CLASS =
            "\\t\\n\\u000b\\u000c\\r\\u001c-\\u001f \\u0085\\u00a0\\u1680\\u2000-\\u200a" +
                "\\u2028\\u2029\\u202f\\u205f\\u3000"
    }
}
