package com.wevisone

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The prompt the app builds equals the reference's prompt ids (r1's B arm, the HF processor) for the fixture rows,
 * and the answer ids decode to the reference's answer text, which the app reads back into the same boxes and shapes
 * as the package's parsers. The 29-row file needs `-Pwevisone.golden`; the committed public rows run with it too.
 */
class TokenizerTest {
    private fun tokenizer(): QwenBpeTokenizer? = Golden.file("tokenizer.json")?.let { QwenBpeTokenizer.fromFile(it) }

    private fun check(rowsFile: File, tok: QwenBpeTokenizer): Int {
        val rows = JSONObject(rowsFile.readText()).getJSONArray("rows")
        var promptOk = 0
        var decodeOk = 0
        var parseOk = 0
        for (i in 0 until rows.length()) {
            val r = rows.getJSONObject(i)
            val want = r.getJSONArray("prompt_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val got = tok.encode(Prompts.BEFORE_IMAGE) + IntArray(WeVisOneEngine.N_IMAGE) { WeVisOneEngine.IMAGE_PAD } +
                tok.encode(Prompts.afterImage(r.getString("instruction")))
            if (got.contentEquals(want)) promptOk++ else println("PROMPT DIFF ${r.getString("id")}: ${got.toList()} vs ${want.toList()}")
            val newIds = r.getJSONArray("new_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val text = tok.decode(newIds)
            if (text == r.getString("response")) decodeOk++ else println("DECODE DIFF ${r.getString("id")}: $text")
            val answer = Answer(text)
            val seg = r.getJSONArray("seg_boxes").let { a -> (0 until a.length()).map { k -> a.getJSONArray(k).let { b -> List(b.length()) { b.getInt(it) } } } }
            val shapes = r.getJSONArray("shapes").let { a ->
                (0 until a.length()).map { k ->
                    val s = a.getJSONObject(k)
                    val kind = s.getString("kind").let { if (it == "polygon") "box" else it }
                    Triple(s.getString("label"), kind, s.getJSONArray("bins").let { b -> List(b.length()) { b.getInt(it) } })
                }
            }
            val mineSeg = answer.segBoxes.map { it.toList() }
            val mineShapes = answer.shapes.map { Triple(it.label, it.kind, it.bins.toList()) }
            // the package groups shapes by label (dict order = first appearance); compare as such
            if (mineSeg == seg && mineShapes.groupBy { it.first } == shapes.groupBy { it.first }) parseOk++
            else println("PARSE DIFF ${r.getString("id")}: $mineSeg / $mineShapes vs $seg / $shapes")
        }
        println("TOKENIZER ${rowsFile.name}: prompt ids $promptOk/${rows.length()}, answers decode $decodeOk/${rows.length()}, parsed $parseOk/${rows.length()}")
        assertEquals("prompt ids", rows.length(), promptOk)
        assertEquals("decode", rows.length(), decodeOk)
        assertEquals("parse", rows.length(), parseOk)
        return rows.length()
    }

    @Test
    fun allFixtureRows() {
        val tok = tokenizer()
        val rows = Golden.file("tok_rows.json")
        assumeTrue("needs -Pwevisone.golden", tok != null && rows != null)
        assertEquals(29, check(rows!!, tok!!))
    }

    @Test
    fun publicRowsCommitted() {
        val tok = tokenizer()
        assumeTrue("needs the tokenizer (-Pwevisone.golden)", tok != null)
        val rows = File(Golden.moduleRoot, "app/src/test/resources/tok_rows_public.json")
        check(rows, tok!!)
    }

    @Test
    fun tokenizerCounts() {
        val tok = tokenizer()
        assumeTrue(tok != null)
        println("TOKENIZER vocab ${tok!!.vocabSize} merges ${tok.mergeCount} added ${tok.addedCount}")
        assertEquals(151_643, tok.vocabSize)
        assertEquals(1_036, tok.addedCount)
        assertEquals(152_678, tok.addedId("<OBJ>"))
    }
}
