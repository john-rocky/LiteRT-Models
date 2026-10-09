package com.audio8tts

import org.json.JSONObject

/**
 * The fixed prompt fragments and tokenizer test vectors of `demo/prompt_constants.json` (ids produced by the
 * `tokenizers` library from the shipped tokenizer.json), packaged as an asset at build time.
 */
class PromptConstants(json: String) {
    class Vector(val name: String, val text: String, val ids: IntArray)

    val fragments: Map<String, IntArray>
    val fragmentTexts: Map<String, String>
    val vectors: List<Vector>
    val ids: Map<String, Int>
    val source: String

    init {
        val o = JSONObject(json)
        source = o.optString("source")
        val f = o.getJSONObject("fragments")
        val frag = LinkedHashMap<String, IntArray>()
        val texts = LinkedHashMap<String, String>()
        for (k in f.keys()) {
            val e = f.getJSONObject(k)
            frag[k] = ints(e.getJSONArray("ids"))
            texts[k] = e.getString("text")
        }
        fragments = frag
        fragmentTexts = texts
        val tv = o.getJSONObject("test_vectors")
        vectors = tv.keys().asSequence().map { k ->
            val e = tv.getJSONObject(k)
            Vector(k, e.getString("text"), ints(e.getJSONArray("ids")))
        }.toList()
        val idObj = o.getJSONObject("ids")
        ids = idObj.keys().asSequence().associateWith { idObj.getInt(it) }
    }

    private fun ints(a: org.json.JSONArray) = IntArray(a.length()) { a.getInt(it) }
}
