package com.opendecision

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Every request of the conversion run's fixtures (1,809 on the author's public test files) through the Kotlin
 * tokenizer and builder: ids, question spans and option spans must equal the official `Collator`'s
 * (captured in fixtures/oracle.json), and the smallest window must be the one the oracle marks as fitting.
 */
class InputsParityTest {
  @Test
  fun idsAndSpansEqualTheOfficialCollator() {
    val root = ExternalTestData.resolve()
    val inputs = DecisionInputs(DecisionTokenizer(ExternalTestData.tokenizer(root)))
    val requests = JSONObject(File(root, "fixtures/requests.json").readText()).getJSONArray("requests")
    val oracle = JSONObject(File(root, "fixtures/oracle.json").readText()).getJSONArray("records")
    val byId = HashMap<String, JSONObject>()
    for (i in 0 until oracle.length()) byId[oracle.getJSONObject(i).getString("id")] = oracle.getJSONObject(i)
    val failures = JSONArray()
    var compared = 0
    for (i in 0 until requests.length()) {
      val r = requests.getJSONObject(i)
      val o = byId[r.getString("id")] ?: continue
      val questions = parse(r.getJSONArray("questions"))
      val encoded = inputs.encode(r.getString("state"), questions)
      val ids = o.getJSONArray("ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
      val qSpans = o.getJSONArray("q_spans").let { a -> List(a.length()) { a.getJSONArray(it).let { s -> DecisionInputs.Span(s.getInt(0), s.getInt(1)) } } }
      val oSpans =
        o.getJSONArray("opt_spans").let { a ->
          List(a.length()) { q -> a.getJSONArray(q).let { b -> List(b.length()) { b.getJSONArray(it).let { s -> DecisionInputs.Span(s.getInt(0), s.getInt(1)) } } } }
        }
      val sameIds = encoded.inputIds.contentEquals(ids)
      val sameSpans = encoded.questionSpans == qSpans && encoded.optionSpans == oSpans
      val expectedWindow = if (o.getJSONObject("fits").getBoolean("s256")) 256 else 512
      val window = inputs.prepare(encoded).window
      compared++
      if (!sameIds || !sameSpans || window != expectedWindow) {
        failures.put(JSONObject().put("id", r.getString("id")).put("ids_equal", sameIds).put("spans_equal", sameSpans).put("window", window).put("expected_window", expectedWindow))
      }
    }
    ExternalTestData.reportFile("inputs_parity.json")
      .writeText(JSONObject().put("compared", compared).put("passed", compared - failures.length()).put("failures", failures).toString(2) + "\n")
    println("INPUTS_PARITY compared=$compared passed=${compared - failures.length()} failures=${failures.length()}")
    assertEquals("mismatches: ${failures.toString().take(2000)}", 0, failures.length())
  }

  private fun parse(array: JSONArray): List<Question> =
    List(array.length()) { i ->
      val q = array.getJSONObject(i)
      val options = q.optJSONArray("options")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList()
      when (Question.Kind.fromKey(q.getString("type"))) {
        Question.Kind.NOUL -> Question.noul(q.getString("instructions"))
        Question.Kind.CHOICE -> Question.choice(q.getString("instructions"), options)
        Question.Kind.SCORE -> Question.score(q.getString("instructions"), options)
      }
    }
}
