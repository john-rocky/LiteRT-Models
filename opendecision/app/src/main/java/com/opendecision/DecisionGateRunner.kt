package com.opendecision

import android.content.Context
import android.util.Log
import java.io.File
import java.util.Locale
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-only fixture gate: `files/gate_fixtures.json` holds requests with the ids, text spans, oracle logits and
 * answers the official implementation produced on the desktop. For every request the on-device tokenizer and
 * builder are compared with the captured ids and spans, then the graph runs on the requested backend and its
 * logits are compared with the oracle (same argmax per question, max probability difference at T = 1.05).
 * Reports go to `files/gate/`. Launch: `am start -n com.opendecision/.MainActivity --ez gate true --es accel GPU`.
 */
class DecisionGateRunner(private val context: Context, private val model: () -> DecisionModel) {
  data class Report(val path: String, val passed: Boolean, val error: String?)

  fun run(requestedBackend: String?): List<Report> {
    val fixtures = JSONObject(File(context.filesDir, FIXTURES_FILE).readText())
    val requests = fixtures.getJSONArray("requests")
    val backends =
      if (requestedBackend == null) listOf(DecisionModel.Backend.GPU)
      else listOf(DecisionModel.Backend.valueOf(requestedBackend.uppercase(Locale.ROOT)))
    val outDir = File(context.filesDir, "gate").apply { mkdirs() }
    val reports = ArrayList<Report>()
    for (backend in backends) {
      val rows = JSONArray()
      var idsEqual = 0
      var spansEqual = 0
      var questions = 0
      var argmaxEqual = 0
      var maxDp = 0.0
      var error: String? = null
      val graphMs = ArrayList<Double>()
      try {
        for (i in 0 until requests.length()) {
          val r = requests.getJSONObject(i)
          val questionList = parseQuestions(r.getJSONArray("questions"))
          val encoded = model().inputs.encode(r.getString("state"), questionList)
          val expectedIds = r.getJSONArray("ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
          val sameIds = encoded.inputIds.contentEquals(expectedIds)
          val expectedQ = r.getJSONArray("q_spans")
          val expectedO = r.getJSONArray("opt_spans")
          var sameSpans = encoded.questionSpans.size == expectedQ.length()
          for (q in 0 until minOf(expectedQ.length(), encoded.questionSpans.size)) {
            val e = expectedQ.getJSONArray(q)
            sameSpans = sameSpans && encoded.questionSpans[q] == DecisionInputs.Span(e.getInt(0), e.getInt(1))
            val eo = expectedO.getJSONArray(q)
            sameSpans = sameSpans && encoded.optionSpans[q].size == eo.length()
            for (o in 0 until minOf(eo.length(), encoded.optionSpans[q].size)) {
              val s = eo.getJSONArray(o)
              sameSpans = sameSpans && encoded.optionSpans[q][o] == DecisionInputs.Span(s.getInt(0), s.getInt(1))
            }
          }
          idsEqual += if (sameIds) 1 else 0
          spansEqual += if (sameSpans) 1 else 0
          val prepared = model().inputs.prepare(encoded)
          val (logits, ms) = model().run(prepared, backend)
          graphMs.add(ms)
          val answers = DecisionDecoder.decode(logits, questionList)
          val oracleLogits = r.getJSONArray("logits")
          val perQuestion = JSONArray()
          for ((qi, answer) in answers.withIndex()) {
            val ref = oracleLogits.getJSONArray(qi).let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            val refProbs = DecisionDecoder.softmax(ref, DecisionDecoder.TEMPERATURE)
            var dp = 0.0
            for (k in refProbs.indices) dp = maxOf(dp, abs(refProbs[k] - answer.probabilities[k]))
            val same = DecisionDecoder.argmax(refProbs) == answer.best
            questions++
            if (same) argmaxEqual++
            maxDp = maxOf(maxDp, dp)
            perQuestion.put(JSONObject().put("kind", answer.question.kind.key).put("same_argmax", same).put("max_dp", dp))
          }
          rows.put(
            JSONObject()
              .put("id", r.getString("id"))
              .put("ids_equal", sameIds)
              .put("spans_equal", sameSpans)
              .put("window", prepared.window)
              .put("graph_ms", ms)
              .put("questions", perQuestion)
          )
          if (i % 25 == 0) Log.i(LOG_TAG, "DECISION_APP_GATE $backend row $i / ${requests.length()}")
        }
      } catch (failure: Throwable) {
        error = failure.toString()
        Log.e(LOG_TAG, "DECISION_APP_GATE failed", failure)
      }
      val sorted = graphMs.drop(1).sorted()
      val passed = error == null && idsEqual == requests.length() && spansEqual == requests.length() && argmaxEqual == questions
      val summary =
        JSONObject()
          .put("backend", backend.name)
          .put("precision", backend.precision)
          .put("litert", DecisionModel.LITERT_VERSION)
          .put("requests", requests.length())
          .put("ids_equal", idsEqual)
          .put("spans_equal", spansEqual)
          .put("questions", questions)
          .put("argmax_equal", argmaxEqual)
          .put("max_dp", maxDp)
          .put("graph_ms_median_after_first", if (sorted.isEmpty()) JSONObject.NULL else sorted[sorted.size / 2])
          .put("passed", passed)
          .put("error", error ?: JSONObject.NULL)
      val file = File(outDir, "gate_${backend.name.lowercase(Locale.ROOT)}.json")
      file.writeText(JSONObject().put("summary", summary).put("rows", rows).toString())
      Log.i(LOG_TAG, "DECISION_APP_GATE $summary")
      reports.add(Report(file.absolutePath, passed, error))
    }
    return reports
  }

  private fun parseQuestions(array: JSONArray): List<Question> =
    List(array.length()) { i ->
      val q = array.getJSONObject(i)
      val kind = Question.Kind.fromKey(q.getString("type"))
      val options = q.optJSONArray("options")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList()
      when (kind) {
        Question.Kind.NOUL -> Question.noul(q.getString("instructions"))
        Question.Kind.CHOICE -> Question.choice(q.getString("instructions"), options)
        Question.Kind.SCORE -> Question.score(q.getString("instructions"), options)
      }
    }

  companion object {
    const val LOG_TAG = "DecisionGate"
    const val FIXTURES_FILE = "gate_fixtures.json"
  }
}
