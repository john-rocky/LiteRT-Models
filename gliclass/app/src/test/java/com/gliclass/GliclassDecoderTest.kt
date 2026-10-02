package com.gliclass

import kotlin.math.abs
import kotlin.math.nextDown
import kotlin.math.nextUp
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** `_postprocess_logits` semantics on the oracle logits, plus the tie/threshold/repeat rules. */
class GliclassDecoderTest {
  @Test
  fun oracleLogitsGiveTheOfficialResultsForEveryRequest() {
    val fixtures = OracleFixtures.load(ExternalTestData.resolve())
    assertEquals(552, fixtures.size)
    val failures = JSONArray()
    var passed = 0
    var maxScore = 0.0
    var maxProbability = 0.0
    for (fixture in fixtures) {
      // Slots past the labels hold scorer(0) in the graph; NaN proves they are never read.
      val slots = FloatArray(GliclassInputs.LABEL_SLOTS) { Float.NaN }
      fixture.logits.copyInto(slots)
      val single = GliclassDecoder.decide(slots, fixture.labels, GliclassDecoder.Mode.SINGLE_LABEL)
      val multi =
        GliclassDecoder.decide(slots, fixture.labels, GliclassDecoder.Mode.MULTI_LABEL, 0.5)
      val checks =
        mutableListOf(
          compare(single, fixture.single),
          compare(multi, fixture.multi),
        )
      fixture.multiThresholdZero?.let {
        checks.add(
          compare(
            GliclassDecoder.decide(slots, fixture.labels, GliclassDecoder.Mode.MULTI_LABEL, 0.0),
            it,
          )
        )
      }
      val probability =
        maxOf(
          maxAbs(single.probabilities, fixture.softmax),
          maxAbs(multi.probabilities, fixture.sigmoid),
        )
      maxProbability = maxOf(maxProbability, probability)
      maxScore = maxOf(maxScore, checks.maxOf { it.second })
      if (checks.all { it.first } && checks.all { it.second <= SCORE_TOLERANCE }) {
        passed++
      } else {
        failures.put(
          JSONObject()
            .put("id", fixture.id)
            .put("labels_equal", JSONArray(checks.map { it.first }))
            .put("max_abs_dscore", checks.maxOf { it.second })
            .put("single", JSONArray(single.predictions.map { "${it.label}=${it.score}" }))
            .put("multi", JSONArray(multi.predictions.map { "${it.label}=${it.score}" }))
        )
      }
    }
    ExternalTestData.reportFile("decoder_parity.json")
      .writeText(
        JSONObject()
          .put("test", "GliclassDecoderTest")
          .put("fixtures", fixtures.size)
          .put("passed", passed)
          .put("max_abs_dscore_vs_pipeline", maxScore)
          .put("max_abs_dprob_vs_oracle", maxProbability)
          .put("score_tolerance", SCORE_TOLERANCE)
          .put("input", "oracle logits (official fp32), slots past the labels = NaN")
          .put("failures", failures)
          .toString(1) + "\n"
      )
    println(
      "GLICLASS_DECODER passed=$passed/${fixtures.size} max_dscore=$maxScore" +
        " max_dprob=$maxProbability"
    )
    assertEquals("requests whose decisions match the pipeline: $failures", fixtures.size, passed)
  }

  @Test
  fun thresholdComparesFloatProbabilityWithDoubleThreshold() {
    // 0.7f is the float just below 0.7: Python rejects it at threshold 0.7, a Float threshold
    // would accept it.
    val logit = findLogitWithSigmoid(0.7f)
    val decision =
      GliclassDecoder.decide(
        floatArrayOf(logit, 2f),
        listOf("a", "b"),
        GliclassDecoder.Mode.MULTI_LABEL,
        0.7,
      )
    assertEquals(0.7f, decision.probabilities[0])
    assertTrue(0.7f.toDouble() < 0.7)
    assertEquals(listOf("b"), decision.predictions.map { it.label })
    // Nothing reaches the threshold: the pipeline returns no label.
    val none =
      GliclassDecoder.decide(
        floatArrayOf(-5f, -6f),
        listOf("a", "b"),
        GliclassDecoder.Mode.MULTI_LABEL,
      )
    assertTrue(none.predictions.isEmpty())
    assertTrue(none.chosen.isEmpty())
  }

  @Test
  fun tiesKeepTheFirstLabelAndRepeatedLabelsFollowThePipelineDict() {
    val tie =
      GliclassDecoder.decide(
        floatArrayOf(1f, 1f, 0f),
        listOf("x", "y", "z"),
        GliclassDecoder.Mode.SINGLE_LABEL,
      )
    assertEquals(listOf("x"), tie.predictions.map { it.label })
    // {label: score}: "a" keeps its first position with the last score.
    val repeated =
      GliclassDecoder.decide(
        floatArrayOf(3f, 1f, -3f),
        listOf("a", "b", "a"),
        GliclassDecoder.Mode.MULTI_LABEL,
        0.5,
      )
    assertEquals(listOf("b"), repeated.predictions.map { it.label })
    assertEquals(listOf(0, 1), repeated.chosen)
    val softmax = GliclassDecoder.softmax(floatArrayOf(0f, 0f))
    assertTrue(softmax.all { it == 0.5f })
  }

  private fun compare(
    decision: GliclassDecoder.Decision,
    expected: List<OracleFixtures.Prediction>,
  ): Pair<Boolean, Double> {
    val labelsEqual = decision.predictions.map { it.label } == expected.map { it.label }
    val score =
      if (labelsEqual) {
        decision.predictions.zip(expected).maxOfOrNull { (a, b) ->
          abs(a.score.toDouble() - b.score)
        } ?: 0.0
      } else {
        Double.POSITIVE_INFINITY
      }
    return labelsEqual to score
  }

  private fun maxAbs(actual: FloatArray, expected: FloatArray): Double =
    expected.indices.maxOf { abs(actual[it].toDouble() - expected[it].toDouble()) }

  private fun findLogitWithSigmoid(target: Float): Float {
    var candidate = 0.8472978f
    repeat(4096) {
      val probability = GliclassDecoder.sigmoid(floatArrayOf(candidate))[0]
      if (probability == target) {
        return candidate
      }
      candidate = if (probability < target) candidate.nextUp() else candidate.nextDown()
    }
    error("No float logit maps to $target")
  }

  private companion object {
    const val SCORE_TOLERANCE = 1e-6
  }
}
