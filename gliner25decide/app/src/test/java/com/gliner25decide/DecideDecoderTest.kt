package com.gliner25decide

import kotlin.math.nextDown
import kotlin.math.nextUp
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * host_decide.py semantics on the oracle logits, plus the edge rules that decide ties/thresholds.
 */
class DecideDecoderTest {
  @Test
  fun oracleLogitsGiveOfficialDecisionsForEveryFixture() {
    val root = ExternalTestData.resolve()
    val fixtures = OracleFixtures.load(root)
    assertEquals(361, fixtures.size)
    val failures = JSONArray()
    var passed = 0
    var maxProbability = 0.0
    var multiLabelHeads = 0
    for (fixture in fixtures) {
      // Slots past the last label hold classifier(0) in the graph; NaN proves they are unread.
      val slots = FloatArray(DecideInputs.LABEL_SLOTS) { Float.NaN }
      fixture.logits.copyInto(slots)
      val decisions = DecideDecoder.decode(slots, fixture.tasks)
      val official = DecideGateFixtures.decisionsEqual(decisions, fixture.official)
      val recorded = decisions.map { it.labels } == fixture.taskResults.map { it.decision }
      val probability =
        DecideGateFixtures.maxAbsDifference(
          DecideGateFixtures.probabilities(decisions),
          fixture.probabilities,
        )
      maxProbability = maxOf(maxProbability, probability)
      multiLabelHeads += fixture.tasks.count { it.multiLabel }
      if (official && recorded && probability <= PROBABILITY_TOLERANCE) {
        passed++
      } else {
        failures.put(
          JSONObject()
            .put("id", fixture.id)
            .put("decisions_equal_official", official)
            .put("decisions_equal_recorded", recorded)
            .put("max_abs_dprob", probability)
            .put("kotlin", DecideGateFixtures.decisionsJson(decisions))
        )
      }
    }
    val report =
      JSONObject()
        .put("test", "DecideDecoderTest")
        .put("fixtures", fixtures.size)
        .put("passed", passed)
        .put("failed", fixtures.size - passed)
        .put("multi_label_heads", multiLabelHeads)
        .put("max_abs_dprob_vs_oracle", maxProbability)
        .put("probability_tolerance", PROBABILITY_TOLERANCE)
        .put("input", "oracle task_results logits (official fp32), slots past the labels = NaN")
        .put("failures", failures)
    ExternalTestData.reportFile("decide_decoder_parity.json").writeText(report.toString(2) + "\n")
    println("DECIDE_DECODER passed=$passed/${fixtures.size} max_dprob=$maxProbability")
    assertEquals(fixtures.size, passed)
  }

  @Test
  fun thresholdComparesFloatProbabilityWithDoubleThreshold() {
    // 0.7f is the float just below 0.7 (0.699999988…): Python rejects it at cls_threshold 0.7,
    // a Float threshold would accept it.
    val logit = findLogitWithSigmoid(0.7f)
    val task = Task("t", listOf("a", "b"), multiLabel = true, clsThreshold = 0.7)
    val decision = DecideDecoder.decide(floatArrayOf(logit, 2f), task)
    assertEquals(0.7f, decision.probabilities[0])
    assertTrue(0.7f.toDouble() < 0.7)
    assertEquals(listOf("b"), decision.labels)
    // Nothing reaches 0.7: gliner2 falls back to the argmax label alone.
    val none = DecideDecoder.decide(floatArrayOf(logit, -20f), task)
    assertEquals(listOf("a"), none.labels)
    val lower = DecideDecoder.decide(floatArrayOf(logit, 2f), task.copy(clsThreshold = 0.6))
    assertEquals(listOf("a", "b"), lower.labels)
  }

  @Test
  fun argmaxKeepsTheFirstMaximumAndActivationCanBeForced() {
    val tie = DecideDecoder.decide(floatArrayOf(1f, 1f, 0f), Task("t", listOf("x", "y", "z")))
    assertEquals(listOf("x"), tie.labels)
    val forced =
      DecideDecoder.decide(
        floatArrayOf(0f, 2f),
        Task("t", listOf("x", "y"), activation = Activation.SIGMOID),
      )
    assertEquals(1f / (1f + kotlin.math.exp(-2.0).toFloat()), forced.probabilities[1])
    assertEquals(listOf("y"), forced.labels)
    val softmax = DecideDecoder.softmax(floatArrayOf(0f, 0f))
    assertTrue(softmax.all { it == 0.5f })
  }

  private fun findLogitWithSigmoid(target: Float): Float {
    var candidate = 0.8472978f
    repeat(4096) {
      val probability = DecideDecoder.sigmoid(floatArrayOf(candidate))[0]
      if (probability == target) {
        return candidate
      }
      candidate = if (probability < target) candidate.nextUp() else candidate.nextDown()
    }
    error("No float logit maps to $target")
  }

  companion object {
    const val PROBABILITY_TOLERANCE = 1e-6
  }
}
