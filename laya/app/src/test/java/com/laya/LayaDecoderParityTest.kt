// SPDX-License-Identifier: Apache-2.0
package com.laya

import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LayaDecoderParityTest {
  @Test
  fun everyCapturedOfficialAnswerMatchesExactlyAtIdentityTemperature() {
    val details = mutableListOf<Map<String, Any?>>()
    val failures = mutableListOf<Map<String, Any?>>()
    var maxProbabilityError = 0.0
    for (window in listOf(256, 512)) {
      val rows = LayaTestData.rows(window)
      assertEquals(201, rows.size)
      for (row in rows) {
        val question = LayaPromptBuilder.normalize(LayaJson.asObject(row["question"]))
        val raw = LayaTestData.floats(row["raw_logits"])
        val act = LayaTestData.floats(row["raw_act_logits"])
        assertEquals((row["K"] as Number).toInt(), question.optionCount)
        assertEquals((row["qtype"] as Number).toInt(), question.qtype)
        val actual = LayaDecoder.decode(raw, act, question)
        val difference = LayaTestData.difference(row["official_answer"], actual)
        val expectedP = LayaTestData.floats(row["probabilities"])
        val p = LayaDecoder.probabilities(raw, question)
        for (i in p.indices) maxProbabilityError =
          maxOf(maxProbabilityError, abs(p[i].toDouble() - expectedP[i].toDouble()))
        val detail =
          linkedMapOf<String, Any?>(
            "row_id" to row["row_id"],
            "window" to window,
            "exact" to (difference == null),
            "finite" to
              (p.all { it.isFinite() } && raw.all { it.isFinite() } && act.all { it.isFinite() }),
            "actual_answer" to actual,
          )
        if (difference != null) {
          detail["difference"] = difference
          detail["expected_answer"] = row["official_answer"]
          detail["probabilities"] = p
          failures.add(detail)
        }
        details.add(detail)
      }
    }
    LayaTestData.report(
      "jvm_decoder_official_parity.json",
      linkedMapOf(
        "status" to if (failures.isEmpty()) "PASS" else "FAIL",
        "rows" to details.size,
        "exact" to details.size - failures.size,
        "all_finite" to details.all { it["finite"] == true },
        "max_abs_probability_error_unrounded" to maxProbabilityError,
        "failures" to failures,
        "details" to details,
      ),
    )
    assertEquals(402, details.size)
    assertTrue("Non-finite host arithmetic", details.all { it["finite"] == true })
    LayaTestData.assertNoFailures("Official decoder parity", failures)
  }

  @Test
  fun everyCalibratedAnswerMatchesPythonHostExactly() {
    val calibration =
      LayaCalibration.load(LayaTestData.required("host_assets/laya_ml_calibration.json"))
    val references =
      LayaJson.asArray(
          LayaJson.asObject(LayaTestData.read("fixtures/calibrated_reference.json"))["rows"]
        )
        .map(LayaJson::asObject)
    assertEquals(402, references.size)
    val captured =
      listOf(256, 512).flatMap(LayaTestData::rows).associateBy { "${it["window"]}/${it["row_id"]}" }
    val failures = mutableListOf<Map<String, Any?>>()
    var maxProbabilityError = 0.0
    var maxFeatureError = 0.0
    val details =
      references.map { reference ->
        val key = "${reference["window"]}/${reference["row_id"]}"
        val row = captured[key] ?: error("Missing captured raw logits for $key")
        val question = LayaPromptBuilder.normalize(LayaJson.asObject(row["question"]))
        val raw = LayaTestData.floats(row["raw_logits"])
        val actual =
          LayaDecoder.decode(raw, LayaTestData.floats(row["raw_act_logits"]), question, calibration)
        val p = LayaDecoder.probabilities(raw, question, calibration)
        val features = LayaDecoder.actFeatures(raw)
        val expectedP = LayaTestData.floats(reference["probabilities"])
        for (i in p.indices) maxProbabilityError =
          maxOf(maxProbabilityError, abs(p[i].toDouble() - expectedP[i].toDouble()))
        val featureList = LayaJson.asArray(reference["act_features"])
        val expectedFeatures =
          LayaTestData.floats(
            if (featureList.firstOrNull() is List<*>) featureList[0] else featureList
          )
        for (i in features.indices) maxFeatureError =
          maxOf(maxFeatureError, abs(features[i].toDouble() - expectedFeatures[i].toDouble()))
        val difference = LayaTestData.difference(reference["answer"], actual)
        val detail =
          linkedMapOf<String, Any?>(
            "row_id" to row["row_id"],
            "window" to row["window"],
            "exact" to (difference == null),
            "finite" to (p.all { it.isFinite() } && features.all { it.isFinite() }),
            "actual_answer" to actual,
          )
        if (difference != null) {
          detail["difference"] = difference
          detail["expected_answer"] = reference["answer"]
          detail["probabilities"] = p
          failures.add(detail)
        }
        detail
      }
    LayaTestData.report(
      "jvm_decoder_calibrated_parity.json",
      linkedMapOf(
        "status" to if (failures.isEmpty()) "PASS" else "FAIL",
        "rows" to details.size,
        "exact" to details.size - failures.size,
        "all_finite" to details.all { it["finite"] == true },
        "max_abs_probability_error_unrounded" to maxProbabilityError,
        "max_abs_act_feature_error" to maxFeatureError,
        "failures" to failures,
        "details" to details,
      ),
    )
    assertTrue("Non-finite calibrated arithmetic", details.all { it["finite"] == true })
    LayaTestData.assertNoFailures("Calibrated decoder parity", failures)
  }

  @Test
  fun bucketPrecedenceTemperatureFloorAndFirstTieFollowContract() {
    val calibration = LayaCalibration(doubleArrayOf(2.0, 3.0, 4.0), mapOf("choice:2" to 0.0))
    assertEquals(0.0, calibration.temperature("choice", 2), 0.0)
    assertEquals(2.0, calibration.temperature("choice", 12), 0.0)
    assertEquals("choice:6-10", LayaCalibration.bucket("choice", 6))
    val q = LayaQuestion("choice", "", linkedMapOf("first" to null, "second" to null))
    val p = LayaDecoder.probabilities(floatArrayOf(0f, 0.001f), q, calibration)
    assertEquals(0.7310586f.toDouble(), p[1].toDouble(), 1e-7)
    val tied = LayaDecoder.decode(floatArrayOf(0f, 0f), floatArrayOf(0f, 0f), q)
    assertEquals("first", tied["choice"])
    assertEquals(0.0, tied["confidence"])
    assertEquals(1.0, LayaDecoder.confidence(floatArrayOf(1f)), 0.0)
  }

  @Test
  fun roundingUsesTheExactBinaryValueAndKeepsNegativeZero() {
    assertEquals(0.0624, LayaDecoder.round4(0.06245), 0.0)
    assertEquals(0.0625, LayaDecoder.round4(Math.nextUp(0.06245)), 0.0)
    assertEquals(0.0625, LayaDecoder.round4(0.0625), 0.0)
    assertEquals(
      java.lang.Double.doubleToRawLongBits(-0.0),
      java.lang.Double.doubleToRawLongBits(LayaDecoder.round4(-0.000001)),
    )
  }
}
