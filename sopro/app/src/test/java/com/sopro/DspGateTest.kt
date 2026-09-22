package com.sopro

import kotlin.math.abs
import kotlin.math.sqrt
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test

internal object DspGateSupport {
  val metadata: JSONObject
    get() = FixtureData.index.getJSONObject("dsp")

  fun host(): HostDsp {
    val names =
      listOf(
        "speaker_window",
        "speaker_melbank",
        "semantic_window",
        "semantic_melbank",
        "acoustic_window",
        "acoustic_melbank",
        "istft_window",
        "resample_24_16_kernel",
        "mel_mean",
        "mel_std",
      )
    return HostDsp(
      names.associateWith { FixtureData.floats("dsp/constants/$it") },
      FixtureData.ints("dsp/constants/resample_24_16_width")[0],
    )
  }

  fun compare(expected: FloatArray, actual: FloatArray): Map<String, Any> {
    if (actual.size != expected.size)
      return mapOf(
        "same_length" to false,
        "expected_samples" to expected.size,
        "actual_samples" to actual.size,
        "finite" to actual.all { it.isFinite() },
      )
    var difference = 0.0
    var scale = 0.0
    var square = 0.0
    var sx = 0.0
    var sy = 0.0
    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for (i in expected.indices) {
      val x = expected[i].toDouble()
      val y = actual[i].toDouble()
      difference = maxOf(difference, abs(x - y))
      scale = maxOf(scale, abs(x))
      square += y * y
      sx += x
      sy += y
      sxx += x * x
      syy += y * y
      sxy += x * y
    }
    val n = expected.size.toDouble()
    val denominator = sqrt(maxOf(0.0, sxx - sx * sx / n) * maxOf(0.0, syy - sy * sy / n))
    val corr =
      if (denominator > 0) (sxy - sx * sy / n) / denominator
      else if (difference == 0.0) 1.0 else 0.0
    return mapOf(
      "same_length" to true,
      "count" to actual.size,
      "finite" to actual.all { it.isFinite() },
      "max_abs_diff" to difference,
      "corr" to corr,
      "expected_absmax" to scale,
      "actual_rms" to sqrt(square / n),
    )
  }

  fun passed(metric: Map<String, Any>, maximum: Double, minimumCorr: Double = -1.0): Boolean =
    metric["same_length"] == true &&
      metric["finite"] == true &&
      (metric["max_abs_diff"] as Double) <= maximum &&
      (metric["corr"] as Double) >= minimumCorr

  fun save(
    gate: String,
    rows: List<Map<String, Any?>>,
    pass: Boolean,
    informational: Boolean = false,
  ) {
    val report =
      linkedMapOf<String, Any?>(
        "status" to if (pass) "PASS" else "FAIL",
        "runtime" to "JVM",
        "device" to "Mac desktop JVM; timings contended",
        "informational" to informational,
        "rows" to rows,
      )
    FixtureData.metrics(gate, report)
    println(JSONObject(report).toString())
  }
}

class HostDspTest {
  private fun refs() = DspGateSupport.metadata.getJSONArray("references")

  @Test
  fun resample() {
    val dsp = DspGateSupport.host()
    val references = refs()
    val rows = mutableListOf<Map<String, Any?>>()
    var allPass = references.length() == 3
    for (i in 0 until references.length()) {
      val ref = references.getJSONObject(i)
      val keys = ref.getJSONObject("arrays")
      val actual = dsp.resample24to16(FixtureData.floats(keys.getString("reference_wav24")))
      val metric =
        DspGateSupport.compare(FixtureData.floats(keys.getString("reference_wav16")), actual)
      val pass = DspGateSupport.passed(metric, 1e-5)
      allPass = allPass && pass
      rows.add(mapOf("reference" to ref.getString("id"), "metrics" to metric, "pass" to pass))
      FixtureData.writeFloats(ref.getString("id") + "_reference_wav16", actual)
    }
    DspGateSupport.save("resample", rows, allPass)
    assertTrue("24→16 kHz max error must be <= 1e-5 for 3 references", allPass)
  }

  @Test
  fun speakerMel() {
    melGate("speaker")
  }

  @Test
  fun semanticMel() {
    melGate("semantic")
  }

  @Test
  fun acousticMel() {
    melGate("acoustic")
  }

  @Test
  fun normalizeFullHero() {
    val dsp = DspGateSupport.host()
    val entries = DspGateSupport.metadata.getJSONArray("normalization")
    val rows = mutableListOf<Map<String, Any?>>()
    var allFinite = entries.length() == 1
    for (i in 0 until entries.length()) {
      val entry = entries.getJSONObject(i)
      val raw = FixtureData.floats(entry.getString("raw"))
      val expected = FixtureData.floats(entry.getString("normalized"))
      val actual = dsp.normalizeReference(raw, entry.getInt("sample_rate_hz"))
      val metric = DspGateSupport.compare(expected, actual.wav)
      val fixed = dsp.fixedReference(raw)
      val bucketExact =
        fixed.wav.contentEquals(actual.wav.copyOf(240000)) && fixed.levelDb == actual.levelDb
      allFinite =
        allFinite && metric["finite"] == true && metric["same_length"] == true && bucketExact
      rows.add(
        mapOf(
          "reference" to entry.getString("id"),
          "metrics" to metric,
          "level_db" to actual.levelDb,
          "python_level_db" to entry.getDouble("level_db"),
          "level_db_abs_diff" to abs(actual.levelDb - entry.getDouble("level_db")),
          "normalization_before_bucket_exact" to bucketExact,
          "scope" to
            "Numerical differences informational; round 1 defines no normalization tolerance",
        )
      )
    }
    DspGateSupport.save("normalize_reference", rows, allFinite, true)
    assertTrue("Full-clip normalization must be finite and run before the fixed bucket", allFinite)
  }

  private fun melGate(kind: String) {
    val dsp = DspGateSupport.host()
    val references = refs()
    val rows = mutableListOf<Map<String, Any?>>()
    var allPass = references.length() == 3
    for (i in 0 until references.length()) {
      val ref = references.getJSONObject(i)
      val keys = ref.getJSONObject("arrays")
      val acoustic = kind == "acoustic"
      val input =
        FixtureData.floats(keys.getString(if (acoustic) "reference_wav24" else "reference_wav16"))
      val actual = if (acoustic) dsp.acousticMelNormalized(input) else dsp.mel(input, kind)
      val outputName = if (acoustic) "ref_mel_normalized" else kind + "_mel"
      val expected = FixtureData.floats(keys.getString(outputName))
      val metric = DspGateSupport.compare(expected, actual)
      val pass =
        if (acoustic) DspGateSupport.passed(metric, .02, .99999)
        else metric["same_length"] == true && metric["finite"] == true
      allPass = allPass && pass
      val frames = actual.size / if (acoustic) 100 else 80
      var worst = 0
      var error = 0f
      for (p in actual.indices) {
        val d = abs(actual[p] - expected[p])
        if (d > error) {
          error = d
          worst = p / frames
        }
      }
      rows.add(
        mapOf(
          "reference" to ref.getString("id"),
          "metrics" to metric,
          "worst_band" to worst,
          "pass" to pass,
          "downstream_gate" to
            if (acoustic) "6 teacher-forced streaming waveforms" else "$kind fp32 LiteRT graph",
        )
      )
      FixtureData.writeFloats(ref.getString("id") + "_" + outputName, actual)
    }
    DspGateSupport.save(kind + "_mel", rows, allPass, kind != "acoustic")
    assertTrue("$kind mel gate, downstream gates are separately reported", allPass)
  }
}

class StreamingIstftTest {
  @Test
  fun replayActualGraphInvocations() {
    val entries = DspGateSupport.metadata.getJSONArray("static_stream_replay")
    val rows = mutableListOf<Map<String, Any?>>()
    var allPass = entries.length() == 2
    for (i in 0 until entries.length()) {
      val entry = entries.getJSONObject(i)
      val calls = entry.getJSONArray("calls")
      val retained = entry.getJSONArray("retained")
      var callIndex = 0
      var outputIndex = 0
      var exact = true
      StaticStreamFeatures.run(
        FixtureData.floats(entry.getString("mel")),
        entry.getInt("frames"),
        invoke = { mode, actual ->
          val call = calls.getJSONObject(callIndex++)
          exact = exact && mode == call.getString("mode")
          val inputs = call.getJSONArray("inputs")
          exact = exact && actual.size == inputs.length()
          for (j in 0 until inputs.length()) exact =
            exact && actual[j].contentEquals(FixtureData.floats(inputs.getString(j)))
          val outputs = call.getJSONArray("outputs")
          List(outputs.length()) { FixtureData.floats(outputs.getString(it)) }
        },
        emit = { actual, flush ->
          val expected = retained.getJSONObject(outputIndex++)
          exact =
            exact &&
              flush == expected.getBoolean("flush") &&
              actual.contentEquals(FixtureData.floats(expected.getString("features")))
        },
      )
      exact = exact && callIndex == calls.length() && outputIndex == retained.length()
      allPass = allPass && exact
      rows.add(
        mapOf(
          "utterance" to entry.getString("id"),
          "real_frames" to entry.getInt("frames"),
          "remainder" to entry.getInt("frames") % 64,
          "graph_calls" to callIndex,
          "retained_chunks" to outputIndex,
          "all_input_state_and_feature_bits_exact" to exact,
          "pass" to exact,
        )
      )
    }
    DspGateSupport.save("static_stream_driver", rows, allPass)
    assertTrue(
      "Exact graph invocation/state/tail replay for partial and full final chunks",
      allPass,
    )
  }

  @Test
  fun chunkedIstft() {
    val rows = mutableListOf<Map<String, Any?>>()
    val entries = DspGateSupport.metadata.getJSONArray("streaming_istft")
    var allPass = entries.length() == 24
    for (i in 0 until entries.length()) {
      val entry = entries.getJSONObject(i)
      val istft = StreamingIstft(FixtureData.floats("dsp/constants/istft_window"))
      val chunks = entry.getJSONArray("chunks")
      val output = mutableListOf<FloatArray>()
      val chunkRows = mutableListOf<Map<String, Any?>>()
      var pass = true
      for (j in 0 until chunks.length()) {
        val chunk = chunks.getJSONObject(j)
        val features = FixtureData.floats(chunk.getString("features"))
        val actual = istft.process(features, chunk.getBoolean("flush"))
        val metric = DspGateSupport.compare(FixtureData.floats(chunk.getString("waveform")), actual)
        val chunkPass = DspGateSupport.passed(metric, 1e-5)
        chunkRows.add(
          mapOf(
            "chunk" to j,
            "frames" to chunk.getInt("frames"),
            "metrics" to metric,
            "pass" to chunkPass,
          )
        )
        output.add(actual)
        pass = pass && chunkPass
      }
      val concatenated = FloatArray(output.sumOf { it.size })
      var offset = 0
      for (chunk in output) {
        chunk.copyInto(concatenated, offset)
        offset += chunk.size
      }
      val metric =
        DspGateSupport.compare(FixtureData.floats(entry.getString("total")), concatenated)
      pass = pass && DspGateSupport.passed(metric, 1e-5)
      allPass = allPass && pass
      rows.add(
        mapOf(
          "utterance" to entry.getString("id"),
          "metrics" to metric,
          "chunks" to chunkRows,
          "pass" to pass,
        )
      )
    }
    DspGateSupport.save("streaming_istft", rows, allPass)
    assertTrue(
      "24 streaming iSTFT utterances must preserve chunk/total lengths and max error <= 1e-5",
      allPass,
    )
  }
}

class PostProcessTest {
  @Test
  fun recordedPostprocess() {
    val rows = mutableListOf<Map<String, Any?>>()
    val entries = DspGateSupport.metadata.getJSONArray("postprocess")
    var allPass = entries.length() == 24
    for (i in 0 until entries.length()) {
      val entry = entries.getJSONObject(i)
      val actual =
        PostProcess.postprocessSegment(
          FixtureData.floats(entry.getString("raw")),
          entry.getDouble("reference_level_db"),
        )
      val metric = DspGateSupport.compare(FixtureData.floats(entry.getString("final")), actual.wav)
      val trim = entry.getJSONObject("trim")
      val exact =
        actual.trim.leadCutSamples == trim.getInt("lead_cut_samples") &&
          actual.trim.trailEndAfterLeadSamples == trim.getInt("trail_end_after_lead_samples") &&
          actual.trim.finalSamples == trim.getInt("final_samples")
      val pass = exact && DspGateSupport.passed(metric, 1e-6)
      allPass = allPass && pass
      rows.add(
        mapOf(
          "utterance" to entry.getString("id"),
          "trim_exact" to exact,
          "trim" to actual.trim.asMap(),
          "metrics" to metric,
          "pass" to pass,
        )
      )
    }
    DspGateSupport.save("postprocess", rows, allPass)
    assertTrue("24 postprocessing waveforms must match trim points and max error <= 1e-6", allPass)
  }
}
