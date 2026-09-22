package com.sopro

import java.io.File
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class CoreGateTest {
  private val f
    get() = FixtureData

  private fun objects(array: JSONArray) = (0 until array.length()).map { array.getJSONObject(it) }

  private fun integerArray(array: JSONArray) = IntArray(array.length()) { array.getInt(it) }

  @Test
  fun tokenizerGate() {
    val tokenizer = SpTokenizer(File(f.root, "../../hf/host_assets/tokenizer.model"))
    val metadata = f.index.getJSONObject("tokenizer")
    val sentenceRows = objects(metadata.getJSONArray("sentences"))
    val failures = ArrayList<Map<String, Any>>()
    var sentenceExact = 0
    for (row in sentenceRows) {
      val actual = tokenizer.encode(row.getString("text"), row.getString("lang"))
      val expected = f.ints(row.getString("key"))
      if (
        actual.contentEquals(expected) &&
          SpTokenizer.languageTag(row.getString("lang")) == row.getString("lang_tag")
      )
        sentenceExact++
      else
        failures +=
          mapOf(
            "id" to row.getString("id"),
            "expected" to expected.toList(),
            "actual" to actual.toList(),
          )
    }
    val stress = JSONArray(File(f.root, metadata.getString("stress_path")).readText())
    val totals = LinkedHashMap<String, Int>()
    for (lang in listOf("en", "pt", "fr", "de")) {
      var exact = 0
      for (row in objects(stress)) {
        val expected = integerArray(row.getJSONObject("encodings").getJSONArray(lang))
        val actual = tokenizer.encode(row.getString("text"), lang)
        if (actual.contentEquals(expected)) exact++
        else
          failures +=
            mapOf(
              "stress_id" to row.getInt("id"),
              "lang" to lang,
              "expected" to expected.toList(),
              "actual" to actual.toList(),
            )
      }
      totals[lang] = exact
    }
    val pass = sentenceExact == 14 && totals.values.all { it == 300 }
    f.metrics(
      "tokenizer",
      mapOf(
        "status" to if (pass) "PASS" else "FAIL",
        "sentences_exact" to sentenceExact,
        "sentences_total" to 14,
        "stress_exact_per_language" to totals,
        "stress_per_language" to 300,
        "model" to metadata.getJSONObject("details"),
        "failures" to failures,
      ),
    )
    assertTrue("Tokenizer mismatches: ${failures.size}", pass)
  }

  @Test
  fun samplerGate() {
    val rows = ArrayList<Map<String, Any>>()
    val disagreements = ArrayList<Map<String, Any>>()
    var predictions = 0
    var exact = 0
    var worst = 0.0
    for (row in objects(f.index.getJSONObject("sampler").getJSONArray("utterances"))) {
      val key = row.getString("key")
      val logits = f.floats("$key/logits")
      val expected = f.floats("$key/probabilities")
      val draws = f.doubles("$key/draws")
      val allow = f.ints("$key/allow_eos")
      val picks = f.ints("$key/picks")
      val distance = f.doubles("$key/cdf_boundary_distance")
      var rowWorst = 0.0
      var rowExact = 0
      for (i in draws.indices) {
        val probabilities =
          Sampler.probabilities(logits.copyOfRange(i * 4377, (i + 1) * 4377), allow[i] != 0)
        for (j in probabilities.indices) {
          assertTrue(probabilities[j].isFinite())
          rowWorst = maxOf(rowWorst, abs(probabilities[j].toDouble() - expected[i * 4377 + j]))
        }
        val pick = Sampler.inverseCdf(probabilities, draws[i])
        if (pick == picks[i]) {
          rowExact++
          exact++
        } else {
          disagreements +=
            mapOf(
              "id" to row.getString("id"),
              "prediction" to i,
              "draw" to draws[i],
              "expected" to picks[i],
              "actual" to pick,
              "cdf_boundary_distance" to distance[i],
              "near_tie" to (distance[i] <= 1e-6),
            )
        }
      }
      predictions += draws.size
      worst = maxOf(worst, rowWorst)
      rows +=
        mapOf(
          "id" to row.getString("id"),
          "predictions" to draws.size,
          "exact_picks" to rowExact,
          "probabilities_max_abs_diff" to rowWorst,
        )
    }
    // Exercise stop semantics independently of teacher replay (which always advances all fixed
    // tokens).
    val eos =
      FloatArray(4377) { -100f }
        .also {
          it[Sampler.EOS] = 100f
          it[3] = 0f
        }
    var calls = 0
    val stopped =
      Sampler.generate(
        eos,
        {
          calls++
          eos
        },
        UniformSource { .5 },
      )
    val stopCorrect =
      stopped.tokens.size == 9 &&
        stopped.drawCount == 10 &&
        stopped.stepCalls == 9 &&
        calls == 9 &&
        stopped.stopReason == "eos"
    val speech = FloatArray(4377) { -100f }.also { it[3] = 100f }
    calls = 0
    val limited =
      Sampler.generate(
        speech,
        {
          calls++
          speech
        },
        UniformSource { .5 },
      )
    val limitCorrect =
      limited.tokens.size == 704 &&
        limited.stepCalls == 703 &&
        calls == 703 &&
        limited.drawCount == 704
    val pass =
      worst <= 1e-6 && disagreements.all { it["near_tie"] == true } && stopCorrect && limitCorrect
    f.metrics(
      "sampler",
      mapOf(
        "status" to if (pass) "PASS" else "FAIL",
        "predictions" to predictions,
        "exact_picks" to exact,
        "probabilities_max_abs_diff" to worst,
        "near_ties" to disagreements,
        "eos_stop_semantics" to stopCorrect,
        "max_704_semantics" to limitCorrect,
        "rows" to rows,
      ),
    )
    assertTrue("Sampler bound or loop semantics failed", pass)
  }

  @Test
  fun arHostGate() {
    val (cosine, sine) = ArHost.rotaryCosSin()
    val cosDiff = f.maxDiff(cosine, f.floats("core/ar/cos"))
    val sinDiff = f.maxDiff(sine, f.floats("core/ar/sin"))
    val metadata = f.index.getJSONObject("ar")
    var prefillExact = 0
    var stepExact = 0
    var prefixExact = 0
    for (n in integerArray(metadata.getJSONArray("prefill_lengths"))) if (
      ArHost.prefillBias(n).contentEquals(f.floats("core/ar/prefill_bias_$n"))
    )
      prefillExact++
    for (p in integerArray(metadata.getJSONArray("step_positions"))) if (
      ArHost.stepBias(p).contentEquals(f.floats("core/ar/step_bias_$p"))
    )
      stepExact++
    val rows = ArrayList<Map<String, Any>>()
    ArHost(
        File(f.root, "../../exports/host_assets/ar_tables_fp32.bin"),
        File(f.root, "../../exports/host_assets/ar_tables_fp32.json"),
      )
      .use { host ->
        for (row in objects(metadata.getJSONArray("utterances"))) {
          val ref = row.getString("reference")
          val key = row.getString("key")
          val prefix =
            host.assemblePrefix(
              f.floats("core/ar/$ref/style"),
              f.ints("$key/text_ids"),
              f.ints("core/ar/$ref/ref_tokens"),
            )
          val expected = f.floats("$key/prefix")
          val same = prefix.length == row.getInt("length") && prefix.values.contentEquals(expected)
          if (same) prefixExact++
          rows +=
            mapOf(
              "id" to row.getString("id"),
              "length" to prefix.length,
              "bit_exact" to same,
              "max_abs_diff" to f.maxDiff(prefix.values, expected),
            )
        }
      }
    // Head-stride indexing is the principal packed-cache hazard: preserve all preceding rows.
    val cache = ArHost.PackedKv()
    val k = FloatArray(96 * 256 * 64) { (it % 1021).toFloat() }
    val v = FloatArray(k.size) { -(it % 997).toFloat() }
    cache.initialize(k, v, 150)
    val kr = FloatArray(96 * 64) { 2000f + it }
    val vr = FloatArray(kr.size) { -2000f - it }
    cache.writeRow(kr, vr)
    var cacheCorrect = cache.position == 151
    for (head in 0 until 96) for (d in 0 until 64) {
      cacheCorrect =
        cacheCorrect && cache.keys[(head * 1024 + 149) * 64 + d] == k[(head * 256 + 149) * 64 + d]
      cacheCorrect =
        cacheCorrect && cache.values[(head * 1024 + 149) * 64 + d] == v[(head * 256 + 149) * 64 + d]
      cacheCorrect = cacheCorrect && cache.keys[(head * 1024 + 150) * 64 + d] == kr[head * 64 + d]
      cacheCorrect = cacheCorrect && cache.values[(head * 1024 + 150) * 64 + d] == vr[head * 64 + d]
      cacheCorrect = cacheCorrect && cache.keys[(head * 1024 + 151) * 64 + d] == 0f
    }
    val pass =
      cosDiff <= 1e-6 &&
        sinDiff <= 1e-6 &&
        prefillExact == 4 &&
        stepExact == 4 &&
        prefixExact == 24 &&
        cacheCorrect
    f.metrics(
      "ar_host",
      mapOf(
        "status" to if (pass) "PASS" else "FAIL",
        "rope_cos_max_abs_diff" to cosDiff,
        "rope_sin_max_abs_diff" to sinDiff,
        "prefill_bias_exact" to prefillExact,
        "step_bias_exact" to stepExact,
        "prefix_bit_exact" to prefixExact,
        "packed_cache_stride_correct" to cacheCorrect,
        "rows" to rows,
      ),
    )
    assertTrue("AR host contract mismatch", pass)
  }

  @Test
  fun acousticHostGate() {
    val metadata = f.index.getJSONObject("acoustic")
    val rows = ArrayList<Map<String, Any>>()
    val prepared = HashMap<String, AcousticHost.Prepared>()
    var exact = 0
    for (row in objects(metadata.getJSONArray("utterances"))) {
      val key = row.getString("key")
      val p =
        AcousticHost.prepareInputs(
          f.ints("$key/ref_semantic_tokens"),
          f.ints("$key/sampled_semantic_tokens"),
          f.floats("$key/x0"),
          f.floats("$key/cond_vec"),
          f.floats("$key/ref_mel_normalized"),
          row.getInt("prompt_frames"),
        )
      prepared[key] = p
      val same =
        p.frameToToken.contentEquals(f.ints("$key/frame_to_token")) &&
          p.tokenMask.contentEquals(f.floats("$key/token_mask")) &&
          p.condMask.contentEquals(f.floats("$key/cond_mask")) &&
          p.keyBias.contentEquals(f.floats("$key/key_bias")) &&
          p.tokens.contentEquals(f.ints("$key/semantic_tokens")) &&
          p.x.contentEquals(f.floats("$key/x")) &&
          p.condMel.contentEquals(f.floats("$key/cond_mel")) &&
          p.validFrames == row.getInt("valid_frames") &&
          p.validTokens == row.getInt("valid_tokens") &&
          p.frames == row.getInt("frames")
      if (same) exact++
      rows +=
        mapOf(
          "id" to row.getString("id"),
          "exact" to same,
          "bucket_frames" to p.frames,
          "valid_frames" to p.validFrames,
          "valid_tokens" to p.validTokens,
        )
    }
    var eulerWorst = 0.0
    val eulerRows = ArrayList<Map<String, Any>>()
    for (row in objects(metadata.getJSONArray("euler"))) {
      val key = row.getString("key")
      val p = prepared.getValue(row.getString("input_key"))
      val actual =
        AcousticHost.eulerUpdate(
          f.floats("$key/x"),
          f.floats("$key/velocity"),
          p.x,
          p.condMel,
          p.condMask,
          row.getDouble("t0").toFloat(),
          row.getDouble("t1").toFloat(),
        )
      val diff = f.maxDiff(actual, f.floats("$key/expected"))
      eulerWorst = maxOf(eulerWorst, diff)
      eulerRows +=
        mapOf(
          "id" to row.getString("id"),
          "t0" to row.getDouble("t0"),
          "t1" to row.getDouble("t1"),
          "max_abs_diff" to diff,
        )
    }
    val gridDiff = f.maxDiff(AcousticHost.timeGrid(), f.floats("core/acoustic/time_grid"))
    val pass = exact == 28 && eulerWorst <= 1e-6 && gridDiff <= 1e-6
    f.metrics(
      "acoustic_host",
      mapOf(
        "status" to if (pass) "PASS" else "FAIL",
        "exact_utterances" to exact,
        "total_utterances" to 28,
        "euler_max_abs_diff" to eulerWorst,
        "euler_utterances" to 2,
        "time_grid_max_abs_diff" to gridDiff,
        "rows" to rows,
        "euler_rows" to eulerRows,
      ),
    )
    assertTrue("Acoustic host mismatch", pass)
  }
}
