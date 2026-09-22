package com.sopro

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test

class R6ContractTest {
  @Test
  fun exactOneHotAndHostSemanticIndexing() {
    val f = FixtureData
    val directory = File(f.root.parentFile, "r6")
    val index = JSONObject(File(directory, "semantic.json").readText())
    fun buffer(spec: JSONObject, dtype: String): ByteBuffer {
      check(spec.getString("dtype") == dtype)
      val bytes = File(directory, spec.getString("path")).readBytes()
      check(bytes.size == spec.getInt("bytes"))
      val hash =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
      check(hash == spec.getString("sha256"))
      return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    }
    val semanticRows = mutableListOf<Map<String, Any>>()
    var semanticExact = 0
    val references = index.getJSONArray("references")
    for (i in 0 until references.length()) {
      val row = references.getJSONObject(i)
      val logitsBuffer = buffer(row.getJSONObject("digit_logits"), "float32")
      val logits = FloatArray(logitsBuffer.remaining() / 4) { logitsBuffer.float }
      val expectedBuffer = buffer(row.getJSONObject("expected_tokens"), "int32")
      val expected = IntArray(expectedBuffer.remaining() / 4) { expectedBuffer.int }
      val currentBuffer = buffer(row.getJSONObject("current_fp32_tokens"), "int32")
      val current = IntArray(currentBuffer.remaining() / 4) { currentBuffer.int }
      val actual = SemanticTokens.decode(logits)
      val exact = actual.indices.count { actual[it] == expected[it] && actual[it] == current[it] }
      semanticExact += exact
      semanticRows +=
        mapOf(
          "reference" to row.getString("id"),
          "exact_tokens" to exact,
          "total_tokens" to expected.size,
          "minimum_digit_margin" to row.getJSONObject("minimum_digit_margin").getDouble("margin"),
        )
    }
    val equalLogits = FloatArray(235 * 27)
    val allEqualChoosesFirst = SemanticTokens.decode(equalLogits).all { it == 0 }
    val tiedNegative = FloatArray(235 * 27) { -4f }
    for (frame in 0 until 235) {
      tiedNegative[frame * 27 + 2] = -1f
      tiedNegative[frame * 27 + 5] = -1f
    }
    val interiorTieChoosesFirst = SemanticTokens.decode(tiedNegative).all { it == 2 }
    fun rejected(action: () -> Unit): Boolean =
      try {
        action()
        false
      } catch (_: IllegalArgumentException) {
        true
      }
    val invalidInputsRejected =
      listOf(
        rejected { SemanticTokens.decode(FloatArray(27)) },
        rejected { SemanticTokens.decode(equalLogits.copyOf().also { it[0] = Float.NaN }) },
        rejected {
          SemanticTokens.decode(equalLogits.copyOf().also { it[26] = Float.POSITIVE_INFINITY })
        },
        rejected { ArHost.lastOneHot(0) },
        rejected { ArHost.lastOneHot(257) },
        rejected { AcousticHost.semanticOneHot(intArrayOf(-1)) },
        rejected { AcousticHost.semanticOneHot(intArrayOf(4375)) },
        rejected { AcousticHost.frameOneHot(intArrayOf(-1), 512) },
        rejected { AcousticHost.frameOneHot(intArrayOf(512), 512) },
      )
    var prefixExact = 0
    var prefixWorst = 0.0
    val prefixes = f.index.getJSONObject("ar").getJSONArray("utterances")
    for (i in 0 until prefixes.length()) {
      val row = prefixes.getJSONObject(i)
      val prefix = f.floats("${row.getString("key")}/prefix")
      val length = row.getInt("length")
      val padded = prefix.copyOf(256 * 512)
      val selector = ArHost.lastOneHot(length)
      val selected =
        FloatArray(512) { channel ->
          var value = 0f
          for (position in 0 until 256) value +=
            selector[position] * padded[position * 512 + channel]
          value
        }
      val expected = prefix.copyOfRange((length - 1) * 512, length * 512)
      val difference = f.maxDiff(expected, selected)
      prefixWorst = maxOf(prefixWorst, difference)
      if (difference == 0.0) prefixExact++
    }
    val prefixBoundaries =
      listOf(1, 256).all { length ->
        val selector = ArHost.lastOneHot(length)
        selector.count { it == 1f } == 1 &&
          selector[length - 1] == 1f &&
          selector.all { it == 0f || it == 1f }
      }
    // Replay one real utterance in each bucket. Nonconstant signed channels expose
    // both row/column transposition and padded/last-token off-by-one errors.
    val acousticRows = mutableListOf<Map<String, Any>>()
    var oneHotWorst = 0.0
    val acoustic = f.index.getJSONObject("acoustic").getJSONArray("utterances")
    for (frames in listOf(2048, 4096)) {
      val row =
        (0 until acoustic.length())
          .map { acoustic.getJSONObject(it) }
          .first { it.getInt("frames") == frames }
      val key = row.getString("key")
      val tokens = f.ints("$key/semantic_tokens")
      val frameMap = f.ints("$key/frame_to_token")
      val semantic = AcousticHost.semanticOneHot(tokens)
      val frame = AcousticHost.frameOneHot(frameMap, tokens.size)
      var selectedTokens = 0
      var selectedFrames = 0
      for (position in tokens.indices) {
        var nonzeros = 0
        for (token in 0 until 4375) if (semantic[position * 4375 + token] != 0f) {
          check(semantic[position * 4375 + token] == 1f && token == tokens[position])
          nonzeros++
        }
        if (nonzeros == 1) selectedTokens++
      }
      for (position in frameMap.indices) {
        var nonzeros = 0
        var gathered = 0f
        for (token in tokens.indices) {
          val coefficient = frame[position * tokens.size + token]
          if (coefficient != 0f) {
            check(coefficient == 1f && token == frameMap[position])
            nonzeros++
          }
          gathered += coefficient * (token * 0.25f - 17f)
        }
        if (nonzeros == 1) selectedFrames++
        oneHotWorst =
          maxOf(
            oneHotWorst,
            kotlin.math.abs(gathered.toDouble() - (frameMap[position] * 0.25f - 17f)),
          )
      }
      acousticRows +=
        mapOf(
          "id" to row.getString("id"),
          "frames" to frames,
          "tokens_capacity" to tokens.size,
          "semantic_selector_rows_exact" to selectedTokens,
          "frame_selector_rows_exact" to selectedFrames,
          "semantic_onehot_bytes" to semantic.size * 4,
          "frame_onehot_bytes" to frame.size * 4,
        )
    }
    val vocabularyEdges =
      AcousticHost.semanticOneHot(intArrayOf(0, 4374)).let {
        it[0] == 1f && it[4375 + 4374] == 1f && it.count { value -> value == 1f } == 2
      }
    val pass =
      semanticExact == 705 &&
        allEqualChoosesFirst &&
        interiorTieChoosesFirst &&
        invalidInputsRejected.all { it } &&
        prefixExact == 24 &&
        prefixWorst == 0.0 &&
        prefixBoundaries &&
        vocabularyEdges &&
        oneHotWorst == 0.0 &&
        acousticRows.all {
          it["semantic_selector_rows_exact"] == it["tokens_capacity"] &&
            it["frame_selector_rows_exact"] == it["frames"]
        }
    f.metrics(
      "r6_contract",
      mapOf(
        "status" to if (pass) "PASS" else "FAIL",
        "device" to "Mac JVM CPU",
        "semantic_tokens_exact" to semanticExact,
        "semantic_tokens_total" to 705,
        "semantic_rows" to semanticRows,
        "exact_tie_first_index" to (allEqualChoosesFirst && interiorTieChoosesFirst),
        "invalid_inputs_rejected" to invalidInputsRejected.count { it },
        "invalid_inputs_total" to invalidInputsRejected.size,
        "prefix_rows_exact" to prefixExact,
        "prefix_max_abs_diff" to prefixWorst,
        "prefix_boundary_lengths" to listOf(1, 256),
        "onehot_gather_max_abs_diff" to oneHotWorst,
        "acoustic_rows" to acousticRows,
        "vocabulary_boundary_tokens" to listOf(0, 4374),
      ),
    )
    assertTrue("R6 semantic host decoding or exact onehot selection failed", pass)
  }
}
