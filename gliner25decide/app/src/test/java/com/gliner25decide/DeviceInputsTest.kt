package com.gliner25decide

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The Kotlin host's graph inputs against the exact bytes the S26 native-runner gate consumed
 * (`results/device/inputs_s{128,256,512}/`: fp16 table + numpy upcast, 42 fixtures per window).
 * Byte equality here means the app feeds the graph what the native runner fed it.
 */
class DeviceInputsTest {
  @Test
  fun hostInputsEqualTheRound3DeviceInputBytes() {
    val root = ExternalTestData.resolve()
    DecideInputs.WINDOWS.forEach {
      ExternalTestData.requireFiles(root, "results/device/inputs_s$it/manifest.json")
    }
    val oracle = OracleFixtures.load(root).associateBy { it.id }
    val inputs = DecideInputs(GlinerTokenizer(ExternalTestData.tokenizer(root)))
    val failures = JSONArray()
    val counts = linkedMapOf<Int, Int>()
    var pairs = 0
    DecideInputs.EmbeddingTable(ExternalTestData.embeddingTable(root)).use { table ->
      for (window in DecideInputs.WINDOWS) {
        val directory = File(root, "results/device/inputs_s$window")
        val manifest = JSONObject(File(directory, "manifest.json").readText())
        assertEquals(window, manifest.getInt("seq"))
        val entries = manifest.getJSONArray("fixtures")
        var identical = 0
        for (index in 0 until entries.length()) {
          val entry = entries.getJSONObject(index)
          val fixture = requireNotNull(oracle[entry.getString("fixture_id")])
          val prepared = inputs.prepare(fixture.text, fixture.tasks, window)
          val files = entry.getJSONObject("files")
          val tensors =
            linkedMapOf(
              "inputs_embeds" to table.lookup(prepared.inputIds),
              "attention_mask" to prepared.attentionMask,
              "label_routing" to prepared.labelRouting,
            )
          val different = tensors.filter { (name, values) ->
            !File(directory, files.getJSONObject(name).getString("file"))
              .readBytes()
              .contentEquals(littleEndian(values))
          }
          if (different.isEmpty()) {
            identical++
          } else {
            failures.put(
              JSONObject()
                .put("window", window)
                .put("id", fixture.id)
                .put("different", JSONArray(different.keys.toList()))
            )
          }
          pairs++
        }
        counts[window] = identical
      }
    }
    ExternalTestData.reportFile("device_inputs_bytes.json")
      .writeText(
        JSONObject()
          .put("test", "DeviceInputsTest")
          .put("pairs", pairs)
          .put("identical_by_window", JSONObject(counts.mapKeys { "s${it.key}" }))
          .put(
            "compared",
            "inputs_embeds (fp16 table + host upcast), attention_mask, label_routing: whole files",
          )
          .put("failures", failures)
          .toString(2) + "\n"
      )
    println("DEVICE_INPUTS pairs=$pairs identical=$counts failures=$failures")
    assertEquals(126, pairs)
    assertEquals(0, failures.length())
  }

  private fun littleEndian(values: FloatArray): ByteArray {
    val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buffer.asFloatBuffer().put(values)
    return buffer.array()
  }
}
