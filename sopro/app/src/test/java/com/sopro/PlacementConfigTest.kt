// SPDX-License-Identifier: Apache-2.0
package com.sopro

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PlacementConfigTest {
  @Test
  fun deviceFallbackStorageAndOverlaySelection() {
    val actual = PlacementConfig.automatic("qcom", "SM8850")
    val expected =
      mapOf(
        "speaker_encoder" to SoproEngine.Backend.GPU32,
        "semantic_encoder" to SoproEngine.Backend.GPU32,
        "acoustic_condition" to SoproEngine.Backend.GPU,
        "acoustic_velocity" to SoproEngine.Backend.GPU,
        "acoustic_condition_t4096" to SoproEngine.Backend.GPU,
        "acoustic_velocity_t4096" to SoproEngine.Backend.GPU,
      )
    assertEquals(expected, actual)
    val fallbackCases =
      listOf(
        "tensor" to "Tensor G3",
        "akita" to "",
        "qcom" to "SM8650",
        "qcom" to "",
        "unknown" to "unknown",
      )
    fallbackCases.forEach { (hardware, soc) ->
      assertTrue(PlacementConfig.automatic(hardware, soc).isEmpty())
    }
    assertEquals(expected, PlacementConfig.automatic(" sm8850 ", ""))
    assertTrue(PlacementConfig.placement(PlacementConfig.Mode.CPU, "qcom", "SM8850").isEmpty())
    val gpuAr = PlacementConfig.placement(PlacementConfig.Mode.GPU_AR, "qcom", "SM8850")
    assertEquals(expected + ("ar_merged" to SoproEngine.Backend.GPU32), gpuAr)
    for (graph in
      listOf(
        "ar_merged",
        "style_prefix",
        "vocoder_stream_start",
        "vocoder_stream_step",
        "vocoder_stream_flush",
      )) {
      assertEquals(SoproEngine.Backend.CPU, actual[graph] ?: SoproEngine.Backend.CPU)
    }
    assertEquals(
      SoproEngine.Precision.SHIP,
      PlacementConfig.precision(PlacementConfig.Mode.AUTOMATIC),
    )
    assertEquals(
      SoproEngine.Precision.WFP16,
      PlacementConfig.precision(PlacementConfig.Mode.GPU_AR),
    )
    val directory =
      File(FixtureData.root.parentFile.parentFile, "results/jvm_r9_catalog_${System.nanoTime()}")
    check(directory.mkdirs())
    fun row(graph: String, storage: String, generation: String, explicitStorage: Boolean = true) =
      JSONObject()
        .put("graph", graph)
        .put(
          "path",
          if (generation == "base") "$storage/$graph.tflite"
          else "$generation/$storage/$graph.tflite",
        )
        .also { if (explicitStorage) it.put("storage", storage) }
    fun write(name: String, rows: List<JSONObject>) =
      File(directory, name).writeText(JSONObject().put("models", JSONArray(rows)).toString())
    fun catalog(
      precision: SoproEngine.Precision = SoproEngine.Precision.SHIP,
      set: String = "r9",
      style: PlacementConfig.StyleVariant = PlacementConfig.StyleVariant.FP32,
    ) = ModelCatalog(directory, precision, set, style)
    var missingChecks = 0
    fun missing(expectedName: String, block: () -> Unit) {
      var name: String? = null
      try {
        block()
      } catch (failure: MissingModelFile) {
        name = failure.filename
      }
      assertEquals(expectedName, name)
      missingChecks++
    }
    try {
      missing("contract_r9.json") { catalog() }
      write(
        "contract.json",
        listOf(
          row("speaker_encoder", "wfp16", "base", false),
          row("style_prefix", "fp32", "base", false),
          row("style_prefix", "wfp16", "base", false),
        ),
      )
      write(
        "contract_r6.json",
        listOf(
          row("semantic_encoder", "wfp16", "r6"),
          row("ar_merged", "int8", "r6"),
          row("ar_merged", "wfp16", "r6"),
          row("style_prefix", "wfp16", "r6"),
        ),
      )
      write(
        "contract_r9.json",
        listOf(
          row("style_prefix", "fp32", "r9"),
          row("style_prefix", "wfp16", "r9"),
          row("vocoder_stream_step", "wfp16", "r9"),
        ),
      )
      val selected = catalog()
      assertEquals("r9/fp32/style_prefix.tflite", selected.spec("style_prefix").getString("path"))
      assertEquals(
        "r9/wfp16/vocoder_stream_step.tflite",
        selected.spec("vocoder_stream_step").getString("path"),
      )
      assertEquals(
        "r6/wfp16/semantic_encoder.tflite",
        selected.spec("semantic_encoder").getString("path"),
      )
      assertEquals("r6/int8/ar_merged.tflite", selected.spec("ar_merged").getString("path"))
      assertEquals(
        "wfp16/speaker_encoder.tflite",
        selected.spec("speaker_encoder").getString("path"),
      )
      assertEquals(
        "r9/wfp16/style_prefix.tflite",
        catalog(style = PlacementConfig.StyleVariant.WFP16).spec("style_prefix").getString("path"),
      )
      assertEquals(
        "r9/fp32/style_prefix.tflite",
        catalog(SoproEngine.Precision.FP32, style = PlacementConfig.StyleVariant.WFP16)
          .spec("style_prefix")
          .getString("path"),
      )
      assertEquals(
        "r6/wfp16/style_prefix.tflite",
        catalog(set = "r6").spec("style_prefix").getString("path"),
      )
      assertEquals(
        "r6/wfp16/ar_merged.tflite",
        catalog(SoproEngine.Precision.WFP16).spec("ar_merged").getString("path"),
      )
      missing("host_assets.json") { selected.requireFiles(listOf("style_prefix")) }
      ModelCatalog.hostAssetPaths.forEach { path ->
        File(directory, path).apply {
          parentFile!!.mkdirs()
          writeBytes(byteArrayOf())
        }
      }
      missing("style_prefix.tflite") { selected.requireFiles(listOf("style_prefix", "ar_merged")) }
      File(directory, "r9/fp32/style_prefix.tflite").apply {
        parentFile!!.mkdirs()
        writeBytes(byteArrayOf())
      }
      missing("ar_merged.tflite") { selected.requireFiles(listOf("style_prefix", "ar_merged")) }
      File(directory, "r6/int8/ar_merged.tflite").apply {
        parentFile!!.mkdirs()
        writeBytes(byteArrayOf())
      }
      selected.requireFiles(listOf("style_prefix", "ar_merged"))
      var rejectedHistorical = false
      try {
        catalog(set = "published")
      } catch (_: IllegalArgumentException) {
        rejectedHistorical = true
      }
      assertTrue(rejectedHistorical)
      FixtureData.metrics(
        "r9_placement",
        mapOf(
          "status" to "PASS",
          "device" to "Mac JVM CPU",
          "hybrid_gpu_graphs_exact" to actual.size,
          "hybrid_cpu_graphs_exact" to 5,
          "cpu_fallback_cases" to fallbackCases.size,
          "gpu_ar_optional" to true,
          "catalog_storage_and_overlay_cases" to 9,
          "missing_filename_checks" to missingChecks,
          "historical_contract_rejected" to rejectedHistorical,
        ),
      )
    } finally {
      directory.deleteRecursively()
    }
  }
}
