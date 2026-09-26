package com.gliformer

import android.content.Context
import org.json.JSONObject

/** Release implementation keeps interaction reporting disabled without a runtime switch. */
@Suppress("UNUSED_PARAMETER")
internal class GliformerFirstTapReport(context: Context) {
  fun ready(
    launchStartedNs: Long,
    backend: GliformerExtractor.Backend,
    tableStorage: GliformerInputs.EmbeddingTable.Storage,
    loadMs: Double,
    warmup: GliformerExtractor.WarmupReport,
  ) = Unit

  fun begin(
    id: Int,
    tapNs: Long,
    backend: GliformerExtractor.Backend,
    storage: GliformerInputs.EmbeddingTable.Storage,
    loadMs: Double,
    warmMs: Double,
  ) = Unit

  fun windowWarmup(id: Int, loadMs: Double, warmMs: Double) = Unit

  fun result(id: Int, resultNs: Long, result: GliformerExtractor.Result) = Unit

  fun statePublished(id: Int, stateNs: Long) = Unit

  fun rendered(id: Int, renderedNs: Long): JSONObject? = null

  fun failure(id: Int, failure: Throwable) = Unit

  fun write(report: JSONObject) = Unit
}
