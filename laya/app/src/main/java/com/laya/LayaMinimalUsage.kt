// SPDX-License-Identifier: Apache-2.0
package com.laya

import android.content.Context
import androidx.annotation.WorkerThread

/** Compiled documentation example; the interactive app does not call this function. */
object LayaMinimalUsage {
  /** Call off the main thread after installing the files listed in README.md. */
  @WorkerThread
  fun classify(context: Context): Map<String, Any?> {
    // Constructor loads tokenizer.json, calibration, and the read-only fp16 table.
    LayaEngine(context, LayaEngine.Storage.WFP16).use { engine ->
      val question =
        linkedMapOf<String, Any?>(
          "type" to "choice",
          "instructions" to "What does the customer need?",
          "criteria" to linkedMapOf("refund" to "money returned", "help" to "technical help"),
        )
      val backend = LayaEngine.Backend.GPU
      // CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32).
      engine.initialize(backend)
      val row = engine.prepare("同じ支払いが二重に請求されました。返金をお願いします。", question)
      // Gathers fp16 rows (PAD id 0), writes fp32 buffers, runs main + act, reads outputs.
      val raw = engine.runRaw(row, backend)
      check(raw.finite)
      return LayaDecoder.decode(raw.markerLogits, raw.actLogits, row.question, engine.calibration)
    }
  }
}
