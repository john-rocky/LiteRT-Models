// SPDX-License-Identifier: Apache-2.0
package com.julia1

import android.content.Context
import androidx.annotation.WorkerThread

/** Compiled documentation example; the interactive app does not call this function. */
object JuliaMinimalUsage {
  /** Call off the main thread after installing the files listed in README.md. */
  @WorkerThread
  fun decide(context: Context): JuliaDecoder.Answer {
    // The constructor loads tokenizer.json and maps the read-only float16 token table.
    JuliaEngine(context).use { engine ->
      val question =
        JuliaQuestion(
          type = "choice",
          instructions = "Which team should handle this request?",
          criteria =
            linkedMapOf(
              "billing" to "Billing and payment disputes",
              "shipping" to "Shipping and delivery",
              "access" to "Account access and login",
            ),
        )
      val backend = JuliaEngine.Backend.GPU
      // CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32).
      engine.initialize(backend)
      val state = "Order #4417 was charged twice. The customer wants the extra charge refunded."
      // Strict encoding: the request must fit the window, or an EncodingException is thrown.
      val row = engine.prepare(state, question)
      // Gathers fp16 rows (PAD id 0), writes fp32 buffers, runs the graph, reads token_logits.
      val raw = engine.runRaw(row, backend)
      check(raw.finite)
      // Softmax at temperature 1 over the marker logits; answer.choice is the winning ID.
      return JuliaDecoder.decode(raw.markerLogits, question)
    }
  }
}
