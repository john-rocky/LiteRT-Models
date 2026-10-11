// SPDX-License-Identifier: Apache-2.0
package com.phocinae

import android.content.Context

/** Bridges debug launch extras to the fixture runner without shipping it in release. */
object PhocinaeGateEntry {
  /** Location and completion state of the on-device JSON report. */
  data class Result(val path: String, val status: String, val error: String? = null)

  /** Runs on the ViewModel's confined model dispatcher. */
  fun run(context: Context, engine: PhocinaeEngine, backend: PhocinaeEngine.Backend, window: Int): Result {
    val report = PhocinaeGateRunner(context, engine).run(backend, window)
    return Result(report.path, report.status, report.error)
  }
}
