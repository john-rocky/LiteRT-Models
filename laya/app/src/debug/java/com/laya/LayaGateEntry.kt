// SPDX-License-Identifier: Apache-2.0
package com.laya

import android.content.Context

/** Bridges debug launch extras to the fixture runner without shipping it in release. */
object LayaGateEntry {
  /** Location and completion state of the on-device JSON report. */
  data class Result(val path: String, val status: String, val error: String? = null)

  /** Runs on the ViewModel's confined model dispatcher. */
  fun run(context: Context, engine: LayaEngine, backend: LayaEngine.Backend, window: Int): Result {
    val report = LayaGateRunner(context, engine).run(backend, window)
    return Result(report.path, report.status, report.error)
  }
}
