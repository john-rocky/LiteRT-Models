// SPDX-License-Identifier: Apache-2.0
package com.julia1

import android.content.Context

/** Bridges debug launch extras to the fixture runner without shipping it in release. */
object JuliaGateEntry {
  /** Location and completion state of the on-device JSON report. */
  data class Result(val path: String, val status: String, val error: String? = null)

  /** Runs on the ViewModel's confined model dispatcher. */
  fun run(
    context: Context,
    engine: JuliaEngine,
    backend: JuliaEngine.Backend,
    window: Int,
  ): Result {
    val report = JuliaGateRunner(context, engine).run(backend, window)
    return Result(report.path, report.status, report.error)
  }
}
