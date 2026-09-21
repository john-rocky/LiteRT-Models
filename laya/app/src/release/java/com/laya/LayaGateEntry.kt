// SPDX-License-Identifier: Apache-2.0
package com.laya

import android.content.Context

/** Release has no fixture reader or device gate implementation. */
object LayaGateEntry {
  /** Shared launch-interface type; no release path creates a gate result. */
  data class Result(val path: String, val status: String, val error: String? = null)

  /** Defensive guard; MainActivity and MainViewModel reject gate extras in release. */
  @Suppress("UNUSED_PARAMETER")
  fun run(context: Context, engine: LayaEngine, backend: LayaEngine.Backend, window: Int): Result {
    error("Gate execution is available only in debug builds")
  }
}
