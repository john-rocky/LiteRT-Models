// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.content.Context
import org.json.JSONObject

object SoproGateEntry {
  data class Result(val path: String, val status: String, val error: String? = null)

  fun run(
    context: Context,
    backend: SoproEngine.Backend,
    precision: SoproEngine.Precision,
    placement: Map<String, SoproEngine.Backend>,
    onlyId: String?,
    manifest: String,
    mode: String = "teacher",
    seed: Long = 5000L,
    config: String = "{}",
  ): Result {
    return DeviceGateProducer(context)
      .run(mode, JSONObject(config), backend, precision, placement, onlyId, manifest, seed)
  }
}
