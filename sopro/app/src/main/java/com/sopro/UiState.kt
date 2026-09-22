// SPDX-License-Identifier: Apache-2.0
package com.sopro

import androidx.compose.runtime.Immutable

@Immutable data class GraphPlacement(val graph: String, val backend: SoproEngine.Backend)

@Immutable
data class UiState(
  val text: String = "",
  val language: String = "en",
  val backend: SoproEngine.Backend = SoproEngine.Backend.CPU,
  val placementMode: PlacementConfig.Mode = PlacementConfig.Mode.AUTOMATIC,
  val styleVariant: PlacementConfig.StyleVariant = PlacementConfig.StyleVariant.FP32,
  val deviceHybridAvailable: Boolean = false,
  val graphPlacements: List<GraphPlacement> = emptyList(),
  val modelsReady: Boolean = false,
  val readyMs: Double? = null,
  val busy: Boolean = false,
  val status: String = "",
  val error: String? = null,
  val generatedSamples: Int = 0,
  val gateMode: Boolean = false,
  val referenceKind: String = "demo",
  val referenceLabel: String = "",
  val referenceSamples: Int = 240000,
  val recording: Boolean = false,
  val lastWavPath: String? = null,
  val lastSidecarPath: String? = null,
  val ttfaMs: Double? = null,
  val ttfaToOnsetMs: Double? = null,
  val totalWallMs: Double? = null,
  val rtf: Double? = null,
  val placement: String = "",
  val seed: Long? = null,
)
