package com.gliformer

import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
  val text: String,
  val backend: GliformerExtractor.Backend = GliformerExtractor.Backend.GPU,
  val phase: Phase = Phase.LOADING,
  val result: UiResult? = null,
  val errorMessage: String? = null,
  val requestId: Int = 0,
  val startupWarmupMs: Double = 0.0,
) {
  enum class Phase {
    LOADING,
    WARMING,
    READY,
    EXTRACTING,
    ERROR,
  }

  val busy: Boolean
    get() = phase == Phase.LOADING || phase == Phase.WARMING || phase == Phase.EXTRACTING
}

@Immutable
data class UiResult(
  val text: String,
  val entities: List<GliformerDecoder.Entity>,
  val backend: GliformerExtractor.Backend,
  val window: Int,
  val encodedTokens: Int,
  val words: Int,
  val timings: GliformerExtractor.Timings,
)
