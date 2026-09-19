package com.gliner25

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

@Immutable
data class UiState(
  val inputText: String,
  val accelerator: GlinerExtractor.Backend = GlinerExtractor.Backend.GPU,
  @param:StringRes val statusMessage: Int = R.string.status_loading,
  val busy: Boolean = false,
  val errorMessage: String? = null,
  val result: ExtractionUiResult? = null,
  val gateMode: Boolean = false,
  val gateFiles: List<String> = emptyList(),
)

@Immutable
data class EntityUiSpan(
  val label: String,
  val text: String,
  val start: Int,
  val end: Int,
  val confidence: Float,
)

@Immutable
data class ExtractionUiResult(
  val text: String,
  val spans: List<EntityUiSpan>,
  val highlightedSpans: List<EntityUiSpan>,
  val accelerator: GlinerExtractor.Backend,
  val window: Int,
  val encodedTokens: Int,
  val textWords: Int,
  val tokenizeEmbedMs: Double,
  val graphMs: Double,
  val decodeMs: Double,
)
