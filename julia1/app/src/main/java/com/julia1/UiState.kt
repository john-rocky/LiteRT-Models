// SPDX-License-Identifier: Apache-2.0
package com.julia1

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/** Bundled question sets; their schemas are in presets.json, the example texts in strings.xml. */
enum class Preset(
  val assetKey: String,
  @param:StringRes val title: Int,
  @param:StringRes val example: Int,
) {
  TICKET("ticket", R.string.preset_ticket, R.string.example_ticket),
  AGENT("agent", R.string.preset_agent, R.string.example_agent),
  REVIEW("review", R.string.preset_review, R.string.example_review),
}

/** One decoded option, with the caller's ID or a localized true/false label. */
@Immutable
data class ProbabilityUiRow(
  val probability: Double,
  val label: String = "",
  @param:StringRes val labelResource: Int? = null,
  val scoreLevel: Int? = null,
  val description: String? = null,
)

/** Display data for one question; numerical values come from the host decoder. */
@Immutable
data class AnswerUiRow(
  val questionId: String,
  val instructions: String,
  val type: String,
  val choice: String? = null,
  val score: Double? = null,
  val trueProbability: Double? = null,
  val probabilities: List<ProbabilityUiRow>,
  val totalMs: Double,
  val tokenCount: Int,
  val window: Int,
)

/** Immutable UI snapshot emitted by the model-owning ViewModel. */
@Immutable
data class UiState(
  val inputText: String,
  val preset: Preset = Preset.TICKET,
  val accelerator: JuliaEngine.Backend = JuliaEngine.Backend.GPU,
  val busy: Boolean = true,
  val ready: Boolean = false,
  val gateMode: Boolean = false,
  @param:StringRes val statusMessage: Int = R.string.status_loading_tokenizer,
  val answers: List<AnswerUiRow> = emptyList(),
  val runTotalMs: Double? = null,
  val runTokenCount: Int = 0,
  val launchToReadyMs: Double? = null,
  val errorMessage: String? = null,
  val canFallbackToCpu: Boolean = false,
  val gateFile: String? = null,
)
