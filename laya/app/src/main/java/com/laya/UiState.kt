// SPDX-License-Identifier: Apache-2.0
package com.laya

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/** Bundled upstream question sets; their schemas remain in presets.json. */
enum class Preset(val assetKey: String, @param:StringRes val title: Int) {
  EMAIL("email", R.string.preset_email),
  TRIAGE("triage", R.string.preset_triage),
  MODERATION("moderation", R.string.preset_moderation),
}

/** Language of the invented example currently loaded in the state editor. */
enum class ExampleLanguage(@param:StringRes val title: Int) {
  JA(R.string.language_japanese),
  EN(R.string.language_english),
}

/** One decoded option, with the original label or a localized boolean/score label. */
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
  val confidence: Double,
  val totalMs: Double,
  val tokenCount: Int,
)

/** Immutable UI snapshot emitted by the model-owning ViewModel. */
@Immutable
data class UiState(
  val inputSubject: String,
  val inputText: String,
  val preset: Preset = Preset.EMAIL,
  val language: ExampleLanguage = ExampleLanguage.JA,
  val accelerator: LayaEngine.Backend = LayaEngine.Backend.GPU,
  val calibrated: Boolean = true,
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
