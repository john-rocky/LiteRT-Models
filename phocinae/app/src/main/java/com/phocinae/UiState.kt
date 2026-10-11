// SPDX-License-Identifier: Apache-2.0
package com.phocinae

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/** Bundled question sets (three typed questions each); their schemas live in presets.json. */
enum class Preset(val assetKey: String, @param:StringRes val title: Int, @param:StringRes val example: Int) {
  DEPLOY("deploy", R.string.preset_deploy, R.string.example_deploy),
  SUPPORT("support", R.string.preset_support, R.string.example_support),
  INVOICE("invoice", R.string.preset_invoice, R.string.example_invoice),
  SECURITY("security", R.string.preset_security, R.string.example_security),
}

/** One decoded option: the original label, a localized yes/no label, or a rubric level. */
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
  /** The answer as shown: a choice label, a rubric level text, or Yes / No. */
  val answer: String,
  /** The probability of the shown answer, 0..1. */
  val answerProbability: Double,
  val probabilities: List<ProbabilityUiRow>,
  val confidence: Double,
  val totalMs: Double,
  val tokenCount: Int,
)

/** Immutable UI snapshot emitted by the model-owning ViewModel. */
@Immutable
data class UiState(
  val inputText: String,
  val preset: Preset = Preset.DEPLOY,
  val accelerator: PhocinaeEngine.Backend = PhocinaeEngine.Backend.GPU,
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
  /** True when this APK packages the Qualcomm NPU libraries (see the README). */
  val npuAvailable: Boolean = false,
  val gateFile: String? = null,
)
