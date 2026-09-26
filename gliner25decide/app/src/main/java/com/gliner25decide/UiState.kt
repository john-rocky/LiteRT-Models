package com.gliner25decide

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/** Everything [com.gliner25decide.view.DecideScreen] renders. */
@Immutable
data class UiState(
  val inputText: String,
  val tasksText: String,
  val accelerator: DecideClassifier.Backend = DecideClassifier.Backend.GPU,
  @param:StringRes val statusMessage: Int = R.string.status_loading,
  val busy: Boolean = false,
  val errorMessage: String? = null,
  val result: ClassificationUiResult? = null,
  val gateMode: Boolean = false,
  val gateFiles: List<String> = emptyList(),
)

/** One head's decision; single-label heads have one label and one probability. */
@Immutable
data class DecisionUiRow(
  val task: String,
  val multiLabel: Boolean,
  val labels: List<String>,
  val probabilities: List<Float>,
)

/** One classified request: decisions per head, the graph window and the timed phases. */
@Immutable
data class ClassificationUiResult(
  val rows: List<DecisionUiRow>,
  val accelerator: DecideClassifier.Backend,
  val window: Int,
  val encodedTokens: Int,
  val labelCount: Int,
  val tokenizeEmbedMs: Double,
  val graphMs: Double,
  val decodeMs: Double,
)
