package com.opendecision

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/** Everything [com.opendecision.view.DecisionScreen] renders. */
@Immutable
data class UiState(
  val stateText: String,
  val questionsText: String,
  val backend: DecisionModel.Backend = DecisionModel.Backend.GPU,
  @param:StringRes val statusMessage: Int = R.string.status_loading,
  val busy: Boolean = false,
  val errorMessage: String? = null,
  val result: DecisionUiResult? = null,
  val gateMode: Boolean = false,
  val gateFiles: List<String> = emptyList(),
)

/** One answered question: the kind, the instructions, the answer line and every option's probability. */
@Immutable
data class AnswerUiRow(
  val kind: Question.Kind,
  val instructions: String,
  val answer: String,
  val options: List<String>,
  val probabilities: List<Double>,
  val best: Int,
)

/** One answered request: rows per question, the window and the timed phases. */
@Immutable
data class DecisionUiResult(
  val rows: List<AnswerUiRow>,
  val backend: DecisionModel.Backend,
  val window: Int,
  val encodedTokens: Int,
  val optionCount: Int,
  val tokenizeEmbedMs: Double,
  val graphMs: Double,
  val decodeMs: Double,
)
