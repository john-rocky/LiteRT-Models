package com.gliclass

import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/** Everything [com.gliclass.view.GliclassScreen] renders. */
@Immutable
data class UiState(
  val inputText: String,
  val labelsText: String,
  val promptText: String = "",
  val mode: GliclassDecoder.Mode = GliclassDecoder.Mode.SINGLE_LABEL,
  val threshold: Float = GliclassDecoder.DEFAULT_THRESHOLD.toFloat(),
  val accelerator: GliclassClassifier.Backend = GliclassClassifier.Backend.GPU,
  @param:StringRes val statusMessage: Int = R.string.status_loading,
  val busy: Boolean = false,
  val errorMessage: String? = null,
  val result: ClassificationUiResult? = null,
  val gateMode: Boolean = false,
  val gateFiles: List<String> = emptyList(),
)

/** One label's probability; [chosen] marks the labels the decision returns. */
@Immutable data class LabelScoreUiRow(val label: String, val score: Float, val chosen: Boolean)

/** One classified request: every label's score, the graph window and the timed phases. */
@Immutable
data class ClassificationUiResult(
  val mode: GliclassDecoder.Mode,
  val threshold: Double,
  val rows: List<LabelScoreUiRow>,
  val accelerator: GliclassClassifier.Backend,
  val window: Int,
  val encodedTokens: Int,
  val labelCount: Int,
  val tokenizeEmbedMs: Double,
  val graphMs: Double,
  val decodeMs: Double,
) {
  companion object {
    /** Screen form of a classifier result. */
    fun from(result: GliclassClassifier.Result): ClassificationUiResult {
      val decision = result.decision
      return ClassificationUiResult(
        decision.mode,
        decision.threshold,
        decision.labels.indices.map {
          LabelScoreUiRow(decision.labels[it], decision.probabilities[it], it in decision.chosen)
        },
        result.backend,
        result.window,
        result.encodedTokens,
        decision.labels.size,
        result.timing.tokenizeEmbedMs,
        result.timing.graphMs,
        result.timing.decodeMs,
      )
    }
  }
}
