package com.gliclass.view

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Divider
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Slider
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.gliclass.ClassificationUiResult
import com.gliclass.GliclassClassifier
import com.gliclass.GliclassDecoder
import com.gliclass.LabelScoreUiRow
import com.gliclass.R
import com.gliclass.UiState

/**
 * The sample's single screen: backend choice, text, labels and optional prompt, single- or
 * multi-label mode with its threshold, Classify, then every label's score with the chosen labels
 * highlighted, the graph window and the timings.
 */
@Composable
fun GliclassScreen(
  state: UiState,
  onTextChanged: (String) -> Unit,
  onLabelsChanged: (String) -> Unit,
  onPromptChanged: (String) -> Unit,
  onModeChanged: (GliclassDecoder.Mode) -> Unit,
  onThresholdChanged: (Float) -> Unit,
  onAcceleratorChanged: (GliclassClassifier.Backend) -> Unit,
  onClassify: () -> Unit,
) {
  Scaffold(
    modifier = Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { contentPadding ->
    Column(
      Modifier.padding(contentPadding)
        .fillMaxSize()
        .verticalScroll(rememberScrollState())
        .padding(20.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Text(stringResource(state.statusMessage), style = MaterialTheme.typography.subtitle1)
      state.errorMessage?.let {
        Text(stringResource(R.string.error_message, it), color = MaterialTheme.colors.error)
      }
      if (state.busy) {
        LinearProgressIndicator(Modifier.fillMaxWidth())
      }
      if (state.gateMode) {
        Text(stringResource(R.string.gate_description))
        state.gateFiles.forEach { Text(it, style = MaterialTheme.typography.caption) }
        state.result?.let { ResultContent(it) }
      } else {
        Text(stringResource(R.string.accelerator_title), style = MaterialTheme.typography.subtitle2)
        Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
          GliclassClassifier.Backend.entries.forEach { backend ->
            Choice(
              acceleratorTitle(backend),
              state.accelerator == backend,
              !state.busy,
            ) {
              onAcceleratorChanged(backend)
            }
          }
        }
        OutlinedTextField(
          value = state.inputText,
          onValueChange = onTextChanged,
          modifier = Modifier.fillMaxWidth(),
          enabled = !state.busy,
          minLines = 4,
          maxLines = 8,
          label = { Text(stringResource(R.string.input_label)) },
        )
        OutlinedTextField(
          value = state.labelsText,
          onValueChange = onLabelsChanged,
          modifier = Modifier.fillMaxWidth(),
          enabled = !state.busy,
          minLines = 3,
          maxLines = 10,
          label = { Text(stringResource(R.string.labels_label)) },
        )
        Text(stringResource(R.string.labels_help), style = MaterialTheme.typography.caption)
        OutlinedTextField(
          value = state.promptText,
          onValueChange = onPromptChanged,
          modifier = Modifier.fillMaxWidth(),
          enabled = !state.busy,
          singleLine = true,
          label = { Text(stringResource(R.string.prompt_label)) },
        )
        Text(stringResource(R.string.prompt_help), style = MaterialTheme.typography.caption)
        Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
          GliclassDecoder.Mode.entries.forEach { mode ->
            Choice(modeTitle(mode), state.mode == mode, !state.busy) { onModeChanged(mode) }
          }
        }
        if (state.mode == GliclassDecoder.Mode.MULTI_LABEL) {
          Text(stringResource(R.string.threshold_value, state.threshold))
          Slider(
            value = state.threshold,
            onValueChange = onThresholdChanged,
            enabled = !state.busy,
            valueRange = THRESHOLD_MIN..THRESHOLD_MAX,
            steps = THRESHOLD_STEPS,
          )
        }
        Button(
          onClick = onClassify,
          enabled = !state.busy && state.inputText.isNotBlank() && state.labelsText.isNotBlank(),
          modifier = Modifier.fillMaxWidth(),
        ) {
          Text(stringResource(R.string.classify))
        }
        state.result?.let { ResultContent(it) }
      }
    }
  }
}

@Composable
private fun Choice(title: String, selected: Boolean, enabled: Boolean, onSelect: () -> Unit) {
  Row(verticalAlignment = Alignment.CenterVertically) {
    RadioButton(selected = selected, enabled = enabled, onClick = onSelect)
    Text(title)
  }
}

@Composable
private fun acceleratorTitle(backend: GliclassClassifier.Backend): String =
  stringResource(
    when (backend) {
      GliclassClassifier.Backend.GPU -> R.string.accelerator_gpu
      GliclassClassifier.Backend.CPU -> R.string.accelerator_cpu
    }
  )

@Composable
private fun modeTitle(mode: GliclassDecoder.Mode): String =
  stringResource(
    if (mode == GliclassDecoder.Mode.SINGLE_LABEL) R.string.mode_single else R.string.mode_multi
  )

@Composable
private fun ResultContent(result: ClassificationUiResult) {
  Divider()
  Text(
    if (result.mode == GliclassDecoder.Mode.SINGLE_LABEL) {
      stringResource(R.string.scores_title_single)
    } else {
      stringResource(R.string.scores_title_multi, result.threshold)
    },
    style = MaterialTheme.typography.h6,
  )
  if (result.rows.none { it.chosen }) {
    Text(stringResource(R.string.no_label_chosen), style = MaterialTheme.typography.body2)
  }
  SelectionContainer {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
      result.rows.forEach { ScoreRow(it) }
    }
  }
  Divider()
  Text(
    stringResource(
      R.string.result_window,
      result.window,
      result.encodedTokens,
      result.labelCount,
      acceleratorTitle(result.accelerator),
    ),
    style = MaterialTheme.typography.caption,
  )
  Text(
    stringResource(R.string.result_timing, result.tokenizeEmbedMs, result.graphMs, result.decodeMs),
    style = MaterialTheme.typography.caption,
  )
}

@Composable
private fun ScoreRow(row: LabelScoreUiRow) {
  Column {
    Text(
      stringResource(R.string.label_score, row.label, row.score),
      fontWeight = if (row.chosen) FontWeight.Bold else FontWeight.Normal,
      color = if (row.chosen) MaterialTheme.colors.primary else MaterialTheme.colors.onSurface,
    )
    LinearProgressIndicator(progress = row.score, modifier = Modifier.fillMaxWidth())
  }
}

private const val THRESHOLD_MIN = 0.05f
private const val THRESHOLD_MAX = 0.95f
private const val THRESHOLD_STEPS = 17
