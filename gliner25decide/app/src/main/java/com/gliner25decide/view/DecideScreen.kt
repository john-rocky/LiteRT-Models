package com.gliner25decide.view

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
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.gliner25decide.ClassificationUiResult
import com.gliner25decide.DecideClassifier
import com.gliner25decide.DecisionUiRow
import com.gliner25decide.R
import com.gliner25decide.UiState

/**
 * The sample's single screen: backend choice, text and task editors, Classify, then the decision of
 * every task with its probability, the graph window and the timings.
 */
@Composable
fun DecideScreen(
  state: UiState,
  onTextChanged: (String) -> Unit,
  onTasksChanged: (String) -> Unit,
  onAcceleratorChanged: (DecideClassifier.Backend) -> Unit,
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
      verticalArrangement = Arrangement.spacedBy(16.dp),
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
      } else {
        Text(stringResource(R.string.accelerator_title), style = MaterialTheme.typography.subtitle2)
        Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
          DecideClassifier.Backend.entries.forEach { backend ->
            Row(verticalAlignment = Alignment.CenterVertically) {
              RadioButton(
                selected = state.accelerator == backend,
                enabled = !state.busy,
                onClick = { onAcceleratorChanged(backend) },
              )
              Text(acceleratorTitle(backend))
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
          value = state.tasksText,
          onValueChange = onTasksChanged,
          modifier = Modifier.fillMaxWidth(),
          enabled = !state.busy,
          minLines = 3,
          maxLines = 10,
          textStyle = MaterialTheme.typography.body2.copy(fontFamily = FontFamily.Monospace),
          label = { Text(stringResource(R.string.tasks_label)) },
        )
        Text(stringResource(R.string.tasks_help), style = MaterialTheme.typography.caption)
        Button(
          onClick = onClassify,
          enabled = !state.busy && state.inputText.isNotBlank() && state.tasksText.isNotBlank(),
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
private fun acceleratorTitle(backend: DecideClassifier.Backend): String =
  stringResource(
    if (backend == DecideClassifier.Backend.GPU) {
      R.string.accelerator_gpu
    } else {
      R.string.accelerator_cpu
    }
  )

@Composable
private fun ResultContent(result: ClassificationUiResult) {
  Divider()
  Text(stringResource(R.string.decisions_title), style = MaterialTheme.typography.h6)
  SelectionContainer {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
      result.rows.forEach { DecisionRow(it) }
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
private fun DecisionRow(row: DecisionUiRow) {
  Column {
    Text(
      stringResource(
        if (row.multiLabel) {
          R.string.task_multi_label
        } else {
          R.string.task_single_label
        },
        row.task,
      ),
      style = MaterialTheme.typography.subtitle2,
    )
    row.labels.zip(row.probabilities).forEach { (label, probability) ->
      Text(
        stringResource(R.string.decision_label, label, probability),
        fontWeight = FontWeight.Medium,
        color = MaterialTheme.colors.primary,
      )
    }
  }
}
