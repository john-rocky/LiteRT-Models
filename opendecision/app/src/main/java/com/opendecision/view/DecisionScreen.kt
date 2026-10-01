package com.opendecision.view

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
import androidx.compose.material.OutlinedButton
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
import com.opendecision.AnswerUiRow
import com.opendecision.DecisionModel
import com.opendecision.DecisionUiResult
import com.opendecision.MainViewModel
import com.opendecision.R
import com.opendecision.UiState

/** The sample's single screen: backend, presets, state and question editors, Decide, then one row per question. */
@Composable
fun DecisionScreen(
  state: UiState,
  onStateChanged: (String) -> Unit,
  onQuestionsChanged: (String) -> Unit,
  onBackendChanged: (DecisionModel.Backend) -> Unit,
  onPreset: (MainViewModel.Preset) -> Unit,
  onDecide: () -> Unit,
) {
  Scaffold(
    modifier = Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { contentPadding ->
    Column(
      Modifier.padding(contentPadding).fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
      verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
      Text(stringResource(state.statusMessage), style = MaterialTheme.typography.subtitle1)
      state.errorMessage?.let { Text(stringResource(R.string.error_message, it), color = MaterialTheme.colors.error) }
      if (state.busy) {
        LinearProgressIndicator(Modifier.fillMaxWidth())
      }
      if (state.gateMode) {
        Text(stringResource(R.string.gate_description))
        state.gateFiles.forEach { Text(it, style = MaterialTheme.typography.caption) }
      } else {
        Text(stringResource(R.string.backend_title), style = MaterialTheme.typography.subtitle2)
        Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
          DecisionModel.Backend.entries.forEach { backend ->
            Row(verticalAlignment = Alignment.CenterVertically) {
              RadioButton(selected = state.backend == backend, enabled = !state.busy, onClick = { onBackendChanged(backend) })
              Text(backendTitle(backend))
            }
          }
        }
        Text(stringResource(R.string.presets_title), style = MaterialTheme.typography.subtitle2)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          MainViewModel.Preset.entries.forEach { preset ->
            OutlinedButton(onClick = { onPreset(preset) }, enabled = !state.busy) { Text(stringResource(preset.title)) }
          }
        }
        OutlinedTextField(
          value = state.stateText,
          onValueChange = onStateChanged,
          modifier = Modifier.fillMaxWidth(),
          enabled = !state.busy,
          minLines = 4,
          maxLines = 8,
          label = { Text(stringResource(R.string.state_label)) },
        )
        OutlinedTextField(
          value = state.questionsText,
          onValueChange = onQuestionsChanged,
          modifier = Modifier.fillMaxWidth(),
          enabled = !state.busy,
          minLines = 3,
          maxLines = 10,
          textStyle = MaterialTheme.typography.body2.copy(fontFamily = FontFamily.Monospace),
          label = { Text(stringResource(R.string.questions_label)) },
        )
        Text(stringResource(R.string.questions_help), style = MaterialTheme.typography.caption)
        Button(
          onClick = onDecide,
          enabled = !state.busy && state.stateText.isNotBlank() && state.questionsText.isNotBlank(),
          modifier = Modifier.fillMaxWidth(),
        ) {
          Text(stringResource(R.string.decide))
        }
        state.result?.let { ResultContent(it) }
      }
    }
  }
}

@Composable
private fun backendTitle(backend: DecisionModel.Backend): String =
  stringResource(if (backend == DecisionModel.Backend.GPU) R.string.backend_gpu else R.string.backend_cpu)

@Composable
private fun ResultContent(result: DecisionUiResult) {
  Divider()
  Text(stringResource(R.string.answers_title), style = MaterialTheme.typography.h6)
  SelectionContainer {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) { result.rows.forEach { AnswerRow(it) } }
  }
  Divider()
  Text(
    stringResource(R.string.result_window, result.window, result.encodedTokens, result.optionCount, backendTitle(result.backend)),
    style = MaterialTheme.typography.caption,
  )
  Text(
    stringResource(R.string.result_timing, result.tokenizeEmbedMs, result.graphMs, result.decodeMs),
    style = MaterialTheme.typography.caption,
  )
}

@Composable
private fun AnswerRow(row: AnswerUiRow) {
  Column {
    Text("${row.kind.key}: ${row.instructions}", style = MaterialTheme.typography.subtitle2)
    Text(row.answer, fontWeight = FontWeight.Medium, color = MaterialTheme.colors.primary)
    row.options.zip(row.probabilities).forEachIndexed { index, (option, probability) ->
      Text(
        stringResource(R.string.option_probability, option, probability),
        style = MaterialTheme.typography.body2,
        fontWeight = if (index == row.best) FontWeight.SemiBold else FontWeight.Normal,
      )
    }
  }
}
