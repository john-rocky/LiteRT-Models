package com.gliner25.view

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Divider
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.unit.dp
import com.gliner25.ExtractionUiResult
import com.gliner25.GlinerExtractor
import com.gliner25.R
import com.gliner25.UiState

@Composable
@OptIn(ExperimentalLayoutApi::class)
fun GlinerScreen(
  state: UiState,
  onTextChanged: (String) -> Unit,
  onAcceleratorChanged: (GlinerExtractor.Backend) -> Unit,
  onExtract: () -> Unit,
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
          GlinerExtractor.Backend.entries.forEach { backend ->
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
        FlowRow(
          Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.spacedBy(8.dp),
          verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          EntityLabel.entries.forEach { label ->
            Surface(color = label.color.copy(alpha = 0.12f), shape = RoundedCornerShape(16.dp)) {
              Text(
                stringResource(label.title),
                color = label.color,
                modifier = Modifier.padding(10.dp, 6.dp),
              )
            }
          }
        }
        Button(
          onClick = onExtract,
          enabled = !state.busy && state.inputText.isNotBlank(),
          modifier = Modifier.fillMaxWidth(),
        ) {
          Text(stringResource(R.string.extract))
        }
        state.result?.let { ResultContent(it) }
      }
    }
  }
}

@Composable
private fun acceleratorTitle(backend: GlinerExtractor.Backend): String =
  stringResource(
    if (backend == GlinerExtractor.Backend.GPU) {
      R.string.accelerator_gpu
    } else {
      R.string.accelerator_cpu
    }
  )

@Composable
private fun ResultContent(result: ExtractionUiResult) {
  val annotated =
    remember(result) {
      buildAnnotatedString {
        append(result.text)
        result.highlightedSpans.forEach { span ->
          addStyle(
            SpanStyle(background = EntityLabel.fromKey(span.label).color.copy(alpha = 0.23f)),
            result.text.offsetByCodePoints(0, span.start),
            result.text.offsetByCodePoints(0, span.end),
          )
        }
      }
    }
  Divider()
  Text(stringResource(R.string.entities_title), style = MaterialTheme.typography.h6)
  SelectionContainer { Text(annotated, style = MaterialTheme.typography.body1) }
  if (result.spans.isEmpty()) {
    Text(stringResource(R.string.no_entities))
  }
  result.spans.forEach { span ->
    val label = EntityLabel.fromKey(span.label)
    Text(
      stringResource(
        R.string.entity_row,
        stringResource(label.title),
        span.text,
        span.confidence,
        span.start,
        span.end,
      ),
      color = label.color,
    )
  }
  Divider()
  Text(
    stringResource(
      R.string.result_window,
      result.window,
      result.encodedTokens,
      result.textWords,
      acceleratorTitle(result.accelerator),
    ),
    style = MaterialTheme.typography.caption,
  )
  Text(
    stringResource(R.string.result_timing, result.tokenizeEmbedMs, result.graphMs, result.decodeMs),
    style = MaterialTheme.typography.caption,
  )
}
