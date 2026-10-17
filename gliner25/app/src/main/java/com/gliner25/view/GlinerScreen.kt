package com.gliner25.view

import androidx.annotation.StringRes
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.unit.dp
import com.gliner25.ExtractionUiResult
import com.gliner25.GlinerExtractor
import com.gliner25.GlinerInputs
import com.gliner25.GlinerModel
import com.gliner25.R
import com.gliner25.UiState

@Composable
@OptIn(ExperimentalLayoutApi::class)
fun GlinerScreen(
  state: UiState,
  onTextChanged: (String) -> Unit,
  onModelChanged: (GlinerModel) -> Unit,
  onSplitterChanged: (GlinerInputs.Splitter) -> Unit,
  onAcceleratorChanged: (GlinerExtractor.Backend) -> Unit,
  onExtract: () -> Unit,
) {
  val scrollState = rememberScrollState()
  // A new result scrolls into view when the controls and input fill the screen.
  LaunchedEffect(state.result) {
    if (state.result != null) {
      withFrameNanos {}
      scrollState.animateScrollTo(scrollState.maxValue)
    }
  }
  Scaffold(
    modifier = Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { contentPadding ->
    Column(
      Modifier.padding(contentPadding)
        .fillMaxSize()
        .verticalScroll(scrollState)
        .padding(horizontal = 16.dp, vertical = 12.dp),
      verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
      Text(
        stringResource(
          R.string.status_line,
          stringResource(state.model.title),
          stringResource(state.statusMessage),
        ),
        style = MaterialTheme.typography.subtitle1,
      )
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
        ChoiceRow(
          R.string.model_label,
          GlinerModel.entries,
          state.model,
          !state.busy,
          { stringResource(it.shortTitle) },
          onModelChanged,
        )
        if (state.model.splitters.size > 1) {
          ChoiceRow(
            R.string.splitter_label,
            state.model.splitters,
            state.splitter,
            !state.busy,
            { splitterTitle(it) },
            onSplitterChanged,
          )
        }
        ChoiceRow(
          R.string.accelerator_title,
          GlinerExtractor.Backend.entries,
          state.accelerator,
          !state.busy,
          { acceleratorTitle(it) },
          onAcceleratorChanged,
        )
        OutlinedTextField(
          value = state.inputText,
          onValueChange = onTextChanged,
          modifier = Modifier.fillMaxWidth(),
          enabled = !state.busy,
          minLines = 3,
          maxLines = 8,
          label = { Text(stringResource(R.string.input_label)) },
        )
        FlowRow(
          Modifier.fillMaxWidth(),
          horizontalArrangement = Arrangement.spacedBy(6.dp),
          verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
          EntityLabel.entries.forEach { label ->
            Surface(color = label.color.copy(alpha = 0.12f), shape = RoundedCornerShape(16.dp)) {
              Text(
                stringResource(label.title),
                color = label.color,
                style = MaterialTheme.typography.body2,
                modifier = Modifier.padding(8.dp, 4.dp),
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

/** One labelled single-choice row; each option's radio button and label select it together. */
@Composable
private fun <T> ChoiceRow(
  @StringRes title: Int,
  options: List<T>,
  selected: T,
  enabled: Boolean,
  label: @Composable (T) -> String,
  onSelect: (T) -> Unit,
) {
  Row(verticalAlignment = Alignment.CenterVertically) {
    Text(
      stringResource(title),
      Modifier.width(76.dp),
      style = MaterialTheme.typography.subtitle2,
    )
    options.forEach { option ->
      Row(
        Modifier.heightIn(min = 40.dp)
          .selectable(
            selected = option == selected,
            enabled = enabled,
            role = Role.RadioButton,
            onClick = { onSelect(option) },
          )
          .padding(end = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
      ) {
        RadioButton(selected = option == selected, onClick = null, enabled = enabled)
        Text(label(option), Modifier.padding(start = 6.dp))
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
private fun splitterTitle(splitter: GlinerInputs.Splitter): String =
  stringResource(
    when (splitter) {
      GlinerInputs.Splitter.WHITESPACE -> R.string.splitter_whitespace
      GlinerInputs.Splitter.CHAR -> R.string.splitter_char
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
  Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
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
  }
  Divider()
  Text(
    if (result.model.splitters.size > 1) {
      stringResource(
        R.string.result_window_splitter,
        result.window,
        result.encodedTokens,
        result.textWords,
        acceleratorTitle(result.accelerator),
        splitterTitle(result.splitter),
      )
    } else {
      stringResource(
        R.string.result_window,
        result.window,
        result.encodedTokens,
        result.textWords,
        acceleratorTitle(result.accelerator),
      )
    },
    style = MaterialTheme.typography.body2,
  )
  Text(
    stringResource(R.string.result_timing, result.tokenizeEmbedMs, result.graphMs, result.decodeMs),
    style = MaterialTheme.typography.body2,
  )
}
