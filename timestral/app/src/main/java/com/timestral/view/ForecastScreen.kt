package com.timestral.view

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.testTagsAsResourceId
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.timestral.Backend
import com.timestral.Example
import com.timestral.ForecastView
import com.timestral.R
import com.timestral.UiState

/**
 * The one screen: the series (pasted, opened from a file or an example), the horizon, the
 * accelerator, the Forecast button, then the chart and the numbers of the last forecast.
 */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun ForecastScreen(
  state: UiState,
  onSeries: (String) -> Unit,
  onHorizon: (String) -> Unit,
  onBackend: (Backend) -> Unit,
  onExample: (Example) -> Unit,
  onOpenFile: () -> Unit,
  onForecast: () -> Unit,
  modifier: Modifier = Modifier,
) {
  val scroll = rememberScrollState()
  val keyboard = LocalSoftwareKeyboardController.current
  // A finished forecast or a refusal scrolls into view below the inputs.
  LaunchedEffect(state.result, state.error) {
    if (state.result != null || state.error != null) scroll.animateScrollTo(scroll.maxValue)
  }
  Scaffold(
    modifier = modifier.statusBarsPadding().semantics { testTagsAsResourceId = true },
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { padding ->
    Column(
      modifier =
        Modifier.padding(padding)
          .fillMaxSize()
          .verticalScroll(scroll)
          .navigationBarsPadding()
          .imePadding()
          .padding(horizontal = 16.dp, vertical = 12.dp),
      verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
      if (state.missingFiles.isNotEmpty()) {
        Text(
          stringResource(R.string.missing_files, state.missingFiles.size),
          color = errorRed,
          style = MaterialTheme.typography.body2,
          modifier = Modifier.testTag("missing_files"),
        )
      }
      OutlinedTextField(
        value = state.seriesText,
        onValueChange = onSeries,
        label = { Text(stringResource(R.string.series_label)) },
        placeholder = { Text(stringResource(R.string.series_placeholder)) },
        maxLines = 6,
        modifier = Modifier.fillMaxWidth().heightIn(min = 96.dp).testTag("series"),
      )
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = onOpenFile, modifier = Modifier.testTag("open_file")) {
          Text(stringResource(R.string.open_file))
        }
        ExampleMenu(state.examples, onExample)
        OutlinedButton(onClick = { onSeries("") }, modifier = Modifier.testTag("clear")) {
          Text(stringResource(R.string.clear))
        }
      }
      OutlinedTextField(
        value = state.horizonText,
        onValueChange = onHorizon,
        label = { Text(stringResource(R.string.horizon_label)) },
        singleLine = true,
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        modifier = Modifier.fillMaxWidth().testTag("horizon"),
      )
      Text(stringResource(R.string.accelerator), style = MaterialTheme.typography.subtitle2)
      BackendChoice(state, onBackend)
      Button(
        onClick = {
          keyboard?.hide()
          onForecast()
        },
        enabled = !state.running,
        modifier = Modifier.fillMaxWidth().testTag("forecast"),
      ) {
        Text(stringResource(R.string.forecast))
      }
      state.status?.let { status ->
        Row(verticalAlignment = Alignment.CenterVertically) {
          CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
          Spacer(Modifier.width(8.dp))
          Text(status, style = MaterialTheme.typography.body2, modifier = Modifier.testTag("status"))
        }
      }
      state.error?.let { error ->
        Text(error, color = errorRed, modifier = Modifier.testTag("error"))
      }
      state.result?.let { ResultSection(it) }
    }
  }
}

@Composable
private fun ExampleMenu(examples: List<Example>, onExample: (Example) -> Unit) {
  var expanded by remember { mutableStateOf(false) }
  Box {
    OutlinedButton(onClick = { expanded = true }, modifier = Modifier.testTag("examples")) {
      Text(stringResource(R.string.examples))
    }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      for (example in examples) {
        DropdownMenuItem(
          onClick = {
            expanded = false
            onExample(example)
          },
          modifier = Modifier.testTag("example_${example.asset.substringBefore('.')}"),
        ) {
          Text(example.title)
        }
      }
    }
  }
}

/** One row per accelerator; the whole row is the touch target (not only the radio button). */
@Composable
private fun BackendChoice(state: UiState, onBackend: (Backend) -> Unit) {
  Column(Modifier.selectableGroup()) {
    for (backend in Backend.entries) {
      val enabled = !state.running && (backend != Backend.NPU || state.npuUnavailable == null)
      Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier =
          Modifier.fillMaxWidth()
            .selectable(
              selected = state.backend == backend,
              enabled = enabled,
              role = Role.RadioButton,
              onClick = { onBackend(backend) },
            )
            .padding(vertical = 2.dp)
            .testTag("backend_${backend.name.lowercase()}"),
      ) {
        RadioButton(selected = state.backend == backend, onClick = null, enabled = enabled)
        Spacer(Modifier.width(8.dp))
        Text(
          when (backend) {
            Backend.CPU -> stringResource(R.string.backend_cpu)
            Backend.GPU -> stringResource(R.string.backend_gpu)
            Backend.NPU -> stringResource(R.string.backend_npu)
          },
          color = if (enabled) Color.Unspecified else Color.Gray,
        )
      }
    }
    state.npuUnavailable?.let { reason ->
      Text(
        reason,
        style = MaterialTheme.typography.caption,
        color = Color.Gray,
        modifier = Modifier.padding(start = 40.dp).testTag("npu_unavailable"),
      )
    }
  }
}

@Composable
private fun ResultSection(result: ForecastView) {
  Column(verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.testTag("result")) {
    ForecastChart(result.chart, modifier = Modifier.fillMaxWidth().height(240.dp))
    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
      LegendItem(historyGray, stringResource(R.string.legend_history))
      LegendItem(medianBlue, stringResource(R.string.legend_median))
      LegendItem(bandBlue, stringResource(R.string.legend_band))
    }
    for (line in result.summary) {
      Text(line, style = MaterialTheme.typography.body2)
    }
    for (note in result.notes) {
      Text(note, style = MaterialTheme.typography.caption, color = Color.DarkGray)
    }
  }
}

@Composable
private fun LegendItem(color: Color, label: String) {
  Row(verticalAlignment = Alignment.CenterVertically) {
    Box(Modifier.size(12.dp).background(color))
    Spacer(Modifier.width(4.dp))
    Text(label, style = MaterialTheme.typography.caption)
  }
}
