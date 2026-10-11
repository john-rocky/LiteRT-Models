// SPDX-License-Identifier: Apache-2.0
package com.phocinae.view

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.Divider
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Switch
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.phocinae.AnswerUiRow
import com.phocinae.PhocinaeEngine
import com.phocinae.Preset
import com.phocinae.ProbabilityUiRow
import com.phocinae.R
import com.phocinae.UiState

/**
 * Product screen for local multilingual classification; all model work belongs to the ViewModel.
 */
@Composable
fun PhocinaeScreen(
  state: UiState,
  onInputChange: (String) -> Unit,
  onPreset: (Preset) -> Unit,
  onAccelerator: (PhocinaeEngine.Backend) -> Unit,
  onCalibration: (Boolean) -> Unit,
  onRun: () -> Unit,
) {
  val editable = !state.busy && !state.gateMode
  Scaffold(
    modifier = Modifier.statusBarsPadding().navigationBarsPadding().imePadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { insets ->
    Column(modifier = Modifier.fillMaxSize().padding(insets)) {
      StatusHeader(state, onAccelerator)
      LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
      ) {
        if (state.gateMode) {
          item { state.gateFile?.let { Text(stringResource(R.string.gate_file, it)) } }
        } else {
          item { PresetPicker(state.preset, editable, onPreset) }
          item {
            OutlinedTextField(
              value = state.inputText,
              onValueChange = onInputChange,
              label = { Text(stringResource(R.string.state_label)) },
              modifier = Modifier.fillMaxWidth(),
              enabled = editable,
              minLines = 5,
              maxLines = 10,
            )
          }
          item { AcceleratorPicker(state.accelerator, editable, state.npuAvailable, onAccelerator) }
          item {
            Row(
              modifier =
                Modifier.fillMaxWidth()
                  .toggleable(
                    value = state.calibrated,
                    enabled = editable,
                    role = Role.Switch,
                    onValueChange = onCalibration,
                  ),
              horizontalArrangement = Arrangement.SpaceBetween,
              verticalAlignment = Alignment.CenterVertically,
            ) {
              Column {
                Text(
                  stringResource(R.string.calibration_title),
                  style = MaterialTheme.typography.subtitle2,
                )
                Text(
                  stringResource(
                    if (state.calibrated) R.string.calibrated else R.string.uncalibrated
                  ),
                  style = MaterialTheme.typography.caption,
                )
              }
              Switch(checked = state.calibrated, onCheckedChange = null, enabled = editable)
            }
          }
          item {
            Button(
              onClick = onRun,
              enabled = editable && state.ready,
              modifier = Modifier.fillMaxWidth(),
            ) {
              Text(stringResource(R.string.run))
            }
          }
          state.runTotalMs?.let { totalMs ->
            item {
              Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                  stringResource(
                    R.string.run_summary,
                    state.answers.size,
                    totalMs,
                    state.runTokenCount,
                  ),
                  style = MaterialTheme.typography.subtitle1,
                )
                Text(
                  stringResource(R.string.token_count_hint),
                  style = MaterialTheme.typography.caption,
                )
              }
            }
          }
          if (state.answers.isNotEmpty()) {
            item {
              Text(stringResource(R.string.results_title), style = MaterialTheme.typography.h6)
            }
          }
          items(state.answers, key = { it.questionId }) { result -> AnswerCard(result) }
        }
      }
    }
  }
}

@Composable
private fun StatusHeader(state: UiState, onAccelerator: (PhocinaeEngine.Backend) -> Unit) {
  Column(
    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
    verticalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    Text(stringResource(state.statusMessage), style = MaterialTheme.typography.subtitle1)
    state.launchToReadyMs?.let {
      Text(stringResource(R.string.launch_to_ready, it), style = MaterialTheme.typography.caption)
    }
    if (state.busy) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    state.errorMessage?.let { message ->
      Text(stringResource(R.string.error_message, message), color = MaterialTheme.colors.error)
      if (state.canFallbackToCpu) {
        OutlinedButton(onClick = { onAccelerator(PhocinaeEngine.Backend.CPU) }, enabled = !state.busy) {
          Text(stringResource(R.string.use_cpu))
        }
      }
    }
  }
  Divider()
}

@Composable
private fun PresetPicker(selected: Preset, enabled: Boolean, onPreset: (Preset) -> Unit) {
  var expanded by remember { mutableStateOf(false) }
  Box {
    OutlinedButton(
      onClick = { expanded = true },
      enabled = enabled,
      modifier = Modifier.fillMaxWidth(),
    ) {
      Text(stringResource(R.string.preset_selected, stringResource(selected.title)))
    }
    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
      Preset.entries.forEach { preset ->
        DropdownMenuItem(
          onClick = {
            expanded = false
            onPreset(preset)
          }
        ) {
          Text(stringResource(preset.title))
        }
      }
    }
  }
}

@Composable
private fun AcceleratorPicker(
  selected: PhocinaeEngine.Backend,
  enabled: Boolean,
  npuAvailable: Boolean,
  onAccelerator: (PhocinaeEngine.Backend) -> Unit,
) {
  Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
    Text(stringResource(R.string.accelerator_title), style = MaterialTheme.typography.subtitle2)
    Row(modifier = Modifier.fillMaxWidth().selectableGroup()) {
      PhocinaeEngine.Backend.entries.forEach { backend ->
        // The NPU needs vendor libraries that are not in git; without them the option is disabled.
        val available = enabled && (backend != PhocinaeEngine.Backend.NPU || npuAvailable)
        Row(
          modifier =
            Modifier.weight(1f)
              .selectable(
                selected = selected == backend,
                enabled = available,
                role = Role.RadioButton,
                onClick = { onAccelerator(backend) },
              )
              .padding(vertical = 4.dp),
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          RadioButton(selected = selected == backend, onClick = null, enabled = available)
          Text(
            stringResource(
              when (backend) {
                PhocinaeEngine.Backend.GPU -> R.string.accelerator_gpu
                PhocinaeEngine.Backend.GPU_FP16 -> R.string.accelerator_gpu_fp16
                PhocinaeEngine.Backend.NPU -> R.string.accelerator_npu
                PhocinaeEngine.Backend.CPU -> R.string.accelerator_cpu
              }
            )
          )
        }
      }
    }
  }
}

@Composable
private fun AnswerCard(result: AnswerUiRow) {
  Card(modifier = Modifier.fillMaxWidth(), elevation = 2.dp) {
    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
      Text(result.instructions, style = MaterialTheme.typography.subtitle1)
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Text(
          result.answer,
          style = MaterialTheme.typography.h5,
          color = MaterialTheme.colors.primary,
          modifier = Modifier.weight(1f),
        )
        Text(
          stringResource(R.string.answer_probability, result.answerProbability * 100.0),
          style = MaterialTheme.typography.h5,
          color = MaterialTheme.colors.primary,
        )
      }
      // `map` is inline, so the composable string lookups are allowed inside it; joinToString is not.
      val otherLabels =
        result.probabilities
          .sortedByDescending { it.probability }
          .drop(1)
          .map { option ->
            val label =
              when {
                option.labelResource != null -> stringResource(option.labelResource)
                option.scoreLevel != null && option.description != null -> option.description
                option.scoreLevel != null -> stringResource(R.string.answer_level, option.scoreLevel)
                else -> option.label
              }
            "$label " + stringResource(R.string.answer_probability, option.probability * 100.0)
          }
      val others = otherLabels.joinToString(" · ")
      if (others.isNotEmpty()) {
        Text(stringResource(R.string.other_options, others), style = MaterialTheme.typography.caption)
      }
      Divider()
      Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(
          stringResource(R.string.result_confidence, result.confidence),
          style = MaterialTheme.typography.caption,
        )
        Text(
          stringResource(R.string.result_timing, result.totalMs),
          style = MaterialTheme.typography.caption,
        )
      }
    }
  }
}
