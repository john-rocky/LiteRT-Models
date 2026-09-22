// SPDX-License-Identifier: Apache-2.0
package com.sopro.view

import android.Manifest
import android.content.ClipData
import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.CircularProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TextButton
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.sopro.BuildConfig
import com.sopro.PlacementConfig
import com.sopro.R
import com.sopro.SoproEngine
import com.sopro.UiState
import java.io.File

@Composable
fun SoproScreen(
  state: UiState,
  onText: (String) -> Unit,
  onLanguage: (String) -> Unit,
  onPlacement: (PlacementConfig.Mode) -> Unit,
  onSynthesize: () -> Unit,
  onStop: () -> Unit,
  onDemo: () -> Unit,
  onPick: (android.net.Uri) -> Unit,
  onRecord: () -> Unit,
  onPermissionDenied: () -> Unit,
  onPlayAgain: () -> Unit,
  onExport: (android.net.Uri) -> Unit,
  onStyleVariant: (PlacementConfig.StyleVariant) -> Unit,
) {
  val context = LocalContext.current
  val editable = !state.busy && !state.gateMode
  val picker =
    rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
      uri?.let(onPick)
    }
  val recorder =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) onRecord() else onPermissionDenied()
    }
  val saver =
    rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
      uri?.let(onExport)
    }
  var showPlacement by remember { mutableStateOf(false) }
  val shareTitle = stringResource(R.string.share_title)
  Scaffold(
    modifier = Modifier.statusBarsPadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { padding ->
    Column(
      Modifier.padding(padding).fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Text(state.status)
      state.error?.let { Text(it, color = MaterialTheme.colors.error) }
      if (state.busy) CircularProgressIndicator()
      Text(stringResource(R.string.reference_heading), style = MaterialTheme.typography.h6)
      Text(stringResource(R.string.voice_permission_notice), style = MaterialTheme.typography.body2)
      Text(state.referenceLabel)
      Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedButton(onClick = onDemo, enabled = editable && state.referenceKind != "demo") {
          Text(stringResource(R.string.demo_voice))
        }
        OutlinedButton(onClick = { picker.launch(arrayOf("audio/*")) }, enabled = editable) {
          Text(stringResource(R.string.pick_audio))
        }
      }
      OutlinedButton(
        onClick = { recorder.launch(Manifest.permission.RECORD_AUDIO) },
        enabled = editable,
      ) {
        Text(stringResource(R.string.record_audio))
      }
      if (state.referenceKind == "demo")
        Text(stringResource(R.string.voice_attribution), style = MaterialTheme.typography.caption)
      Text(
        stringResource(R.string.reference_length, state.referenceSamples / 24000.0),
        style = MaterialTheme.typography.caption,
      )
      if (state.referenceSamples < 240000)
        Text(
          stringResource(R.string.reference_padding_notice),
          style = MaterialTheme.typography.caption,
        )
      Text(stringResource(R.string.language_heading), style = MaterialTheme.typography.h6)
      Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        listOf("en", "pt", "fr", "de").forEach { lang ->
          OutlinedButton(
            onClick = { onLanguage(lang) },
            enabled = editable && state.language != lang,
          ) {
            Text(
              stringResource(
                when (lang) {
                  "pt" -> R.string.lang_pt
                  "fr" -> R.string.lang_fr
                  "de" -> R.string.lang_de
                  else -> R.string.lang_en
                }
              )
            )
          }
        }
      }
      OutlinedTextField(
        value = state.text,
        onValueChange = onText,
        enabled = editable,
        modifier = Modifier.fillMaxWidth(),
        label = { Text(stringResource(R.string.text_label)) },
        minLines = 4,
      )
      Text(stringResource(R.string.placement_heading), style = MaterialTheme.typography.h6)
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(
          onClick = { onPlacement(PlacementConfig.Mode.AUTOMATIC) },
          enabled = editable && state.placementMode != PlacementConfig.Mode.AUTOMATIC,
        ) {
          Text(stringResource(R.string.placement_automatic))
        }
        OutlinedButton(
          onClick = { onPlacement(PlacementConfig.Mode.CPU) },
          enabled = editable && state.placementMode != PlacementConfig.Mode.CPU,
        ) {
          Text(stringResource(R.string.placement_all_cpu))
        }
      }
      OutlinedButton(
        onClick = { onPlacement(PlacementConfig.Mode.GPU_AR) },
        enabled = editable && state.placementMode != PlacementConfig.Mode.GPU_AR,
      ) {
        Text(stringResource(R.string.placement_gpu_ar))
      }
      Text(
        stringResource(
          when (state.placementMode) {
            PlacementConfig.Mode.AUTOMATIC ->
              if (state.deviceHybridAvailable) R.string.placement_hybrid_description
              else R.string.placement_cpu_fallback
            PlacementConfig.Mode.CPU -> R.string.placement_cpu_description
            PlacementConfig.Mode.GPU_AR -> R.string.placement_gpu_ar_description
            PlacementConfig.Mode.CUSTOM -> R.string.placement_custom_description
          }
        ),
        style = MaterialTheme.typography.caption,
      )
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(
          onClick = onSynthesize,
          enabled = editable && state.modelsReady && state.text.isNotBlank(),
        ) {
          Text(stringResource(R.string.synthesize))
        }
        OutlinedButton(onClick = onStop, enabled = state.busy && !state.gateMode) {
          Text(stringResource(R.string.stop))
        }
      }
      state.readyMs?.let {
        Text(stringResource(R.string.stat_ready_ms, it), style = MaterialTheme.typography.caption)
      }
      state.ttfaMs?.let { Text(stringResource(R.string.stat_ttfa, it)) }
      state.ttfaToOnsetMs?.let { Text(stringResource(R.string.stat_onset, it)) }
      state.totalWallMs?.let { Text(stringResource(R.string.stat_wall, it)) }
      state.rtf?.let { Text(stringResource(R.string.stat_rtf, it)) }
      if (state.placement.isNotEmpty())
        Text(stringResource(R.string.stat_placement, state.placement))
      state.seed?.let {
        Text(stringResource(R.string.stat_seed, it), style = MaterialTheme.typography.caption)
      }
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = onPlayAgain, enabled = editable && state.lastWavPath != null) {
          Text(stringResource(R.string.play_again))
        }
        OutlinedButton(
          onClick = {
            val file = state.lastWavPath?.let(::File) ?: return@OutlinedButton
            saver.launch(file.name)
          },
          enabled = editable && state.lastWavPath != null,
        ) {
          Text(stringResource(R.string.save))
        }
        OutlinedButton(
          onClick = {
            val file = state.lastWavPath?.let(::File) ?: return@OutlinedButton
            val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
            val send =
              Intent(Intent.ACTION_SEND)
                .setType("audio/wav")
                .putExtra(Intent.EXTRA_STREAM, uri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                .apply { clipData = ClipData.newRawUri("", uri) }
            context.startActivity(Intent.createChooser(send, shareTitle))
          },
          enabled = editable && state.lastWavPath != null,
        ) {
          Text(stringResource(R.string.share))
        }
      }
      if (state.lastWavPath != null)
        Text(stringResource(R.string.saved_locally), style = MaterialTheme.typography.caption)
      if (BuildConfig.DEBUG) {
        TextButton(onClick = { showPlacement = !showPlacement }) {
          Text(stringResource(R.string.debug_placement))
        }
        if (showPlacement) {
          Text(stringResource(R.string.placement_details), style = MaterialTheme.typography.caption)
          state.graphPlacements.forEach { row ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
              Text(stringResource(graphLabel(row.graph)), style = MaterialTheme.typography.caption)
              Text(
                stringResource(
                  when (row.backend) {
                    SoproEngine.Backend.CPU -> R.string.cpu
                    SoproEngine.Backend.GPU -> R.string.gpu
                    SoproEngine.Backend.GPU32 -> R.string.gpu_fp32
                  }
                ),
                style = MaterialTheme.typography.caption,
              )
            }
          }
          Text(
            stringResource(R.string.style_storage_heading),
            style = MaterialTheme.typography.caption,
          )
          Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
              onClick = { onStyleVariant(PlacementConfig.StyleVariant.FP32) },
              enabled = editable && state.styleVariant != PlacementConfig.StyleVariant.FP32,
            ) {
              Text(stringResource(R.string.style_fp32))
            }
            OutlinedButton(
              onClick = { onStyleVariant(PlacementConfig.StyleVariant.WFP16) },
              enabled = editable && state.styleVariant != PlacementConfig.StyleVariant.WFP16,
            ) {
              Text(stringResource(R.string.style_wfp16))
            }
          }
        }
      }
    }
  }
}

private fun graphLabel(graph: String): Int =
  when (graph) {
    "speaker_encoder" -> R.string.graph_speaker
    "semantic_encoder" -> R.string.graph_semantic
    "style_prefix" -> R.string.graph_style
    "ar_merged" -> R.string.graph_ar
    "acoustic_condition" -> R.string.graph_condition
    "acoustic_condition_t4096" -> R.string.graph_condition_long
    "acoustic_velocity" -> R.string.graph_velocity
    "acoustic_velocity_t4096" -> R.string.graph_velocity_long
    "vocoder_stream_start" -> R.string.graph_vocoder_start
    "vocoder_stream_step" -> R.string.graph_vocoder_step
    "vocoder_stream_flush" -> R.string.graph_vocoder_flush
    else -> error("Unknown graph $graph")
  }
