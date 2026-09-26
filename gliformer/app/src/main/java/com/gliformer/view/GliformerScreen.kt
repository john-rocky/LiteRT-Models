package com.gliformer.view

import android.os.SystemClock
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
import androidx.compose.material.RadioButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.unit.dp
import com.gliformer.GliformerExtractor
import com.gliformer.R
import com.gliformer.UiState

@Composable
fun GliformerScreen(
  state: UiState,
  onInputChanged: (String) -> Unit,
  onBackendSelected: (GliformerExtractor.Backend) -> Unit,
  onExtract: (Long) -> Unit,
  onRetry: () -> Unit,
  onResultRendered: (Int, Long) -> Unit,
) {
  Scaffold(
    modifier = Modifier.statusBarsPadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { padding ->
    Column(
      modifier =
        Modifier.fillMaxSize()
          .padding(padding)
          .padding(16.dp)
          .verticalScroll(rememberScrollState()),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Text(
        stringResource(
          when (state.phase) {
            UiState.Phase.LOADING -> R.string.status_loading
            UiState.Phase.WARMING -> R.string.status_warming
            UiState.Phase.READY -> R.string.status_ready
            UiState.Phase.EXTRACTING -> R.string.status_extracting
            UiState.Phase.ERROR -> R.string.status_error
          }
        ),
        style = MaterialTheme.typography.h6,
      )
      if (state.busy) CircularProgressIndicator()
      Text(stringResource(R.string.fixed_labels), style = MaterialTheme.typography.body2)
      OutlinedTextField(
        value = state.text,
        onValueChange = onInputChanged,
        label = { Text(stringResource(R.string.input_label)) },
        modifier = Modifier.fillMaxWidth(),
        enabled = !state.busy,
        minLines = 3,
        maxLines = 5,
      )
      Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        GliformerExtractor.Backend.entries.forEach { backend ->
          Row {
            RadioButton(
              selected = state.backend == backend,
              enabled = !state.busy,
              onClick = { onBackendSelected(backend) },
            )
            Text(
              stringResource(
                if (backend == GliformerExtractor.Backend.GPU) R.string.backend_gpu
                else R.string.backend_cpu
              ),
              modifier = Modifier.padding(top = 12.dp),
            )
          }
        }
      }
      Button(
        onClick = { onExtract(SystemClock.elapsedRealtimeNanos()) },
        enabled = state.phase == UiState.Phase.READY && state.text.isNotBlank(),
        modifier = Modifier.fillMaxWidth(),
      ) {
        Text(stringResource(R.string.extract))
      }
      state.errorMessage?.let {
        Text(it, color = MaterialTheme.colors.error)
        OutlinedButton(onClick = onRetry, enabled = !state.busy) {
          Text(stringResource(R.string.retry))
        }
      }
      state.result?.let { result ->
        val highlighted =
          remember(result) {
            buildAnnotatedString {
              append(result.text)
              result.entities.forEach { entity ->
                val start = result.text.offsetByCodePoints(0, entity.start)
                val end = result.text.offsetByCodePoints(0, entity.end)
                addStyle(SpanStyle(background = entityColor(entity.label)), start, end)
              }
            }
          }
        Column(
          modifier =
            Modifier.fillMaxWidth().drawWithContent {
              drawContent()
              if (state.phase == UiState.Phase.READY) {
                onResultRendered(state.requestId, SystemClock.elapsedRealtimeNanos())
              }
            },
          verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
          Text(
            stringResource(R.string.window_info, result.window, result.encodedTokens, result.words),
            style = MaterialTheme.typography.subtitle2,
          )
          Text(
            stringResource(
              R.string.timing_info,
              result.timings.tokenizeLookupMs,
              result.timings.graphMs,
              result.timings.decodeMs,
            ),
            style = MaterialTheme.typography.caption,
          )
          Text(highlighted, style = MaterialTheme.typography.body1)
          if (result.entities.isEmpty()) Text(stringResource(R.string.no_entities))
          result.entities.forEach { entity ->
            Text(
              stringResource(
                R.string.entity_info,
                entity.text,
                entity.label,
                entity.score * 100f,
                entity.start,
                entity.end,
              ),
              style = MaterialTheme.typography.body2,
            )
          }
          Text(
            stringResource(R.string.offset_explanation),
            style = MaterialTheme.typography.caption,
          )
        }
      }
    }
  }
}
