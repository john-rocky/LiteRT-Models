package com.d1omni.view

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.d1omni.R
import com.d1omni.UiState

/** The app's colors (Material 1). */
@Composable
fun ApplicationTheme(content: @Composable () -> Unit) {
  MaterialTheme(colors = MaterialTheme.colors.copy(primary = Indigo), content = content)
}

/** The title, the status line and the engine line. */
@Composable
fun StatusScreen(state: UiState, modifier: Modifier = Modifier) {
  Surface(modifier = modifier.fillMaxSize()) {
    Column(modifier = Modifier.safeDrawingPadding().padding(16.dp)) {
      Text(stringResource(R.string.app_name), style = MaterialTheme.typography.h6)
      Spacer(Modifier.height(12.dp))
      Text(
        state.status,
        style = MaterialTheme.typography.body1,
        color = if (state.error) ErrorRed else Color.Unspecified,
      )
      if (state.engineLine.isNotEmpty()) {
        Spacer(Modifier.height(8.dp))
        Text(state.engineLine, style = MaterialTheme.typography.body2)
      }
    }
  }
}

private val Indigo = Color(0xFF3949AB)
private val ErrorRed = Color(0xFFB00020)
