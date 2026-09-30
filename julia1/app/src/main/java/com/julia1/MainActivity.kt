// SPDX-License-Identifier: Apache-2.0
package com.julia1

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.julia1.view.ApplicationTheme
import com.julia1.view.JuliaScreen

/** Thin Compose host; the ViewModel owns initialization and inference. */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    val launchedAtNs = System.nanoTime()
    super.onCreate(savedInstanceState)
    viewModel.start(
      gate = BuildConfig.DEBUG && intent.getBooleanExtra("gate", false),
      accelerator = intent.getStringExtra("accel"),
      requestedWindow = intent.getStringExtra("window"),
      launchedAtNs = launchedAtNs,
    )
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        JuliaScreen(
          state = state,
          onInputChange = viewModel::setInputText,
          onPreset = viewModel::selectPreset,
          onAccelerator = viewModel::selectAccelerator,
          onRun = viewModel::run,
        )
      }
    }
  }
}
