// SPDX-License-Identifier: Apache-2.0
package com.phocinae

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.phocinae.view.ApplicationTheme
import com.phocinae.view.PhocinaeScreen

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
      requestedStorage = intent.getStringExtra("storage"),
      launchedAtNs = launchedAtNs,
    )
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        PhocinaeScreen(
          state = state,
          onInputChange = viewModel::setInputText,
          onPreset = viewModel::selectPreset,
          onAccelerator = viewModel::selectAccelerator,
          onCalibration = viewModel::setCalibrated,
          onRun = viewModel::run,
        )
      }
    }
  }
}
