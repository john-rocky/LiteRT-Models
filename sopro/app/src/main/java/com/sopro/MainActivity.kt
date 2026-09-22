// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sopro.view.ApplicationTheme
import com.sopro.view.SoproScreen

class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    val activityStartedNanos = System.nanoTime()
    val activityStartedUnixMs = System.currentTimeMillis()
    super.onCreate(savedInstanceState)
    viewModel.start(intent, activityStartedNanos, activityStartedUnixMs)
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        SoproScreen(
          state,
          viewModel::setText,
          viewModel::selectLanguage,
          viewModel::selectPlacement,
          viewModel::synthesize,
          viewModel::stop,
          viewModel::selectDemo,
          viewModel::pickReference,
          viewModel::recordReference,
          viewModel::permissionDenied,
          viewModel::playAgain,
          viewModel::exportLastWav,
          viewModel::selectStyleVariant,
        )
      }
    }
  }
}
