// SPDX-License-Identifier: Apache-2.0
package com.phocinae

import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
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
    val gate = BuildConfig.DEBUG && intent.getBooleanExtra("gate", false)
    keepVisible(gate)
    viewModel.start(
      gate = gate,
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

  /**
   * On a locked test phone the measured process must stay in the foreground (top-app): in the debug build, and for
   * every gate launch, show above the keyguard, turn the screen on and keep it on. A normal launch of the release
   * build is unchanged.
   */
  private fun keepVisible(gate: Boolean) {
    val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
    if (!debuggable && !gate) return
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
  }
}
