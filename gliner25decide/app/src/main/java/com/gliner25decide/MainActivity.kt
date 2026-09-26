package com.gliner25decide

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliner25decide.view.ApplicationTheme
import com.gliner25decide.view.DecideScreen

/**
 * Thin host of the Compose screen. A debug build also accepts diagnostic extras: `gate` (fixture
 * gate), `firsttap` (first classification after startup), `paced` with `count` and `interval_ms`,
 * and `accel` (`GPU` or `CPU`).
 */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val diagnostic =
      BuildConfig.DEBUG &&
        listOf(EXTRA_GATE, EXTRA_FIRST_TAP, EXTRA_PACED).any { intent.getBooleanExtra(it, false) }
    if (diagnostic && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      // Debug measurements on a locked test phone: behind a secure keyguard the process is not
      // top-app, so show above it; a normal launch is unchanged.
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    viewModel.start(
      intent.getBooleanExtra(EXTRA_GATE, false),
      intent.getStringExtra(EXTRA_ACCELERATOR),
      BuildConfig.DEBUG && intent.getBooleanExtra(EXTRA_FIRST_TAP, false),
      if (BuildConfig.DEBUG && intent.getBooleanExtra(EXTRA_PACED, false)) {
        intent.getIntExtra(EXTRA_COUNT, MainViewModel.DEFAULT_PACED_COUNT)
      } else {
        0
      },
      intent.getIntExtra(EXTRA_INTERVAL_MS, MainViewModel.DEFAULT_PACED_INTERVAL_MS).toLong(),
    )
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        DecideScreen(
          state,
          viewModel::setInputText,
          viewModel::setTasksText,
          viewModel::selectAccelerator,
          viewModel::classify,
        )
      }
    }
  }

  private companion object {
    const val EXTRA_GATE = "gate"
    const val EXTRA_FIRST_TAP = "firsttap"
    const val EXTRA_PACED = "paced"
    const val EXTRA_ACCELERATOR = "accel"
    const val EXTRA_COUNT = "count"
    const val EXTRA_INTERVAL_MS = "interval_ms"
  }
}
