package com.gliclass

import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliclass.view.ApplicationTheme
import com.gliclass.view.GliclassScreen

/**
 * Thin host of the Compose screen. The debug build accepts `gate` (fixture gate) with `backend`
 * (`gpu` or `cpu`) and `report` (file name in `files/`); the debug and benchmark builds accept
 * `firsttap` (first classification after startup) and `paced` with `count` and `interval_ms`.
 */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    val createdAt = SystemClock.elapsedRealtime()
    super.onCreate(savedInstanceState)
    val gate = BuildConfig.DEBUG && intent.getBooleanExtra(EXTRA_GATE, false)
    val diagnostics = GliclassDiagnostics.enabled
    val firstTap = diagnostics && intent.getBooleanExtra(EXTRA_FIRST_TAP, false)
    val pacedCount =
      if (diagnostics && intent.getBooleanExtra(EXTRA_PACED, false)) {
        intent.getIntExtra(EXTRA_COUNT, MainViewModel.DEFAULT_PACED_COUNT)
      } else {
        0
      }
    if (
      (gate || firstTap || pacedCount > 0) && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1
    ) {
      // Measurements on a locked test phone: behind a secure keyguard the process is not top-app,
      // so show above it; a normal launch is unchanged.
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    viewModel.start(
      gate,
      intent.getStringExtra(EXTRA_BACKEND),
      intent.getStringExtra(EXTRA_REPORT),
      firstTap,
      pacedCount,
      intent.getIntExtra(EXTRA_INTERVAL_MS, MainViewModel.DEFAULT_PACED_INTERVAL_MS).toLong(),
      createdAt,
    )
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        GliclassScreen(
          state,
          viewModel::setInputText,
          viewModel::setLabelsText,
          viewModel::setPromptText,
          viewModel::setMode,
          viewModel::setThreshold,
          viewModel::selectAccelerator,
          viewModel::classify,
        )
      }
    }
  }

  private companion object {
    const val EXTRA_GATE = "gate"
    const val EXTRA_BACKEND = "backend"
    const val EXTRA_REPORT = "report"
    const val EXTRA_FIRST_TAP = "firsttap"
    const val EXTRA_PACED = "paced"
    const val EXTRA_COUNT = "count"
    const val EXTRA_INTERVAL_MS = "interval_ms"
  }
}
