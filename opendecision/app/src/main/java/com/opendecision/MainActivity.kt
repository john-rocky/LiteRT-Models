package com.opendecision

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.opendecision.view.ApplicationTheme
import com.opendecision.view.DecisionScreen

/**
 * Thin host of the Compose screen. A debug build also accepts diagnostic extras: `gate` (fixture gate against
 * the captured Python ids, spans and logits) and `accel` (`GPU` or `CPU`).
 */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val gate = BuildConfig.DEBUG && intent.getBooleanExtra(EXTRA_GATE, false)
    if (gate && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      // Debug measurements on a locked test phone: behind a secure keyguard the process is not top-app.
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    viewModel.start(gate, intent.getStringExtra(EXTRA_ACCELERATOR))
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        DecisionScreen(
          state,
          viewModel::setStateText,
          viewModel::setQuestionsText,
          viewModel::selectBackend,
          viewModel::loadPreset,
          viewModel::decide,
        )
      }
    }
  }

  private companion object {
    const val EXTRA_GATE = "gate"
    const val EXTRA_ACCELERATOR = "accel"
  }
}
