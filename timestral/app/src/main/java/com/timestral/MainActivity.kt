package com.timestral

import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.timestral.view.ApplicationTheme
import com.timestral.view.ForecastScreen

/** Hosts the forecast screen and the file picker. */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    keepVisibleInDebugBuilds()
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      val picker =
        rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
          uri?.let(viewModel::loadDocument)
        }
      ApplicationTheme {
        ForecastScreen(
          state,
          onSeries = viewModel::setSeries,
          onHorizon = viewModel::setHorizon,
          onBackend = viewModel::selectBackend,
          onExample = viewModel::loadExample,
          onOpenFile = { picker.launch(DOCUMENT_TYPES) },
          onForecast = viewModel::forecast,
        )
      }
    }
  }

  /**
   * A debug build stays on screen above the keyguard and keeps the screen on, so a test phone with
   * a locked screen can still be driven over adb; release builds behave as usual.
   */
  private fun keepVisibleInDebugBuilds() {
    if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
  }

  companion object {
    /** The picker offers text and CSV files (some providers label CSV as other types). */
    private val DOCUMENT_TYPES =
      arrayOf(
        "text/plain",
        "text/csv",
        "text/comma-separated-values",
        "application/csv",
        "application/vnd.ms-excel",
        "text/*",
      )
  }
}
