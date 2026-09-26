package com.gliformer

import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliformer.view.ApplicationTheme
import com.gliformer.view.GliformerScreen

class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    val launchNs = SystemClock.elapsedRealtimeNanos()
    super.onCreate(savedInstanceState)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    viewModel.start(intent.getStringExtra("table"), launchNs)
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        GliformerScreen(
          state,
          viewModel::setInputText,
          viewModel::selectBackend,
          viewModel::extract,
          viewModel::retry,
          viewModel::resultRendered,
        )
      }
    }
  }
}
