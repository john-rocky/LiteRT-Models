package com.gliner25

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.gliner25.view.ApplicationTheme
import com.gliner25.view.GlinerScreen

class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    viewModel.start(
      intent.getBooleanExtra("gate", false),
      intent.getStringExtra("accel"),
      intent.getStringExtra("set"),
    )
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        GlinerScreen(
          state,
          viewModel::setInputText,
          viewModel::selectAccelerator,
          viewModel::extract,
        )
      }
    }
  }
}
