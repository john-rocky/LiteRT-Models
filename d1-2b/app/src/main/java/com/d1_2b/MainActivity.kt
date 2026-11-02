package com.d1_2b

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {
  private val model: MainViewModel by viewModels()

  private val pickPhoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> model.setPhoto(uri) }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    keepVisible(intent)
    setContent {
      val state by model.state.collectAsStateWithLifecycle()
      ExtractionScreen(state, model) {
        pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
      }
    }
    if (savedInstanceState == null) runFrom(intent)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    keepVisible(intent)
    runFrom(intent)
  }

  /** `adb shell am start -n <package>/.MainActivity --es run <file>`: a run file of the app's files directory. */
  private fun runFrom(intent: Intent?) {
    intent?.getStringExtra("run")?.let { model.run(it) }
  }

  /**
   * A run launch drives the app from adb, often on a locked test phone: the app then shows above the
   * keyguard, turns the screen on and keeps it on. A launch from the launcher is unchanged.
   */
  private fun keepVisible(intent: Intent?) {
    if (intent?.getStringExtra("run") == null) return
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
  }
}
