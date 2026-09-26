package com.gliformer

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.gliformer.view.ApplicationTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Debug-only entry point; the interactive ViewModel never starts during gates. */
class GliformerGateActivity : ComponentActivity() {
  private var extractor: GliformerExtractor? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    var status by mutableStateOf(getString(R.string.gate_running))
    setContent {
      ApplicationTheme {
        Scaffold(modifier = Modifier.statusBarsPadding()) { padding ->
          Text(status, modifier = Modifier.padding(padding).padding(24.dp))
        }
      }
    }
    lifecycleScope.launch(Dispatchers.Default) {
      val runner = GliformerGateRunner(this@GliformerGateActivity, intent)
      val summary = runner.run { extractor = it }
      withContext(Dispatchers.Main) {
        status =
          getString(
            R.string.gate_complete,
            summary.optString("status"),
            runner.reportName,
            summary.optInt("completed_inputs"),
            summary.optInt("selected_inputs"),
          )
      }
    }
  }

  override fun onDestroy() {
    extractor?.close()
    super.onDestroy()
  }
}
