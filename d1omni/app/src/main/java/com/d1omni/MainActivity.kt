package com.d1omni

import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.d1omni.view.ApplicationTheme
import com.d1omni.view.StatusScreen

/**
 * Hosts the Compose screen (`launchMode="singleTop"`: a later intent reaches the running activity
 * through [onNewIntent]). Extras ([D1Launch.parse]):
 * - every launch: `[--es backend gpu|cpu]`, `[--es precision fp32|fp16acc]` (the GPU precision
 *   of the decision graphs; default fp32) and `[--es precision_audio fp16acc|fp32]` (the audio
 *   graph's; default [D1AudioEngine.DEFAULT_PRECISION])
 * - debug build: `--ez gate true --es fixture <rows file in files/> --es report <name.json>
 *   [--ei limit n] [--ei resident 128]` (the gate: every row of the file on its bucket, with
 *   `resident` compiled first and kept; a rows file of kind audio runs its clips' wavs through the
 *   audio graph first)
 * - debug build: `--ez timing true --es rows <rows file in files/> --es report <name.json>
 *   [--ei warmup 5] [--ei reps 20] [--ei cool_ms 120000] [--es sets card3,one]`
 */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    val launch = parse(intent)
    keepVisible(launch)
    viewModel.start(launch)
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme { StatusScreen(state) }
    }
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    val launch = parse(intent)
    keepVisible(launch)
    viewModel.newIntent(launch)
  }

  /**
   * On a locked test phone the measured process must stay in the foreground (top-app): in the
   * debug build, and for every gate and timing launch, show above the keyguard, turn the screen on
   * and keep it on. A normal launch of a release build is unchanged.
   */
  private fun keepVisible(launch: D1Launch) {
    val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
    if (!debuggable && launch is D1Launch.Normal) return
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
  }

  private fun parse(intent: Intent): D1Launch = D1Launch.parse(IntentExtras(intent), BuildConfig.DEBUG)

  /** The extras of [intent] as [D1Launch.parse] reads them. */
  private class IntentExtras(private val intent: Intent) : D1Extras {
    override fun has(name: String): Boolean = intent.hasExtra(name)

    override fun string(name: String): String? = intent.getStringExtra(name)

    override fun int(name: String, default: Int): Int = intent.getIntExtra(name, default)

    override fun boolean(name: String, default: Boolean): Boolean =
      intent.getBooleanExtra(name, default)
  }
}
