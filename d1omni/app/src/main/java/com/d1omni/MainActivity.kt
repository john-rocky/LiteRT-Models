package com.d1omni

import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.d1omni.view.ApplicationTheme
import com.d1omni.view.InboxScreen
import com.d1omni.view.PresentationScreen
import com.d1omni.view.StatusScreen

/**
 * Hosts the Compose screen (`launchMode="singleTop"`: a later intent reaches the running activity
 * through [onNewIntent]). Extras ([D1Launch.parse]):
 * - a normal launch and an autoplay: `[--es backend gpu|cpu]`, `[--es precision fp32|fp16acc]`
 *   (every kind of graph), `[--es precision_audio fp16acc|fp32]` and `[--es precision_vision
 *   fp16acc|fp32]` (one kind; defaults: decision graphs FP32, audio graph, tower and projector
 *   FP16_WITH_FP32_ACCUM); they apply when the launch starts the app
 * - the demo: `--ez autoplay true --es fixture <name in files/ or inbox_demo.json> [--ei delay_ms
 *   1000] [--ei gap_ms 1500]` (the inbox on the presentation screen; the voice note plays through
 *   the speaker; the run JSON goes to files/d1omni-demo-<epoch ms>.json, logcat tag D1OmniDemo)
 * - gate and timing launches: `[--es precision fp32|fp16acc]` (the decision graphs) and
 *   `[--es precision_audio fp16acc|fp32]` (the audio graph)
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
    // Light system bars whatever the phone's dark mode: every screen of the app is white, and the status bar's
    // icons (the airplane mode icon among them) must stay readable on it.
    enableEdgeToEdge(
      statusBarStyle = SystemBarStyle.light(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT),
      navigationBarStyle = SystemBarStyle.light(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT),
    )
    super.onCreate(savedInstanceState)
    val launch = parse(intent)
    keepVisible(launch)
    viewModel.start(launch)
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      LaunchedEffect(state.presentation != null) { showNavigationBar(state.presentation == null) }
      ApplicationTheme {
        val presentation = state.presentation
        val inbox = state.inbox
        when {
          presentation != null ->
            PresentationScreen(presentation, viewModel::onPresentationLayout, viewModel::leavePresentation)
          inbox != null ->
            InboxScreen(inbox, state.status, state.engine, state.error, state.ready, viewModel::decide)
          else -> StatusScreen(state)
        }
      }
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
   * debug build, and for every gate, timing and autoplay launch, show above the keyguard, turn the
   * screen on and keep it on. A normal launch of a release build is unchanged.
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

  /** The presentation hides the navigation bar; the status bar stays (its airplane icon shows the phone is offline). */
  private fun showNavigationBar(show: Boolean) {
    val controller = WindowCompat.getInsetsController(window, window.decorView)
    controller.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    if (show) {
      controller.show(WindowInsetsCompat.Type.navigationBars())
    } else {
      controller.hide(WindowInsetsCompat.Type.navigationBars())
    }
  }

  // vision (round 3): the picture runs (`--ez vgate true --es fixture <rows> --es report <name>
  // [--ei limit n]`, `--ez vtiming true --es rows <rows> --es report <name> [--ei warmup 5]
  // [--ei reps 20] [--ei cool_ms 120000] [--es sets dogs2]`) first, then the launches above
  private fun parse(intent: Intent): D1Launch =
    D1Launch.Vision.parse(IntentExtras(intent), BuildConfig.DEBUG)
      ?: D1Launch.parse(IntentExtras(intent), BuildConfig.DEBUG)
  // end vision (round 3)

  /** The extras of [intent] as [D1Launch.parse] reads them. */
  private class IntentExtras(private val intent: Intent) : D1Extras {
    override fun has(name: String): Boolean = intent.hasExtra(name)

    override fun string(name: String): String? = intent.getStringExtra(name)

    override fun int(name: String, default: Int): Int = intent.getIntExtra(name, default)

    override fun boolean(name: String, default: Boolean): Boolean =
      intent.getBooleanExtra(name, default)
  }
}
