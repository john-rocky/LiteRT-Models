package com.kev

import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kev.view.ApplicationTheme
import com.kev.view.KevScreen
import com.kev.view.PresentationScreen

/**
 * Hosts the Compose screen (`launchMode="singleTop"`: a demo intent reaches the running activity
 * through [onNewIntent], without a new process). Extras ([KevLaunch.parse]):
 * - every launch: `[--es graph auto|rows|pair]` (auto: the form with the smaller predicted time;
 *   rows: one row graph per question; pair: the shared-state pair, or a failure that says why),
 *   `[--es backend gpu|npu|cpu]` (npu: L64 / L128 / L256 on the Qualcomm HTP, the other graphs on
 *   the GPU; it needs the NPU libraries in the APK; a normal or autoplay launch without it keeps
 *   the choice made on screen), `[--es precision fp32|fp16acc]` (GPU precision of every graph of
 *   the process; without it each graph runs at its own default, `KevPrecision.defaultFor`) and
 *   `[--es share auto|on|off]` (whether a pair holds one copy of its weights for both signatures on
 *   the GPU; auto: without sharing when memory allows, `KevPairShare.AUTO`)
 * - `--ez autoplay true --es fixture <files/ path> --ei delay_ms 1500 --ei gap_ms 800
 *   [--ei window 512]` (without `window`, the questions run on the plan a Decide would use)
 * - debug build: `--ez gate true --es backend gpu|npu|cpu --es report <name.json> [--ei window 512]
 *   [--ei limit n]` (with `--es graph pair`, the gate runs on the pair; `[--ei ls 128]` names the
 *   pair by its state length, else the first installed one)
 * - debug and benchmark builds: `--ez timing true --es rows <files/ path> --es backend gpu|npu|cpu
 *   --es report <name.json> [--ei window 512] [--ez clear_cache true]
 *   [--es sets <name[,name…]>|none] [--ez request_path false] [--ei cool_ms 60000]` (one set per
 *   launch keeps every set at the same starting temperature; with `--es graph pair` the sets run on
 *   the pair, `[--ei ls 128]` as for the gate; `cool_ms` waits before each set for the GPU to cool
 *   after the compile; `clear_cache` empties the cache directory, NPU compilations included)
 * - gate and timing with the NPU libraries: `[--es npu_perf burst|none|<mode>]` (the HTP
 *   performance mode of every graph, default burst), `[--es npu_opt default|o3|prepare]` (the
 *   optimization level of the NPU graphs) and `[--es pair_state direct|copy]` (copy: the pair's
 *   state goes through the host)
 *
 * Gate and timing sets compile the one graph they name, on the named backend as it is: the pair,
 * the `window`, or without it the smallest installed window.
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
      LaunchedEffect(state.presentation != null) { showNavigationBar(state.presentation == null) }
      ApplicationTheme {
        val presentation = state.presentation
        if (presentation != null) {
          PresentationScreen(
            presentation,
            viewModel::onPresentationLayout,
            viewModel::leavePresentation,
          )
        } else {
          KevScreen(
            state,
            onExample = viewModel::selectExample,
            onState = viewModel::setState,
            onQuestionId = viewModel::setQuestionId,
            onQuestionType = viewModel::setQuestionType,
            onInstructions = viewModel::setInstructions,
            onOptions = viewModel::setOptions,
            onAddQuestion = viewModel::addQuestion,
            onRemoveQuestion = viewModel::removeQuestion,
            onBackend = viewModel::selectBackend,
            onDecide = viewModel::decide,
            onToggleResponse = viewModel::toggleResponse,
          )
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
   * On a locked test phone the measured process must stay in the foreground (top-app): in the debug
   * build, and for every autoplay, gate and timing launch, show above the keyguard, turn the screen
   * on and keep it on. A normal launch of the other builds is unchanged.
   */
  private fun keepVisible(launch: KevLaunch) {
    val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
    if (!debuggable && launch is KevLaunch.Normal) return
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    }
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
  }

  /** The presentation layout hides the navigation bar; the status bar stays visible. */
  private fun showNavigationBar(show: Boolean) {
    val controller = WindowCompat.getInsetsController(window, window.decorView)
    controller.systemBarsBehavior =
      WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    if (show) {
      controller.show(WindowInsetsCompat.Type.navigationBars())
    } else {
      controller.hide(WindowInsetsCompat.Type.navigationBars())
    }
  }

  private fun parse(intent: Intent): KevLaunch =
    KevLaunch.parse(
      IntentExtras(intent),
      KevLaunchContext(
        debug = BuildConfig.DEBUG,
        diagnostics = BuildConfig.DEBUG || BuildConfig.BUILD_TYPE == "benchmark",
        installedWindows = KevFiles.installedWindows(filesDir),
        installedPairs = KevFiles.installedPairs(filesDir),
        npuAvailable = KevNpu.librariesInstalled(this),
      ),
    )

  /** The extras of [intent] as [KevLaunch.parse] reads them. */
  private class IntentExtras(private val intent: Intent) : KevExtras {
    override fun has(name: String): Boolean = intent.hasExtra(name)

    override fun string(name: String): String? = intent.getStringExtra(name)

    override fun int(name: String, default: Int): Int = intent.getIntExtra(name, default)

    override fun boolean(name: String, default: Boolean): Boolean =
      intent.getBooleanExtra(name, default)
  }
}
