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
 * through [onNewIntent], without a new process). Extras:
 * - every launch: `[--es graph auto|rows|pair]` (auto: the form with the smaller predicted time;
 *   rows: one row graph per question; pair: the shared-state pair, or a failure that says why) and
 *   `[--es precision fp32|fp16acc]` (GPU precision of every graph of the process; fp32 by default)
 * - `--ez autoplay true --es fixture <files/ path> --ei delay_ms 1500 --ei gap_ms 800
 *   [--ei window 512]` (without `window`, the questions run on the plan a Decide would use)
 * - debug build: `--ez gate true --es backend gpu|cpu --es report <name.json> [--ei window 512]
 *   [--ei limit n]` (with `--es graph pair`, the gate runs on the pair)
 * - debug and benchmark builds: `--ez timing true --es rows <files/ path> --es backend gpu|cpu --es
 *   report <name.json> [--ei window 512] [--ez clear_cache true] [--es sets <name[,name…]>|none]
 *   [--ez request_path false]` (one set per launch keeps every set at the same starting
 *   temperature; with `--es graph pair` the sets run on the pair)
 *
 * Gate and timing sets compile the one graph they name: the pair, the `window`, or without it the
 * smallest installed window.
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

  private fun parse(intent: Intent): KevLaunch {
    val named = if (intent.hasExtra(EXTRA_WINDOW)) intent.getIntExtra(EXTRA_WINDOW, 0) else null
    // Gate and timing runs compile one window: the named one, else the smallest installed one.
    val window =
      named ?: KevFiles.installedWindows(filesDir).firstOrNull() ?: KevFiles.DEFAULT_INSTALL.first()
    val diagnostics = BuildConfig.DEBUG || BuildConfig.BUILD_TYPE == "benchmark"
    val autoplay = intent.getBooleanExtra(EXTRA_AUTOPLAY, false)
    val graphName = intent.getStringExtra(EXTRA_GRAPH)
    val graph =
      KevGraphMode.of(graphName)
        ?: return KevLaunch.Invalid(KevLaunch.graphInvalid(graphName.orEmpty()), autoplay)
    val precisionName = intent.getStringExtra(EXTRA_PRECISION)
    val precision =
      KevLaunch.precision(precisionName)
        ?: return KevLaunch.Invalid(KevLaunch.precisionInvalid(precisionName.orEmpty()), autoplay)
    // The pair a gate or timing run on the pair compiles: an installed shape (PAIRS order).
    val pair = KevFiles.installedPairs(filesDir).firstOrNull() ?: KevFiles.PAIRS.first()
    return when {
      autoplay -> {
        val fixture = intent.getStringExtra(EXTRA_FIXTURE)
        when {
          fixture.isNullOrEmpty() -> KevLaunch.Invalid("no fixture extra", autoplay = true)
          named != null && !KevLaunch.windowValid(named) ->
            KevLaunch.Invalid(KevLaunch.windowInvalid(named), autoplay = true)
          named != null && graph == KevGraphMode.PAIR ->
            KevLaunch.Invalid("window $named and graph pair exclude each other", autoplay = true)
          else ->
            KevLaunch.Autoplay(
              fixture,
              intent.getIntExtra(EXTRA_DELAY_MS, DEFAULT_DELAY_MS).toLong(),
              intent.getIntExtra(EXTRA_GAP_MS, DEFAULT_GAP_MS).toLong(),
              named,
              graph,
              precision,
            )
        }
      }
      BuildConfig.DEBUG && intent.getBooleanExtra(EXTRA_GATE, false) -> {
        val backend = KevLaunch.backend(intent.getStringExtra(EXTRA_BACKEND))
        val gateGraph =
          if (graph == KevGraphMode.PAIR) KevGraphKey.Pair(pair) else KevGraphKey.Window(window)
        val report =
          intent.getStringExtra(EXTRA_REPORT)
            ?: "app_gate_${backend?.name?.lowercase()}_${KevLaunch.reportLabel(gateGraph)}.json"
        when {
          backend == null -> KevLaunch.Invalid("backend must be gpu or cpu", autoplay = false)
          !KevLaunch.windowValid(window) ->
            KevLaunch.Invalid(KevLaunch.windowInvalid(window), autoplay = false)
          !KevLaunch.reportNameValid(report) ->
            KevLaunch.Invalid("invalid report name $report", autoplay = false)
          else ->
            KevLaunch.Gate(
              backend,
              precision,
              report,
              gateGraph,
              intent.getIntExtra(EXTRA_LIMIT, 0),
            )
        }
      }
      diagnostics && intent.getBooleanExtra(EXTRA_TIMING, false) -> {
        val backend = KevLaunch.backend(intent.getStringExtra(EXTRA_BACKEND))
        val rows = intent.getStringExtra(EXTRA_ROWS)
        val setGraph =
          if (graph == KevGraphMode.PAIR) KevGraphKey.Pair(pair) else KevGraphKey.Window(window)
        val report =
          intent.getStringExtra(EXTRA_REPORT)
            ?: "app_timing_${backend?.name?.lowercase()}_${KevLaunch.reportLabel(setGraph)}.json"
        when {
          backend == null -> KevLaunch.Invalid("backend must be gpu or cpu", autoplay = false)
          rows.isNullOrEmpty() -> KevLaunch.Invalid("no rows extra", autoplay = false)
          !KevLaunch.windowValid(window) ->
            KevLaunch.Invalid(KevLaunch.windowInvalid(window), autoplay = false)
          !KevLaunch.reportNameValid(report) ->
            KevLaunch.Invalid("invalid report name $report", autoplay = false)
          else ->
            KevLaunch.Timing(
              rows,
              backend,
              precision,
              report,
              window,
              graph,
              intent.getBooleanExtra(EXTRA_CLEAR_CACHE, false),
              intent.getStringExtra(EXTRA_SETS)?.let(KevLaunch::setNames),
              intent.getBooleanExtra(EXTRA_REQUEST_PATH, true),
            )
        }
      }
      else -> KevLaunch.Normal(graph, precision)
    }
  }

  private companion object {
    const val EXTRA_AUTOPLAY = "autoplay"
    const val EXTRA_FIXTURE = "fixture"
    const val EXTRA_DELAY_MS = "delay_ms"
    const val EXTRA_GAP_MS = "gap_ms"
    const val EXTRA_WINDOW = "window"
    const val EXTRA_GATE = "gate"
    const val EXTRA_TIMING = "timing"
    const val EXTRA_BACKEND = "backend"
    const val EXTRA_REPORT = "report"
    const val EXTRA_LIMIT = "limit"
    const val EXTRA_ROWS = "rows"
    const val EXTRA_CLEAR_CACHE = "clear_cache"
    const val EXTRA_SETS = "sets"
    const val EXTRA_REQUEST_PATH = "request_path"
    const val EXTRA_GRAPH = "graph"
    const val EXTRA_PRECISION = "precision"
    const val DEFAULT_DELAY_MS = 1500
    const val DEFAULT_GAP_MS = 800
  }
}
