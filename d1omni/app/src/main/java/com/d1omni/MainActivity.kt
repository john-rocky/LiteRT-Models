package com.d1omni

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.d1omni.view.AppActions
import com.d1omni.view.AppScreen
import com.d1omni.view.ApplicationTheme
import com.d1omni.view.EditorActions
import com.d1omni.view.StatusScreen

/**
 * Hosts the Compose screen (`launchMode="singleTop"`: a later intent reaches the running activity through
 * [onNewIntent]): the microphone permission (asked the first time Record is tapped), the photo permission (asked the
 * first time Pick photo opens the Recent photos sheet), the system photo picker (Browse…, no permission) and the system
 * file picker for a WAV file. Extras ([D1Launch.parse]):
 * - a normal launch and an autoplay: `[--es backend gpu|cpu]`, `[--es precision fp32|fp16acc]` (every kind of graph),
 *   `[--es precision_audio fp16acc|fp32]` and `[--es precision_vision fp16acc|fp32]` (one kind; defaults: decision
 *   graphs FP32, audio graph, tower and projector FP16_WITH_FP32_ACCUM); they apply when the launch starts the app
 * - the reproduction run: `--ez autoplay true [--ei delay_ms 1000] [--ei gap_ms 1500]` (Load sample, then Decide on the
 *   voice, photo and message screens in turn, then the summary; logcat tag D1OmniDemo, one run JSON per Decide)
 * - debug build: `--ez gate true …`, `--ez timing true …`, `--ez vgate true …`, `--ez vtiming true …` (the fixture gates
 *   and timing protocols, scripts/TEST_DATA.md)
 */
class MainActivity : ComponentActivity() {
  private val viewModel: MainViewModel by viewModels { MainViewModel.getFactory(this) }

  // The ActivityResult launchers below are ComponentActivity's own registry; the app has no Fragment on its classpath,
  // which the androidx.activity lint check reads as a Fragment older than 1.3.0.
  @SuppressLint("InvalidFragmentVersionForActivityResult")
  private val recordPermission =
    registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) viewModel.startRecording() else viewModel.recordPermissionDenied()
    }

  @SuppressLint("InvalidFragmentVersionForActivityResult")
  private val pickPhoto =
    registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let(viewModel::pickedPhoto) }

  @SuppressLint("InvalidFragmentVersionForActivityResult")
  private val photosPermission =
    registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) viewModel.openRecentPhotos() else viewModel.photosPermissionDenied()
    }

  @SuppressLint("InvalidFragmentVersionForActivityResult")
  private val pickAudio =
    registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(viewModel::pickedAudio) }

  override fun onCreate(savedInstanceState: Bundle?) {
    // Light system bars whatever the phone's dark mode: the screen is white, and the status bar's icons (the airplane
    // mode icon among them) must stay readable on it.
    enableEdgeToEdge(
      statusBarStyle = SystemBarStyle.light(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT),
      navigationBarStyle = SystemBarStyle.light(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT),
    )
    super.onCreate(savedInstanceState)
    val launch = parse(intent)
    keepVisible(launch)
    hideNavigationBar()
    reportScreen()
    viewModel.start(launch)
    val actions =
      AppActions(
        onTab = viewModel::selectTab,
        onLoadSample = viewModel::loadSample,
        onRecord = ::record,
        onStop = viewModel::stopRecording,
        onPickAudio = { pickAudio.launch(WAV_TYPES) },
        onPlay = viewModel::play,
        onPickPhoto = ::recentPhotos,
        onRecentPhoto = viewModel::pickedRecent,
        onBrowsePhotos = {
          viewModel.closeRecentPhotos()
          pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        },
        onCloseRecent = viewModel::closeRecentPhotos,
        onMessage = viewModel::setMessage,
        onDecide = viewModel::decide,
        editor =
          EditorActions(
            onToggle = viewModel::toggleEditor,
            onId = viewModel::setQuestionId,
            onType = viewModel::setQuestionType,
            onInstructions = viewModel::setInstructions,
            onOptions = viewModel::setOptions,
            onAdd = viewModel::addQuestion,
            onRemove = viewModel::removeQuestion,
            onReset = viewModel::resetQuestions,
          ),
        onPill = viewModel::onPillLayout,
        onAnswerLayout = viewModel::onAnswerLayout,
      )
    setContent {
      val state by viewModel.uiState.collectAsStateWithLifecycle()
      ApplicationTheme {
        if (state.diagnostics) StatusScreen(state) else AppScreen(state, actions)
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

  override fun onWindowFocusChanged(hasFocus: Boolean) {
    super.onWindowFocusChanged(hasFocus)
    if (hasFocus) hideNavigationBar()
  }

  private fun record() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
      viewModel.startRecording()
    } else {
      recordPermission.launch(Manifest.permission.RECORD_AUDIO)
    }
  }

  /** Pick photo: the Recent photos sheet, after the read permission (asked the first time). */
  private fun recentPhotos() {
    val permission =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) Manifest.permission.READ_MEDIA_IMAGES
      else Manifest.permission.READ_EXTERNAL_STORAGE
    if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
      viewModel.openRecentPhotos()
    } else {
      photosPermission.launch(permission)
    }
  }

  /** The screen's size in px, its density and font scale, for the run JSON's layout. */
  private fun reportScreen() {
    val metrics = resources.displayMetrics
    val (width, height) =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        windowManager.currentWindowMetrics.bounds.let { it.width() to it.height() }
      } else {
        metrics.widthPixels to metrics.heightPixels
      }
    viewModel.onScreen(width, height, metrics.density, resources.configuration.fontScale)
  }

  /**
   * On a locked test phone the app must stay in the foreground (top-app) while a harness drives it: in the debug build,
   * and for every gate, timing and autoplay launch, show above the keyguard, turn the screen on and keep it on. A normal
   * launch of a release build is unchanged.
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

  /** The navigation bar stays hidden (a swipe shows it for a moment); the status bar stays (its airplane icon shows the phone is offline). */
  private fun hideNavigationBar() {
    val controller = WindowCompat.getInsetsController(window, window.decorView)
    controller.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    controller.hide(WindowInsetsCompat.Type.navigationBars())
  }

  // vision (round 3): the picture runs (`--ez vgate true …`, `--ez vtiming true …`) first, then the launches above
  private fun parse(intent: Intent): D1Launch =
    D1Launch.Vision.parse(IntentExtras(intent), BuildConfig.DEBUG)
      ?: D1Launch.parse(IntentExtras(intent), BuildConfig.DEBUG)
  // end vision (round 3)

  /** The extras of [intent] as [D1Launch.parse] reads them. */
  private class IntentExtras(private val intent: Intent) : D1Extras {
    override fun has(name: String): Boolean = intent.hasExtra(name)

    override fun string(name: String): String? = intent.getStringExtra(name)

    override fun int(name: String, default: Int): Int = intent.getIntExtra(name, default)

    override fun boolean(name: String, default: Boolean): Boolean = intent.getBooleanExtra(name, default)
  }

  private companion object {
    /** The WAV types the file picker offers (16 kHz mono 16-bit PCM is what the app reads). */
    val WAV_TYPES = arrayOf("audio/wav", "audio/x-wav", "audio/vnd.wave", "audio/wave")
  }
}
