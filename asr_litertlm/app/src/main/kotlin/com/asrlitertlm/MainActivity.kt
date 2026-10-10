package com.asrlitertlm

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Future

/** How a microphone test ended (see [MainActivity.runMicTest]). */
sealed interface MicTestResult {
  data class Finished(val outcome: TranscribeOutcome) : MicTestResult

  data class Failed(val message: String) : MicTestResult
}

/**
 * Speech to text on the phone with one of three LiteRT-LM bundles: pick a model, hold the microphone button and speak
 * (up to 30 s), or play one of three bundled clips. The bundle's language model and audio encoder run on the CPU
 * (4 threads); every number on screen is measured by this app: "engine load" = Engine.initialize() plus the first
 * conversation, "transcribe" = the wall time of sendMessage(), RTF = that time over the clip length.
 *
 * The bundles are read from /sdcard/Android/data/com.asrlitertlm/files/<file name> (adb push, see README). Launch
 * extras, for recordings and the device check (all optional):
 * ```
 *   --es model <id>          qwen3-asr-1.7b | fun-asr-nano-2512 | confucius4-r2t2 (default: the last one used)
 *   --es backend gpu         language model on the GPU (audio encoder stays on the CPU)
 *   --ez autoplay true       after READY: wait delay_ms, then clips zh -> en -> ja (each played, then transcribed)
 *   --ei delay_ms 3000       with gap_ms between one transcript and the next clip
 *   --ei gap_ms 2500
 *   --ei mic_test_ms 9000    after READY and delay_ms: record that long without touching the screen, then transcribe
 *   --ez mic_test_play true  and play clip en through the speaker meanwhile (needs RECORD_AUDIO granted already)
 * ```
 * Logs go to the tag AsrLitertlm: one `STATE <name>` line per screen state, `LOAD` and `TRANSCRIBED` lines with the
 * measured times.
 */
class MainActivity : Activity() {
  private val main = Handler(Looper.getMainLooper())
  private val player = ClipPlayer()
  private lateinit var recorder: MicRecorder
  private val onEvent: (AsrEvent) -> Unit = { handle(it) }

  private var busy = false
  private var lmBackend = LmBackend.CPU
  private var selected: ModelProfile? = null
  private var micCallback: ((MicTestResult) -> Unit)? = null
  private var extrasPending = false
  private var autoplayIndex = -1
  private var pendingNotice: String? = null

  private lateinit var pill: TextView
  private lateinit var engineLine: TextView
  private lateinit var pickerRows: Map<ModelProfile, TextView>
  private lateinit var sourceLine: TextView
  private lateinit var bar: PlayBar
  private lateinit var body: TextView
  private lateinit var languageLine: TextView
  private lateinit var timeLine: TextView
  private lateinit var deviceLine: TextView
  private lateinit var clipButtons: List<TextView>
  private lateinit var micButton: TextView

  private val deviceName: String
    get() = if (Build.MODEL.startsWith("SM-S942")) "Galaxy S26" else Build.MODEL

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // Show over the lock screen and keep the display on: a hidden activity's process moves to the background CPU
    // set (little cores), where the same model runs about 10x slower.
    setShowWhenLocked(true)
    setTurnScreenOn(true)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    AsrSession.init(this)
    recorder = MicRecorder(this)
    setContentView(buildUi())
    AsrSession.addListener(onEvent)
    startFrom(intent)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    if (!busy) startFrom(intent)
  }

  override fun onResume() {
    super.onResume()
    refreshPicker()
  }

  override fun onDestroy() {
    AsrSession.removeListener(onEvent)
    recorder.stop()
    player.stop()
    main.removeCallbacksAndMessages(null)
    // The engine holds gigabytes: free it when the screen goes away for good.
    if (isFinishing) AsrSession.release()
    super.onDestroy()
  }

  // ---- model --------------------------------------------------------------------------------------------------

  /** Picks the model from the launch extras, the last one used, or the first on the phone, and loads it. */
  private fun startFrom(launch: Intent) {
    lmBackend = if (launch.getStringExtra(EXTRA_BACKEND) == "gpu") LmBackend.GPU else LmBackend.CPU
    deviceLine.text = statusLine()
    extrasPending = launch.getBooleanExtra(EXTRA_AUTOPLAY, false) || launch.getIntExtra(EXTRA_MIC_TEST_MS, 0) > 0
    val requested = ModelProfile.byId(launch.getStringExtra(EXTRA_MODEL))
    val saved = ModelProfile.byId(getPreferences(MODE_PRIVATE).getString(PREF_MODEL, null))
    val choice =
      listOfNotNull(requested, saved).firstOrNull { AsrSession.isPresent(it) }
        ?: ModelProfile.ALL.firstOrNull { AsrSession.isPresent(it) }
    when {
      choice == null -> showNoModel()
      requested != null && choice != requested -> {
        // Shown once the model that is on the phone is ready (the load clears the screen).
        pendingNotice = AsrEngine.missingBundleMessage(requested, AsrSession.modelDir)
        selectModel(choice)
      }
      else -> selectModel(choice)
    }
  }

  /**
   * Loads [profile] (the picker's tap). Returns the load, or null when the bundle is not on the phone (the screen
   * then says which file to push where). A model that is already loaded is not loaded again.
   */
  fun selectModel(profile: ModelProfile, backend: LmBackend = lmBackend): Future<LoadInfo>? {
    if (!AsrSession.isPresent(profile)) {
      refreshPicker()
      showText(AsrEngine.missingBundleMessage(profile, AsrSession.modelDir))
      return null
    }
    selected = profile
    lmBackend = backend
    getPreferences(MODE_PRIVATE).edit().putString(PREF_MODEL, profile.id).apply()
    refreshPicker()
    val current = AsrSession.current
    if (current != null && current.profile == profile && current.lmBackend == backend) {
      showReady(current)
      return CompletableFuture.completedFuture(current)
    }
    setBusy(true)
    return AsrSession.load(profile, backend)
  }

  private fun handle(event: AsrEvent) {
    when (event) {
      is AsrEvent.Loading -> {
        setPill("LOADING", C_IDLE)
        engineLine.text =
          String.format(
            Locale.US,
            "loading %s (%.2f GB) · LM %s",
            event.profile.displayName,
            AsrSession.bundleFile(event.profile).length() / 1e9,
            backendLabel(event.lmBackend),
          )
        clearResult()
      }
      is AsrEvent.Loaded -> {
        setBusy(false)
        showReady(event.info)
      }
      is AsrEvent.LoadFailed -> {
        setBusy(false)
        setPill("ERROR", C_LIVE)
        // A missing file leaves the model loaded before in place.
        engineLine.text = AsrSession.current?.let { "${it.profile.displayName} still loaded" } ?: "no model loaded"
        showText(event.message)
      }
      is AsrEvent.Transcribing -> setPill("TRANSCRIBING", C_WORK)
      is AsrEvent.Transcribed -> {
        show(event.outcome)
        setBusy(false)
        if (event.source == SOURCE_MIC) finishMicTest(MicTestResult.Finished(event.outcome))
        continueAutoplay(event.source)
      }
      is AsrEvent.TranscribeFailed -> {
        setBusy(false)
        setPill("ERROR", C_LIVE)
        showText(event.message)
        if (event.source == SOURCE_MIC) finishMicTest(MicTestResult.Failed(event.message))
        autoplayIndex = -1
      }
      is AsrEvent.Released -> {
        setPill("RELEASED", C_IDLE)
        engineLine.text = "${event.profile.displayName} closed"
        setBusy(busy)
      }
    }
  }

  private fun showReady(info: LoadInfo) {
    selected = info.profile
    setPill("READY", C_IDLE)
    engineLine.text =
      String.format(
        Locale.US,
        "engine load %.2f s · LM %s",
        info.loadMs / 1000,
        backendLabel(info.lmBackend),
      )
    deviceLine.text = statusLine()
    refreshPicker()
    pendingNotice?.let {
      pendingNotice = null
      showText(it)
    }
    if (extrasPending) {
      extrasPending = false
      runLaunchExtras()
    }
  }

  private fun showNoModel() {
    setPill("NO MODEL", C_LIVE)
    engineLine.text = "no model on this phone"
    val files = ModelProfile.ALL.joinToString("\n") { "· ${it.fileName}  (${it.hubRepo})" }
    showText("Push at least one bundle to ${AsrSession.modelDir.path}/ with adb, then reopen the app:\n$files")
    refreshPicker()
  }

  // ---- clips --------------------------------------------------------------------------------------------------

  /** A bundled clip: play it through the speaker, then transcribe it. */
  private fun playClip(index: Int) {
    if (busy || AsrSession.current == null) return
    val clip = Clip.ALL[index]
    val file = AsrSession.clipFile(clip)
    setBusy(true)
    clearResult()
    setPill("● PLAYING", C_LIVE)
    sourceLine.text = String.format(Locale.US, "clip %s · %.1f s", clip.label, Wav.seconds(file))
    bar.set(0f, C_LIVE)
    bar.visibility = View.VISIBLE
    player.play(
      file,
      onProgress = { bar.fraction = it },
      onDone = { failure ->
        bar.set(1f, C_IDLE)
        if (failure != null) {
          setBusy(false)
          setPill("ERROR", C_LIVE)
          showText(failure)
          autoplayIndex = -1
        } else {
          AsrSession.transcribe(file, SOURCE_CLIP + clip.id)
        }
      },
    )
  }

  private fun runLaunchExtras() {
    val delay = intent.getIntExtra(EXTRA_DELAY_MS, 3000).toLong()
    val micTestMs = intent.getIntExtra(EXTRA_MIC_TEST_MS, 0)
    if (intent.getBooleanExtra(EXTRA_AUTOPLAY, false)) {
      Log.i(TAG, "AUTOPLAY_START delay_ms=$delay")
      autoplayIndex = 0
      main.postDelayed({ playClip(0) }, delay)
    } else if (micTestMs > 0) {
      val play = intent.getBooleanExtra(EXTRA_MIC_TEST_PLAY, false)
      main.postDelayed({ runMicTest(micTestMs.toLong(), play) }, delay)
    }
  }

  private fun continueAutoplay(source: String) {
    if (autoplayIndex < 0 || !source.startsWith(SOURCE_CLIP)) return
    autoplayIndex++
    if (autoplayIndex >= Clip.ALL.size) {
      autoplayIndex = -1
      Log.i(TAG, "AUTOPLAY_DONE")
      return
    }
    val next = autoplayIndex
    main.postDelayed({ playClip(next) }, intent.getIntExtra(EXTRA_GAP_MS, 2500).toLong())
  }

  // ---- microphone ---------------------------------------------------------------------------------------------

  /** Hold to talk: recording starts on press and ends on release (or at 30 s). */
  @SuppressLint("ClickableViewAccessibility") // the button also takes a click: performClick() on release
  private fun onMicTouch(view: View, event: MotionEvent): Boolean {
    when (event.actionMasked) {
      MotionEvent.ACTION_DOWN -> {
        if (busy || AsrSession.current == null) return true
        if (!recorder.hasPermission()) {
          requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_MIC)
          return true
        }
        startRecording(playClip = false)
      }
      MotionEvent.ACTION_UP -> {
        view.performClick()
        recorder.stop()
      }
      MotionEvent.ACTION_CANCEL -> recorder.stop()
    }
    return true
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode != REQUEST_MIC) return
    if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
      showText("Microphone allowed. Hold the button and speak.")
    } else {
      setPill("NO MIC", C_LIVE)
      showText(MicRecorder.PERMISSION_MESSAGE)
    }
  }

  /**
   * The microphone path without touching the screen: record for [durationMs] (playing clip en through the speaker
   * when [playClip]), then transcribe like the button does. [onFinished] gets the outcome or the failure sentence;
   * without the permission the sentence comes back at once and nothing is recorded.
   */
  fun runMicTest(durationMs: Long, playClip: Boolean, onFinished: ((MicTestResult) -> Unit)? = null) {
    Log.i(TAG, "MIC_TEST ms=$durationMs play=$playClip")
    micCallback = onFinished
    if (!recorder.hasPermission()) {
      setPill("NO MIC", C_LIVE)
      showText(MicRecorder.PERMISSION_MESSAGE)
      finishMicTest(MicTestResult.Failed(MicRecorder.PERMISSION_MESSAGE))
      return
    }
    if (busy || AsrSession.current == null) {
      finishMicTest(MicTestResult.Failed("The app is busy or no model is loaded."))
      return
    }
    if (!startRecording(playClip)) return
    main.postDelayed({ recorder.stop() }, durationMs)
  }

  private fun startRecording(playClip: Boolean): Boolean {
    val wav = File(cacheDir, "mic.wav")
    val failure =
      recorder.start(
        wav,
        onProgress = { seconds -> main.post { showRecording(seconds) } },
        onDone = { result -> main.post { onRecorded(result) } },
      )
    if (failure != null) {
      setPill("ERROR", C_LIVE)
      showText(failure)
      finishMicTest(MicTestResult.Failed(failure))
      return false
    }
    setBusy(true)
    micButton.isEnabled = true
    micButton.alpha = 1f
    clearResult()
    setPill("● LISTENING", C_LIVE)
    showRecording(0.0)
    bar.set(0f, C_LIVE)
    bar.visibility = View.VISIBLE
    if (playClip) {
      val clip = AsrSession.clipFile(Clip.ALL[1])
      main.postDelayed({ player.play(clip, onProgress = {}, onDone = {}) }, MIC_TEST_PLAY_DELAY_MS)
    }
    return true
  }

  private fun showRecording(seconds: Double) {
    sourceLine.text = String.format(Locale.US, "mic · %.1f s of %d s", seconds, MicRecorder.MAX_SECONDS)
    bar.fraction = (seconds / MicRecorder.MAX_SECONDS).toFloat()
    micButton.text = String.format(Locale.US, "Release to transcribe · %.1f s", seconds)
  }

  private fun onRecorded(result: MicResult) {
    micButton.text = MIC_LABEL
    bar.set(1f, C_IDLE)
    when (result) {
      is MicResult.Failed -> {
        setBusy(false)
        setPill("ERROR", C_LIVE)
        showText(result.message)
        finishMicTest(MicTestResult.Failed(result.message))
      }
      is MicResult.Recorded -> {
        Log.i(
          TAG,
          String.format(
            Locale.US,
            "MIC_RECORDED seconds=%.2f peak_dbfs=%.1f rms_dbfs=%.1f at_limit=%b",
            result.seconds,
            result.levels.peakDbfs,
            result.levels.rmsDbfs,
            result.stoppedAtLimit,
          ),
        )
        sourceLine.text =
          String.format(Locale.US, "mic · %.1f s", result.seconds) +
            if (result.stoppedAtLimit) " · stopped at the ${MicRecorder.MAX_SECONDS} s limit" else ""
        setBusy(true)
        AsrSession.transcribe(result.wav, SOURCE_MIC)
      }
    }
  }

  private fun finishMicTest(result: MicTestResult) {
    val callback = micCallback ?: return
    micCallback = null
    callback(result)
  }

  // ---- screen -------------------------------------------------------------------------------------------------

  private fun show(outcome: TranscribeOutcome) {
    when (outcome) {
      is TranscribeOutcome.NoSpeech -> {
        setPill("NO SPEECH", C_IDLE)
        body.text = "No speech heard."
        languageLine.text =
          String.format(
            Locale.US,
            "The recording peaked at %.0f dBFS (RMS %.0f dBFS); the model was not called.",
            outcome.levels.peakDbfs,
            outcome.levels.rmsDbfs,
          )
        timeLine.text = ""
      }
      is TranscribeOutcome.Text -> {
        val transcript = outcome.transcript
        val answer = transcript.answer
        setPill("DONE", C_DONE)
        body.textLocale = localeFor(answer)
        body.text = answer.text.ifEmpty { "(no speech in the answer)" }
        languageLine.text =
          when {
            transcript.profile.answerFormat == AnswerFormat.PLAIN -> ""
            answer.language.isEmpty() -> "detected language: none"
            else -> "detected language: ${answer.language}"
          }
        val numbers =
          String.format(
            Locale.US,
            "transcribe %.2f s / clip %.2f s = RTF %.2f",
            transcript.sendMs / 1000,
            transcript.sentSeconds,
            transcript.rtf,
          )
        timeLine.text =
          SpannableStringBuilder(numbers).apply {
            setSpan(ForegroundColorSpan(C_TEXT), 0, numbers.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            setSpan(StyleSpan(Typeface.BOLD), 0, numbers.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
          }
      }
    }
  }

  private fun showText(text: String) {
    body.textLocale = Locale.US
    body.text = text
    languageLine.text = ""
    timeLine.text = ""
  }

  private fun clearResult() {
    sourceLine.text = ""
    body.text = ""
    languageLine.text = ""
    timeLine.text = ""
    bar.visibility = View.INVISIBLE
  }

  /** The glyph forms of Han characters follow the language the model named, or the script of the text. */
  private fun localeFor(answer: Answer): Locale =
    when {
      answer.language == "Japanese" || answer.text.any { it in '぀'..'ヿ' } -> Locale.JAPANESE
      answer.language == "Cantonese" -> Locale.TRADITIONAL_CHINESE
      answer.language == "Chinese" ||
        answer.text.any { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.HAN } ->
        Locale.SIMPLIFIED_CHINESE
      else -> Locale.US
    }

  private fun refreshPicker() {
    if (!::pickerRows.isInitialized) return
    val loaded = AsrSession.current?.profile
    for ((profile, row) in pickerRows) {
      val present = AsrSession.isPresent(profile)
      val bytes = if (present) AsrSession.bundleFile(profile).length() else profile.hubBytes
      val mark = if (profile == selected) "●  " else "○  "
      val state =
        when {
          !present -> "missing"
          profile == loaded -> "loaded"
          else -> "on phone"
        }
      val head = mark + profile.displayName
      val text = String.format(Locale.US, "%s   %.2f GB · %s", head, bytes / 1e9, state)
      row.text =
        SpannableStringBuilder(text).apply {
          setSpan(StyleSpan(Typeface.BOLD), 0, head.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
      row.setTextColor(if (present) C_TEXT else C_SUB)
      row.background = rounded(if (profile == selected) C_SELECTED else C_BUTTON, C_BUTTON_EDGE)
      row.isEnabled = !busy
    }
  }

  private fun setBusy(on: Boolean) {
    busy = on
    val ready = !on && AsrSession.current != null
    for (button in clipButtons + micButton) {
      button.isEnabled = ready
      button.alpha = if (ready) 1f else 0.4f
    }
    refreshPicker()
  }

  private fun setPill(label: String, color: Int) {
    pill.text = label
    pill.background = GradientDrawable().apply {
      cornerRadius = dp(999f)
      setColor(color)
    }
    Log.i(TAG, "STATE ${label.removePrefix("● ").replace(' ', '_')}")
  }

  private fun statusLine(): String =
    String.format(
      Locale.US,
      "%s · LiteRT-LM %s · LM %s, audio CPU",
      deviceName,
      BuildConfig.LITERTLM_VERSION,
      if (lmBackend == LmBackend.GPU) "GPU" else "CPU",
    )

  private fun backendLabel(backend: LmBackend): String =
    if (backend == LmBackend.GPU) "GPU" else "CPU ${AsrEngine.CPU_THREADS} threads"

  private fun buildUi(): View {
    val side = dp(20f).toInt()
    val root =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setBackgroundColor(C_BG)
        setOnApplyWindowInsetsListener { view, insets ->
          val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
          view.setPadding(side + bars.left, dp(14f).toInt() + bars.top, side + bars.right, dp(14f).toInt() + bars.bottom)
          insets
        }
      }
    val title = label(15f, C_TEXT, bold = true).apply { text = "Speech to text on the phone · LiteRT-LM" }
    val picker = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    pickerRows =
      ModelProfile.ALL.associateWith { profile ->
        label(14f, C_TEXT).apply {
          setPadding(dp(14f).toInt(), dp(10f).toInt(), dp(14f).toInt(), dp(10f).toInt())
          maxLines = 1
          setOnClickListener { if (!busy) selectModel(profile) }
          picker.addView(this, layoutParams(top = 8f))
        }
      }
    pill =
      label(13f, C_TEXT, bold = true).apply {
        letterSpacing = 0.08f
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(12f).toInt(), 0, dp(12f).toInt(), 0)
      }
    engineLine = label(14f, C_SUB)
    sourceLine = label(17f, C_SUB)
    bar = PlayBar(this, C_BUTTON_EDGE).apply { visibility = View.INVISIBLE }
    body = label(24f, C_TEXT).apply { setLineSpacing(0f, 1.1f) }
    languageLine = label(17f, C_TEXT)
    timeLine = label(15f, C_SUB)
    deviceLine = label(13f, C_SUB)
    val attribution = label(13f, C_SUB).apply { text = ATTRIBUTION }

    val result =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        addView(sourceLine, layoutParams(top = 14f))
        addView(
          bar,
          LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(6f).toInt()).apply {
            topMargin = dp(10f).toInt()
          },
        )
        addView(body, layoutParams(top = 14f))
        addView(languageLine, layoutParams(top = 14f))
      }
    val scroll =
      ScrollView(this).apply {
        isVerticalScrollBarEnabled = false
        addView(result)
      }
    val clipRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    clipButtons =
      Clip.ALL.mapIndexed { index, clip ->
        val seconds = Wav.seconds(AsrSession.clipFile(clip))
        button(String.format(Locale.US, "%s · %.1f s", clip.label, seconds)) { playClip(index) }
          .also {
            clipRow.addView(
              it,
              LinearLayout.LayoutParams(0, dp(52f).toInt(), 1f).apply {
                if (index > 0) marginStart = dp(10f).toInt()
              },
            )
          }
      }
    micButton = button(MIC_LABEL) {}.apply { setOnTouchListener { view, event -> onMicTouch(view, event) } }
    setBusy(false)

    root.addView(title)
    root.addView(picker, layoutParams(top = 4f))
    root.addView(
      pill,
      LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(28f).toInt()).apply {
        topMargin = dp(14f).toInt()
      },
    )
    root.addView(engineLine, layoutParams(top = 6f))
    root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    root.addView(clipRow, layoutParams(top = 10f))
    root.addView(
      micButton,
      LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(64f).toInt()).apply {
        topMargin = dp(10f).toInt()
      },
    )
    // Footer: the measured seconds, then device, runtime and backend, then the clips' source.
    root.addView(timeLine, layoutParams(top = 12f))
    root.addView(deviceLine, layoutParams(top = 4f))
    root.addView(attribution, layoutParams(top = 2f))
    return root
  }

  private fun layoutParams(top: Float) =
    LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
      topMargin = dp(top).toInt()
    }

  private fun label(sizeSp: Float, color: Int, bold: Boolean = false) =
    TextView(this).apply {
      setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
      setTextColor(color)
      if (bold) typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

  private fun button(text: String, onClick: () -> Unit) =
    label(16f, C_TEXT, bold = true).apply {
      this.text = text
      gravity = Gravity.CENTER
      background = rounded(C_BUTTON, C_BUTTON_EDGE)
      setOnClickListener { onClick() }
    }

  private fun rounded(fill: Int, edge: Int) =
    GradientDrawable().apply {
      cornerRadius = dp(14f)
      setColor(fill)
      setStroke(dp(1f).toInt(), edge)
    }

  private fun dp(value: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, resources.displayMetrics)

  companion object {
    private const val TAG = "AsrLitertlm"
    const val EXTRA_MODEL = "model"
    const val EXTRA_BACKEND = "backend"
    const val EXTRA_AUTOPLAY = "autoplay"
    const val EXTRA_DELAY_MS = "delay_ms"
    const val EXTRA_GAP_MS = "gap_ms"
    const val EXTRA_MIC_TEST_MS = "mic_test_ms"
    const val EXTRA_MIC_TEST_PLAY = "mic_test_play"
    const val SOURCE_MIC = "mic"
    const val SOURCE_CLIP = "clip:"
    private const val PREF_MODEL = "model"
    private const val REQUEST_MIC = 1
    private const val MIC_TEST_PLAY_DELAY_MS = 500L
    private const val MIC_LABEL = "Hold to talk (up to 30 s)"
    // The clips are FLEURS recordings with one linear gain each: CC BY 4.0 asks to say so.
    private const val ATTRIBUTION = "clips: FLEURS #1698, CC BY 4.0, level-normalized"
    private const val C_BG = 0xFF0E1116.toInt()
    private const val C_IDLE = 0xFF5F6368.toInt()
    private const val C_LIVE = 0xFFE53935.toInt()
    private const val C_WORK = 0xFF1565C0.toInt()
    private const val C_DONE = 0xFF2E7D32.toInt()
    private const val C_TEXT = 0xFFE6E8EB.toInt()
    private const val C_SUB = 0xFF8A919C.toInt()
    private const val C_BUTTON = 0xFF1A1F27.toInt()
    private const val C_SELECTED = 0xFF1E3A5F.toInt()
    private const val C_BUTTON_EDGE = 0xFF2C333D.toInt()
  }
}
