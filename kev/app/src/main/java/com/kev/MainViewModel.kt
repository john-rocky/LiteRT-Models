package com.kev

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.annotation.StringRes
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Owns the [KevEngine] and runs every model call on [KevRuntime.dispatcher], one job at a time
 * ([engineLock]), because LiteRT reuses its native buffers. Publishes [UiState] for [MainActivity]:
 * the editable sample, the demo autoplay, and the debug gate and timing runs.
 */
class MainViewModel(private val context: Context) : ViewModel() {
  private val worker = CoroutineScope(SupervisorJob() + KevRuntime.dispatcher)
  private val engineLock = Mutex()
  private var engine: KevEngine? = null
  private var started = false
  @Volatile private var cleared = false
  @Volatile private var demoLayout: KevDemoLayout? = null
  private var ticker: Job? = null
  private var nextKey = FIRST_ADDED_KEY
  private var lastWarmupMs = 0.0

  /** The bundled requests (`res/raw`): invented support ticket, incident report and review. */
  private val examples: List<KevFixture> = EXAMPLES.map { id ->
    KevFixture.parse(context.resources.openRawResource(id).use { it.readBytes() }.decodeToString())
  }

  private val mutableState =
    MutableStateFlow(UiState(draft = KevDrafts.fromRequest(examples[0].request)))

  /** Immutable screen state, replaced after every event. */
  val uiState: StateFlow<UiState> = mutableState.asStateFlow()

  /** Starts once per ViewModel with the launch intent's request. */
  fun start(launch: KevLaunch) {
    if (started || cleared) return
    started = true
    when (launch) {
      is KevLaunch.Gate -> runGate(launch)
      is KevLaunch.Timing -> runTiming(launch)
      is KevLaunch.Autoplay -> {
        loadInteractive(launch.window)
        autoplay(launch, SystemClock.elapsedRealtimeNanos())
      }
      is KevLaunch.Invalid -> {
        reportInvalid(launch)
        loadInteractive(KevFiles.DEFAULT_WINDOW)
      }
      KevLaunch.Normal -> loadInteractive(KevFiles.DEFAULT_WINDOW)
    }
  }

  /** A launch intent delivered to the running activity (`singleTop`). */
  fun newIntent(launch: KevLaunch) {
    if (cleared) return
    val received = SystemClock.elapsedRealtimeNanos()
    when (launch) {
      is KevLaunch.Autoplay -> autoplay(launch, received)
      is KevLaunch.Invalid -> reportInvalid(launch)
      is KevLaunch.Gate,
      is KevLaunch.Timing ->
        Log.i(KevGateRunner.LOG_TAG, "failed the app is running; force-stop it and launch again")
      KevLaunch.Normal -> Unit
    }
  }

  // ---- Editor ----

  fun selectExample(index: Int) = edit {
    it.copy(
      example = index,
      draft = KevDrafts.fromRequest(examples[index].request, nextKeys(examples[index])),
    )
  }

  fun setState(text: String) = edit { it.copy(example = null, draft = it.draft.copy(state = text)) }

  fun setQuestionId(key: Long, id: String) = editQuestion(key) { it.copy(id = id) }

  fun setQuestionType(key: Long, type: QuestionType) = editQuestion(key) { it.copy(type = type) }

  fun setInstructions(key: Long, text: String) = editQuestion(key) { it.copy(instructions = text) }

  fun setOptions(key: Long, text: String) = editQuestion(key) { it.copy(options = text) }

  fun addQuestion() = edit {
    val question = QuestionDraft(nextKey++, "", QuestionType.NOUL, "", "")
    it.copy(example = null, draft = it.draft.copy(questions = it.draft.questions + question))
  }

  fun removeQuestion(key: Long) = edit {
    it.copy(
      example = null,
      draft = it.draft.copy(questions = it.draft.questions.filterNot { q -> q.key == key }),
    )
  }

  fun toggleResponse() = mutableState.update { it.copy(showResponse = !it.showResponse) }

  private fun editQuestion(key: Long, change: (QuestionDraft) -> QuestionDraft) = edit {
    it.copy(
      example = null,
      draft =
        it.draft.copy(
          questions = it.draft.questions.map { q -> if (q.key == key) change(q) else q }
        ),
    )
  }

  private fun edit(change: (UiState) -> UiState) {
    if (uiState.value.editable) mutableState.update { change(it).copy(requestError = null) }
  }

  private fun nextKeys(fixture: KevFixture): Long = nextKey.also {
    nextKey += fixture.request.questions.size
  }

  // ---- Engine ----

  /** Recompiles the resident window on [backend]; GPU falls back to CPU if it fails. */
  fun selectBackend(backend: KevDecider.Backend) {
    val state = uiState.value
    if (!state.canDecide || backend == state.backendChoice) return
    mutableState.update { it.copy(backendChoice = backend) }
    val loaded = engine ?: return
    worker.launch {
      engineLock.withLock {
        guarded(onFailure = ::showError) {
          setLoading(LoadStage.GRAPH, loaded.window, switching = false)
          loaded.ensureGraph(context, loaded.window, backend, cpuFallback = true)
          setReady(loaded)
        }
      }
    }
  }

  private fun loadInteractive(window: Int) {
    val missing = KevFiles.missing(context.filesDir, window)
    if (missing.isNotEmpty()) {
      mutableState.update { it.copy(status = KevStatus.MissingFiles(missing)) }
      KevDemo.failed("missing ${missing.joinToString(" ")}")
      return
    }
    worker.launch {
      engineLock.withLock {
        guarded(
          onFailure = { message ->
            showError(message)
            KevDemo.failed("engine $message")
          }
        ) {
          val loaded =
            KevEngine.load(context, window, uiState.value.backendChoice, cpuFallback = true) { stage
              ->
              setLoading(stage, window, switching = false)
            }
          engine = loaded
          loaded.gpuFailure?.let { KevDemo.gpuFallback(it) }
          val warmupMs = warmUp(loaded)
          setReady(loaded, warmupMs)
          KevDemo.engineReady(KevAnswerView.wholeMillis(loaded.loadMs))
        }
      }
    }
  }

  /**
   * One untimed pass of the bundled ticket's question 1, so a request does not pay the graph's
   * start-up costs.
   */
  private fun warmUp(loaded: KevEngine): Double {
    val start = System.nanoTime()
    val prepared = loaded.pipeline.prepare(examples[0].request)
    if (prepared.rows[0].length <= loaded.window) loaded.pipeline.run(prepared, 0, loaded.decider)
    return KevPipeline.millis(System.nanoTime() - start)
  }

  // ---- Decide ----

  /** Answers the edited request: tokenize once, then one graph call and the head per question. */
  fun decide() {
    val state = uiState.value
    if (!state.canDecide) return
    val request =
      try {
        KevDrafts.toRequest(state.draft)
      } catch (failure: KevDraftException) {
        mutableState.update { it.copy(requestError = draftMessage(failure)) }
        return
      } catch (failure: IllegalArgumentException) {
        mutableState.update { it.copy(requestError = failure.message) }
        return
      }
    val loaded = engine ?: return
    mutableState.update {
      it.copy(
        requestError = null,
        responseJson = null,
        showResponse = false,
        status = KevStatus.Running(0, request.questions.size),
      )
    }
    worker.launch {
      engineLock.withLock {
        guarded(onFailure = { message -> requestFailed(loaded, message) }) {
          val prepared = loaded.pipeline.prepare(request)
          val window = requiredWindow(prepared) ?: return@guarded
          if (window != loaded.window || loaded.decider.isClosed) {
            setLoading(LoadStage.GRAPH, window, switching = window != loaded.window)
            loaded.ensureGraph(context, window, uiState.value.backendChoice, cpuFallback = true)
            mutableState.update { it.copy(engine = engineUi(loaded)) }
          }
          val cards = request.questions.map { pendingCard(it) }.toMutableList()
          mutableState.update { it.copy(cards = cards.toList(), footerLines = emptyList()) }
          val results = ArrayList<KevQuestionResult>()
          var workMs = prepared.tokenizeMs
          for (index in request.questions.indices) {
            mutableState.update {
              it.copy(status = KevStatus.Running(index, request.questions.size))
            }
            cards[index] = cards[index].copy(state = CardState.RUNNING)
            mutableState.update { it.copy(cards = cards.toList()) }
            val start = System.nanoTime()
            val outcome = answerQuestion(loaded, prepared, index)
            cards[index] = outcome.card
            workMs += KevPipeline.millis(System.nanoTime() - start)
            mutableState.update { it.copy(cards = cards.toList()) }
            outcome.result?.let { results.add(it) }
              ?: return@guarded requestFailed(loaded, outcome.card.error ?: "")
          }
          val answers = loaded.pipeline.answers(prepared, results)
          val totalMs = KevAnswerView.wholeMillis(workMs)
          mutableState.update {
            it.copy(
              status = KevStatus.Done(totalMs, results.size),
              engine = engineUi(loaded),
              footerLines = footerLines(loaded, totalMs),
              responseJson = KevJson.writeIndented(loaded.pipeline.response(prepared, answers)),
            )
          }
        }
      }
    }
  }

  /** The window the request's longest row needs, or null after showing why it cannot run. */
  private fun requiredWindow(prepared: KevPrepared): Int? {
    val window = prepared.window
    if (window == null) {
      requestRejected(
        string(R.string.error_over_largest, prepared.longestRow, KevEncoder.WINDOWS.last())
      )
      return null
    }
    val file = File(context.filesDir, KevFiles.graph(window))
    if (window != engine?.window && !file.isFile) {
      requestRejected(string(R.string.error_needs_window, window, file.name))
      return null
    }
    return window
  }

  private class QuestionOutcome(val card: AnswerCardUi, val result: KevQuestionResult?)

  private fun answerQuestion(
    loaded: KevEngine,
    prepared: KevPrepared,
    index: Int,
  ): QuestionOutcome {
    val question = prepared.request.questions[index]
    val card = pendingCard(question)
    return try {
      val result = loaded.pipeline.run(prepared, index, loaded.decider)
      val answer = singleAnswer(result)
      val view = KevAnswerView.of(answer, result.meta, result.probabilities)
      val ms = string(R.string.card_ms, KevAnswerView.wholeMillis(result.inferMs))
      QuestionOutcome(card.copy(state = CardState.DONE, view = view, msText = ms), result)
    } catch (failure: KevNonFiniteException) {
      QuestionOutcome(
        card.copy(
          state = CardState.FAILED,
          error = string(R.string.error_nonfinite, failure.questionId, failure.count),
        ),
        null,
      )
    }
  }

  private fun singleAnswer(result: KevQuestionResult): Map<*, *> =
    KevAnswers.toAnswers(listOf(result.probabilities), listOf(result.meta)).getValue(result.meta.id)
      as Map<*, *>

  private fun pendingCard(question: KevQuestion) =
    AnswerCardUi(
      question.id,
      question.type,
      KevRecords.render(question.instructions).ifEmpty { question.id },
      CardState.PENDING,
    )

  private fun requestRejected(message: String) {
    mutableState.update { state ->
      state.copy(requestError = message, status = engine?.let { readyStatus(it) } ?: state.status)
    }
  }

  private fun requestFailed(loaded: KevEngine, message: String) {
    mutableState.update {
      it.copy(requestError = message, status = readyStatus(loaded), engine = engineUi(loaded))
    }
  }

  // ---- Autoplay (demo recording) ----

  /** Thrown inside the autoplay to end it with `failed <reason>`. */
  private class AutoplayFailure(reason: String) : Exception(reason)

  private fun autoplay(launch: KevLaunch.Autoplay, receivedNanos: Long) {
    KevDemo.autoplayStart(launch.fixture)
    if (uiState.value.mode != LaunchMode.INTERACTIVE) {
      KevDemo.failed("busy with a ${uiState.value.mode.name.lowercase()} run")
      return
    }
    worker.launch {
      engineLock.withLock {
        try {
          runAutoplay(launch, receivedNanos)
        } catch (failure: AutoplayFailure) {
          autoplayFailed(failure.message.orEmpty())
        } catch (failure: Exception) {
          autoplayFailed(KevDecider.describe(failure))
        } catch (failure: LinkageError) {
          autoplayFailed("native runtime ${KevDecider.describe(failure)}")
        }
      }
    }
  }

  private suspend fun runAutoplay(launch: KevLaunch.Autoplay, receivedNanos: Long) {
    val file =
      filesPath(launch.fixture)
        ?: throw AutoplayFailure("fixture outside the app files dir: ${launch.fixture}")
    if (!file.isFile) throw AutoplayFailure("missing fixture ${file.path}")
    val fixture =
      try {
        KevFixture.parse(file.readText())
      } catch (failure: IllegalArgumentException) {
        throw AutoplayFailure("fixture does not parse: ${failure.message}")
      }
    val loaded =
      engine ?: throw AutoplayFailure("engine not ready: ${statusText(uiState.value.status)}")
    if (launch.window != loaded.window || loaded.decider.isClosed) {
      if (!File(context.filesDir, KevFiles.graph(launch.window)).isFile) {
        throw AutoplayFailure("missing ${KevFiles.graph(launch.window)}")
      }
      setLoading(LoadStage.GRAPH, launch.window, switching = true)
      loaded.ensureGraph(context, launch.window, uiState.value.backendChoice, cpuFallback = true)
      setReady(loaded)
    }
    demoLayout = null
    val requestStart = System.nanoTime()
    val prepared = loaded.pipeline.prepare(fixture.request)
    val tooLong = prepared.rows.indexOfFirst { it.length > loaded.window }
    if (tooLong >= 0) {
      throw AutoplayFailure(
        "row ${prepared.meta[tooLong].id} is ${prepared.rows[tooLong].length} tokens > L${loaded.window}"
      )
    }
    val cgroup = KevDevice.cgroup()
    var waitedNanos = 0L
    suspend fun waitFor(ms: Long) {
      val start = System.nanoTime()
      if (ms > 0) delay(ms)
      waitedNanos += System.nanoTime() - start
    }
    // delay_ms counts from the intent's arrival to the ticket on screen.
    waitFor(launch.delayMs - (SystemClock.elapsedRealtimeNanos() - receivedNanos) / NANOS_PER_MILLI)
    val title = string(R.string.presentation_title)
    val cards = fixture.request.questions.map { pendingCard(it) }.toMutableList()
    var presentation =
      PresentationUi(
        title,
        KevRecords.render(fixture.request.state),
        cards.toList(),
        presentationFooter(loaded, null),
      )
    mutableState.update {
      it.copy(presentation = presentation, status = KevStatus.Running(0, cards.size))
    }
    val questions = ArrayList<KevDemoQuestion>()
    var lastAnswerNanos = requestStart
    for (index in cards.indices) {
      waitFor(launch.gapMs)
      cards[index] = cards[index].copy(state = CardState.RUNNING)
      presentation = presentation.copy(cards = cards.toList())
      mutableState.update {
        it.copy(presentation = presentation, status = KevStatus.Running(index, cards.size))
      }
      val start = System.nanoTime()
      val outcome = answerQuestion(loaded, prepared, index)
      lastAnswerNanos = System.nanoTime()
      val totalNanos = lastAnswerNanos - start
      cards[index] = outcome.card
      presentation = presentation.copy(cards = cards.toList())
      mutableState.update { it.copy(presentation = presentation) }
      val result = outcome.result ?: throw AutoplayFailure(outcome.card.error.orEmpty())
      val inferMs = KevAnswerView.wholeMillis(result.inferMs)
      KevDemo.questionDone(result.meta.id, inferMs)
      questions.add(
        KevDemoQuestion(
          result,
          singleAnswer(result),
          requireNotNull(outcome.card.view).shownCompact(),
          requireNotNull(outcome.card.msText),
          KevAnswerView.wholeMillis(prepared.questionTokenizeMs(index)),
          inferMs,
          KevAnswerView.wholeMillis(result.headMs),
          KevAnswerView.wholeMillis(KevPipeline.millis(totalNanos)),
        )
      )
    }
    // From the start of tokenizing to the last answer, without the presentation waits.
    val requestTotalMs =
      KevAnswerView.wholeMillis(KevPipeline.millis(lastAnswerNanos - requestStart - waitedNanos))
    val footer = presentationFooter(loaded, requestTotalMs)
    presentation = presentation.copy(footerLines = footer)
    mutableState.update {
      it.copy(presentation = presentation, status = KevStatus.Done(requestTotalMs, questions.size))
    }
    // Let the final frame reach the screen before the layout is recorded.
    delay(LAYOUT_SETTLE_MS)
    val run =
      KevDemoRun.build(
        KevDemoRunInput(
          fixtureId = fixture.id,
          fixturePath = file.path,
          deviceModel = Build.MODEL,
          deviceManufacturer = Build.MANUFACTURER,
          deviceShownAs = KevDevice.marketName(),
          deviceAndroidRelease = Build.VERSION.RELEASE,
          litert = KevDecider.LITERT_VERSION,
          accelerator = backendName(loaded.backend),
          graphFile = loaded.decider.file.name,
          window = loaded.window,
          graphBytes = loaded.decider.file.length(),
          engineLoadMs = KevAnswerView.wholeMillis(loaded.loadMs),
          warmupMs = KevAnswerView.wholeMillis(lastWarmupMs),
          title = title,
          footerLines = footer,
          delayMs = launch.delayMs,
          gapMs = launch.gapMs,
          tokenizeMs = KevAnswerView.wholeMillis(prepared.tokenizeMs),
          requestTotalMs = requestTotalMs,
          questions = questions,
          airplaneMode = KevDevice.airplaneMode(context),
          cgroup = cgroup,
          cgroupEnd = KevDevice.cgroup(),
          layout = demoLayout,
        )
      )
    val written = KevDemo.writeRun(context, run)
    KevDemo.autoplayDone(written.path)
  }

  private fun autoplayFailed(reason: String) {
    KevDemo.failed(reason)
    mutableState.update { state ->
      val presentation = state.presentation?.copy(failure = reason)
      state.copy(
        presentation =
          presentation
            ?: PresentationUi(
              string(R.string.presentation_title),
              "",
              emptyList(),
              emptyList(),
              reason,
            ),
        status = engine?.let { readyStatus(it) } ?: state.status,
      )
    }
  }

  /** The presentation screen reports where it drew the cards. */
  fun onPresentationLayout(layout: KevDemoLayout) {
    demoLayout = layout
  }

  /** Back from the presentation layout to the editable sample. */
  fun leavePresentation() {
    if (uiState.value.status is KevStatus.Running) return
    mutableState.update { it.copy(presentation = null) }
  }

  // ---- Gate and timing (debug) ----

  private fun runGate(launch: KevLaunch.Gate) {
    mutableState.update {
      it.copy(
        mode = LaunchMode.GATE,
        backendChoice = launch.backend,
        diagnostics = string(R.string.gate_running),
      )
    }
    worker.launch {
      engineLock.withLock {
        guarded(onFailure = ::diagnosticsFailed) {
          val summary =
            KevGateRunner(context)
              .run(
                KevGateRunner.Args(launch.backend, launch.report, launch.window, launch.limit),
                loadEngine = { diagnosticEngine(launch.window, launch.backend) },
                progress = { step ->
                  mutableState.update {
                    it.copy(diagnostics = string(R.string.gate_progress, step))
                  }
                },
              )
          diagnosticsDone(summary)
        }
      }
    }
  }

  private fun runTiming(launch: KevLaunch.Timing) {
    mutableState.update {
      it.copy(
        mode = LaunchMode.TIMING,
        backendChoice = launch.backend,
        diagnostics = string(R.string.timing_running),
      )
    }
    val rows = filesPath(launch.rows)
    if (rows == null) {
      Log.i(KevGateRunner.LOG_TAG, "failed rows outside the app files dir: ${launch.rows}")
      mutableState.update {
        it.copy(diagnostics = string(R.string.diagnostics_failed, launch.rows))
      }
      return
    }
    worker.launch {
      engineLock.withLock {
        guarded(onFailure = ::diagnosticsFailed) {
          val summary =
            KevTimingRunner(context)
              .run(
                KevTimingRunner.Args(
                  rows,
                  launch.backend,
                  launch.report,
                  launch.window,
                  launch.clearCache,
                  launch.sets,
                  launch.requestPath,
                ),
                loadEngine = { diagnosticEngine(launch.window, launch.backend) },
                progress = { step ->
                  mutableState.update {
                    it.copy(diagnostics = string(R.string.timing_progress, step))
                  }
                },
              )
          diagnosticsDone(summary)
        }
      }
    }
  }

  /** The engine for a gate or timing run: the requested window and backend, no CPU fallback. */
  private fun diagnosticEngine(window: Int, backend: KevDecider.Backend): KevEngine {
    val missing = KevFiles.missing(context.filesDir, window)
    check(missing.isEmpty()) { "missing ${missing.joinToString(" ")}" }
    engine?.close()
    engine = null
    return KevEngine.load(context, window, backend, cpuFallback = false) { stage ->
        setLoading(stage, window, false)
      }
      .also {
        engine = it
        setReady(it)
        KevDemo.engineReady(KevAnswerView.wholeMillis(it.loadMs))
      }
  }

  private fun diagnosticsDone(summary: KevGateRunner.Summary) {
    mutableState.update {
      it.copy(
        diagnostics =
          string(R.string.diagnostics_done, summary.status, summary.path) +
            (summary.error?.let { e -> "\n$e" } ?: "")
      )
    }
  }

  /** A gate or timing run that could not write its report. */
  private fun diagnosticsFailed(message: String) {
    Log.i(KevGateRunner.LOG_TAG, "failed $message")
    mutableState.update { it.copy(diagnostics = message) }
  }

  private fun reportInvalid(launch: KevLaunch.Invalid) {
    if (launch.autoplay) KevDemo.failed(launch.reason)
    else Log.i(KevGateRunner.LOG_TAG, "failed ${launch.reason}")
    mutableState.update { it.copy(requestError = launch.reason) }
  }

  // ---- State helpers ----

  private fun setLoading(stage: LoadStage, window: Int, switching: Boolean) {
    mutableState.update {
      it.copy(status = KevStatus.Loading(stage, window, switching, SystemClock.elapsedRealtime()))
    }
    startTicker()
  }

  /** Advances the elapsed seconds of a Loading status once a second. */
  private fun startTicker() {
    if (ticker?.isActive == true) return
    ticker = viewModelScope.launch {
      while (isActive) {
        delay(TICK_MS)
        val status = uiState.value.status as? KevStatus.Loading ?: break
        val elapsed = ((SystemClock.elapsedRealtime() - status.startedAt) / TICK_MS).toInt()
        mutableState.update { state ->
          val current = state.status
          if (current is KevStatus.Loading)
            state.copy(status = current.copy(elapsedSeconds = elapsed))
          else state
        }
      }
    }
  }

  private fun setReady(loaded: KevEngine, warmupMs: Double? = null) {
    if (warmupMs != null) lastWarmupMs = warmupMs
    mutableState.update { it.copy(status = readyStatus(loaded), engine = engineUi(loaded)) }
  }

  private fun readyStatus(loaded: KevEngine) =
    KevStatus.Ready(
      loaded.backend,
      loaded.window,
      KevAnswerView.wholeMillis(loaded.loadMs),
      KevAnswerView.wholeMillis(loaded.decider.compileMs),
    )

  private fun engineUi(loaded: KevEngine) =
    EngineUi(
      loaded.backend,
      loaded.window,
      KevAnswerView.wholeMillis(loaded.loadMs),
      KevAnswerView.wholeMillis(loaded.decider.compileMs),
      loaded.gpuFailure,
    )

  private fun showError(message: String) {
    mutableState.update {
      it.copy(status = KevStatus.Error(message), engine = engine?.let { e -> engineUi(e) })
    }
  }

  /** Runs [block], turning failures (including a missing native library) into [onFailure]. */
  private inline fun guarded(onFailure: (String) -> Unit, block: () -> Unit) {
    try {
      block()
    } catch (failure: Exception) {
      onFailure(KevDecider.describe(failure))
    } catch (failure: LinkageError) {
      onFailure("Native runtime: ${KevDecider.describe(failure)}")
    }
  }

  /**
   * The footer under the answers, one line per group so that a 360 dp wide screen does not wrap
   * them: device and Android version; LiteRT and backend; graph and the request total once known.
   */
  private fun footerLines(loaded: KevEngine, totalMs: Long?): List<String> =
    listOf(
      string(R.string.footer_device, KevDevice.displayName(), Build.VERSION.RELEASE),
      string(R.string.footer_litert, KevDecider.LITERT_VERSION, backendName(loaded.backend)),
      if (totalMs == null) string(R.string.footer_graph, loaded.window)
      else string(R.string.footer_graph_total, loaded.window, totalMs),
    )

  /**
   * The presentation footer: three short lines that fit a 360 dp wide screen (device name, LiteRT
   * and backend; graph; the request total, empty until the end). Model code and Android version are
   * in the run JSON only.
   */
  private fun presentationFooter(loaded: KevEngine, totalMs: Long?): List<String> =
    listOf(
      string(
        R.string.footer_runtime,
        KevDevice.marketName(),
        KevDecider.LITERT_VERSION,
        backendName(loaded.backend),
      ),
      string(R.string.footer_graph, loaded.window),
      if (totalMs == null) "" else string(R.string.presentation_total, totalMs),
    )

  private fun backendName(backend: KevDecider.Backend): String =
    string(if (backend == KevDecider.Backend.GPU) R.string.backend_gpu else R.string.backend_cpu)

  private fun statusText(status: KevStatus): String = status.javaClass.simpleName

  private fun draftMessage(failure: KevDraftException): String =
    when (failure.problem) {
      DraftProblem.NO_QUESTIONS -> string(R.string.draft_no_questions)
      DraftProblem.EMPTY_ID -> string(R.string.draft_empty_id, failure.question)
      DraftProblem.DUPLICATE_ID ->
        string(R.string.draft_duplicate_id, failure.question, failure.detail)
      DraftProblem.NO_OPTIONS -> string(R.string.draft_no_options, failure.question)
      DraftProblem.TOO_MANY_OPTIONS ->
        string(R.string.draft_too_many_options, failure.question, KevRequest.MAX_OPTIONS)
      DraftProblem.EMPTY_OPTION_NAME ->
        string(R.string.draft_empty_option_name, failure.question, failure.detail)
      DraftProblem.DUPLICATE_OPTION ->
        string(R.string.draft_duplicate_option, failure.question, failure.detail)
      DraftProblem.BAD_NOUL_OPTION ->
        string(R.string.draft_bad_noul_option, failure.question, failure.detail)
    }

  /**
   * [path] inside `files/`: a bare name or an absolute path there; null when it points elsewhere.
   */
  private fun filesPath(path: String): File? {
    val base = context.filesDir.canonicalFile
    val file = (if (path.startsWith("/")) File(path) else File(base, path)).canonicalFile
    return file.takeIf { it.toPath().startsWith(base.toPath()) }
  }

  private fun string(@StringRes id: Int, vararg args: Any): String = context.getString(id, *args)

  override fun onCleared() {
    cleared = true
    // Close after any in-flight native call, on the same worker thread.
    worker.launch {
      engineLock.withLock {
        try {
          engine?.close()
          engine = null
        } catch (failure: Exception) {
          Log.e(KevDemo.LOG_TAG, "Engine cleanup failed", failure)
        }
      }
      worker.cancel()
    }
    super.onCleared()
  }

  companion object {
    private val EXAMPLES =
      listOf(R.raw.example_ticket, R.raw.example_incident, R.raw.example_review)
    private const val FIRST_ADDED_KEY = 1_000L
    private const val TICK_MS = 1_000L
    private const val NANOS_PER_MILLI = 1_000_000L
    private const val LAYOUT_SETTLE_MS = 300L

    /** Factory that builds the ViewModel with the application context. */
    fun getFactory(context: Context): ViewModelProvider.Factory =
      object : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
          require(modelClass.isAssignableFrom(MainViewModel::class.java))
          @Suppress("UNCHECKED_CAST")
          return MainViewModel(context.applicationContext) as T
        }
      }
  }
}
