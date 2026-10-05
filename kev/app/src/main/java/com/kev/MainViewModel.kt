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

  /** How Decide plans a request (the `graph` extra of the launch that started the app). */
  private var interactiveGraph = KevGraphMode.AUTO

  /**
   * The GPU precision of every graph of this process (the `precision` extra), or null when each
   * graph runs at its own default ([KevPrecision.defaultFor]).
   */
  private var precision: KevPrecision? = null

  /** Whether a pair compiles with constant tensor sharing (the `share` extra; auto by default). */
  private var pairShare = KevPairShare.AUTO

  /** The APK carries the NPU libraries: the NPU choice works. */
  private val npuAvailable = KevNpu.librariesInstalled(context)

  /** The Qualcomm options of every graph of a normal or autoplay launch (null without them). */
  private val npuOptions: KevNpuOptions? = if (npuAvailable) KevNpuOptions() else null

  /** Where the "Run on" choice is kept between launches. */
  private val preferences =
    context.getSharedPreferences(KevBackendChoice.PREFERENCES, Context.MODE_PRIVATE)

  /** The bundled requests (`res/raw`): invented support ticket, incident report and review. */
  private val examples: List<KevFixture> = EXAMPLES.map { id ->
    KevFixture.parse(context.resources.openRawResource(id).use { it.readBytes() }.decodeToString())
  }

  private val mutableState =
    MutableStateFlow(
      UiState(draft = KevDrafts.fromRequest(examples[0].request), npuAvailable = npuAvailable)
    )

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
        usePrecision(launch.precision)
        pairShare = launch.share
        useBackend(launch.backend)
        // The graphs compiled at startup are the ones the fixture's request runs on.
        loadInteractive { autoplayStartup(launch) }
        autoplay(launch, SystemClock.elapsedRealtimeNanos())
      }
      is KevLaunch.Invalid -> {
        reportInvalid(launch)
        useBackend(null)
        loadInteractive { editorStartup() }
      }
      is KevLaunch.Normal -> {
        interactiveGraph = launch.graph
        usePrecision(launch.precision)
        pairShare = launch.share
        useBackend(launch.backend)
        loadInteractive { editorStartup() }
      }
    }
  }

  private fun usePrecision(value: KevPrecision?) {
    precision = value
    mutableState.update { it.copy(precision = value) }
  }

  /**
   * The backend a normal or autoplay launch starts on ([KevBackendChoice]): its `backend` extra,
   * else the kept choice; a kept NPU that this APK cannot run is replaced by the GPU.
   */
  private fun useBackend(extra: KevDecider.Backend?) {
    val kept = preferences.getString(KevBackendChoice.KEY, null)
    if (KevBackendChoice.keptUnusable(kept, npuAvailable)) keep(KevDecider.Backend.GPU)
    val backend = KevBackendChoice.resolve(extra, kept, npuAvailable)
    mutableState.update { it.copy(backendChoice = backend) }
  }

  private fun keep(backend: KevDecider.Backend) {
    preferences.edit().putString(KevBackendChoice.KEY, backend.wireName).apply()
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
      is KevLaunch.Normal -> Unit
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

  /**
   * Closes the compiled graphs and compiles the plan of the request in the editor on [backend]
   * (nothing when the editor does not hold a valid request); NPU falls back to GPU and GPU to CPU
   * if it fails. The choice is kept for the next launch.
   */
  fun selectBackend(backend: KevDecider.Backend) {
    val state = uiState.value
    if (!state.canDecide || backend == state.backendChoice) return
    if (backend == KevDecider.Backend.NPU && !npuAvailable) return
    keep(backend)
    mutableState.update { it.copy(backendChoice = backend) }
    val loaded = engine ?: return
    val request = runCatching { KevDrafts.toRequest(state.draft) }.getOrNull()
    worker.launch {
      engineLock.withLock {
        guarded(onFailure = ::showError) {
          loaded.switchBackend(backend)
          mutableState.update { it.copy(engine = engineUi(loaded)) }
          if (request != null) {
            val plan = loaded.plan(loaded.pipeline.prepare(request), interactiveGraph)
            if (plan is KevPlan.Ready) prepareGraphs(loaded, plan, switching = false)
          }
          setReady(loaded)
        }
      }
    }
  }

  /** The startup request of a normal launch: the bundled example the editor opens with. */
  private fun editorStartup() = StartupRequest(examples[0].request, interactiveGraph, null)

  /**
   * The startup request of an autoplay launch: the fixture's, on the autoplay's plan, or the
   * editor's when the fixture cannot be read (the autoplay then fails with its own reason).
   */
  private fun autoplayStartup(launch: KevLaunch.Autoplay): StartupRequest {
    val fixture =
      filesPath(launch.fixture)
        ?.takeIf { it.isFile }
        ?.let { file ->
          runCatching { KevFixture.parse(file.readText()) }.getOrNull()
        } ?: return editorStartup()
    val mode = if (launch.window != null) KevGraphMode.ROWS else launch.graph
    return StartupRequest(fixture.request, mode, launch.window)
  }

  /** A request whose plan the engine compiles while it loads. */
  private class StartupRequest(val request: KevRequest, val mode: KevGraphMode, val window: Int?)

  /**
   * Loads tokenizer and head, then compiles the graphs of the [startup] request's plan and warms
   * them up: `ENGINE_READY` comes after that compile.
   */
  private fun loadInteractive(startup: () -> StartupRequest) {
    val missing = KevFiles.missing(context.filesDir)
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
            KevEngine.load(
              context,
              uiState.value.backendChoice,
              precision,
              pairShare,
              cpuFallback = true,
              npuOptions,
            ) { stage ->
              setLoading(stage, null, switching = false)
            }
          engine = loaded
          val warmupMs = compileAtStartup(loaded, startup())
          loaded.markLoaded()
          loaded.gpuFailure?.let { KevDemo.gpuFallback(it) }
          setReady(loaded, warmupMs)
          KevDemo.engineReady(KevAnswerView.wholeMillis(loaded.loadMs ?: 0.0))
        }
      }
    }
  }

  /**
   * Compiles the plan of [startup] and makes one untimed call on each graph (rows: the earliest
   * question of each window; pair: the state and question 1), so that a request does not pay the
   * graphs' start-up costs. Returns the warm-up wall time; nothing compiles when no form takes the
   * request.
   */
  private fun compileAtStartup(loaded: KevEngine, startup: StartupRequest): Double {
    val prepared = loaded.pipeline.prepare(startup.request)
    val plan = loaded.plan(prepared, startup.mode, startup.window)
    KevDemo.plan(plan, loaded.lastPlanInputs)
    if (plan !is KevPlan.Ready) return 0.0
    val graphs = prepareGraphs(loaded, plan, switching = false)
    val start = System.nanoTime()
    // A warm-up call that returns non-finite values only warms up: the request reports them.
    runCatching {
      when (val runners = graphs.runners) {
        is KevRunners.Rows ->
          for (window in graphs.windows.distinct()) {
            val index = graphs.windows.indexOf(window)
            loaded.pipeline.run(prepared, index, runners.graphs[index])
          }
        is KevRunners.Pair -> {
          loaded.pipeline.runState(prepared, runners.graph)
          loaded.pipeline.runBranch(prepared, 0, runners.graph)
        }
      }
    }
      .exceptionOrNull()
      ?.let { if (it !is KevNonFiniteException) throw it }
    return KevPipeline.millis(System.nanoTime() - start)
  }

  // ---- Decide ----

  /**
   * Answers the edited request: tokenize once, then per question one row graph call and the head,
   * or (pair plan) the state call once and per question one branch call and the head.
   */
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
          val plan = readyPlan(loaded, loaded.plan(prepared, interactiveGraph)) ?: return@guarded
          val graphs = prepareGraphs(loaded, plan, switching = true)
          val cards = request.questions.map { pendingCard(it) }.toMutableList()
          val pair = graphs.pair
          var stateLine = pair?.let { StateLineUi(prepared.encoded.stateIds.size) }
          mutableState.update {
            it.copy(cards = cards.toList(), stateLine = stateLine, footerLines = emptyList())
          }
          val results = ArrayList<KevQuestionResult>()
          var workMs = prepared.tokenizeMs
          if (pair != null) {
            val start = System.nanoTime()
            val stateResult = loaded.pipeline.runState(prepared, pair)
            workMs += KevPipeline.millis(System.nanoTime() - start)
            stateLine = StateLineUi(stateResult.tokens, KevAnswerView.wholeMillis(stateResult.ms))
            mutableState.update { it.copy(stateLine = stateLine) }
          }
          for (index in request.questions.indices) {
            mutableState.update {
              it.copy(status = KevStatus.Running(index, request.questions.size))
            }
            cards[index] = cards[index].copy(state = CardState.RUNNING)
            mutableState.update { it.copy(cards = cards.toList()) }
            val start = System.nanoTime()
            val outcome = answerQuestion(loaded, prepared, index, graphs)
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
              footerLines = footerLines(loaded, graphs, totalMs),
              responseJson = KevJson.writeIndented(loaded.pipeline.response(prepared, answers)),
            )
          }
        }
      }
    }
  }

  /**
   * [plan] when a form takes the request, or null after showing why it cannot run; logs the plan
   * (`PLAN` under [KevDemo.LOG_TAG]).
   */
  private fun readyPlan(loaded: KevEngine, plan: KevPlan): KevPlan.Ready? {
    KevDemo.plan(plan, loaded.lastPlanInputs)
    return when (plan) {
      is KevPlan.Ready -> plan
      is KevPlan.NoWindow,
      is KevPlan.NoPair -> {
        requestRejected(planMessage(plan))
        null
      }
    }
  }

  /** Why [plan] cannot run, as the screen says it. */
  private fun planMessage(plan: KevPlan): String =
    when (plan) {
      is KevPlan.Ready -> ""
      is KevPlan.NoWindow -> {
        val window = plan.missing.window
        if (window == null) {
          string(R.string.error_over_largest, plan.missing.rowTokens, KevFiles.WINDOWS.last())
        } else {
          string(R.string.error_needs_window, window, KevFiles.graph(window))
        }
      }
      is KevPlan.NoPair ->
        when (val miss = plan.miss) {
          KevPairMiss.NotInstalled -> string(R.string.error_no_pair)
          is KevPairMiss.StateTooLong -> string(R.string.error_pair_state, miss.tokens, miss.window)
          is KevPairMiss.QuestionTooLong -> {
            val draft = uiState.value.draft.questions.getOrNull(miss.index)
            string(
              R.string.error_pair_question,
              draft?.id ?: (miss.index + 1).toString(),
              miss.tokens,
              miss.window,
            )
          }
        }
    }

  /**
   * Closes and compiles what [plan] needs, the status line counting the seconds of each compile,
   * and logs what was done (`WINDOWS` under [KevDemo.LOG_TAG]); returns the request's graphs.
   */
  private fun prepareGraphs(
    loaded: KevEngine,
    plan: KevPlan.Ready,
    switching: Boolean,
  ): KevRequestGraphs {
    val graphs =
      loaded.prepare(plan) { step ->
        KevDemo.compileStart(step)
        setLoading(LoadStage.GRAPH, step.graph, switching, step)
      }
    graphs.compiles.forEach { KevDemo.compiled(it) }
    if (graphs.compiled.isNotEmpty() || graphs.closed.isNotEmpty()) {
      mutableState.update { it.copy(engine = engineUi(loaded)) }
    }
    KevDemo.windows(graphs, loaded.resident)
    return graphs
  }

  private class QuestionOutcome(val card: AnswerCardUi, val result: KevQuestionResult?)

  /** Question [index] on its row graph, or its branch on the pair after the state call. */
  private fun answerQuestion(
    loaded: KevEngine,
    prepared: KevPrepared,
    index: Int,
    graphs: KevRequestGraphs,
  ): QuestionOutcome {
    val question = prepared.request.questions[index]
    val card = pendingCard(question)
    return try {
      val result =
        when (val runners = graphs.runners) {
          is KevRunners.Rows -> loaded.pipeline.run(prepared, index, runners.graphs[index])
          is KevRunners.Pair -> loaded.pipeline.runBranch(prepared, index, runners.graph)
        }
      val answer = singleAnswer(result)
      val view = KevAnswerView.of(answer, result.meta, result.probabilities)
      val ms = string(R.string.card_ms, KevAnswerView.wholeMillis(result.inferMs))
      QuestionOutcome(
        card.copy(
          state = CardState.DONE,
          view = view,
          msText = ms,
          form = result.form,
          window = result.window,
          backend = graphs.backendOf(index),
        ),
        result,
      )
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
    // The plan tokenizes once; the timed request below tokenizes again, as from its text.
    val plan = autoplayPlan(loaded, loaded.pipeline.prepare(fixture.request), launch)
    val planInputs = loaded.lastPlanInputs
    val graphs = prepareGraphs(loaded, plan, switching = true)
    if (graphs.compiled.isNotEmpty()) setReady(loaded)
    demoLayout = null
    val requestStart = System.nanoTime()
    val prepared = loaded.pipeline.prepare(fixture.request)
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
    val pair = graphs.pair
    var presentation =
      PresentationUi(
        title,
        KevRecords.render(fixture.request.state),
        cards.toList(),
        presentationFooter(loaded, graphs, null),
        state = pair?.let { StateLineUi(prepared.encoded.stateIds.size) },
      )
    mutableState.update {
      it.copy(presentation = presentation, status = KevStatus.Running(0, cards.size))
    }
    var stateResult: KevStateResult? = null
    var lastAnswerNanos = requestStart
    if (pair != null) {
      waitFor(launch.gapMs)
      val state = loaded.pipeline.runState(prepared, pair)
      lastAnswerNanos = System.nanoTime()
      stateResult = state
      presentation =
        presentation.copy(state = StateLineUi(state.tokens, KevAnswerView.wholeMillis(state.ms)))
      mutableState.update { it.copy(presentation = presentation) }
      KevDemo.stateDone(state.tokens, KevAnswerView.wholeMillis(state.ms))
    }
    val questions = ArrayList<KevDemoQuestion>()
    for (index in cards.indices) {
      waitFor(launch.gapMs)
      cards[index] = cards[index].copy(state = CardState.RUNNING)
      presentation = presentation.copy(cards = cards.toList())
      mutableState.update {
        it.copy(presentation = presentation, status = KevStatus.Running(index, cards.size))
      }
      val start = System.nanoTime()
      val outcome = answerQuestion(loaded, prepared, index, graphs)
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
          graphs.precisionOf(index).takeIf { graphs.backendOf(index) == KevDecider.Backend.GPU },
          graphs.backendOf(index),
        )
      )
    }
    // From the start of tokenizing to the last answer, without the presentation waits.
    val requestTotalMs =
      KevAnswerView.wholeMillis(KevPipeline.millis(lastAnswerNanos - requestStart - waitedNanos))
    val footer = presentationFooter(loaded, graphs, requestTotalMs)
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
          accelerator = backendName(graphs),
          precision = gpuPrecisionName(graphs),
          precisionRequested = precision?.wireName,
          graph =
            graphFile(
              graphs.used.last(),
              graphs.precisions.last().takeIf { graphs.backends.last() == KevDecider.Backend.GPU },
              graphs.pairShare,
            ),
          form = plan.form,
          windowsUsed = graphs.used.filterIsInstance<KevGraphKey.Window>().map { it.window },
          resident =
            loaded.resident.indices.map { index ->
              val graph = loaded.resident[index]
              graphFile(
                graph,
                loaded.residentPrecisions[index].takeIf {
                  loaded.residentBackends[index] == KevDecider.Backend.GPU
                },
                loaded.pair?.shareConstants?.takeIf { graph is KevGraphKey.Pair },
              )
            },
          compiled = graphs.compiled,
          availableBytesBeforeCompile = graphs.availableBytes,
          closed = graphs.closed,
          secondRefused = graphs.secondRefused,
          plan = KevDemoPlan(launch.graphRequested(), plan.form, plan.prediction, planInputs),
          state = stateResult,
          engineLoadMs = KevAnswerView.wholeMillis(loaded.loadMs ?: 0.0),
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
          accelerators = acceleratorsJson(loaded, graphs),
        )
      )
    val written = KevDemo.writeRun(context, run)
    KevDemo.autoplayDone(written.path)
  }

  /** The plan mode an autoplay asked for: `rows` when it names a window. */
  private fun KevLaunch.Autoplay.graphRequested(): KevGraphMode =
    if (window != null) KevGraphMode.ROWS else graph

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

  /**
   * The plan of an autoplay run: every question on the named window, or the plan of the launch's
   * graph mode (the plan a Decide would use for `auto`).
   */
  private fun autoplayPlan(
    loaded: KevEngine,
    prepared: KevPrepared,
    launch: KevLaunch.Autoplay,
  ): KevPlan.Ready {
    val window = launch.window
    if (window != null) {
      if (!File(context.filesDir, KevFiles.graph(window)).isFile) {
        throw AutoplayFailure("missing ${KevFiles.graph(window)}")
      }
      val tooLong = prepared.rows.indexOfFirst { it.length > window }
      if (tooLong >= 0) {
        throw AutoplayFailure(
          "row ${prepared.meta[tooLong].id} is ${prepared.rows[tooLong].length} tokens > L$window"
        )
      }
    }
    val plan = loaded.plan(prepared, launch.graphRequested(), window)
    KevDemo.plan(plan, loaded.lastPlanInputs)
    return when (plan) {
      is KevPlan.Ready -> plan
      is KevPlan.NoWindow ->
        throw AutoplayFailure(
          plan.missing.window?.let { "missing ${KevFiles.graph(it)}" }
            ?: "a row is ${plan.missing.rowTokens} tokens > L${KevFiles.WINDOWS.last()}"
        )
      is KevPlan.NoPair ->
        throw AutoplayFailure(
          when (val miss = plan.miss) {
            KevPairMiss.NotInstalled -> "no shared-state pair installed"
            is KevPairMiss.StateTooLong -> "the state is ${miss.tokens} tokens > Ls ${miss.window}"
            is KevPairMiss.QuestionTooLong ->
              "question ${prepared.meta[miss.index].id} is ${miss.tokens} tokens > Lq ${miss.window}"
          }
        )
    }
  }

  private fun graphFile(
    graph: KevGraphKey,
    precision: KevPrecision?,
    share: Boolean?,
  ): KevGraphFile {
    val file = File(context.filesDir, graph.file)
    return KevGraphFile(graph, file.length(), precision, share)
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
    usePrecision(launch.precision)
    pairShare = launch.share
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
                KevGateRunner.Args(
                  launch.backend,
                  launch.precision,
                  launch.report,
                  launch.graph,
                  launch.limit,
                ),
                loadEngine = {
                  diagnosticEngine(launch.graph, launch.backend, launch.npu, launch.stateCopy)
                },
                prepare = { loaded, graph -> diagnosticGraph(loaded, graph) },
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
    usePrecision(launch.precision)
    pairShare = launch.share
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
    val setGraph =
      if (launch.graph == KevGraphMode.PAIR) KevGraphKey.Pair(launch.pair)
      else KevGraphKey.Window(launch.window)
    worker.launch {
      engineLock.withLock {
        guarded(onFailure = ::diagnosticsFailed) {
          val summary =
            KevTimingRunner(context)
              .run(
                KevTimingRunner.Args(
                  rows,
                  launch.backend,
                  launch.precision,
                  launch.report,
                  setGraph,
                  launch.graph,
                  launch.window.takeIf { launch.windowNamed },
                  launch.clearCache,
                  launch.sets,
                  launch.requestPath,
                  launch.coolMs,
                ),
                loadEngine = {
                  diagnosticEngine(setGraph, launch.backend, launch.npu, launch.stateCopy)
                },
                prepare = { loaded, graph -> diagnosticGraph(loaded, graph) },
                preparePlan = { loaded, plan -> diagnosticPlan(loaded, plan) },
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

  /**
   * The engine of a gate or timing run: tokenizer and head of [backend] at this launch's precision,
   * no fallback, with the launch's Qualcomm options [npu] (null without the NPU libraries) and pair
   * handoff ([stateCopy]); the run compiles its graphs with [diagnosticGraph] and [diagnosticPlan].
   */
  private fun diagnosticEngine(
    graph: KevGraphKey,
    backend: KevDecider.Backend,
    npu: KevNpuOptions?,
    stateCopy: Boolean,
  ): KevEngine {
    val missing = KevFiles.missing(context.filesDir, graph)
    check(missing.isEmpty()) { "missing ${missing.joinToString(" ")}" }
    engine?.close()
    engine = null
    return KevEngine.load(
        context,
        backend,
        precision,
        pairShare,
        cpuFallback = false,
        npuOptions = npu,
        pairStateCopy = stateCopy,
      ) { stage ->
        setLoading(stage, null, false)
      }
      .also { engine = it }
  }

  /**
   * The one [graph] of a gate or timing run resident (`ENGINE_READY` after the initial compile).
   */
  private fun diagnosticGraph(loaded: KevEngine, graph: KevGraphKey): KevRequestGraphs {
    setLoading(LoadStage.GRAPH, graph, false)
    val graphs =
      when (graph) {
        is KevGraphKey.Window -> loaded.prepareWindow(graph.window)
        is KevGraphKey.Pair -> loaded.preparePair(graph.shape)
      }
    graphs.compiles.forEach { KevDemo.compiled(it) }
    KevDemo.windows(graphs, loaded.resident)
    diagnosticReady(loaded)
    return graphs
  }

  /** The graphs of [plan] resident (the timing request path). */
  private fun diagnosticPlan(loaded: KevEngine, plan: KevPlan.Ready): KevRequestGraphs {
    val graphs = prepareGraphs(loaded, plan, switching = false)
    diagnosticReady(loaded)
    return graphs
  }

  /** Ready after a diagnostic compile; `ENGINE_READY` after the initial one. */
  private fun diagnosticReady(loaded: KevEngine) {
    val initial = loaded.loadMs == null
    loaded.markLoaded()
    setReady(loaded)
    if (initial) KevDemo.engineReady(KevAnswerView.wholeMillis(loaded.loadMs ?: 0.0))
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

  private fun setLoading(
    stage: LoadStage,
    graph: KevGraphKey?,
    switching: Boolean,
    step: KevCompileStep? = null,
  ) {
    mutableState.update {
      it.copy(
        status =
          KevStatus.Loading(
            stage,
            graph,
            switching,
            SystemClock.elapsedRealtime(),
            backend = step?.backend,
            npuFirst = step?.npuFirst == true,
          )
      )
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
      loaded.residentBackends,
      loaded.resident,
      KevAnswerView.wholeMillis(loaded.loadMs ?: 0.0),
      KevAnswerView.wholeMillis(loaded.compileMs),
    )

  private fun engineUi(loaded: KevEngine) =
    EngineUi(
      loaded.requestedBackend,
      loaded.forcedPrecision,
      loaded.resident,
      loaded.residentPrecisions,
      loaded.residentBackends,
      KevEngineTimes.of(loaded.tokenizerMs, loaded.headMs, loaded.residentCompileMs),
      loaded.gpuFailure,
      loaded.npuFailure,
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
   * them: device and Android version; LiteRT, where the graphs ran and their precision; the graphs
   * the questions ran on and the request total.
   */
  private fun footerLines(
    loaded: KevEngine,
    graphs: KevRequestGraphs,
    totalMs: Long,
  ): List<String> =
    listOf(
      string(R.string.footer_device, KevDevice.displayName(), Build.VERSION.RELEASE),
      string(R.string.footer_litert, KevDecider.LITERT_VERSION, backendName(graphs)),
      string(R.string.footer_graph_total, graphsText(graphs), totalMs),
    )

  /**
   * The presentation footer: three short lines that fit a 360 dp wide screen (device name, LiteRT,
   * where the graphs ran and their precision; the graphs of the questions; the request total, empty
   * until the end). Model code and Android version are in the run JSON only.
   */
  private fun presentationFooter(
    loaded: KevEngine,
    graphs: KevRequestGraphs,
    totalMs: Long?,
  ): List<String> =
    listOf(
      string(
        R.string.footer_runtime,
        KevDevice.marketName(),
        KevDecider.LITERT_VERSION,
        backendName(graphs),
      ),
      string(R.string.footer_graph, graphsText(graphs)),
      if (totalMs == null) "" else string(R.string.presentation_total, totalMs),
    )

  /**
   * "L256 + L512" or "S128+Q64": the graphs of the request by their names on screen; graphs on
   * different backends each with its own ("L128 NPU + L512 GPU"), GPU graphs at different
   * precisions each with its own ("L128 FP16 (FP32 accum) + L256 FP32").
   */
  private fun graphsText(graphs: KevRequestGraphs): String {
    val eachBackend = graphs.backends.distinct().size > 1
    val gpu = gpuPrecisions(graphs)
    val eachPrecision = gpu.isNotEmpty() && KevPrecision.common(gpu, null) == null
    return graphs.used
      .mapIndexed { index, graph ->
        val backend = graphs.backends[index]
        listOfNotNull(
            when (graph) {
              is KevGraphKey.Window -> string(R.string.window_name, graph.window)
              is KevGraphKey.Pair ->
                string(R.string.pair_name, graph.shape.stateLength, graph.shape.questionLength)
            },
            shortBackendName(backend).takeIf { eachBackend },
            precisionName(graphs.precisions[index]).takeIf {
              eachPrecision && backend == KevDecider.Backend.GPU
            },
          )
          .joinToString(" ")
      }
      .joinToString(WINDOW_SEPARATOR)
  }

  /** The GPU precisions of the graphs of [graphs] that ran on the GPU, in the order of `used`. */
  private fun gpuPrecisions(graphs: KevRequestGraphs): List<KevPrecision> =
    graphs.precisions.filterIndexed { index, _ ->
      graphs.backends[index] == KevDecider.Backend.GPU
    }

  /**
   * The run JSON's `runtime.precision`: the GPU graphs' precision (`fp32` or `fp16acc`), `mixed`
   * when they differ, null when no graph ran on the GPU.
   */
  private fun gpuPrecisionName(graphs: KevRequestGraphs): String? {
    val gpu = gpuPrecisions(graphs)
    if (gpu.isEmpty()) return null
    return KevPrecision.common(gpu, null)?.wireName ?: MIXED_PRECISION
  }

  /**
   * Where the graphs of [graphs] ran ([backendName] of each backend, with the GPU graphs' common
   * precision), joined by " + " when they ran on different ones.
   */
  private fun backendName(graphs: KevRequestGraphs): String =
    graphs.backends.distinct().joinToString(WINDOW_SEPARATOR) { backend ->
      backendName(
        backend,
        if (backend == KevDecider.Backend.GPU) KevPrecision.common(gpuPrecisions(graphs), null)
        else null,
      )
    }

  /**
   * "GPU FP32", "GPU FP16 (FP32 accum)", "GPU" (graphs at different precisions), "NPU" or "CPU 4
   * threads".
   */
  private fun backendName(backend: KevDecider.Backend, precision: KevPrecision?): String =
    string(
      when {
        backend == KevDecider.Backend.CPU -> R.string.backend_cpu
        backend == KevDecider.Backend.NPU -> R.string.backend_npu
        precision == KevPrecision.FP16_FP32_ACCUM -> R.string.backend_gpu_fp16acc
        precision == KevPrecision.FP32 -> R.string.backend_gpu
        else -> R.string.backend_gpu_any
      }
    )

  /** "GPU", "NPU" or "CPU". */
  private fun shortBackendName(backend: KevDecider.Backend): String =
    string(
      when (backend) {
        KevDecider.Backend.GPU -> R.string.run_on_gpu
        KevDecider.Backend.NPU -> R.string.run_on_npu
        KevDecider.Backend.CPU -> R.string.run_on_cpu
      }
    )

  /** "FP32" or "FP16 (FP32 accum)". */
  private fun precisionName(precision: KevPrecision): String =
    string(
      when (precision) {
        KevPrecision.FP32 -> R.string.precision_fp32
        KevPrecision.FP16_FP32_ACCUM -> R.string.precision_fp16acc
      }
    )

  /**
   * The run JSON's `accelerators`: the backend chosen, whether the APK carries the NPU libraries,
   * where each graph of the request ran, the resident graphs, this request's compiles and the NPU
   * compile record of each resident NPU graph (cache state, log lines, cache files).
   */
  private fun acceleratorsJson(
    loaded: KevEngine,
    graphs: KevRequestGraphs,
  ): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "requested" to loaded.requestedBackend.wireName,
      "npu_libraries" to npuAvailable,
      "graphs" to
        graphs.used.mapIndexed { index, graph ->
          linkedMapOf("graph" to graph.label, "ran_on" to graphs.backends[index].wireName)
        },
      "resident" to
        loaded.resident.zip(loaded.residentBackends) { graph, backend ->
          linkedMapOf("graph" to graph.label, "ran_on" to backend.wireName)
        },
      "compiles" to graphs.compiles.map { it.toJson() },
      "resident_npu" to
        loaded.residentNpu.map { (graph, npu) ->
          linkedMapOf<String, Any?>("graph" to graph.label).apply { putAll(npu.toJson()) }
        },
    )

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
    private const val WINDOW_SEPARATOR = " + "

    /** The run JSON's `runtime.precision` when the questions ran on graphs at different ones. */
    private const val MIXED_PRECISION = "mixed"

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
