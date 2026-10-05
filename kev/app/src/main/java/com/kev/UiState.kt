package com.kev

import androidx.compose.runtime.Immutable

/** What the engine is doing, shown in the status line. */
sealed interface KevStatus {
  /** Files `scripts/install_to_device.sh` puts in `files/` are missing. */
  data class MissingFiles(val files: List<String>) : KevStatus

  /**
   * Loading [stage]; [graph] is the graph being compiled (null before the graph stage) on
   * [backend], [switching] = compiling it for a request the resident graphs do not take, [npuFirst]
   * = a first NPU compile of its file (minutes), not a load from LiteRT's JIT cache.
   */
  data class Loading(
    val stage: LoadStage,
    val graph: KevGraphKey?,
    val switching: Boolean,
    val startedAt: Long,
    val elapsedSeconds: Int = 0,
    val backend: KevDecider.Backend? = null,
    val npuFirst: Boolean = false,
  ) : KevStatus

  /** Idle with the [resident] graphs compiled (windows ascending, then the pair) on [backends]. */
  data class Ready(
    val backends: List<KevDecider.Backend>,
    val resident: List<KevGraphKey>,
    val loadMs: Long,
    val compileMs: Long,
  ) : KevStatus

  /** Question [questionIndex] (0-based) of [total] is in the graph. */
  data class Running(val questionIndex: Int, val total: Int) : KevStatus

  /**
   * A request finished: [requestTotalMs] from tokenizing to the last head, [questions] answered.
   */
  data class Done(val requestTotalMs: Long, val questions: Int) : KevStatus

  data class Error(val text: String) : KevStatus
}

/**
 * The engine as the status line shows it: where the resident graphs (windows ascending, then the
 * pair) run and at which GPU precision, and their load and compile times.
 */
@Immutable
data class EngineUi(
  /** The backend chosen for the engine (the graphs it maps elsewhere run on the GPU). */
  val requested: KevDecider.Backend,
  /** The launch's precision for every graph, or null when each graph runs at its own default. */
  val forcedPrecision: KevPrecision?,
  val resident: List<KevGraphKey>,
  /** The GPU precision of each [resident] graph, in that order. */
  val precisions: List<KevPrecision>,
  /** Where each [resident] graph runs, in that order. */
  val backends: List<KevDecider.Backend>,
  /** The times of the [resident] graphs; null while no graph is compiled. */
  val times: KevEngineTimes?,
  /** GPU's error when the app fell back to CPU. */
  val gpuFailure: String?,
  /** NPU's error when the app fell back to GPU. */
  val npuFailure: String? = null,
)

/**
 * The times of the engine line: [loadMs] = tokenizer + head + the compiles of the resident graphs,
 * [compileMs] = those compiles alone. Only the graphs compiled now count, so after a switch to
 * another backend the line shows that backend's compiles, not the ones before the switch.
 */
@Immutable
data class KevEngineTimes(val loadMs: Long, val compileMs: Long) {
  companion object {
    /** The times with graphs of [residentCompileMs] compiled, or null when no graph is. */
    fun of(tokenizerMs: Double, headMs: Double, residentCompileMs: List<Double>): KevEngineTimes? {
      if (residentCompileMs.isEmpty()) return null
      val compileMs = residentCompileMs.sum()
      return KevEngineTimes(
        KevAnswerView.wholeMillis(tokenizerMs + headMs + compileMs),
        KevAnswerView.wholeMillis(compileMs),
      )
    }
  }
}

enum class CardState {
  PENDING,
  RUNNING,
  DONE,
  FAILED,
}

/** One question's answer card. [view], [msText], [form] and [window] exist once it is done. */
@Immutable
data class AnswerCardUi(
  val qid: String,
  val type: QuestionType,
  val question: String,
  val state: CardState,
  val view: KevAnswerView? = null,
  /** The card's time: input writes + `run()` + read-back, whole milliseconds. */
  val msText: String? = null,
  /** How the question ran: its own row graph, or its branch on the pair. */
  val form: KevForm? = null,
  /** The window the question ran in (L of its row graph, or the pair's Lq), next to [msText]. */
  val window: Int? = null,
  /** Where the question's graph ran, after [window]. */
  val backend: KevDecider.Backend? = null,
  val error: String? = null,
)

/**
 * The pair's state call of a request: `[state] + state tokens` ([tokens]) and its time once it ran
 * ([ms], whole milliseconds; null while it is pending).
 */
@Immutable data class StateLineUi(val tokens: Int, val ms: Long? = null)

/** The read-only demo layout shown while an autoplay intent runs. */
@Immutable
data class PresentationUi(
  val title: String,
  val ticket: String,
  val cards: List<AnswerCardUi>,
  val footerLines: List<String>,
  val failure: String? = null,
  /** The state line above the cards when the request runs on the pair. */
  val state: StateLineUi? = null,
)

/** How the app was launched: the editable sample, or a debug gate / timing run. */
enum class LaunchMode {
  INTERACTIVE,
  GATE,
  TIMING,
}

/** Everything the screen renders. */
@Immutable
data class UiState(
  val draft: RequestDraft,
  val example: Int? = 0,
  val status: KevStatus = KevStatus.Loading(LoadStage.TOKENIZER, null, false, 0L),
  val engine: EngineUi? = null,
  val backendChoice: KevDecider.Backend = KevDecider.Backend.GPU,
  /** The APK carries the NPU libraries: the NPU choice works. */
  val npuAvailable: Boolean = false,
  /**
   * The GPU precision the launch set for every graph (the `precision` extra; there is no control on
   * screen), or null when each graph runs at its own default.
   */
  val precision: KevPrecision? = null,
  /** The state line above the cards when the last request ran on the pair. */
  val stateLine: StateLineUi? = null,
  val cards: List<AnswerCardUi> = emptyList(),
  val footerLines: List<String> = emptyList(),
  val responseJson: String? = null,
  val showResponse: Boolean = false,
  val requestError: String? = null,
  val presentation: PresentationUi? = null,
  val mode: LaunchMode = LaunchMode.INTERACTIVE,
  val diagnostics: String? = null,
) {
  /** The request can be edited: the editable sample, not while a request runs. */
  val editable: Boolean
    get() = mode == LaunchMode.INTERACTIVE && presentation == null && status !is KevStatus.Running

  /** Decide and the backend choice work: an engine is loaded and idle. */
  val canDecide: Boolean
    get() = editable && engine != null && status !is KevStatus.Loading
}
