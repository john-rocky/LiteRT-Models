package com.kev

import androidx.compose.runtime.Immutable

/** What the engine is doing, shown in the status line. */
sealed interface KevStatus {
  /** Files `scripts/install_to_device.sh` puts in `files/` are missing. */
  data class MissingFiles(val files: List<String>) : KevStatus

  /**
   * Loading [stage]; [graph] is the graph being compiled (null before the graph stage), [switching]
   * = compiling it for a request the resident graphs do not take.
   */
  data class Loading(
    val stage: LoadStage,
    val graph: KevGraphKey?,
    val switching: Boolean,
    val startedAt: Long,
    val elapsedSeconds: Int = 0,
  ) : KevStatus

  /** Idle with the [resident] graphs compiled (windows ascending, then the pair). */
  data class Ready(
    val backend: KevDecider.Backend,
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
 * The engine as the status line shows it: backend and GPU precision, the resident graphs (windows
 * ascending, then the pair), the load time and the resident graphs' compile time.
 */
@Immutable
data class EngineUi(
  val backend: KevDecider.Backend,
  val precision: KevPrecision,
  val resident: List<KevGraphKey>,
  val loadMs: Long,
  val compileMs: Long,
  /** GPU's error when the app fell back to CPU. */
  val gpuFailure: String?,
)

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
  /** The GPU precision of this launch (the `precision` extra; there is no control on screen). */
  val precision: KevPrecision = KevPrecision.FP32,
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
