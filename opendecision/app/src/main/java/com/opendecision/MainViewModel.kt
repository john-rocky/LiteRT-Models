package com.opendecision

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import java.util.Locale
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [DecisionModel] and runs every model call on one confined dispatcher, because LiteRT reuses its native
 * input and output buffers. Publishes [UiState] for [MainActivity].
 */
class MainViewModel(private val context: Context) : ViewModel() {
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private var model: DecisionModel? = null
  private var started = false
  @Volatile private var cleared = false
  private val mutableUiState =
    MutableStateFlow(UiState(context.getString(R.string.preset_ticket_state), context.getString(R.string.preset_ticket_questions)))
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  /** Starts once per ViewModel: compiles the GPU graph and warms it up, or runs the debug fixture [gate]. */
  fun start(gate: Boolean, requestedBackend: String?) {
    if (started || cleared) {
      return
    }
    started = true
    if (gate) {
      mutableUiState.update { it.copy(gateMode = true, busy = true, statusMessage = R.string.status_gate_running) }
      modelScope.launch {
        try {
          val reports = DecisionGateRunner(context, ::helper).run(requestedBackend)
          mutableUiState.update { state ->
            state.copy(
              busy = false,
              statusMessage = if (reports.all { it.passed }) R.string.status_gate_passed else R.string.status_gate_failed,
              gateFiles = reports.map { it.path },
              errorMessage = reports.mapNotNull { it.error }.takeIf { it.isNotEmpty() }?.joinToString("\n"),
            )
          }
        } catch (failure: Exception) {
          showFailure(failure)
          Log.e(DecisionGateRunner.LOG_TAG, "Gate setup failed: ${failure.message}")
        } catch (failure: LinkageError) {
          showFailure(failure)
        }
      }
    } else {
      loadBackend(DecisionModel.Backend.valueOf((requestedBackend ?: "GPU").uppercase(Locale.ROOT)))
    }
  }

  fun setStateText(text: String) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update { it.copy(stateText = text, result = null, errorMessage = null) }
    }
  }

  fun setQuestionsText(text: String) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update { it.copy(questionsText = text, result = null, errorMessage = null) }
    }
  }

  /** Replaces the state and the questions with one of the bundled example requests. */
  fun loadPreset(preset: Preset) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update {
        it.copy(
          stateText = context.getString(preset.state),
          questionsText = context.getString(preset.questions),
          result = null,
          errorMessage = null,
        )
      }
    }
  }

  /** Switches to [backend]; the graph is compiled and warmed up before Ready. */
  fun selectBackend(backend: DecisionModel.Backend) {
    if (uiState.value.busy || uiState.value.gateMode || cleared || backend == uiState.value.backend) {
      return
    }
    loadBackend(backend)
  }

  private fun loadBackend(backend: DecisionModel.Backend) {
    mutableUiState.update {
      it.copy(backend = backend, busy = true, errorMessage = null, result = null, statusMessage = R.string.status_loading)
    }
    modelScope.launch {
      try {
        helper().initialize(backend)
        mutableUiState.update { it.copy(statusMessage = R.string.status_warming_up) }
        helper().warmUpForInteraction(
          context.getString(R.string.preset_ticket_state),
          Question.parseLines(context.getString(R.string.preset_ticket_questions)),
          backend,
        )
        mutableUiState.update { it.copy(busy = false, statusMessage = readyStatus(backend)) }
      } catch (failure: Exception) {
        showFailure(failure)
      } catch (failure: LinkageError) {
        showFailure(failure)
      }
    }
  }

  /** Answers the editor's questions about the state. Invalid lines and rejected requests are shown as an error. */
  fun decide() {
    val state = uiState.value
    if (state.busy || state.gateMode || cleared || state.stateText.isBlank()) {
      return
    }
    val questions =
      try {
        Question.parseLines(state.questionsText).also { Question.validate(it) }
      } catch (failure: IllegalArgumentException) {
        mutableUiState.update { it.copy(errorMessage = failure.message, result = null) }
        return
      }
    mutableUiState.update { it.copy(busy = true, errorMessage = null, statusMessage = R.string.status_deciding) }
    modelScope.launch {
      try {
        val output = helper().decide(state.stateText, questions, state.backend)
        val rows =
          output.answers.map { answer ->
            val q = answer.question
            val text =
              when (q.kind) {
                Question.Kind.CHOICE -> context.getString(R.string.answer_choice, answer.label, answer.confidence)
                Question.Kind.SCORE ->
                  context.getString(R.string.answer_score, answer.expectedLevel, q.options.size - 1, answer.label, answer.confidence)
                Question.Kind.NOUL -> context.getString(R.string.answer_noul, answer.yes)
              }
            AnswerUiRow(q.kind, q.instructions, text, q.options, answer.probabilities.toList(), answer.best)
          }
        mutableUiState.update {
          it.copy(
            busy = false,
            statusMessage = readyStatus(output.backend),
            result =
              DecisionUiResult(
                rows,
                output.backend,
                output.window,
                output.encodedTokens,
                output.optionCount,
                output.timing.tokenizeEmbedMs,
                output.timing.graphMs,
                output.timing.decodeMs,
              ),
          )
        }
      } catch (failure: IllegalArgumentException) {
        // Rejected request (too long, too many options): the model stays ready.
        if (model == null) {
          showFailure(failure)
        } else {
          mutableUiState.update {
            it.copy(busy = false, errorMessage = failure.message, statusMessage = readyStatus(state.backend))
          }
        }
      } catch (failure: Exception) {
        showFailure(failure)
      } catch (failure: LinkageError) {
        showFailure(failure)
      }
    }
  }

  private fun helper(): DecisionModel = model ?: DecisionModel(context).also { model = it }

  private fun showFailure(failure: Throwable) {
    mutableUiState.update {
      it.copy(busy = false, errorMessage = failure.message ?: failure.javaClass.simpleName, statusMessage = R.string.status_error)
    }
  }

  override fun onCleared() {
    cleared = true
    modelScope.launch {
      try {
        model?.close()
        model = null
      } catch (failure: Exception) {
        Log.w(TAG, "Close failed: ${failure.message}")
      } finally {
        modelScope.cancel()
      }
    }
  }

  /** The bundled example requests. */
  enum class Preset(val title: Int, val state: Int, val questions: Int) {
    TICKET(R.string.preset_ticket, R.string.preset_ticket_state, R.string.preset_ticket_questions),
    REVIEW(R.string.preset_review, R.string.preset_review_state, R.string.preset_review_questions),
    PASSAGE(R.string.preset_passage, R.string.preset_passage_state, R.string.preset_passage_questions),
  }

  companion object {
    private const val TAG = "DecisionViewModel"

    fun readyStatus(backend: DecisionModel.Backend) =
      if (backend == DecisionModel.Backend.GPU) R.string.status_gpu_ready else R.string.status_cpu_ready

    fun getFactory(context: Context): ViewModelProvider.Factory =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T = MainViewModel(context.applicationContext) as T
      }
  }
}
