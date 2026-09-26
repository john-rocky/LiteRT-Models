package com.gliner25decide

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Owns the [DecideClassifier] and runs every model call on one confined dispatcher, because LiteRT
 * reuses its native input and output buffers. Publishes [UiState] for [MainActivity].
 */
class MainViewModel(private val context: Context) : ViewModel() {
  // Every helper call, including initialization, gate work and close, is confined here.
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private var classifier: DecideClassifier? = null
  private var started = false
  private var firstTapWarmUpMs: Double? = null
  @Volatile private var cleared = false
  private val mutableUiState =
    MutableStateFlow(
      UiState(context.getString(R.string.example_text), context.getString(R.string.example_tasks))
    )
  /** Immutable screen state, updated after every model event. */
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  /**
   * Starts once per ViewModel: loads the GPU backend and warms it up for interactive use. The debug
   * build can instead run the fixture [gate], record the [firstTap] classification after startup,
   * or classify the bundled example [pacedCount] times, one every [pacedIntervalMs].
   */
  fun start(
    gate: Boolean,
    requestedAccelerator: String?,
    firstTap: Boolean = false,
    pacedCount: Int = 0,
    pacedIntervalMs: Long = DEFAULT_PACED_INTERVAL_MS.toLong(),
  ) {
    if (started || cleared) {
      return
    }
    started = true
    if (pacedCount > 0 && !gate && BuildConfig.DEBUG) {
      runPaced(
        DecideClassifier.Backend.valueOf((requestedAccelerator ?: "GPU").uppercase(Locale.ROOT)),
        pacedCount,
        pacedIntervalMs,
      )
    } else if (gate) {
      mutableUiState.update {
        it.copy(gateMode = true, busy = true, statusMessage = R.string.status_gate_running)
      }
      modelScope.launch {
        try {
          val reports = DecideGateRunner(context, ::helper).run(requestedAccelerator)
          mutableUiState.update { state ->
            state.copy(
              busy = false,
              statusMessage =
                if (reports.all { it.passed }) {
                  R.string.status_gate_passed
                } else {
                  R.string.status_gate_failed
                },
              gateFiles = reports.map { it.path },
              errorMessage =
                reports.mapNotNull { it.error }.takeIf { it.isNotEmpty() }?.joinToString("\n"),
            )
          }
        } catch (failure: Exception) {
          showFailure(failure)
          Log.e(DecideGateFixtures.GATE_LOG_TAG, "Gate setup failed: ${failure.message}")
        } catch (failure: LinkageError) {
          showFailure(failure)
          Log.e(DecideGateFixtures.GATE_LOG_TAG, "Native runtime failed: ${failure.message}")
        }
      }
    } else {
      loadAccelerator(DecideClassifier.Backend.GPU, firstTap && BuildConfig.DEBUG)
    }
  }

  /** Replaces the text to classify unless a request is running. */
  fun setInputText(text: String) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update { it.copy(inputText = text, result = null, errorMessage = null) }
    }
  }

  /** Replaces the task editor text unless a request is running. */
  fun setTasksText(text: String) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update { it.copy(tasksText = text, result = null, errorMessage = null) }
    }
  }

  /** Switches to [backend]; the new graph is compiled and warmed up before Ready. */
  fun selectAccelerator(backend: DecideClassifier.Backend) {
    if (
      uiState.value.busy ||
        uiState.value.gateMode ||
        cleared ||
        backend == uiState.value.accelerator
    ) {
      return
    }
    loadAccelerator(backend)
  }

  private fun loadAccelerator(backend: DecideClassifier.Backend, recordFirstTap: Boolean = false) {
    mutableUiState.update {
      it.copy(
        accelerator = backend,
        busy = true,
        errorMessage = null,
        result = null,
        statusMessage = R.string.status_loading,
      )
    }
    modelScope.launch {
      try {
        if (recordFirstTap) {
          DecideFirstTapReport.begin(context)
        }
        helper().initialize(backend)
        mutableUiState.update { it.copy(statusMessage = R.string.status_warming_up) }
        val warmUpMs =
          helper()
            .warmUpForInteraction(
              context.getString(R.string.example_text),
              DecideSchema.parseTaskLines(context.getString(R.string.example_tasks)),
              backend,
            )
        mutableUiState.update { it.copy(busy = false, statusMessage = readyStatus(backend)) }
        if (recordFirstTap) {
          firstTapWarmUpMs = warmUpMs
          classify()
        }
      } catch (failure: Exception) {
        if (recordFirstTap) {
          DecideFirstTapReport.failure(context, null, failure)
        }
        showFailure(failure)
      } catch (failure: LinkageError) {
        if (recordFirstTap) {
          DecideFirstTapReport.failure(context, null, failure)
        }
        showFailure(failure)
      }
    }
  }

  /**
   * Classifies the current text with the tasks parsed from the editor. Invalid task lines and
   * rejected inputs are shown as an error while the model stays ready.
   */
  fun classify() {
    val state = uiState.value
    if (state.busy || state.gateMode || cleared || state.inputText.isBlank()) {
      return
    }
    val tasks =
      try {
        DecideSchema.parseTaskLines(state.tasksText).also {
          DecideSchema.validate(it, DecideInputs.LABEL_SLOTS)
        }
      } catch (failure: IllegalArgumentException) {
        mutableUiState.update { it.copy(errorMessage = failure.message, result = null) }
        return
      }
    val startupWarmUpMs = firstTapWarmUpMs
    firstTapWarmUpMs = null
    mutableUiState.update {
      it.copy(busy = true, errorMessage = null, statusMessage = R.string.status_classifying)
    }
    modelScope.launch {
      try {
        val output = helper().classify(state.inputText, tasks, state.accelerator)
        val result =
          ClassificationUiResult(
            output.decisions.map {
              DecisionUiRow(it.task, it.multiLabel, it.labels, it.chosenProbabilities)
            },
            output.backend,
            output.window,
            output.encodedTokens,
            output.labelCount,
            output.timing.tokenizeEmbedMs,
            output.timing.graphMs,
            output.timing.decodeMs,
          )
        mutableUiState.update {
          it.copy(busy = false, result = result, statusMessage = readyStatus(output.backend))
        }
        if (startupWarmUpMs != null) {
          DecideFirstTapReport.complete(context, startupWarmUpMs, output)
        }
      } catch (failure: IllegalArgumentException) {
        // Rejected input (too long, too many labels): the model stays ready.
        if (startupWarmUpMs != null) {
          DecideFirstTapReport.failure(context, startupWarmUpMs, failure)
        }
        if (classifier == null) {
          showFailure(failure)
        } else {
          mutableUiState.update {
            it.copy(
              busy = false,
              errorMessage = failure.message,
              statusMessage = readyStatus(state.accelerator),
            )
          }
        }
      } catch (failure: Exception) {
        if (startupWarmUpMs != null) {
          DecideFirstTapReport.failure(context, startupWarmUpMs, failure)
        }
        showFailure(failure)
      } catch (failure: LinkageError) {
        if (startupWarmUpMs != null) {
          DecideFirstTapReport.failure(context, startupWarmUpMs, failure)
        }
        showFailure(failure)
      }
    }
  }

  /**
   * Debug only: the normal startup (compile, warm-up), then [count] classifications of the bundled
   * example, one every [intervalMs] after Ready, each timed end to end on the model dispatcher.
   */
  private fun runPaced(backend: DecideClassifier.Backend, count: Int, intervalMs: Long) {
    mutableUiState.update {
      it.copy(
        accelerator = backend,
        gateMode = true,
        busy = true,
        statusMessage = R.string.status_paced_running,
      )
    }
    modelScope.launch {
      val report = DecidePacedReport(context, backend, count, intervalMs)
      try {
        report.begin()
        val text = context.getString(R.string.example_text)
        val tasks = DecideSchema.parseTaskLines(context.getString(R.string.example_tasks))
        val initializeStart = System.nanoTime()
        helper().initialize(backend)
        val initializeMs = (System.nanoTime() - initializeStart) / NANOS_PER_MILLI
        report.startup(initializeMs, helper().warmUpForInteraction(text, tasks, backend))
        val ready = System.nanoTime()
        repeat(count) { index ->
          val wait =
            ready + (index + 1) * TimeUnit.MILLISECONDS.toNanos(intervalMs) - System.nanoTime()
          if (wait > 0) {
            delay(TimeUnit.NANOSECONDS.toMillis(wait))
          }
          val begun = System.nanoTime()
          val output = helper().classify(text, tasks, backend)
          val endToEndMs = (System.nanoTime() - begun) / NANOS_PER_MILLI
          report.request(index, (begun - ready) / NANOS_PER_MILLI, endToEndMs, output)
          mutableUiState.update {
            it.copy(
              result =
                ClassificationUiResult(
                  output.decisions.map { decision ->
                    DecisionUiRow(
                      decision.task,
                      decision.multiLabel,
                      decision.labels,
                      decision.chosenProbabilities,
                    )
                  },
                  output.backend,
                  output.window,
                  output.encodedTokens,
                  output.labelCount,
                  output.timing.tokenizeEmbedMs,
                  output.timing.graphMs,
                  output.timing.decodeMs,
                )
            )
          }
        }
        report.complete()
        mutableUiState.update { it.copy(busy = false, statusMessage = R.string.status_paced_done) }
      } catch (failure: Exception) {
        report.failure(failure)
        showFailure(failure)
      } catch (failure: LinkageError) {
        report.failure(failure)
        showFailure(failure)
      }
    }
  }

  private fun helper(): DecideClassifier =
    classifier ?: DecideClassifier(context).also { classifier = it }

  private fun showFailure(failure: Throwable) {
    mutableUiState.update {
      it.copy(
        busy = false,
        errorMessage = failure.message ?: failure.javaClass.simpleName,
        statusMessage = R.string.status_error,
      )
    }
  }

  override fun onCleared() {
    cleared = true
    // Queue cleanup after any in-flight blocking native call on the same serial dispatcher.
    modelScope.launch {
      try {
        classifier?.close()
        classifier = null
      } catch (failure: Exception) {
        Log.e(LOG_TAG, "Classifier cleanup failed", failure)
      } finally {
        modelScope.cancel()
      }
    }
    super.onCleared()
  }

  companion object {
    /** Requests of a debug paced run when the launch intent does not set a count. */
    const val DEFAULT_PACED_COUNT = 20

    /** Milliseconds between debug paced requests when the launch intent does not set them. */
    const val DEFAULT_PACED_INTERVAL_MS = 2000

    private const val LOG_TAG = "DECIDE"
    private const val NANOS_PER_MILLI = 1_000_000.0

    private fun readyStatus(backend: DecideClassifier.Backend) =
      when (backend) {
        DecideClassifier.Backend.GPU -> R.string.status_gpu_ready
        DecideClassifier.Backend.CPU -> R.string.status_cpu_ready
      }

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
