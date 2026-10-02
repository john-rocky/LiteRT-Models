package com.gliclass

import android.content.Context
import android.os.Process
import android.os.SystemClock
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
 * Owns the [GliclassClassifier] and runs every model call on one confined dispatcher, because
 * LiteRT reuses its native input and output buffers. Publishes [UiState] for [MainActivity].
 */
class MainViewModel(private val context: Context) : ViewModel() {
  // Every helper call, including initialization, gate work and close, is confined here.
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private var classifier: GliclassClassifier? = null
  private var started = false
  private var pendingFirstTap: GliclassDiagnostics.Startup? = null
  @Volatile private var cleared = false
  private val mutableUiState =
    MutableStateFlow(
      UiState(
        context.getString(R.string.example_text),
        context.getString(R.string.example_labels),
      )
    )

  /** Immutable screen state, updated after every model event. */
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  /**
   * Starts once per ViewModel: loads the GPU backend and warms it up for interactive use. The debug
   * build can instead run the fixture [gate] on [backend] into [report]; the debug and benchmark
   * builds can record the [firstTap] classification after startup, or classify the bundled example
   * [pacedCount] times, one every [pacedIntervalMs]. [createdAt] = Activity creation
   * (`SystemClock.elapsedRealtime`).
   */
  fun start(
    gate: Boolean,
    backend: String?,
    report: String?,
    firstTap: Boolean,
    pacedCount: Int,
    pacedIntervalMs: Long,
    createdAt: Long,
  ) {
    if (started || cleared) {
      return
    }
    started = true
    val requested = GliclassClassifier.Backend.valueOf((backend ?: "gpu").uppercase(Locale.ROOT))
    when {
      gate && BuildConfig.DEBUG -> runGate(backend, report)
      pacedCount > 0 && GliclassDiagnostics.enabled ->
        runPaced(requested, pacedCount, pacedIntervalMs, createdAt)
      else ->
        loadAccelerator(requested, if (firstTap && GliclassDiagnostics.enabled) createdAt else null)
    }
  }

  /** Replaces the text to classify unless a request is running. */
  fun setInputText(text: String) = edit { it.copy(inputText = text) }

  /** Replaces the label editor text unless a request is running. */
  fun setLabelsText(text: String) = edit { it.copy(labelsText = text) }

  /** Replaces the optional prompt unless a request is running. */
  fun setPromptText(text: String) = edit { it.copy(promptText = text) }

  /** Switches between single-label and multi-label decisions. */
  fun setMode(mode: GliclassDecoder.Mode) = edit { it.copy(mode = mode) }

  /** Sets the multi-label threshold. */
  fun setThreshold(threshold: Float) = edit { it.copy(threshold = threshold) }

  private fun edit(change: (UiState) -> UiState) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update { change(it).copy(result = null, errorMessage = null) }
    }
  }

  /** Switches to [backend]; the new graph is compiled and warmed up before Ready. */
  fun selectAccelerator(backend: GliclassClassifier.Backend) {
    if (
      uiState.value.busy ||
        uiState.value.gateMode ||
        cleared ||
        backend == uiState.value.accelerator
    ) {
      return
    }
    loadAccelerator(backend, null)
  }

  private fun runGate(backend: String?, report: String?) {
    mutableUiState.update {
      it.copy(gateMode = true, busy = true, statusMessage = R.string.status_gate_running)
    }
    modelScope.launch {
      try {
        val summary = GliclassGateRunner(context, ::helper).run(backend, report)
        mutableUiState.update { state ->
          state.copy(
            busy = false,
            statusMessage =
              if (summary.passed) R.string.status_gate_passed else R.string.status_gate_failed,
            gateFiles = listOf(summary.path),
            errorMessage = summary.error,
          )
        }
      } catch (failure: Exception) {
        showFailure(failure)
        Log.e(GliclassGateFixtures.GATE_LOG_TAG, "Gate setup failed: ${failure.message}")
      } catch (failure: LinkageError) {
        showFailure(failure)
        Log.e(GliclassGateFixtures.GATE_LOG_TAG, "Native runtime failed: ${failure.message}")
      }
    }
  }

  /** Compiles s128 on [backend] and warms it up; [firstTapCreatedAt] also records the first tap. */
  private fun loadAccelerator(backend: GliclassClassifier.Backend, firstTapCreatedAt: Long?) {
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
        val initializeStart = System.nanoTime()
        helper().initialize(backend)
        val initializeMs = (System.nanoTime() - initializeStart) / NANOS_PER_MILLI
        mutableUiState.update { it.copy(statusMessage = R.string.status_warming_up) }
        val warmUpMs =
          helper()
            .warmUpForInteraction(
              context.getString(R.string.example_text),
              LabelEditor.parse(context.getString(R.string.example_labels)),
              null,
              backend,
            )
        mutableUiState.update { it.copy(busy = false, statusMessage = readyStatus(backend)) }
        if (firstTapCreatedAt != null) {
          pendingFirstTap = startup(firstTapCreatedAt, initializeMs, warmUpMs)
          classify()
        }
      } catch (failure: Exception) {
        if (firstTapCreatedAt != null) {
          GliclassDiagnostics.failure(context, "firsttap", failure)
        }
        showFailure(failure)
      } catch (failure: LinkageError) {
        if (firstTapCreatedAt != null) {
          GliclassDiagnostics.failure(context, "firsttap", failure)
        }
        showFailure(failure)
      }
    }
  }

  /**
   * Classifies the current text against the labels parsed from the editor. Rejected inputs (no
   * labels, more than 25 labels, too many tokens) are shown as an error while the model stays
   * ready.
   */
  fun classify() {
    val state = uiState.value
    if (state.busy || state.gateMode || cleared || state.inputText.isBlank()) {
      return
    }
    val labels = LabelEditor.parse(state.labelsText)
    if (labels.isEmpty()) {
      mutableUiState.update {
        it.copy(errorMessage = context.getString(R.string.error_no_labels), result = null)
      }
      return
    }
    val prompt = state.promptText.takeIf { it.isNotEmpty() }
    val startup = pendingFirstTap
    pendingFirstTap = null
    mutableUiState.update {
      it.copy(busy = true, errorMessage = null, statusMessage = R.string.status_classifying)
    }
    modelScope.launch {
      try {
        val begun = System.nanoTime()
        val output =
          helper()
            .classify(
              state.inputText,
              labels,
              prompt,
              state.mode,
              state.threshold.toDouble(),
              state.accelerator,
            )
        val endToEndMs = (System.nanoTime() - begun) / NANOS_PER_MILLI
        mutableUiState.update {
          it.copy(
            busy = false,
            result = ClassificationUiResult.from(output),
            statusMessage = readyStatus(output.backend),
          )
        }
        if (startup != null) {
          GliclassDiagnostics.firstTap(context, startup, endToEndMs, output)
        }
      } catch (failure: IllegalArgumentException) {
        // Rejected input: the model stays ready.
        if (startup != null) {
          GliclassDiagnostics.failure(context, "firsttap", failure)
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
        if (startup != null) {
          GliclassDiagnostics.failure(context, "firsttap", failure)
        }
        showFailure(failure)
      } catch (failure: LinkageError) {
        if (startup != null) {
          GliclassDiagnostics.failure(context, "firsttap", failure)
        }
        showFailure(failure)
      }
    }
  }

  /**
   * Diagnostics: the normal startup (compile, warm-up), then [count] classifications of the bundled
   * example, one every [intervalMs] after Ready, each timed end to end on the model dispatcher.
   */
  private fun runPaced(
    backend: GliclassClassifier.Backend,
    count: Int,
    intervalMs: Long,
    createdAt: Long,
  ) {
    mutableUiState.update {
      it.copy(
        accelerator = backend,
        gateMode = true,
        busy = true,
        statusMessage = R.string.status_paced_running,
      )
    }
    modelScope.launch {
      val report = GliclassDiagnostics.Paced(context, backend, count, intervalMs)
      try {
        val text = context.getString(R.string.example_text)
        val labels = LabelEditor.parse(context.getString(R.string.example_labels))
        val initializeStart = System.nanoTime()
        helper().initialize(backend)
        val initializeMs = (System.nanoTime() - initializeStart) / NANOS_PER_MILLI
        val warmUpMs = helper().warmUpForInteraction(text, labels, null, backend)
        report.startup(startup(createdAt, initializeMs, warmUpMs))
        val ready = System.nanoTime()
        repeat(count) { index ->
          val wait =
            ready + (index + 1) * TimeUnit.MILLISECONDS.toNanos(intervalMs) - System.nanoTime()
          if (wait > 0) {
            delay(TimeUnit.NANOSECONDS.toMillis(wait))
          }
          val begun = System.nanoTime()
          val output =
            helper()
              .classify(
                text,
                labels,
                null,
                GliclassDecoder.Mode.SINGLE_LABEL,
                GliclassDecoder.DEFAULT_THRESHOLD,
                backend,
              )
          val endToEndMs = (System.nanoTime() - begun) / NANOS_PER_MILLI
          report.request(index, (begun - ready) / NANOS_PER_MILLI, endToEndMs, output)
          mutableUiState.update { it.copy(result = ClassificationUiResult.from(output)) }
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

  private fun startup(createdAt: Long, initializeMs: Double, warmUpMs: Double) =
    SystemClock.elapsedRealtime().let { now ->
      GliclassDiagnostics.Startup(
        (now - Process.getStartElapsedRealtime()).toDouble(),
        (now - createdAt).toDouble(),
        initializeMs,
        warmUpMs,
      )
    }

  private fun helper(): GliclassClassifier =
    classifier ?: GliclassClassifier(context).also { classifier = it }

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
    /** Requests of a paced run when the launch intent does not set a count. */
    const val DEFAULT_PACED_COUNT = 20

    /** Milliseconds between paced requests when the launch intent does not set them. */
    const val DEFAULT_PACED_INTERVAL_MS = 2000

    private const val LOG_TAG = "GLICLASS"
    private const val NANOS_PER_MILLI = 1_000_000.0

    private fun readyStatus(backend: GliclassClassifier.Backend) =
      when (backend) {
        GliclassClassifier.Backend.GPU -> R.string.status_gpu_ready
        GliclassClassifier.Backend.CPU -> R.string.status_cpu_ready
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
