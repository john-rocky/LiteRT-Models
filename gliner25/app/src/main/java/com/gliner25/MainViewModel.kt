package com.gliner25

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val context: Context) : ViewModel() {
  // Every helper call, including initialization, gate work and close, is confined here.
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private var extractor: GlinerExtractor? = null
  private var started = false
  private var profileDecoder = false
  @Volatile private var cleared = false
  private val mutableUiState = MutableStateFlow(UiState(context.getString(R.string.example_text)))
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  fun start(
    gate: Boolean,
    requestedAccelerator: String?,
    fixtureSet: String? = null,
    profile: Boolean = false,
  ) {
    if (started || cleared) {
      return
    }
    started = true
    profileDecoder = gate && profile && BuildConfig.PROFILE_ALLOWED
    if (gate) {
      mutableUiState.update {
        it.copy(gateMode = true, busy = true, statusMessage = R.string.status_gate_running)
      }
      modelScope.launch {
        try {
          val reports =
            when (fixtureSet) {
              null -> GlinerGateRunner(context, ::helper).run(requestedAccelerator)
              "f1" ->
                GlinerF1GateRunner(context, ::helper, profileDecoder).run(requestedAccelerator)
              else -> error("Unknown gate set: $fixtureSet")
            }
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
          Log.e("GLINER_GATE", "Gate setup failed: ${failure.message}")
        } catch (failure: LinkageError) {
          showFailure(failure)
          Log.e("GLINER_GATE", "Native runtime failed: ${failure.message}")
        }
      }
    } else {
      loadAccelerator(GlinerExtractor.Backend.GPU)
    }
  }

  fun setInputText(text: String) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update { it.copy(inputText = text, result = null, errorMessage = null) }
    }
  }

  fun selectAccelerator(backend: GlinerExtractor.Backend) {
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

  private fun loadAccelerator(backend: GlinerExtractor.Backend) {
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
        helper().initialize(backend)
        mutableUiState.update { it.copy(busy = false, statusMessage = readyStatus(backend)) }
      } catch (failure: Exception) {
        showFailure(failure)
      } catch (failure: LinkageError) {
        showFailure(failure)
      }
    }
  }

  fun extract() {
    val state = uiState.value
    if (state.busy || state.gateMode || cleared || state.inputText.isBlank()) {
      return
    }
    mutableUiState.update {
      it.copy(busy = true, errorMessage = null, statusMessage = R.string.status_extracting)
    }
    modelScope.launch {
      try {
        val output = helper().extract(state.inputText, state.accelerator)
        val spans =
          output.spans.map { EntityUiSpan(it.label, it.text, it.start, it.end, it.confidence) }
        // Rendering policy only: keep a span iff it does not overlap a higher-confidence span.
        val highlights = mutableListOf<EntityUiSpan>()
        spans
          .sortedWith(
            compareByDescending<EntityUiSpan> { it.confidence }
              .thenBy { it.start }
              .thenBy { it.end }
              .thenBy { it.label }
          )
          .forEach { span ->
            if (highlights.none { span.start < it.end && it.start < span.end }) {
              highlights.add(span)
            }
          }
        val result =
          ExtractionUiResult(
            output.text,
            spans,
            highlights.sortedBy { it.start },
            output.backend,
            output.window,
            output.encodedTokens,
            output.textWords,
            output.timing.tokenizeEmbedMs,
            output.timing.graphMs,
            output.timing.decodeMs,
          )
        mutableUiState.update {
          it.copy(busy = false, result = result, statusMessage = readyStatus(output.backend))
        }
      } catch (failure: Exception) {
        showFailure(failure)
      } catch (failure: LinkageError) {
        showFailure(failure)
      }
    }
  }

  private fun helper(): GlinerExtractor =
    extractor ?: GlinerExtractor(context, profileDecoder).also { extractor = it }

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
        extractor?.close()
        extractor = null
      } catch (failure: Exception) {
        Log.e("GLINER", "Extractor cleanup failed", failure)
      } finally {
        modelScope.cancel()
      }
    }
    super.onCleared()
  }

  companion object {
    private fun readyStatus(backend: GlinerExtractor.Backend) =
      when (backend) {
        GlinerExtractor.Backend.GPU -> R.string.status_gpu_ready
        GlinerExtractor.Backend.CPU -> R.string.status_cpu_ready
      }

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
