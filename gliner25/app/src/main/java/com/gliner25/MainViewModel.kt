package com.gliner25

import android.content.Context
import android.util.Log
import androidx.annotation.StringRes
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
import org.json.JSONArray
import org.json.JSONObject

class MainViewModel(private val context: Context) : ViewModel() {
  // Every helper call, including initialization, gate work and close, is confined here.
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private var extractor: GlinerExtractor? = null
  private var started = false
  private var profileDecoder = false
  private var firstTapWarmUpMs: Double? = null
  @Volatile private var cleared = false
  private val mutableUiState = MutableStateFlow(UiState(context.getString(R.string.example_text)))
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  /**
   * [requestedModel], [requestedSplitter] and [prefilledText] come from debug-only launch extras:
   * `small` or `multi`, `whitespace` or `char`, and a sentence placed in the editor (not run).
   */
  fun start(
    gate: Boolean,
    requestedAccelerator: String?,
    fixtureSet: String? = null,
    profile: Boolean = false,
    firstTap: Boolean = false,
    requestedModel: String? = null,
    requestedSplitter: String? = null,
    prefilledText: String? = null,
  ) {
    if (started || cleared) {
      return
    }
    started = true
    profileDecoder = gate && profile && BuildConfig.PROFILE_ALLOWED
    try {
      val model = requestedModel?.let { GlinerModel.fromId(it) } ?: GlinerModel.SMALL
      val splitter =
        requestedSplitter?.let { GlinerInputs.Splitter.fromId(it) } ?: model.defaultSplitter
      require(splitter in model.splitters) {
        "${model.id} does not support the ${splitter.id} splitter"
      }
      mutableUiState.update {
        it.copy(
          model = model,
          splitter = splitter,
          inputText = prefilledText ?: context.getString(model.exampleText(splitter)),
        )
      }
    } catch (failure: IllegalArgumentException) {
      showFailure(failure)
      return
    }
    if (gate) {
      mutableUiState.update {
        it.copy(gateMode = true, busy = true, statusMessage = R.string.status_gate_running)
      }
      val model = uiState.value.model
      modelScope.launch {
        try {
          val reports =
            when (fixtureSet) {
              null -> GlinerGateRunner(context, ::helper, model).run(requestedAccelerator)
              "f1" -> {
                require(model == GlinerModel.SMALL) { "The f1 set covers GLiNER2.5 Small only." }
                GlinerF1GateRunner(context, ::helper, profileDecoder).run(requestedAccelerator)
              }
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
      loadAccelerator(GlinerExtractor.Backend.GPU, firstTap && BuildConfig.DEBUG)
    }
  }

  fun setInputText(text: String) {
    if (!uiState.value.busy && !uiState.value.gateMode && !cleared) {
      mutableUiState.update { it.copy(inputText = text, result = null, errorMessage = null) }
    }
  }

  /**
   * Switches between GLiNER2.5 Small and Multi. The previous extractor is closed on the model
   * dispatcher before the new one opens, so only one model's graphs and table are resident.
   */
  fun selectModel(model: GlinerModel) {
    val state = uiState.value
    if (state.busy || state.gateMode || cleared || model == state.model) {
      return
    }
    mutableUiState.update {
      it.copy(
        model = model,
        splitter = model.defaultSplitter,
        inputText =
          swapUneditedExample(
            it.inputText,
            it.model.exampleText(it.splitter),
            model.exampleText(model.defaultSplitter),
          ),
        result = null,
        errorMessage = null,
      )
    }
    loadAccelerator(state.accelerator, replaceModel = true)
  }

  /** Chooses the Multi word splitter; the loaded graphs are shared by both splitters. */
  fun selectSplitter(splitter: GlinerInputs.Splitter) {
    val state = uiState.value
    if (
      state.busy ||
        state.gateMode ||
        cleared ||
        splitter == state.splitter ||
        splitter !in state.model.splitters
    ) {
      return
    }
    mutableUiState.update {
      it.copy(
        splitter = splitter,
        inputText =
          swapUneditedExample(
            it.inputText,
            it.model.exampleText(it.splitter),
            it.model.exampleText(splitter),
          ),
        result = null,
        errorMessage = null,
      )
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

  private fun loadAccelerator(
    backend: GlinerExtractor.Backend,
    recordFirstTap: Boolean = false,
    replaceModel: Boolean = false,
  ) {
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
        if (replaceModel) {
          extractor?.close()
          extractor = null
        }
        if (recordFirstTap) {
          GlinerFirstTapReport.begin(context)
        }
        helper().initialize(backend)
        mutableUiState.update { it.copy(statusMessage = R.string.status_warming_up) }
        val state = uiState.value
        val warmUpMs =
          helper()
            .warmUpForInteraction(
              context.getString(state.model.exampleText(state.splitter)),
              backend,
              state.splitter,
            )
        mutableUiState.update { it.copy(busy = false, statusMessage = readyStatus(backend)) }
        if (recordFirstTap) {
          firstTapWarmUpMs = warmUpMs
          extract()
        }
      } catch (failure: Exception) {
        if (recordFirstTap) {
          GlinerFirstTapReport.failure(context, null, failure)
        }
        showFailure(failure)
      } catch (failure: LinkageError) {
        if (recordFirstTap) {
          GlinerFirstTapReport.failure(context, null, failure)
        }
        showFailure(failure)
      }
    }
  }

  fun extract() {
    val state = uiState.value
    if (state.busy || state.gateMode || cleared || state.inputText.isBlank()) {
      return
    }
    val startupWarmUpMs = firstTapWarmUpMs
    firstTapWarmUpMs = null
    mutableUiState.update {
      it.copy(busy = true, errorMessage = null, statusMessage = R.string.status_extracting)
    }
    modelScope.launch {
      try {
        val output = helper().extract(state.inputText, state.accelerator, state.splitter)
        val spans =
          output.spans.map { EntityUiSpan(it.label, it.text, it.start, it.end, it.confidence) }
        // Multi's appended terminal period lies past the typed text; highlights stop at its end.
        val textLength = output.text.codePointCount(0, output.text.length)
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
            highlights
              .filter { it.start < textLength }
              .map { it.copy(end = minOf(it.end, textLength)) }
              .sortedBy { it.start },
            output.backend,
            output.window,
            output.encodedTokens,
            output.textWords,
            output.timing.tokenizeEmbedMs,
            output.timing.graphMs,
            output.timing.decodeMs,
            output.model,
            output.splitter,
          )
        mutableUiState.update {
          it.copy(busy = false, result = result, statusMessage = readyStatus(output.backend))
        }
        if (BuildConfig.DEBUG) {
          Log.i("GLINER_RESULT", resultLog(output).toString())
        }
        if (startupWarmUpMs != null) {
          GlinerFirstTapReport.complete(context, startupWarmUpMs, output)
        }
      } catch (failure: Exception) {
        if (startupWarmUpMs != null) {
          GlinerFirstTapReport.failure(context, startupWarmUpMs, failure)
        }
        showFailure(failure)
      } catch (failure: LinkageError) {
        if (startupWarmUpMs != null) {
          GlinerFirstTapReport.failure(context, startupWarmUpMs, failure)
        }
        showFailure(failure)
      }
    }
  }

  private fun helper(): GlinerExtractor =
    extractor
      ?: GlinerExtractor(context, profileDecoder, uiState.value.model).also { extractor = it }

  /** Keeps edited text; replaces the bundled sentence of the previous model or splitter. */
  private fun swapUneditedExample(
    text: String,
    @StringRes previous: Int,
    @StringRes next: Int,
  ): String =
    if (text == context.getString(previous)) {
      context.getString(next)
    } else {
      text
    }

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

    /** Debug log of one interactive extraction: the numbers and spans shown on screen. */
    private fun resultLog(output: GlinerExtractor.Result): JSONObject =
      JSONObject()
        .put("model", output.model.id)
        .put("splitter", output.splitter.id)
        .put("accelerator", output.backend.name)
        .put("precision", output.backend.precision)
        .put("window", output.window)
        .put("encoded_tokens", output.encodedTokens)
        .put("text_words", output.textWords)
        .put("tokenize_embed_ms", output.timing.tokenizeEmbedMs)
        .put("graph_ms", output.timing.graphMs)
        .put("decode_ms", output.timing.decodeMs)
        .put("text", output.text)
        .put(
          "spans",
          JSONArray(
            output.spans.map {
              JSONObject()
                .put("label", it.label)
                .put("text", it.text)
                .put("start", it.start)
                .put("end", it.end)
                .put("confidence", it.confidence.toDouble())
            }
          ),
        )

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
