// SPDX-License-Identifier: Apache-2.0
package com.julia1

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** Owns the engine; every tokenizer, builder, model and decoder call runs on one dispatcher. */
class MainViewModel(private val context: Context) : ViewModel() {
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private val preferences = context.getSharedPreferences("julia1_preferences", Context.MODE_PRIVATE)
  private var engine: JuliaEngine? = null
  private var presets: Map<String, Any?>? = null
  private var started = false
  private var launchStartedNs = 0L
  @Volatile private var cleared = false
  private val mutableUiState =
    MutableStateFlow(UiState(inputText = context.getString(Preset.TICKET.example)))

  /** State consumed by the screen with lifecycle-aware collection. */
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  /** Starts the debug gate or the interactive pipeline once per ViewModel lifetime. */
  fun start(
    gate: Boolean,
    accelerator: String?,
    requestedWindow: String?,
    launchedAtNs: Long = System.nanoTime(),
  ) {
    if (started || cleared) {
      return
    }
    started = true
    launchStartedNs = launchedAtNs
    if (gate && BuildConfig.DEBUG) {
      mutableUiState.update {
        it.copy(gateMode = true, statusMessage = R.string.status_gate_running)
      }
      modelScope.launch {
        try {
          val backend = JuliaEngine.Backend.fromArgument(accelerator ?: "gpu")
          val window = (requestedWindow ?: JuliaEngine.DEFAULT_WINDOW.toString()).toInt()
          mutableUiState.update { it.copy(accelerator = backend) }
          val result = JuliaGateEntry.run(context, helper(), backend, window)
          mutableUiState.update {
            it.copy(
              busy = false,
              gateFile = result.path,
              errorMessage = result.error,
              statusMessage =
                if (result.status == "MEASURED") {
                  R.string.status_gate_measured
                } else {
                  R.string.status_gate_failed
                },
            )
          }
        } catch (failure: Exception) {
          showFailure(failure)
          Log.e("JULIA1_GATE", "Gate setup failed", failure)
        } catch (failure: LinkageError) {
          showFailure(failure)
          Log.e("JULIA1_GATE", "Native runtime failed", failure)
        }
      }
    } else {
      val backend =
        try {
          JuliaEngine.Backend.fromArgument(
            accelerator ?: preferences.getString("accelerator", "gpu") ?: "gpu"
          )
        } catch (failure: IllegalStateException) {
          showFailure(failure)
          return
        }
      loadAccelerator(backend)
    }
  }

  /** Edits the state text the questions are asked about. */
  fun setInputText(value: String) {
    if (!editable()) {
      return
    }
    mutableUiState.update { it.copy(inputText = value).withoutResults() }
  }

  /** Selects a bundled question set and loads its invented example text. */
  fun selectPreset(value: Preset) {
    if (!editable()) {
      return
    }
    mutableUiState.update {
      it.copy(preset = value, inputText = context.getString(value.example)).withoutResults()
    }
  }

  /** Saves an explicit backend choice; GPU failure never selects CPU automatically. */
  fun selectAccelerator(value: JuliaEngine.Backend) {
    if (!editable() || (value == uiState.value.accelerator && uiState.value.ready)) {
      return
    }
    loadAccelerator(value)
  }

  private fun editable() = !cleared && !uiState.value.busy && !uiState.value.gateMode

  private fun loadAccelerator(backend: JuliaEngine.Backend) {
    val initializationStartedNs = System.nanoTime()
    preferences.edit().putString("accelerator", backend.name.lowercase()).apply()
    mutableUiState.update {
      it
        .withoutResults()
        .copy(
          accelerator = backend,
          busy = true,
          ready = false,
          errorMessage = null,
          canFallbackToCpu = false,
          statusMessage = R.string.status_loading_tokenizer,
        )
    }
    modelScope.launch {
      var compiling = false
      try {
        val missing = JuliaEngine.REQUIRED_FILES.firstOrNull { !File(context.filesDir, it).isFile }
        if (missing != null) {
          mutableUiState.update {
            it.copy(
              busy = false,
              statusMessage = R.string.status_not_installed,
              errorMessage = context.getString(R.string.missing_model_file, missing),
            )
          }
          return@launch
        }
        val helper = helper()
        mutableUiState.update {
          it.copy(
            statusMessage =
              when (backend) {
                JuliaEngine.Backend.GPU -> R.string.status_compiling_gpu
                JuliaEngine.Backend.CPU -> R.string.status_loading_cpu
              }
          )
        }
        compiling = true
        val compileStartedNs = System.nanoTime()
        helper.initialize(backend)
        val compileMs = milliseconds(System.nanoTime() - compileStartedNs)
        compiling = false
        mutableUiState.update { it.copy(statusMessage = R.string.status_warming_up) }
        val warmStartedNs = System.nanoTime()
        val state = uiState.value
        // The warm-up runs the whole preset: tokenizer, builder, table lookup, graph and decoder.
        questions(state.preset).forEach { (_, question) ->
          helper.answer(
            state.inputText,
            JuliaQuestion.fromMap(JuliaJson.asObject(question)),
            backend,
          )
        }
        val readyAtNs = System.nanoTime()
        val warmupMs = milliseconds(readyAtNs - warmStartedNs)
        val launchToReadyMs =
          uiState.value.launchToReadyMs ?: milliseconds(readyAtNs - launchStartedNs)
        mutableUiState.update {
          it.copy(
            busy = false,
            ready = true,
            launchToReadyMs = launchToReadyMs,
            statusMessage = readyStatus(backend),
          )
        }
        Log.i(
          "JULIA1_READY",
          "accelerator=${backend.name.lowercase()} launch_to_ready_ms=$launchToReadyMs " +
            "initialization_ms=${milliseconds(readyAtNs - initializationStartedNs)} " +
            "tokenizer_load_ms=${helper.tokenizerLoadMs} " +
            "embedding_map_ms=${helper.embeddingLoadMs} compile_ms=$compileMs warmup_ms=$warmupMs",
        )
      } catch (failure: Exception) {
        showFailure(failure, canFallbackToCpu = backend != JuliaEngine.Backend.CPU && compiling)
      } catch (failure: LinkageError) {
        showFailure(failure, canFallbackToCpu = backend != JuliaEngine.Backend.CPU && compiling)
      }
    }
  }

  /**
   * Runs the current preset and reports complete host-plus-model time, excluding initialization.
   */
  fun run() {
    val state = uiState.value
    if (!editable() || !state.ready) {
      return
    }
    val runStartedNs = System.nanoTime()
    mutableUiState.update {
      it
        .withoutResults()
        .copy(busy = true, errorMessage = null, statusMessage = R.string.status_running)
    }
    modelScope.launch {
      try {
        val results = mutableListOf<AnswerUiRow>()
        questions(state.preset).forEach { (id, question) ->
          val parsed = JuliaQuestion.fromMap(JuliaJson.asObject(question))
          val output = helper().answer(state.inputText, parsed, state.accelerator)
          results += presentAnswer(id, output)
          mutableUiState.update { it.copy(answers = results.toList()) }
        }
        val runTotalMs = milliseconds(System.nanoTime() - runStartedNs)
        val tokenCount = results.sumOf { it.tokenCount }
        mutableUiState.update {
          it.copy(
            busy = false,
            runTotalMs = runTotalMs,
            runTokenCount = tokenCount,
            statusMessage = readyStatus(state.accelerator),
          )
        }
        Log.i(
          "JULIA1_TIMING",
          "accelerator=${state.accelerator.name.lowercase()} total_ms=$runTotalMs " +
            "question_count=${results.size} token_count=$tokenCount",
        )
      } catch (failure: Exception) {
        showFailure(failure)
      } catch (failure: LinkageError) {
        showFailure(failure)
      }
    }
  }

  private fun presentAnswer(id: String, output: JuliaEngine.AnswerResult): AnswerUiRow {
    val answer = output.answer
    val question = answer.question
    val probabilities =
      when (question.type) {
        "choice" ->
          question.keys.mapIndexed { index, key ->
            ProbabilityUiRow(
              probability = answer.probabilities[index],
              label = key,
              description = question.options[index],
            )
          }
        "score" ->
          question.keys.mapIndexed { index, _ ->
            ProbabilityUiRow(
              probability = answer.probabilities[index],
              scoreLevel = index,
              description = question.options[index],
            )
          }
        else ->
          listOf(
            ProbabilityUiRow(
              probability = answer.probabilities[0],
              labelResource = R.string.option_false,
              description = question.options[0].takeIf { question.criteria != null },
            ),
            ProbabilityUiRow(
              probability = answer.probabilities[1],
              labelResource = R.string.option_true,
              description = question.options[1].takeIf { question.criteria != null },
            ),
          )
      }
    return AnswerUiRow(
      questionId = id,
      instructions = question.instructions,
      type = question.type,
      choice = answer.choice,
      score = answer.score,
      trueProbability = answer.noul,
      probabilities = probabilities,
      totalMs = output.totalMs,
      tokenCount = output.sequence.ids.size,
      window = output.raw.window,
    )
  }

  private fun questions(preset: Preset): Map<String, Any?> {
    val all =
      presets
        ?: context.assets
          .open("presets.json")
          .bufferedReader()
          .use { JuliaJson.asObject(JuliaJson.parse(it.readText())) }
          .also { presets = it }
    return JuliaJson.asObject(all.getValue(preset.assetKey))
  }

  private fun helper(): JuliaEngine = engine ?: JuliaEngine(context).also { engine = it }

  private fun showFailure(failure: Throwable, canFallbackToCpu: Boolean = false) {
    mutableUiState.update {
      it.copy(
        busy = false,
        ready = false,
        statusMessage = R.string.status_error,
        runTotalMs = null,
        errorMessage = failure.message ?: failure.javaClass.simpleName,
        canFallbackToCpu = canFallbackToCpu,
      )
    }
  }

  private fun UiState.withoutResults() =
    copy(answers = emptyList(), runTotalMs = null, runTokenCount = 0)

  override fun onCleared() {
    cleared = true
    // Queue cleanup after any blocking native call on the same confined dispatcher.
    modelScope.launch {
      try {
        engine?.close()
        engine = null
      } catch (failure: Exception) {
        Log.e("JULIA1", "Engine cleanup failed", failure)
      } finally {
        modelScope.cancel()
      }
    }
    super.onCleared()
  }

  companion object {
    private fun milliseconds(nanos: Long) = nanos / 1_000_000.0

    private fun readyStatus(backend: JuliaEngine.Backend) =
      when (backend) {
        JuliaEngine.Backend.GPU -> R.string.status_gpu_ready
        JuliaEngine.Backend.CPU -> R.string.status_cpu_ready
      }

    /** Creates a ViewModel with the application context, never an Activity reference. */
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
