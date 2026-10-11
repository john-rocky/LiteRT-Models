// SPDX-License-Identifier: Apache-2.0
package com.phocinae

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

/**
 * Owns the engine and confines all tokenizer, builder, model, and decoder calls to one dispatcher.
 */
class MainViewModel(private val context: Context) : ViewModel() {
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(SupervisorJob() + modelDispatcher)
  private val preferences = context.getSharedPreferences("phocinae_preferences", Context.MODE_PRIVATE)
  private var engine: PhocinaeEngine? = null
  private var presets: Map<String, Any?>? = null
  private var started = false
  private var storage = PhocinaeEngine.Storage.WFP16
  private var launchStartedNs = 0L
  @Volatile private var cleared = false
  private val mutableUiState =
    MutableStateFlow(
      UiState(
        inputText = context.getString(Preset.DEPLOY.example),
        npuAvailable = PhocinaeEngine.npuLibrariesInstalled(context),
      )
    )

  /** State consumed by the screen with lifecycle-aware collection. */
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()

  /** Starts the debug gate or the interactive pipeline once per ViewModel lifetime. */
  fun start(
    gate: Boolean,
    accelerator: String?,
    requestedWindow: String?,
    requestedStorage: String? = null,
    launchedAtNs: Long = System.nanoTime(),
  ) {
    if (started || cleared) return
    started = true
    launchStartedNs = launchedAtNs
    try {
      storage = PhocinaeEngine.Storage.fromArgument(requestedStorage ?: "wfp16")
    } catch (failure: IllegalStateException) {
      showFailure(failure)
      return
    }
    if (gate && BuildConfig.DEBUG) {
      mutableUiState.update {
        it.copy(gateMode = true, statusMessage = R.string.status_gate_running)
      }
      modelScope.launch {
        try {
          val backend = PhocinaeEngine.Backend.fromArgument(accelerator ?: "gpu")
          val window = (requestedWindow ?: "256").toInt()
          mutableUiState.update { it.copy(accelerator = backend) }
          val result = PhocinaeGateEntry.run(context, helper(), backend, window)
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
          Log.e("PHOCINAE_GATE", "Gate setup failed", failure)
        } catch (failure: LinkageError) {
          showFailure(failure)
          Log.e("PHOCINAE_GATE", "Native runtime failed", failure)
        }
      }
    } else {
      val backend =
        try {
          PhocinaeEngine.Backend.fromArgument(
            accelerator ?: preferences.getString("accelerator", "npu") ?: "npu"
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
    if (!editable()) return
    mutableUiState.update { it.copy(inputText = value).withoutResults() }
  }

  /** Selects a question set and loads its invented example into the editor. */
  fun selectPreset(value: Preset) {
    if (!editable()) return
    mutableUiState.update {
      it.copy(preset = value, inputText = context.getString(value.example)).withoutResults()
    }
  }

  /** Saves an explicit backend choice; GPU failure never selects CPU automatically. */
  fun selectAccelerator(value: PhocinaeEngine.Backend) {
    if (!editable() || (value == uiState.value.accelerator && uiState.value.ready)) return
    loadAccelerator(value)
  }

  /** Selects the calibration JSON or the unchanged decoder's identity temperature. */
  fun setCalibrated(value: Boolean) {
    if (!editable()) return
    mutableUiState.update { it.copy(calibrated = value).withoutResults() }
  }

  private fun editable() = !cleared && !uiState.value.busy && !uiState.value.gateMode

  private fun loadAccelerator(backend: PhocinaeEngine.Backend) {
    val initializationStartedNs = System.nanoTime()
    preferences.edit().putString("accelerator", backend.argument).apply()
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
        val requiredFiles = listOf(helper().mainFilename(WINDOW)) + PhocinaeEngine.REQUIRED_FILES
        val missing = requiredFiles.firstOrNull { !File(context.filesDir, it).isFile }
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
                PhocinaeEngine.Backend.GPU,
                PhocinaeEngine.Backend.GPU_FP16 -> R.string.status_compiling_gpu
                PhocinaeEngine.Backend.NPU -> R.string.status_compiling_npu
                PhocinaeEngine.Backend.CPU -> R.string.status_loading_cpu
              }
          )
        }
        compiling = true
        val compileStartedNs = System.nanoTime()
        helper.initialize(backend, WINDOW)
        val compileMs = milliseconds(System.nanoTime() - compileStartedNs)
        compiling = false
        mutableUiState.update { it.copy(statusMessage = R.string.status_warming_up) }
        val warmStartedNs = System.nanoTime()
        val state = uiState.value
        // Preserve the full preset warm-up, including tokenization, building, both graphs and
        // decode.
        questions(state.preset).forEach { (id, question) ->
          answer(helper, state, id, PhocinaeJson.asObject(question))
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
          "PHOCINAE_READY",
          "accelerator=${backend.argument} storage=${storage.argument} window=$WINDOW " +
            "launch_to_ready_ms=$launchToReadyMs " +
            "initialization_ms=${milliseconds(readyAtNs - initializationStartedNs)} " +
            "tokenizer_load_ms=${helper.tokenizerLoadMs} " +
            "embedding_map_ms=${helper.embeddingLoadMs} compile_ms=$compileMs warmup_ms=$warmupMs",
        )
      } catch (failure: Exception) {
        showFailure(failure, canFallbackToCpu = backend != PhocinaeEngine.Backend.CPU && compiling)
      } catch (failure: LinkageError) {
        showFailure(failure, canFallbackToCpu = backend != PhocinaeEngine.Backend.CPU && compiling)
      }
    }
  }

  /**
   * Runs the current preset and reports complete host-plus-model time, excluding initialization.
   */
  fun run() {
    val state = uiState.value
    if (!editable() || !state.ready) return
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
          val output = answer(helper(), state, id, PhocinaeJson.asObject(question))
          results += presentAnswer(id, output, state.calibrated)
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
          "PHOCINAE_TIMING",
          "accelerator=${state.accelerator.argument} storage=${storage.argument} window=$WINDOW " +
            "total_ms=$runTotalMs question_count=${results.size} token_count=$tokenCount " +
            "calibrated=${state.calibrated}",
        )
      } catch (failure: Exception) {
        showFailure(failure)
      } catch (failure: LinkageError) {
        showFailure(failure)
      }
    }
  }

  private fun answer(
    helper: PhocinaeEngine,
    state: UiState,
    id: String,
    question: Map<String, Any?>,
  ): PhocinaeEngine.AnswerResult {
    val startedNs = System.nanoTime()
    val sequence = helper.prepare(state.inputText, question, WINDOW, questionId = id)
    val preparedNs = System.nanoTime()
    val raw = helper.runRaw(sequence, state.accelerator, WINDOW)
    check(raw.finite) { context.getString(R.string.error_nonfinite) }
    val decodeStartedNs = System.nanoTime()
    val calibration = if (state.calibrated) helper.calibration else PhocinaeCalibration.identity()
    val answer = PhocinaeDecoder.decode(raw.markerLogits, raw.actLogits, sequence.question, calibration)
    val finishedNs = System.nanoTime()
    return PhocinaeEngine.AnswerResult(
      sequence,
      answer,
      raw,
      milliseconds(preparedNs - startedNs),
      milliseconds(finishedNs - decodeStartedNs),
      milliseconds(finishedNs - startedNs),
    )
  }

  private fun presentAnswer(
    id: String,
    output: PhocinaeEngine.AnswerResult,
    calibrated: Boolean,
  ): AnswerUiRow {
    val answer = output.answer
    val question = output.sequence.question
    val probabilities =
      when (question.type) {
        "choice" -> {
          val criteria = PhocinaeJson.asObject(question.criteria)
          PhocinaeJson.asObject(answer.getValue("probabilities")).map { (label, value) ->
            ProbabilityUiRow(
              probability = (value as Number).toDouble(),
              label = label,
              description = criteria[label]?.let(PhocinaePromptBuilder::renderCriterion),
            )
          }
        }
        "score" -> {
          val legend = PhocinaeJson.asObject(answer.getValue("legend"))
          PhocinaeJson.asObject(answer.getValue("probabilities")).map { (level, value) ->
            ProbabilityUiRow(
              probability = (value as Number).toDouble(),
              scoreLevel = level.toInt(),
              description = legend[level]?.let(PhocinaePromptBuilder::renderCriterion),
            )
          }
        }
        else -> {
          val calibration = if (calibrated) helper().calibration else PhocinaeCalibration.identity()
          val values = PhocinaeDecoder.probabilities(output.raw.markerLogits, question, calibration)
          listOf(
            ProbabilityUiRow(
              probability = PhocinaeDecoder.round4(values[0].toDouble()),
              labelResource = R.string.answer_false,
            ),
            ProbabilityUiRow(
              probability = PhocinaeDecoder.round4(values[1].toDouble()),
              labelResource = R.string.answer_true,
            ),
          )
        }
      }
    // The shown answer is the most probable option (the publisher's server answers the same way:
    // choice = argmax, score = the level with the highest probability, noul = p(true) >= 0.5).
    val best = probabilities.maxByOrNull { it.probability } ?: error("no options")
    val shown =
      when (question.type) {
        "choice" -> best.label
        "score" ->
          best.description?.takeIf { it.isNotEmpty() }
            ?: context.getString(R.string.answer_level, checkNotNull(best.scoreLevel))
        else -> {
          val trueProbability = (answer.getValue("noul") as Number).toDouble()
          context.getString(if (trueProbability >= 0.5) R.string.answer_true else R.string.answer_false)
        }
      }
    val shownProbability =
      if (question.type == "noul") {
        val trueProbability = (answer.getValue("noul") as Number).toDouble()
        if (trueProbability >= 0.5) trueProbability else 1.0 - trueProbability
      } else {
        best.probability
      }
    return AnswerUiRow(
      questionId = id,
      instructions = question.instructions,
      type = question.type,
      answer = shown,
      answerProbability = shownProbability,
      probabilities = probabilities,
      confidence = (answer.getValue("confidence") as Number).toDouble(),
      totalMs = output.totalMs,
      tokenCount = output.sequence.ids.size,
    )
  }

  private fun questions(preset: Preset): Map<String, Any?> {
    val all =
      presets
        ?: context.assets
          .open("presets.json")
          .bufferedReader()
          .use { PhocinaeJson.asObject(PhocinaeJson.parse(it.readText())) }
          .also { presets = it }
    return PhocinaeJson.asObject(all.getValue(preset.assetKey))
  }

  private fun helper(): PhocinaeEngine = engine ?: PhocinaeEngine(context, storage).also { engine = it }

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
        Log.e("PHOCINAE", "Engine cleanup failed", failure)
      } finally {
        modelScope.cancel()
      }
    }
    super.onCleared()
  }

  companion object {
    private fun milliseconds(nanos: Long) = nanos / 1_000_000.0

    /** The editor's window: 384 tokens hold 80 % of the publisher's test rows; the published graphs also cover 256 and 512. */
    const val WINDOW = 384

    private fun readyStatus(backend: PhocinaeEngine.Backend) =
      when (backend) {
        PhocinaeEngine.Backend.GPU -> R.string.status_gpu_ready
        PhocinaeEngine.Backend.GPU_FP16 -> R.string.status_gpu_fp16_ready
        PhocinaeEngine.Backend.NPU -> R.string.status_npu_ready
        PhocinaeEngine.Backend.CPU -> R.string.status_cpu_ready
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
