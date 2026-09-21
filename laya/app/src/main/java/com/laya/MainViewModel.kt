// SPDX-License-Identifier: Apache-2.0
package com.laya

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
  private val preferences = context.getSharedPreferences("laya_preferences", Context.MODE_PRIVATE)
  private var engine: LayaEngine? = null
  private var presets: Map<String, Any?>? = null
  private var started = false
  private var storage = LayaEngine.Storage.WFP16
  private var launchStartedNs = 0L
  @Volatile private var cleared = false
  private val mutableUiState =
    MutableStateFlow(
      UiState(
        inputSubject = context.getString(R.string.example_email_ja_subject),
        inputText = context.getString(R.string.example_email_ja_body),
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
      storage = LayaEngine.Storage.fromArgument(requestedStorage ?: "wfp16")
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
          val backend = LayaEngine.Backend.fromArgument(accelerator ?: "gpu")
          val window = (requestedWindow ?: "256").toInt()
          mutableUiState.update { it.copy(accelerator = backend) }
          val result = LayaGateEntry.run(context, helper(), backend, window)
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
          Log.e("LAYA_GATE", "Gate setup failed", failure)
        } catch (failure: LinkageError) {
          showFailure(failure)
          Log.e("LAYA_GATE", "Native runtime failed", failure)
        }
      }
    } else {
      val backend =
        try {
          LayaEngine.Backend.fromArgument(
            accelerator ?: preferences.getString("accelerator", "gpu") ?: "gpu"
          )
        } catch (failure: IllegalStateException) {
          showFailure(failure)
          return
        }
      loadAccelerator(backend)
    }
  }

  /** Edits the email subject without changing the preset schema. */
  fun setSubject(value: String) {
    if (!editable()) return
    mutableUiState.update { it.copy(inputSubject = value).withoutResults() }
  }

  /** Edits the email body, support message, or moderation post. */
  fun setInputText(value: String) {
    if (!editable()) return
    mutableUiState.update { it.copy(inputText = value).withoutResults() }
  }

  /** Replaces the editor contents with this preset's invented example in the selected language. */
  fun selectLanguage(value: ExampleLanguage) {
    if (!editable()) return
    mutableUiState.update { withExample(it.copy(language = value)).withoutResults() }
  }

  /** Selects the unchanged upstream questions and their matching invented example. */
  fun selectPreset(value: Preset) {
    if (!editable()) return
    mutableUiState.update { withExample(it.copy(preset = value)).withoutResults() }
  }

  /** Saves an explicit backend choice; GPU failure never selects CPU automatically. */
  fun selectAccelerator(value: LayaEngine.Backend) {
    if (!editable() || (value == uiState.value.accelerator && uiState.value.ready)) return
    loadAccelerator(value)
  }

  /** Selects the calibration JSON or the unchanged decoder's identity temperature. */
  fun setCalibrated(value: Boolean) {
    if (!editable()) return
    mutableUiState.update { it.copy(calibrated = value).withoutResults() }
  }

  private fun editable() = !cleared && !uiState.value.busy && !uiState.value.gateMode

  private fun loadAccelerator(backend: LayaEngine.Backend) {
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
        val requiredFiles =
          listOf("laya_ml_s256_embeds_${storage.argument}.tflite") + LayaEngine.REQUIRED_FILES
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
              if (backend == LayaEngine.Backend.GPU) {
                R.string.status_compiling_gpu
              } else {
                R.string.status_loading_cpu
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
        // Preserve the full preset warm-up, including tokenization, building, both graphs and
        // decode.
        questions(state.preset).forEach { (id, question) ->
          answer(helper, state, id, LayaJson.asObject(question))
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
          "LAYA_READY",
          "accelerator=${backend.name.lowercase()} storage=${storage.argument} " +
            "launch_to_ready_ms=$launchToReadyMs " +
            "initialization_ms=${milliseconds(readyAtNs - initializationStartedNs)} " +
            "tokenizer_load_ms=${helper.tokenizerLoadMs} " +
            "embedding_map_ms=${helper.embeddingLoadMs} compile_ms=$compileMs warmup_ms=$warmupMs",
        )
      } catch (failure: Exception) {
        showFailure(failure, canFallbackToCpu = backend == LayaEngine.Backend.GPU && compiling)
      } catch (failure: LinkageError) {
        showFailure(failure, canFallbackToCpu = backend == LayaEngine.Backend.GPU && compiling)
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
          val output = answer(helper(), state, id, LayaJson.asObject(question))
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
          "LAYA_TIMING",
          "accelerator=${state.accelerator.name.lowercase()} storage=${storage.argument} " +
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
    helper: LayaEngine,
    state: UiState,
    id: String,
    question: Map<String, Any?>,
  ): LayaEngine.AnswerResult {
    val startedNs = System.nanoTime()
    val sequence = helper.prepare(modelState(state), question, questionId = id)
    val preparedNs = System.nanoTime()
    val raw = helper.runRaw(sequence, state.accelerator)
    check(raw.finite) { context.getString(R.string.error_nonfinite) }
    val decodeStartedNs = System.nanoTime()
    val calibration = if (state.calibrated) helper.calibration else LayaCalibration.identity()
    val answer = LayaDecoder.decode(raw.markerLogits, raw.actLogits, sequence.question, calibration)
    val finishedNs = System.nanoTime()
    return LayaEngine.AnswerResult(
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
    output: LayaEngine.AnswerResult,
    calibrated: Boolean,
  ): AnswerUiRow {
    val answer = output.answer
    val question = output.sequence.question
    val probabilities =
      when (question.type) {
        "choice" -> {
          val criteria = LayaJson.asObject(question.criteria)
          LayaJson.asObject(answer.getValue("probabilities")).map { (label, value) ->
            ProbabilityUiRow(
              probability = (value as Number).toDouble(),
              label = label,
              description = criteria[label]?.let(LayaPromptBuilder::renderCriterion),
            )
          }
        }
        "score" -> {
          val legend = LayaJson.asObject(answer.getValue("legend"))
          LayaJson.asObject(answer.getValue("probabilities")).map { (level, value) ->
            ProbabilityUiRow(
              probability = (value as Number).toDouble(),
              scoreLevel = level.toInt(),
              description = legend[level]?.let(LayaPromptBuilder::renderCriterion),
            )
          }
        }
        else -> {
          val calibration = if (calibrated) helper().calibration else LayaCalibration.identity()
          val values = LayaDecoder.probabilities(output.raw.markerLogits, question, calibration)
          listOf(
            ProbabilityUiRow(
              probability = LayaDecoder.round4(values[0].toDouble()),
              labelResource = R.string.option_false,
            ),
            ProbabilityUiRow(
              probability = LayaDecoder.round4(values[1].toDouble()),
              labelResource = R.string.option_true,
            ),
          )
        }
      }
    return AnswerUiRow(
      questionId = id,
      instructions = question.instructions,
      type = question.type,
      choice = answer["choice"] as? String,
      score = (answer["score"] as? Number)?.toDouble(),
      trueProbability = (answer["noul"] as? Number)?.toDouble(),
      probabilities = probabilities,
      confidence = (answer.getValue("confidence") as Number).toDouble(),
      totalMs = output.totalMs,
      tokenCount = output.sequence.ids.size,
    )
  }

  private fun modelState(state: UiState): Map<String, String> =
    when (state.preset) {
      Preset.EMAIL -> linkedMapOf("subject" to state.inputSubject, "body" to state.inputText)
      Preset.TRIAGE -> linkedMapOf("message" to state.inputText)
      Preset.MODERATION -> linkedMapOf("post" to state.inputText)
    }

  private fun withExample(state: UiState): UiState {
    val japanese = state.language == ExampleLanguage.JA
    val subject =
      if (state.preset == Preset.EMAIL) {
        context.getString(
          if (japanese) R.string.example_email_ja_subject else R.string.example_email_en_subject
        )
      } else {
        ""
      }
    val text =
      when (state.preset) {
        Preset.EMAIL ->
          if (japanese) R.string.example_email_ja_body else R.string.example_email_en_body
        Preset.TRIAGE -> if (japanese) R.string.example_support_ja else R.string.example_support_en
        Preset.MODERATION ->
          if (japanese) R.string.example_moderation_ja else R.string.example_moderation_en
      }
    return state.copy(inputSubject = subject, inputText = context.getString(text))
  }

  private fun questions(preset: Preset): Map<String, Any?> {
    val all =
      presets
        ?: context.assets
          .open("presets.json")
          .bufferedReader()
          .use { LayaJson.asObject(LayaJson.parse(it.readText())) }
          .also { presets = it }
    return LayaJson.asObject(all.getValue(preset.assetKey))
  }

  private fun helper(): LayaEngine = engine ?: LayaEngine(context, storage).also { engine = it }

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
        Log.e("LAYA", "Engine cleanup failed", failure)
      } finally {
        modelScope.cancel()
      }
    }
    super.onCleared()
  }

  companion object {
    private fun milliseconds(nanos: Long) = nanos / 1_000_000.0

    private fun readyStatus(backend: LayaEngine.Backend) =
      when (backend) {
        LayaEngine.Backend.GPU -> R.string.status_gpu_ready
        LayaEngine.Backend.CPU -> R.string.status_cpu_ready
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
