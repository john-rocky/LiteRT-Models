package com.gliformer

import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class MainViewModel(private val context: Context) : ViewModel() {
  private val mutableUiState = MutableStateFlow(UiState(context.getString(R.string.example_text)))
  val uiState: StateFlow<UiState> = mutableUiState.asStateFlow()
  private var extractor: GliformerExtractor? = null
  private var started = false
  private var launchStartedNs = 0L
  private var tapCount = 0
  private var loadMs = 0.0
  private val tapReporter = GliformerFirstTapReport(context)

  fun start(table: String? = null, launchNs: Long = SystemClock.elapsedRealtimeNanos()) {
    if (started) return
    started = true
    launchStartedNs = launchNs
    val storage =
      if (table == "fp32") GliformerInputs.EmbeddingTable.Storage.FP32
      else GliformerInputs.EmbeddingTable.Storage.FP16
    extractor = GliformerExtractor(context.filesDir, storage)
    warmSelectedBackend()
  }

  fun setInputText(text: String) {
    if (!mutableUiState.value.busy) mutableUiState.value = mutableUiState.value.copy(text = text)
  }

  fun selectBackend(backend: GliformerExtractor.Backend) {
    if (mutableUiState.value.busy || backend == mutableUiState.value.backend) return
    mutableUiState.value = mutableUiState.value.copy(backend = backend, result = null)
    warmSelectedBackend()
  }

  fun retry() {
    if (!mutableUiState.value.busy) warmSelectedBackend()
  }

  private fun warmSelectedBackend() {
    val helper = extractor ?: return
    val backend = mutableUiState.value.backend
    mutableUiState.value =
      mutableUiState.value.copy(phase = UiState.Phase.LOADING, errorMessage = null)
    viewModelScope.launch {
      try {
        val loaded = helper.initialize(backend, 128)
        loadMs = loaded.loadMs
        mutableUiState.value = mutableUiState.value.copy(phase = UiState.Phase.WARMING)
        val warmup = helper.warmUp(backend, 128, GliformerExtractor.STARTUP_WARMUP_ITERATIONS)
        mutableUiState.value =
          mutableUiState.value.copy(phase = UiState.Phase.READY, startupWarmupMs = warmup.totalMs)
        tapReporter.ready(launchStartedNs, backend, helper.tableStorage, loadMs, warmup)
      } catch (failure: CancellationException) {
        throw failure
      } catch (failure: Throwable) {
        fail(failure)
      }
    }
  }

  /** Only the screen's actual Button.onClick calls this; startup never invokes extraction taps. */
  fun extract(tapNs: Long) {
    val before = mutableUiState.value
    if (before.phase != UiState.Phase.READY) return
    val helper = extractor ?: return
    val id = ++tapCount
    tapReporter.begin(
      id,
      tapNs,
      before.backend,
      helper.tableStorage,
      loadMs,
      before.startupWarmupMs,
    )
    mutableUiState.value =
      before.copy(phase = UiState.Phase.EXTRACTING, requestId = id, errorMessage = null)
    viewModelScope.launch {
      try {
        val selected = helper.prepare(before.text).window.sequenceLength
        if (!helper.isWarm(before.backend, selected)) {
          mutableUiState.value = mutableUiState.value.copy(phase = UiState.Phase.LOADING)
          val load = helper.initialize(before.backend, selected)
          mutableUiState.value = mutableUiState.value.copy(phase = UiState.Phase.WARMING)
          val warm =
            helper.warmUp(before.backend, selected, GliformerExtractor.STARTUP_WARMUP_ITERATIONS)
          tapReporter.windowWarmup(id, load.loadMs, warm.totalMs)
          mutableUiState.value = mutableUiState.value.copy(phase = UiState.Phase.EXTRACTING)
        }
        val result = helper.extract(before.text, before.backend)
        val resultNs = SystemClock.elapsedRealtimeNanos()
        val shown =
          UiResult(
            result.prepared.text,
            result.entities,
            result.backend,
            result.window,
            result.prepared.encodedLength,
            result.prepared.words.size,
            result.timings,
          )
        tapReporter.result(id, resultNs, result)
        mutableUiState.value =
          mutableUiState.value.copy(phase = UiState.Phase.READY, result = shown)
        tapReporter.statePublished(id, SystemClock.elapsedRealtimeNanos())
      } catch (failure: CancellationException) {
        throw failure
      } catch (failure: Throwable) {
        tapReporter.failure(id, failure)
        fail(failure)
      }
    }
  }

  fun resultRendered(requestId: Int, renderedNs: Long) {
    val report = tapReporter.rendered(requestId, renderedNs) ?: return
    viewModelScope.launch(Dispatchers.IO) { tapReporter.write(report) }
  }

  private fun fail(failure: Throwable) {
    Log.e("GLIFORMER_UI", "Pipeline failed", failure)
    mutableUiState.value =
      mutableUiState.value.copy(
        phase = UiState.Phase.ERROR,
        errorMessage = failure.message ?: failure.javaClass.simpleName,
      )
  }

  companion object {
    fun getFactory(context: Context): ViewModelProvider.Factory =
      object : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
          require(modelClass.isAssignableFrom(MainViewModel::class.java))
          @Suppress("UNCHECKED_CAST")
          return MainViewModel(context.applicationContext) as T
        }
      }
  }

  override fun onCleared() {
    extractor?.close()
    super.onCleared()
  }
}
