package com.d1omni

import android.content.Context
import androidx.compose.runtime.Immutable
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** The screen's state: a title, the status line and the engine line. */
@Immutable
data class UiState(val status: String = "", val engine: String = "", val error: Boolean = false)

/**
 * Owns the engine and runs every graph call on [D1Runtime.dispatcher]. A normal launch loads the
 * tokenizer and the contract and compiles the installed graphs of up to 256 positions; a gate or
 * timing launch (debug build) runs that protocol into its report instead.
 */
class MainViewModel(private val context: Context) : ViewModel() {
  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  private var engine: D1Engine? = null
  private var started = false

  /** Starts what [launch] asks for, once per ViewModel; a later launch intent goes to [newIntent]. */
  fun start(launch: D1Launch) {
    if (started) return
    started = true
    run(launch)
  }

  /** A launch intent that reached the running activity (`singleTop`). */
  fun newIntent(launch: D1Launch) = run(launch)

  private fun run(launch: D1Launch) {
    when (launch) {
      is D1Launch.Invalid -> {
        D1Demo.failed(launch.reason)
        show(launch.reason, error = true)
      }
      is D1Launch.Gate -> diagnostics("Fixture gate") { progress ->
        D1GateRunner(context).run(launch, progress)
      }
      is D1Launch.Timing -> diagnostics("Timing") { progress ->
        D1TimingRunner(context).run(launch, progress)
      }
      is D1Launch.Normal -> load(launch)
      // vision (round 3): the picture gate and timing (debug build)
      is D1Launch.VGate -> diagnostics("Picture gate") { progress ->
        D1VisionGate(context).gate(launch, progress)
      }
      is D1Launch.VTiming -> diagnostics("Picture timing") { progress ->
        D1VisionGate(context).timing(launch, progress)
      }
      // end vision (round 3)
    }
  }

  private fun load(launch: D1Launch.Normal) {
    show("Loading the tokenizer…")
    viewModelScope.launch {
      try {
        val ready =
          withContext(D1Runtime.dispatcher) {
            engine?.close()
            val loaded = D1Engine.load(context, launch.backend, launch.precision, launch.audioPrecision)
            engine = loaded
            val startup = loaded.installed.filter { it <= STARTUP_LARGEST }.sortedDescending()
            require(startup.isNotEmpty()) {
              "Missing ${loaded.fileOf(loaded.contract.buckets.first())}. Run " +
                "scripts/install_to_device.sh, then reopen the app."
            }
            loaded.ensure(startup.take(D1Residency.MAX_RESIDENT), emptyList())
            loaded
          }
        val compileMs = ready.compiles.sumOf { it.compileMs }
        D1Demo.engineReady(Math.round(ready.loadMs + compileMs))
        _uiState.update {
          UiState(
            status = "Ready",
            engine =
              ready.resident.joinToString(" + ") { "L$it" } +
                " · " +
                backendName(ready) +
                " · loaded in %.1f s".format((ready.loadMs + compileMs) / 1000),
          )
        }
      } catch (failure: Exception) {
        val reason = D1Decider.describe(failure)
        D1Demo.failed(reason)
        show(reason, error = true)
      }
    }
  }

  private fun diagnostics(title: String, body: ((String) -> Unit) -> D1GateRunner.Summary) {
    show("$title…")
    viewModelScope.launch {
      val summary =
        withContext(D1Runtime.dispatcher) { body { line -> show("$title: $line") } }
      show("$title: ${summary.status} · ${summary.path}", error = summary.error != null)
    }
  }

  private fun backendName(engine: D1Engine): String {
    val backends = engine.residentBackends.distinct()
    return when {
      backends == listOf(D1Backend.CPU) -> "CPU 4 threads"
      engine.precision == D1Precision.FP32 -> "GPU FP32"
      else -> "GPU FP16 (FP32 accum)"
    }
  }

  private fun show(status: String, error: Boolean = false) {
    _uiState.update { it.copy(status = status, error = error) }
  }

  override fun onCleared() {
    val current = engine
    engine = null
    if (current != null) {
      viewModelScope.launch(D1Runtime.dispatcher) { current.close() }
    }
  }

  companion object {
    /** The largest bucket a normal launch compiles at startup. */
    private const val STARTUP_LARGEST = 256

    fun getFactory(context: Context): ViewModelProvider.Factory = viewModelFactory {
      initializer { MainViewModel(context.applicationContext) }
    }
  }
}
