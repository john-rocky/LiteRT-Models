package com.timestral

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import java.util.Locale
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Owns the [ForecastEngine]: checks the user's series and horizon on the main thread (the failure
 * sentences need no model), then runs the forecast on [TimestralRuntime.dispatcher].
 */
class MainViewModel(private val application: Application) : ViewModel() {
  private val _uiState = MutableStateFlow(UiState(examples = EXAMPLES))
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  private var engine: ForecastEngine? = null

  init {
    viewModelScope.launch {
      val created = withContext(TimestralRuntime.dispatcher) { runCatching { ForecastEngine(application) } }
      val ready = created.getOrNull()
      engine = ready
      _uiState.update { state ->
        state.copy(
          npuUnavailable = npuUnavailable(ready),
          missingFiles = ready?.missingFiles().orEmpty(),
          error = created.exceptionOrNull()?.let { "Could not read the model configuration: ${describe(it)}" },
        )
      }
    }
  }

  fun setSeries(text: String) = _uiState.update { it.copy(seriesText = text, error = null) }

  fun setHorizon(text: String) = _uiState.update { it.copy(horizonText = text, error = null) }

  fun selectBackend(backend: Backend) {
    if (backend == Backend.NPU && _uiState.value.npuUnavailable != null) return
    _uiState.update { it.copy(backend = backend, error = null) }
  }

  /** Loads a bundled example into the text field and sets its horizon. */
  fun loadExample(example: Example) {
    viewModelScope.launch {
      val text =
        withContext(Dispatchers.IO) {
          runCatching {
            application.assets.open("examples/${example.asset}").bufferedReader().use { it.readText() }
          }
        }
      _uiState.update { state ->
        text.fold(
          { state.copy(seriesText = it, horizonText = example.horizon.toString(), error = null) },
          { state.copy(error = "Could not read the example: ${describe(it)}") },
        )
      }
    }
  }

  /** Loads a text or CSV file the user picked into the text field. */
  fun loadDocument(uri: Uri) {
    viewModelScope.launch {
      val text =
        withContext(Dispatchers.IO) {
          runCatching {
            val stream =
              application.contentResolver.openInputStream(uri)
                ?: throw InputException("The file could not be opened.")
            stream.use { input ->
              val bytes = input.readNBytesCompat(MAX_FILE_BYTES + 1)
              if (bytes.size > MAX_FILE_BYTES) {
                throw InputException("The file is larger than ${MAX_FILE_BYTES / 1_000_000} MB.")
              }
              String(bytes, Charsets.UTF_8)
            }
          }
        }
      _uiState.update { state ->
        text.fold(
          { state.copy(seriesText = it, error = null) },
          { state.copy(error = "Could not read the file: ${describe(it)}") },
        )
      }
    }
  }

  /** Checks the inputs and runs one forecast on the chosen backend. */
  fun forecast() {
    val state = _uiState.value
    if (state.running) return
    val parsed: ParsedSeries
    val horizon: Int
    try {
      parsed = SeriesInput.parse(state.seriesText)
      horizon = SeriesInput.horizon(state.horizonText)
    } catch (failure: InputException) {
      _uiState.update { it.copy(error = failure.message, result = null) }
      return
    }
    val ready = engine
    if (ready == null) {
      _uiState.update { it.copy(error = "The model configuration is not loaded yet.") }
      return
    }
    val backend = state.backend
    _uiState.update { it.copy(running = true, error = null, status = statusLine(null, backend, 0)) }
    viewModelScope.launch {
      val start = SystemClock.elapsedRealtime()
      var stage: ForecastStage? = null
      val ticker = launch {
        while (isActive) {
          val seconds = ((SystemClock.elapsedRealtime() - start) / 1000).toInt()
          _uiState.update { it.copy(status = statusLine(stage, backend, seconds)) }
          delay(TICK_MS)
        }
      }
      val outcome =
        withContext(TimestralRuntime.dispatcher) {
          runCatching { ready.forecast(parsed, horizon, backend) { stage = it } }
        }
      ticker.cancel()
      val missing = withContext(Dispatchers.IO) { ready.missingFiles() }
      _uiState.update { current ->
        outcome.fold(
          { result ->
            current.copy(running = false, status = null, result = view(result), missingFiles = missing)
          },
          { failure ->
            current.copy(
              running = false,
              status = null,
              result = null,
              error = failureLine(failure, stage, backend),
              missingFiles = missing,
            )
          },
        )
      }
    }
  }

  override fun onCleared() {
    val closing = engine ?: return
    engine = null
    // CompiledModel calls stay on the LiteRT thread, after any forecast still running there.
    CoroutineScope(SupervisorJob() + TimestralRuntime.dispatcher).launch { closing.close() }
  }

  private fun npuUnavailable(ready: ForecastEngine?): String? {
    if (ready == null) return "The model configuration is not loaded."
    if (!ready.npuLibraries) {
      return "This APK has no NPU libraries: run scripts/fetch_npu_libs.sh before the build (README)."
    }
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && Build.SOC_MANUFACTURER != QUALCOMM) {
      return "The NPU path needs a Qualcomm Snapdragon; this phone's chip is from ${Build.SOC_MANUFACTURER}."
    }
    return null
  }

  private fun statusLine(stage: ForecastStage?, backend: Backend, seconds: Int): String =
    when (stage) {
      ForecastStage.COMPILE ->
        when (backend) {
          Backend.NPU ->
            "Compiling the graph for the NPU (this can take a few minutes)… $seconds s"
          Backend.GPU -> "Compiling the graph for the GPU… $seconds s"
          Backend.CPU -> "Loading the graph on the CPU… $seconds s"
        }
      ForecastStage.RUN -> "Forecasting on the ${backend.label}… $seconds s"
      null -> "Preparing the series… $seconds s"
    }

  private fun failureLine(failure: Throwable, stage: ForecastStage?, backend: Backend): String =
    when (failure) {
      is InputException,
      is MissingFileException,
      is NpuNotUsedException -> failure.message.orEmpty()
      is ForecastException ->
        "The forecast is not finite on the ${backend.label} (${failure.message}). Choose another " +
          "accelerator."
      else ->
        when (stage) {
          ForecastStage.COMPILE ->
            "The ${backend.label} could not compile the graph: ${describe(failure)}. The sample " +
              "does not fall back to another accelerator; choose one above."
          ForecastStage.RUN -> "The ${backend.label} failed while running the graph: ${describe(failure)}"
          null -> "The forecast failed: ${describe(failure)}"
        }
    }

  private fun view(result: ForecastResult): ForecastView {
    val horizon = result.horizon
    val historyCount = minOf(result.series.size, (2 * horizon).coerceIn(MIN_HISTORY, MAX_HISTORY))
    val history = result.series.copyOfRange(result.series.size - historyCount, result.series.size)
    val q = result.quantiles
    val n = TimestralHost.QUANTILES
    val median = FloatArray(horizon) { q[it * n + TimestralHost.MEDIAN] }
    val low = FloatArray(horizon) { q[it * n + TimestralHost.P10] }
    val high = FloatArray(horizon) { q[it * n + TimestralHost.P90] }
    val runOn =
      when (result.backend) {
        Backend.CPU -> "CPU, ${result.threads} threads"
        Backend.GPU -> "GPU, ${result.precision.label}"
        Backend.NPU -> "NPU"
      }
    val spec = result.spec
    val summary = ArrayList<String>()
    summary.add(
      "Next step: median ${value(median[0])} (10–90 %: ${value(low[0])} to ${value(high[0])})"
    )
    summary.add(
      "Step ${number(horizon)}: median ${value(median[horizon - 1])} " +
        "(10–90 %: ${value(low[horizon - 1])} to ${value(high[horizon - 1])})"
    )
    summary.add(
      if (result.callMs.size == 1) {
        "Graph call: ${ms(result.graphMs)} ms on the $runOn (write inputs, run, read back)"
      } else {
        "Graph calls: ${ms(result.graphMs)} ms on the $runOn " +
          "(${result.callMs.joinToString(" + ") { ms(it) }}; each writes inputs, runs, reads back)"
      }
    )
    summary.add("Host: ${ms(result.hostMs)} ms (statistics, inputs, refine, averaging)")
    result.compileMs?.let { summary.add("Compile: ${ms(it)} ms (this forecast compiled the graph)") }
    summary.add(
      "Graph ${spec.name}: ${spec.contextPatches} + ${spec.horizonPatches} patches of 32 steps, " +
        if (spec.rows == 2) "the series and its negation in one call" else "one call each for the series and its negation"
    )
    result.npu?.let { npu ->
      val dispatched = npu.dispatched
      summary.add(
        when {
          dispatched == null -> "NPU: placement not confirmed (${npu.error ?: "no log line"})"
          dispatched.first == dispatched.second ->
            "NPU: the HTP took the whole graph (${dispatched.first} of ${dispatched.second} " +
              "nodes${if (npu.fromCache) ", compiled graph from LiteRT's cache" else ""})"
          else ->
            "NPU: the HTP took ${dispatched.first} of ${dispatched.second} nodes; the rest ran on the CPU"
        }
      )
    }
    return ForecastView(ForecastChartData(history, median, low, high), summary, result.notes)
  }

  companion object {
    /** The bundled example series (synthetic, made by `scripts/make_test_vectors.py`). */
    val EXAMPLES =
      listOf(
        Example("Hourly temperature", "hourly_temperature.csv", 48),
        Example("Daily orders", "daily_orders.csv", 28),
        Example("Sensor with gaps", "sensor_with_gaps.csv", 60),
      )

    private const val QUALCOMM = "QTI"
    private const val TICK_MS = 1000L
    private const val MIN_HISTORY = 64
    private const val MAX_HISTORY = 1024
    private const val MAX_FILE_BYTES = 4_000_000

    fun getFactory(context: Context): ViewModelProvider.Factory =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
          MainViewModel(context.applicationContext as Application) as T
      }

    private fun describe(failure: Throwable): String =
      failure.message?.takeIf { it.isNotBlank() } ?: failure.javaClass.simpleName

    private fun ms(value: Double): String = String.format(Locale.US, "%,.1f", value)

    private fun number(count: Int): String = String.format(Locale.US, "%,d", count)

    /** A quantile value with as many digits as its size needs. */
    fun value(v: Float): String {
      val magnitude = Math.abs(v)
      return when {
        magnitude >= 10_000f -> String.format(Locale.US, "%,.0f", v)
        magnitude >= 100f -> String.format(Locale.US, "%.1f", v)
        magnitude >= 1f || magnitude == 0f -> String.format(Locale.US, "%.2f", v)
        else -> String.format(Locale.US, "%.3g", v)
      }
    }

    /** Reads up to [limit] bytes (InputStream.readNBytes needs API 33). */
    private fun java.io.InputStream.readNBytesCompat(limit: Int): ByteArray {
      val out = java.io.ByteArrayOutputStream()
      val buffer = ByteArray(BUFFER_BYTES)
      while (out.size() < limit) {
        val read = read(buffer, 0, minOf(buffer.size, limit - out.size()))
        if (read < 0) break
        out.write(buffer, 0, read)
      }
      return out.toByteArray()
    }

    private const val BUFFER_BYTES = 64 * 1024
  }
}
