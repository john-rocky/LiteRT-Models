package com.timestral

import androidx.compose.runtime.Immutable

/** A bundled example series: [asset] under `assets/examples/`, its title and a fitting horizon. */
@Immutable
data class Example(val title: String, val asset: String, val horizon: Int)

/**
 * The chart of a forecast: the last [history] values of the series (NaN = missing), then the
 * forecast's median and its 10 % and 90 % quantiles for each step.
 */
@Immutable
data class ForecastChartData(
  val history: FloatArray,
  val median: FloatArray,
  val low: FloatArray,
  val high: FloatArray,
)

/** What the screen shows after a forecast: the chart and the lines under it. */
@Immutable
data class ForecastView(val chart: ForecastChartData, val summary: List<String>, val notes: List<String>)

/** Everything the screen draws. */
@Immutable
data class UiState(
  val seriesText: String = "",
  val horizonText: String = DEFAULT_HORIZON.toString(),
  val backend: Backend = Backend.GPU,
  /** null when the NPU can be chosen, else the reason it cannot. */
  val npuUnavailable: String? = null,
  val examples: List<Example> = emptyList(),
  /** Model files missing from `files/` (the install line under the title). */
  val missingFiles: List<String> = emptyList(),
  val running: Boolean = false,
  /** The progress line while a forecast runs. */
  val status: String? = null,
  /** The sentence of the last failure. */
  val error: String? = null,
  val result: ForecastView? = null,
) {
  companion object {
    const val DEFAULT_HORIZON = 64
  }
}
