package com.timestral

import android.content.Context
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/** One forecast as the screen shows it. */
class ForecastResult(
  /** The user's whole series (NaN = missing). */
  val series: FloatArray,
  val horizon: Int,
  /** `[horizon × 21]` quantiles, step after step. */
  val quantiles: FloatArray,
  val backend: Backend,
  val spec: GraphSpec,
  /** CPU threads (the CPU backend). */
  val threads: Int,
  /** GPU precision (the GPU backend). */
  val precision: GpuPrecision,
  /** Wall time of each graph call: write the inputs, run, read the output back. */
  val callMs: List<Double>,
  /** Wall time of the host steps before and after the graph calls. */
  val hostMs: Double,
  /** Wall time of the compile when this forecast compiled the graph, else null. */
  val compileMs: Double?,
  val npu: NpuEvidence?,
  val notes: List<String>,
) {
  /** The graph calls of the forecast together. */
  val graphMs: Double
    get() = callMs.sum()
}

/** A step of a forecast, for the status line. */
enum class ForecastStage {
  COMPILE,
  RUN,
}

/** A model file the forecast needs is not in the app's files directory. */
class MissingFileException(val names: List<String>) :
  Exception(
    "Model file not found: ${names.joinToString { "files/$it" }}. Install the model with " +
      "scripts/install_to_device.sh (README), then try again."
  )

/**
 * The model files, the host tables and the one compiled graph of the process: a forecast takes the
 * graph of its backend with the fewest tokens that holds its context and horizon, compiles it when
 * another one is resident (closing that one first, so at most one graph holds memory), builds the
 * inputs on the host, runs the graph (once with both rows, or once per row for a one-row graph) and
 * finishes the forecast on the host. Use only on [TimestralRuntime.dispatcher].
 */
class ForecastEngine(private val context: Context) : Closeable {
  val config: TimestralConfig =
    TimestralConfig.parse(
      context.assets.open(TimestralConfig.ASSET).bufferedReader().use { it.readText() }
    )

  /** Whether the APK carries the Qualcomm NPU libraries. */
  val npuLibraries: Boolean = NpuSupport.librariesInstalled(context)

  private var tables: HostTables? = null
  private var graph: TimestralGraph? = null
  private var dumpIndex = 0

  /** The required model files missing from `files/`. */
  fun missingFiles(): List<String> =
    config.requiredFiles.filter { !File(context.filesDir, it).isFile }

  /**
   * Forecasts [horizon] steps of [series] on [backend], calling [onStage] before the compile (when
   * the graph is not resident) and before the run. Throws [InputException] when no graph of
   * [backend] holds the context, [MissingFileException], [ForecastException] (the host's NaN
   * check), [NpuNotUsedException] or LiteRT's exception when the graph does not compile on
   * [backend].
   */
  fun forecast(
    series: ParsedSeries,
    horizon: Int,
    backend: Backend,
    onStage: (ForecastStage) -> Unit,
  ): ForecastResult {
    val backendConfig =
      config.backends[backend] ?: throw IllegalStateException("No graph for ${backend.label}")
    val hostStart = System.nanoTime()
    val prepared = TimestralHost.prepare(series.values, horizon)
    val spec =
      GraphSpec.choose(config.graphs, backend, prepared.contextPatches, prepared.horizonPatches)
        ?: throw noGraph(backend, minOf(series.values.size, TimestralHost.MAX_CONTEXT), prepared)
    val missing =
      listOf(config.forecastQueryFile, config.ropeCosFile, config.ropeSinFile, spec.file).filter {
        !File(context.filesDir, it).isFile
      }
    if (missing.isNotEmpty()) throw MissingFileException(missing)
    val rowsPerCall =
      if (spec.rows == TimestralHost.ROWS) listOf(intArrayOf(0, 1)) else listOf(intArrayOf(0), intArrayOf(1))
    val inputs =
      rowsPerCall.map { TimestralHost.buildInputs(prepared, spec.bucket, hostTables(), spec.mask, it) }
    var hostNs = System.nanoTime() - hostStart
    var compileMs: Double? = null
    val file = File(context.filesDir, spec.file)
    val resident = graph
    val compiled =
      if (resident != null && resident.file == file && resident.backend == backend) {
        resident
      } else {
        resident?.close()
        graph = null
        onStage(ForecastStage.COMPILE)
        TimestralGraph.create(context, file, spec, backendConfig, npuLibraries).also {
          graph = it
          compileMs = it.compileMs
        }
      }
    onStage(ForecastStage.RUN)
    val runs = inputs.map { compiled.run(it) }
    val finishStart = System.nanoTime()
    // A one-row graph's two outputs, one after the other, are the two-row output the host reads.
    val raw = if (runs.size == 1) runs[0].raw else runs[0].raw + runs[1].raw
    val quantiles = TimestralHost.finish(prepared, raw, spec.bucket)
    hostNs += System.nanoTime() - finishStart
    val result =
      ForecastResult(
        series.values,
        horizon,
        quantiles,
        backend,
        spec,
        backendConfig.threads,
        compiled.precision,
        runs.map { it.ms },
        hostNs / 1e6,
        compileMs,
        compiled.npu,
        series.notes,
      )
    if (BuildConfig.DEBUG) dump(result, inputs, raw)
    return result
  }

  override fun close() {
    graph?.close()
    graph = null
  }

  /**
   * Why no graph of [backend] holds the forecast: a context of [contextSize] values longer than the
   * backend's graphs take is a sentence for the screen ([SeriesInput.contextTooLong]).
   */
  private fun noGraph(backend: Backend, contextSize: Int, prepared: PreparedSeries): Exception =
    SeriesInput.contextTooLong(config.graphs, backend, contextSize)
      ?: IllegalStateException(
        "No ${backend.label} graph holds ${prepared.contextPatches} context patches and " +
          "${prepared.horizonPatches} horizon patches"
      )

  private fun hostTables(): HostTables =
    tables
      ?: HostTables(
          readFloats(File(context.filesDir, config.forecastQueryFile)),
          readFloats(File(context.filesDir, config.ropeCosFile)),
          readFloats(File(context.filesDir, config.ropeSinFile)),
          config.ropeFirstPosition,
        )
        .also { tables = it }

  /**
   * Debug builds keep each forecast under `files/forecast_dumps/` (the series, the sha256 of each
   * call's graph inputs, the two-row raw graph output and the quantiles) so the device run can be
   * compared with the Python reference of the host on the same graph output.
   */
  private fun dump(result: ForecastResult, inputs: List<GraphInputs>, raw: FloatArray) {
    val directory =
      File(
        context.filesDir,
        String.format(
          Locale.US,
          "forecast_dumps/%03d_%s",
          dumpIndex++,
          result.backend.name.lowercase(Locale.ROOT),
        ),
      )
    directory.mkdirs()
    writeFloats(File(directory, "series.f32"), result.series)
    writeFloats(File(directory, "raw.f32"), raw)
    writeFloats(File(directory, "quantiles.f32"), result.quantiles)
    val calls = JSONArray()
    for (call in inputs) {
      val hashes = JSONObject()
      for ((name, values) in call.byName()) hashes.put(name, sha256(values))
      calls.put(hashes)
    }
    val spec = result.spec
    val meta =
      JSONObject()
        .put("horizon", result.horizon)
        .put("backend", result.backend.name.lowercase(Locale.ROOT))
        .put("graph", spec.name)
        .put("file", spec.file)
        .put("context_patches", spec.contextPatches)
        .put("horizon_patches", spec.horizonPatches)
        .put("rows", spec.rows)
        .put("mask", spec.mask.toDouble())
        .put("precision", result.precision.wireName)
        .put("threads", result.threads)
        .put("first_observed_patch", inputs[0].firstObservedPatch)
        .put("graph_ms", result.graphMs)
        .put("call_ms", JSONArray(result.callMs))
        .put("host_ms", result.hostMs)
        .put("compile_ms", result.compileMs ?: JSONObject.NULL)
        .put("input_sha256", calls)
        .put("npu_log", result.npu?.lines?.let { JSONArray(it) } ?: JSONObject.NULL)
        .put("time_ms", System.currentTimeMillis())
    File(directory, "meta.json").writeText(meta.toString(1))
  }

  companion object {
    /** Reads a little-endian float32 file. */
    fun readFloats(file: File): FloatArray {
      val bytes = file.readBytes()
      val floats = FloatArray(bytes.size / 4)
      ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
      return floats
    }

    private fun writeFloats(file: File, values: FloatArray) {
      file.writeBytes(bytes(values))
    }

    private fun bytes(values: FloatArray): ByteArray {
      val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
      buffer.asFloatBuffer().put(values)
      return buffer.array()
    }

    private fun sha256(values: FloatArray): String =
      MessageDigest.getInstance("SHA-256").digest(bytes(values)).joinToString("") {
        "%02x".format(it)
      }
  }
}
