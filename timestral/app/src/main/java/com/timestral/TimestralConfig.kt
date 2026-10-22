package com.timestral

import org.json.JSONObject

/** How one backend runs its graphs: the CPU threads and the GPU precision. */
class BackendConfig(val backend: Backend, val threads: Int, val precision: GpuPrecision) {
  /** The GPU precision of [spec] on this backend: the graph's own when it names one. */
  fun precisionOf(spec: GraphSpec): GpuPrecision = spec.gpuPrecision ?: precision
}

/**
 * The graph files and the backends' settings, read from `assets/timestral_forms.json`: each graph
 * names its shape, mask value and the backends that run it, so the forms the device measurements
 * pick can replace the defaults without a code change. A forecast takes the graph of its backend
 * with the fewest tokens that holds its context and horizon ([GraphSpec.choose]).
 */
class TimestralConfig(
  val model: String,
  val forecastQueryFile: String,
  val ropeCosFile: String,
  val ropeSinFile: String,
  val ropeFirstPosition: Int,
  val graphs: List<GraphSpec>,
  val backends: Map<Backend, BackendConfig>,
) {
  /** The host tables, then every graph file, each once. */
  val requiredFiles: List<String>
    get() = (listOf(forecastQueryFile, ropeCosFile, ropeSinFile) + graphs.map { it.file }).distinct()

  companion object {
    /** The configuration asset. */
    const val ASSET = "timestral_forms.json"

    private const val DEFAULT_THREADS = 4

    /** Reads the configuration; throws [IllegalArgumentException] when a field is missing or wrong. */
    fun parse(text: String): TimestralConfig {
      val root = JSONObject(text)
      val tables = root.getJSONObject("host_tables")
      val backendObject = root.getJSONObject("backends")
      val backends =
        Backend.entries
          .filter { backendObject.has(it.name.lowercase()) }
          .associateWith { backend ->
            val entry = backendObject.getJSONObject(backend.name.lowercase())
            BackendConfig(
              backend,
              entry.optInt("threads", DEFAULT_THREADS),
              GpuPrecision.of(entry.optString("precision", GpuPrecision.FP32.wireName)),
            )
          }
      val graphArray = root.getJSONArray("graphs")
      val graphs =
        (0 until graphArray.length()).map { index ->
          val graph = graphArray.getJSONObject(index)
          val names = graph.getJSONArray("backends")
          val runOn =
            (0 until names.length())
              .map { i ->
                val name = names.getString(i)
                Backend.entries.firstOrNull { it.name.equals(name, ignoreCase = true) }
                  ?: throw IllegalArgumentException("Unknown backend \"$name\"")
              }
              .toSet()
          GraphSpec(
            graph.getString("name"),
            graph.getString("file"),
            graph.getInt("context_patches"),
            graph.getInt("horizon_patches"),
            graph.getInt("rows"),
            graph.getDouble("mask").toFloat(),
            runOn,
            graph.optString("gpu_precision").takeIf { it.isNotEmpty() }?.let { GpuPrecision.of(it) },
          )
        }
      for (backend in backends.keys) {
        require(graphs.any { backend in it.backends }) { "No graph runs on ${backend.label}" }
      }
      return TimestralConfig(
        root.getString("model"),
        tables.getString("forecast_query"),
        tables.getString("rope_cos"),
        tables.getString("rope_sin"),
        tables.getInt("rope_first_position"),
        graphs,
        backends,
      )
    }
  }
}
