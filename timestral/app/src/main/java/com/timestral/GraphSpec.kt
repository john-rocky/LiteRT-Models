package com.timestral

/** Where a forecast runs. */
enum class Backend(val label: String) {
  CPU("CPU"),
  GPU("GPU"),
  NPU("NPU"),
}

/** The GPU precision of a graph (the CPU and the NPU ignore it). */
enum class GpuPrecision(val label: String, val wireName: String) {
  FP32("FP32", "fp32"),
  FP16("FP16", "fp16"),
  FP16_FP32_ACCUM("FP16, FP32 accumulation", "fp16acc32");

  companion object {
    fun of(name: String): GpuPrecision =
      entries.firstOrNull { it.wireName == name }
        ?: throw IllegalArgumentException("Unknown GPU precision \"$name\"")
  }
}

/**
 * One installed graph file: its shape ([contextPatches] L, [horizonPatches] H, and [rows] R per
 * call: 2 = the context and its negation in one call, 1 = one call for each), the additive
 * attention mask value its host sends (the exact forms take -1e30; the forms with the float16-safe
 * attention take -3e4, whose float16 value stays finite), the backends that run it, and the GPU
 * precision it needs when that differs from the backend's.
 */
data class GraphSpec(
  val name: String,
  val file: String,
  val contextPatches: Int,
  val horizonPatches: Int,
  val rows: Int,
  val mask: Float,
  val backends: Set<Backend>,
  val gpuPrecision: GpuPrecision? = null,
) {
  init {
    require(rows == 1 || rows == 2) { "Graph $name has $rows rows; 1 or 2 are supported" }
  }

  /** The shape the host builds inputs for. */
  val bucket: Bucket
    get() = Bucket(name, contextPatches, horizonPatches)

  /** Tokens of the graph (S = L + H): what a call costs. */
  val tokens: Int
    get() = contextPatches + horizonPatches

  /** Graph calls per forecast. */
  val calls: Int
    get() = TimestralHost.ROWS / rows

  companion object {
    /**
     * The graph of [specs] that [backend] runs with the fewest tokens among those that hold
     * [contextPatches] and [horizonPatches] (the first listed among equals), or null.
     */
    fun choose(
      specs: List<GraphSpec>,
      backend: Backend,
      contextPatches: Int,
      horizonPatches: Int,
    ): GraphSpec? =
      specs
        .filter {
          backend in it.backends &&
            contextPatches <= it.contextPatches &&
            horizonPatches <= it.horizonPatches
        }
        .minByOrNull { it.tokens }
  }
}
