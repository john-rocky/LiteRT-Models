package com.d1_2b

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File

/** The graphs' shared settings: the GPU at float32 with constant tensor sharing. */
object Graphs {
  const val PREFIX = "d1-2b"
  const val MULTI8 = "${PREFIX}_decide_multi8_fp16fc.tflite"
  const val SEES = 66

  private var environment: Environment? = null

  fun environment(): Environment = environment ?: Environment.create().also { environment = it }

  /**
   * The GPU at float32: the delegate's default (float16 activations) moves the answers outside the
   * checked tolerance. Constant tensor sharing keeps one copy of the weights on the GPU; without it
   * the 2B graph did not finish compiling on a Pixel 11 Pro.
   */
  fun gpuOptions(): CompiledModel.Options =
    CompiledModel.Options(Accelerator.GPU).apply {
      gpuOptions =
        CompiledModel.GpuOptions(
          constantTensorSharing = true,
          precision = CompiledModel.GpuOptions.Precision.FP32,
        )
    }

  /**
   * The two mask inputs of one sequence of `question.size` tokens on a graph of [length]:
   * `sees_q` = [is_state, 1 - is_state, one-hot of the question 1..64] per position, padding rows
   * [0, 1, 0, ...]; `sees_k` = real * [1, is_state, one-hot of the question].
   */
  fun sees(stateLength: Int, question: IntArray, length: Int): Pair<FloatArray, FloatArray> {
    require(question.size <= length) { "a sequence of ${question.size} tokens on a graph of $length" }
    val q = FloatArray(length * SEES)
    val k = FloatArray(length * SEES)
    for (t in 0 until length) q[t * SEES + 1] = 1f
    for (t in question.indices) {
      val state = t < stateLength
      require((question[t] == -1) == state) { "tags: -1 exactly on the state" }
      q[t * SEES] = if (state) 1f else 0f
      q[t * SEES + 1] = if (state) 0f else 1f
      k[t * SEES] = 1f
      k[t * SEES + 1] = if (state) 1f else 0f
      if (!state) {
        require(question[t] in 1..64) { "question tags 1..64" }
        q[t * SEES + 1 + question[t]] = 1f
        k[t * SEES + 1 + question[t]] = 1f
      }
    }
    return q to k
  }
}

/** One signature's compiled buffers, created on first use and kept. */
private class Signature(val model: CompiledModel, val name: String, inputs: List<String>, outputs: List<String>) : Closeable {
  val inputs = LinkedHashMap<String, TensorBuffer>()
  val outputs = LinkedHashMap<String, TensorBuffer>()

  init {
    try {
      inputs.forEach { this.inputs[it] = model.createInputBuffer(it, name) }
      outputs.forEach { this.outputs[it] = model.createOutputBuffer(it, name) }
    } catch (failure: Throwable) {
      close()
      throw failure
    }
  }

  fun run(feeds: Map<String, FloatArray>, output: String): FloatArray {
    for ((key, values) in feeds) inputs.getValue(key).writeFloat(values)
    model.run(inputs, outputs, name)
    // run() only enqueues the work on the GPU; reading the output waits for it.
    return outputs.getValue(output).readFloat()
  }

  override fun close() = (inputs.values + outputs.values).forEach { it.close() }
}

/**
 * The check before a graph is compiled: MemAvailable against what the stage was measured to take
 * from it. A stage that does not fit throws [NotEnoughMemoryException] and builds nothing.
 */
object Memory {
  /** MemAvailable the first load must leave on top of its own need (kept once, before anything loads). */
  const val RESERVE_MB = 800L

  /**
   * What a picture request takes beside a loaded decision graph (the tower's first compile and its
   * buffers), rounded up: MemAvailable 1.46 -> 0.71 GB on a Pixel 11 Pro.
   */
  const val TOWER_NEED_MB = 800L

  /** MemAvailable in kB (`/proc/meminfo`), or -1 when it cannot be read (the check then passes). */
  fun availableKb(): Long =
    runCatching {
        File("/proc/meminfo").useLines { lines ->
          lines.first { it.startsWith("MemAvailable:") }.trim().split(Regex("\\s+"))[1].toLong()
        }
      }
      .getOrDefault(-1L)

  /** Throws [NotEnoughMemoryException] when MemAvailable is under [neededMb]; [then] says what to do. */
  fun ensure(stage: String, neededMb: Long, then: String) {
    val available = availableKb()
    val neededKb = neededMb * 1024
    if (available in 0 until neededKb) throw NotEnoughMemoryException(stage, available, neededKb, then)
  }
}

/** A stage that would not fit: nothing was compiled. */
class NotEnoughMemoryException(stage: String, availableKb: Long, neededKb: Long, then: String) :
  RuntimeException(
    "Not enough memory for the $stage: ${(availableKb / 1048576.0).format(1)} GB available, " +
      "${(neededKb / 1048576.0).format(1)} GB needed; nothing was loaded. $then"
  )

/**
 * The decision graphs: the eight-signature file on one CompiledModel when it is in [dir] (16 GB
 * phones), else one bucket file at a time (12 GB phones: the next bucket closes the last).
 */
class LiteRtDecide(private val dir: File, private val maxTokens: Int) : DecideBackend, Closeable {
  private val multi = File(dir, Graphs.MULTI8).takeIf { it.isFile }
  private val files: Map<Int, File> =
    if (multi != null) emptyMap()
    else
      (dir.listFiles() ?: emptyArray())
        .mapNotNull { f -> BUCKET_FILE.matchEntire(f.name)?.let { it.groupValues[1].toInt() to f } }
        .groupBy({ it.first }, { it.second })
        .mapValues { (_, fs) -> fs.maxBy { it.name.length } } // the query-blocked file when both are there
  private val present: List<Int> = if (multi != null) MULTI8_BUCKETS else files.keys.sorted()
  private val buckets: List<Int> = present.filter { it <= maxTokens }
  private var model: CompiledModel? = null
  private var loadedBucket = 0
  private val signatures = HashMap<Int, Signature>()

  /** What [decide] last did, for the status line. */
  var lastBucket = 0
    private set
  override var lastLoadMs = 0.0
    private set

  val form: String
    get() = if (multi != null) "8 signatures in one file" else "one file per length"

  /** What compiling this form takes from MemAvailable ([MULTI8_NEED_MB] or [SINGLE_NEED_MB]). */
  val needMb: Long
    get() = if (multi != null) MULTI8_NEED_MB else SINGLE_NEED_MB

  override val maxLength: Int
    get() = buckets.lastOrNull() ?: 0

  override val limitNote: String
    get() = present.lastOrNull()?.takeIf { it > maxTokens }?.let { "; the $it-token graph needs a phone with 16 GB of memory" } ?: ""

  init {
    require(buckets.isNotEmpty()) { "No decision graph in ${dir.absolutePath}: see the README for the files to copy." }
  }

  fun bucketFor(tokens: Int): Int =
    buckets.firstOrNull { it >= tokens } ?: throw IllegalArgumentException("a sequence of $tokens tokens fits no graph")

  /** Compiles the graph that serves [bucket] (all of them for the eight-signature file). */
  fun prepare(bucket: Int) {
    lastLoadMs = 0.0
    if (multi != null) {
      if (model == null) {
        val start = System.nanoTime()
        model = CompiledModel.create(multi.absolutePath, Graphs.gpuOptions(), Graphs.environment())
        lastLoadMs = (System.nanoTime() - start) / 1e6
      }
      return
    }
    if (loadedBucket == bucket && model != null) return
    closeModel()
    val start = System.nanoTime()
    model = CompiledModel.create(files.getValue(bucket).absolutePath, Graphs.gpuOptions(), Graphs.environment())
    loadedBucket = bucket
    lastLoadMs = (System.nanoTime() - start) / 1e6
  }

  override fun decide(embeds: FloatArray, tokens: Int, stateLength: Int, question: IntArray): FloatArray {
    val length = bucketFor(tokens)
    prepare(length)
    val signature =
      signatures.getOrPut(length) {
        Signature(
          model!!,
          if (multi != null) MULTI8_SIGNATURES.getValue(length) else "decide",
          listOf("embeds", "sees_q", "sees_k"),
          listOf("hidden"),
        )
      }
    val padded = if (tokens * HIDDEN == embeds.size && tokens == length) embeds else FloatArray(length * HIDDEN).also { System.arraycopy(embeds, 0, it, 0, tokens * HIDDEN) }
    val (q, k) = Graphs.sees(stateLength, question, length)
    lastBucket = length
    return signature.run(mapOf("embeds" to padded, "sees_q" to q, "sees_k" to k), "hidden")
  }

  private fun closeModel() {
    signatures.values.forEach { it.close() }
    signatures.clear()
    model?.close()
    model = null
    loadedBucket = 0
  }

  override fun close() = closeModel()

  companion object {
    /**
     * What the load took from MemAvailable on the tested phones (before the load -> the lowest reading
     * after it), rounded up: the eight-signature file 8.5 GB on a Pixel 11 Pro (9.93 -> 1.44 GB); one per-length
     * file 5.5 GB on a Galaxy S26 (7.19 -> 1.70 GB for the 256-token file with the picture tower and a
     * picture request).
     */
    const val MULTI8_NEED_MB = 8_700L
    const val SINGLE_NEED_MB = 5_700L

    private val BUCKET_FILE = Regex("${Graphs.PREFIX}_decide_L(\\d+)(_qb\\d+)?_fp16fc\\.tflite")
    val MULTI8_SIGNATURES =
      mapOf(
        128 to "decide_L128",
        256 to "decide_L256",
        512 to "decide_L512",
        1024 to "decide_L1024",
        2048 to "decide_L2048",
        4096 to "decide_L4096_qb1024",
        6144 to "decide_L6144_qb1024",
      )
    val MULTI8_BUCKETS = MULTI8_SIGNATURES.keys.sorted()
  }
}

/** The picture tower: one file per canvas, compiled on first use; one kept at a time. */
class LiteRtTower(private val dir: File) : TowerBackend, Closeable {
  private val files: Map<Pair<Int, Int>, File> =
    (dir.listFiles() ?: emptyArray())
      .mapNotNull { f -> TOWER_FILE.matchEntire(f.name)?.let { (it.groupValues[1].toInt() to it.groupValues[2].toInt()) to f } }
      .toMap()
  private var loaded: Pair<Int, Int>? = null
  private var model: CompiledModel? = null
  private var signature: Signature? = null

  var lastCanvas: Pair<Int, Int>? = null
    private set
  override var lastLoadMs = 0.0
    private set

  val available: Boolean
    get() = files.isNotEmpty()

  /** Compiles the canvas [picture] needs unless it is the one loaded; [lastLoadMs] = the compile time (0 if none). */
  fun prepare(picture: Picture) {
    val canvas = Picture.canvasFor(picture, files.keys.toList())
    lastLoadMs = 0.0
    if (loaded == canvas) return
    close()
    Memory.ensure("picture tower", Memory.TOWER_NEED_MB, "Close other apps and press Decide again.")
    val start = System.nanoTime()
    val compiled = CompiledModel.create(files.getValue(canvas).absolutePath, Graphs.gpuOptions(), Graphs.environment())
    model = compiled
    signature = Signature(compiled, "tower", listOf("patches", "valid"), listOf("features"))
    loaded = canvas
    lastLoadMs = (System.nanoTime() - start) / 1e6
  }

  override fun tower(picture: Picture): FloatArray {
    prepare(picture)
    val canvas = loaded!!
    val (patches, valid) = picture.canvas(canvas.first, canvas.second)
    val features = signature!!.run(mapOf("patches" to patches, "valid" to valid), "features")
    lastCanvas = canvas
    return picture.fromCanvas(features, canvas.first, canvas.second)
  }

  override fun close() {
    signature?.close()
    signature = null
    model?.close()
    model = null
    loaded = null
  }

  companion object {
    private val TOWER_FILE = Regex("${Graphs.PREFIX}_tower_G(\\d+)x(\\d+)(?:_qb\\d+)?_fp16fc\\.tflite")
  }
}
