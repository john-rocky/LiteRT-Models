package com.d1_omni_2b

import android.content.Context
import android.os.Build
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File

/**
 * The accelerators of this process: one LiteRT Environment, and whether the Qualcomm NPU is usable. The NPU needs
 * Qualcomm's libraries in the APK (copied into src/main/jniLibs before the build, never committed) and a Qualcomm
 * phone; the decision files in model/npu/sm8850 are compiled ahead of time for the SM8850 (Galaxy S26), so no
 * compiler plugin is loaded.
 */
class Accelerators private constructor(val env: Environment, val npu: Boolean, val note: String) : Closeable {
  override fun close() = env.close()

  companion object {
    /** What an ahead-of-time compiled graph needs: LiteRT's dispatch library and QAIRT's HTP runtime for Hexagon v81. */
    val NPU_LIBS =
      listOf("libLiteRtDispatch_Qualcomm.so", "libQnnHtp.so", "libQnnSystem.so", "libQnnHtpV81Stub.so", "libQnnHtpV81CalculatorStub.so", "libQnnHtpV81Skel.so")

    /** A Qualcomm phone: only there are Qualcomm's libraries opened. */
    private fun qualcomm(): Boolean {
      val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else ""
      return soc.contains("qti", ignoreCase = true) || soc.contains("qualcomm", ignoreCase = true) ||
        (Build.VERSION.SDK_INT < 31 && Build.HARDWARE.contains("qcom", ignoreCase = true))
    }

    fun create(context: Context): Accelerators {
      val libDir = context.applicationInfo.nativeLibraryDir
      val missing = NPU_LIBS.filter { !File(libDir, it).exists() }
      val qualcomm = qualcomm()
      val env =
        if (missing.isEmpty() && qualcomm) Environment.create(mapOf(Environment.Option.DispatchLibraryDir to libDir))
        else Environment.create()
      val available = runCatching { env.getAvailableAccelerators() }.getOrDefault(emptySet())
      val npu = missing.isEmpty() && qualcomm && Accelerator.NPU in available
      val note =
        when {
          !qualcomm -> "this phone has no Qualcomm NPU"
          missing.size == NPU_LIBS.size -> "this APK was built without Qualcomm's NPU libraries"
          missing.isNotEmpty() -> "this APK lacks ${missing.joinToString()}"
          !npu -> "LiteRT did not register this phone's NPU"
          else -> "NPU ready"
        }
      Log.i(TAG, "accelerators $available, npu=$npu ($note), soc " +
        (if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE))
      return Accelerators(env, npu, note)
    }

    const val TAG = "D1OmniDecide"
  }
}

/** The graphs' shared settings: the GPU at float32 with constant tensor sharing; the Qualcomm options when the NPU is in use. */
object Graphs {
  const val PREFIX = "d1-omni-2b"
  const val MULTI8 = "${PREFIX}_decide_multi8_fp16fc.tflite"
  const val NPU_DIR = "npu/sm8850"
  const val TOWER = "${PREFIX}_tower_G24x24_fp16fc.tflite"
  const val SEES = 66

  /**
   * The GPU at float32: the delegate's default (float16 activations) moves the answers outside the checked
   * tolerance. Constant tensor sharing keeps one copy of the weights on the GPU. When the NPU is usable every graph
   * carries the Qualcomm options, because the first model a process loads fixes the NPU's dispatch options.
   */
  fun gpuOptions(acc: Accelerators): CompiledModel.Options =
    CompiledModel.Options(Accelerator.GPU).apply {
      gpuOptions = CompiledModel.GpuOptions(constantTensorSharing = true, precision = CompiledModel.GpuOptions.Precision.FP32)
      if (acc.npu) qualcommOptions = burst()
    }

  /** An ahead-of-time NPU graph: the dispatch library runs the whole graph on the HTP (no CPU fallback). */
  fun npuOptions(): CompiledModel.Options = CompiledModel.Options(Accelerator.NPU).apply { qualcommOptions = burst() }

  // Without BURST the HTP runs at its default clock, several times slower.
  private fun burst() = CompiledModel.QualcommOptions(htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST)

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

/** One signature's compiled buffers, created on first use and kept; inputs and outputs are bound by name. */
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
  /**
   * MemAvailable a decision graph's load must leave on top of its own need: at the first load, and when the Galaxy S26
   * moves from the NPU to a GPU file. Without it, that move after a run with clips and a photo (5.9 GB free at the
   * check, 5,700 MiB needed) took MemAvailable down to 1.2 GB during the load, and the system then killed the app with
   * no other process left to kill.
   */
  const val RESERVE_MB = 800L

  /**
   * What a picture request takes beside a loaded decision graph (the tower's first compile and its
   * buffers), rounded up: MemAvailable 1.46 -> 0.71 GB on a Pixel 11 Pro with the d1-2B tower of the same shape.
   */
  const val TOWER_NEED_MB = 800L

  /**
   * What an audio graph's first compile and its buffers took from MemAvailable in this app on a Galaxy S26, rounded
   * up: 311 MiB for the 504-frame file and 485 MiB for the 2,000-frame file.
   */
  const val AUDIO_NEED_MB = 500L

  /**
   * How long a check waits for MemAvailable to come back after a graph was closed: on a Galaxy S26 the NPU file's
   * memory returned within about a second of its close (4.0 GB right after it, 6.8 GB 0.6 s later).
   */
  const val SETTLE_MS = 3_000L

  /**
   * The graphs this app keeps compiled for later requests (the audio files used last, the picture tower), by name:
   * each entry closes what it keeps and says whether anything was open. A check that falls short closes them before it
   * refuses: on a Pixel 11 Pro the eight-signature decision file left 1.2 to 1.7 GB, and after clips of several
   * lengths and a photo the kept audio files and the tower had taken it down to 0.3 GB, under a 3,000-frame audio
   * file's need (the same request in a fresh start left 1.3 GB).
   */
  private val kept = LinkedHashMap<String, () -> Boolean>()

  /** Registers [release] under [name]; a newer backend replaces an older one of the same name. */
  @Synchronized
  fun keep(name: String, release: () -> Boolean) {
    kept[name] = release
  }

  /** MemAvailable in kB (`/proc/meminfo`), or -1 when it cannot be read (the check then passes). */
  fun availableKb(): Long =
    runCatching {
        File("/proc/meminfo").useLines { lines ->
          lines.first { it.startsWith("MemAvailable:") }.trim().split(Regex("\\s+"))[1].toLong()
        }
      }
      .getOrDefault(-1L)

  /**
   * Throws [NotEnoughMemoryException] when MemAvailable is under [neededMb]; [then] says what to do. With [settleMs], a
   * reading under the need is taken again every 100 ms for up to that long (right after a graph is closed, its memory
   * is not back yet). A reading still under the need first closes the graphs kept for later requests ([keep]) and, when
   * one was open, reads again for up to [SETTLE_MS] before the check fails.
   */
  fun ensure(stage: String, neededMb: Long, then: String, settleMs: Long = 0) {
    val neededKb = neededMb * 1024
    var available = settle(neededKb, settleMs)
    if (available in 0 until neededKb && releaseKept()) available = settle(neededKb, SETTLE_MS)
    if (available in 0 until neededKb) throw NotEnoughMemoryException(stage, available, neededKb, then)
  }

  /** Closes every kept graph; true when any was open. */
  @Synchronized
  private fun releaseKept(): Boolean = kept.values.map { it() }.any { it }

  /** MemAvailable, read again every 100 ms while it is under [neededKb], for up to [settleMs]. */
  private fun settle(neededKb: Long, settleMs: Long): Long {
    var available = availableKb()
    val deadline = System.nanoTime() + settleMs * 1_000_000L
    while (available in 0 until neededKb && System.nanoTime() < deadline) {
      Thread.sleep(100)
      available = availableKb()
    }
    return available
  }
}

/** A stage that would not fit: nothing was compiled. */
class NotEnoughMemoryException(stage: String, availableKb: Long, neededKb: Long, then: String) :
  RuntimeException(
    "Not enough memory for the $stage: ${(availableKb / 1048576.0).format(1)} GB available, " +
      "${(neededKb / 1048576.0).format(1)} GB needed; nothing was loaded. $then"
  )

/** A decision backend the app reports on: which graph and accelerator the last call used. */
interface PhoneDecideBackend : DecideBackend, Closeable {
  val lastBucket: Int
  val lastAccelerator: String
  /** What compiling the largest stage takes from MemAvailable. */
  val needMb: Long
  /** The form for the status line. */
  val form: String

  /** Compiles the graph that serves [bucket] (all of them for the eight-signature file). */
  fun prepare(bucket: Int)

  fun bucketFor(tokens: Int): Int
}

/**
 * The decision graphs on the GPU: the eight-signature file on one CompiledModel when it is in [dir] (16 GB
 * phones), else one bucket file at a time (12 GB phones: the next bucket closes the last).
 */
class LiteRtDecide(private val dir: File, private val maxTokens: Int, private val acc: Accelerators) : PhoneDecideBackend {
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

  override var lastBucket = 0
    private set
  override val lastAccelerator = "GPU"
  override var lastLoadMs = 0.0
    private set

  override val form: String
    get() = if (multi != null) "8 signatures in one file" else "one file per length"

  override val needMb: Long
    get() = if (multi != null) MULTI8_NEED_MB else SINGLE_NEED_MB

  /** Whether any decision file for the GPU is in the folder. */
  val available: Boolean
    get() = buckets.isNotEmpty()

  val loaded: Boolean
    get() = model != null

  override val maxLength: Int
    get() = buckets.lastOrNull() ?: 0

  override val limitNote: String
    get() = present.lastOrNull()?.takeIf { it > maxTokens }?.let { "; the $it-token graph needs a phone with 16 GB of memory" } ?: ""

  override fun bucketFor(tokens: Int): Int =
    buckets.firstOrNull { it >= tokens } ?: throw IllegalArgumentException("a sequence of $tokens tokens fits no graph")

  override fun prepare(bucket: Int) {
    lastLoadMs = 0.0
    if (multi != null) {
      if (model == null) {
        val start = System.nanoTime()
        model = CompiledModel.create(multi.absolutePath, Graphs.gpuOptions(acc), acc.env)
        lastLoadMs = (System.nanoTime() - start) / 1e6
      }
      return
    }
    if (loadedBucket == bucket && model != null) return
    closeModel()
    val start = System.nanoTime()
    model = CompiledModel.create(files.getValue(bucket).absolutePath, Graphs.gpuOptions(acc), acc.env)
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

  fun closeModel() {
    signatures.values.forEach { it.close() }
    signatures.clear()
    model?.close()
    model = null
    loadedBucket = 0
  }

  override fun close() = closeModel()

  companion object {
    /**
     * What the load took from MemAvailable (before the load -> the lowest reading after it), rounded up: this app's
     * load of the eight-signature file on a Pixel 11 Pro (the tokenizer, the table and the file, up to the first
     * request) took 8,631, 8,758 and 8,660 MiB in three loads; one per-length file 5.5 GB on a Galaxy S26 with the
     * d1-2B files of the same graphs (7.19 -> 1.70 GB for the 256-token file with the picture tower and a picture
     * request), and 5,414 MiB for the 1,024-token file in this app on the same phone.
     */
    const val MULTI8_NEED_MB = 8_800L
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

/**
 * The decision graphs compiled ahead of time for the SM8850's NPU (model/npu/sm8850: 128, 256 and 512 tokens), one
 * file compiled at a time, run through LiteRT's dispatch library. The NPU computes in float16.
 */
class NpuDecide(dir: File, private val acc: Accelerators) : Closeable {
  private val files: Map<Int, File> =
    (File(dir, Graphs.NPU_DIR).listFiles() ?: emptyArray())
      .mapNotNull { f -> NPU_FILE.matchEntire(f.name)?.let { it.groupValues[1].toInt() to f } }
      .toMap()
  val buckets: List<Int> = files.keys.sorted()
  private var model: CompiledModel? = null
  private var signature: Signature? = null
  private var loadedBucket = 0
  var lastLoadMs = 0.0
    private set

  val available: Boolean
    get() = buckets.isNotEmpty()

  val maxLength: Int
    get() = buckets.lastOrNull() ?: 0

  val loaded: Boolean
    get() = model != null

  fun bucketFor(tokens: Int): Int = buckets.first { it >= tokens }

  fun prepare(bucket: Int) {
    lastLoadMs = 0.0
    if (loadedBucket == bucket && model != null) return
    closeModel()
    val start = System.nanoTime()
    val compiled = CompiledModel.create(files.getValue(bucket).absolutePath, Graphs.npuOptions(), acc.env)
    try {
      signature = Signature(compiled, "decide", listOf("embeds", "sees_q", "sees_k"), listOf("hidden"))
    } catch (failure: Throwable) {
      compiled.close()
      throw failure
    }
    model = compiled
    loadedBucket = bucket
    lastLoadMs = (System.nanoTime() - start) / 1e6
  }

  fun decide(embeds: FloatArray, tokens: Int, stateLength: Int, question: IntArray): FloatArray {
    val length = bucketFor(tokens)
    prepare(length)
    val padded = FloatArray(length * HIDDEN).also { System.arraycopy(embeds, 0, it, 0, tokens * HIDDEN) }
    val (q, k) = Graphs.sees(stateLength, question, length)
    return signature!!.run(mapOf("embeds" to padded, "sees_q" to q, "sees_k" to k), "hidden")
  }

  fun closeModel() {
    signature?.close()
    signature = null
    model?.close()
    model = null
    loadedBucket = 0
  }

  override fun close() = closeModel()

  companion object {
    /**
     * What this app's load with an NPU file took from MemAvailable on a Galaxy S26 (the tokenizer, the table and the
     * 256-token NPU file; before the load -> the lowest reading before the first request), rounded up: 2,931 and
     * 3,064 MiB in two loads (6.98 -> 4.12 GB and 7.50 -> 4.50 GB).
     */
    const val NEED_MB = 3_100L

    private val NPU_FILE = Regex("${Graphs.PREFIX}_decide_L(\\d+)_fp16fc_SM8850\\.tflite")
  }
}

/**
 * The Galaxy S26's decision backend: the NPU files up to their longest (512 tokens), the GPU per-length files for a
 * longer sequence (the NPU file is closed first, and MemAvailable is checked before the GPU file compiles).
 */
class NpuThenGpuDecide(private val npu: NpuDecide, private val gpu: LiteRtDecide?) : PhoneDecideBackend {
  override var lastBucket = 0
    private set
  override var lastAccelerator = "NPU"
    private set
  override var lastLoadMs = 0.0
    private set

  override val form: String
    get() = "NPU up to ${npu.maxLength} tokens" + if (gpu?.available == true) ", GPU above" else ""

  override val needMb: Long = NpuDecide.NEED_MB

  override val maxLength: Int
    get() = maxOf(npu.maxLength, if (gpu?.available == true) gpu.maxLength else 0)

  override val limitNote: String
    get() =
      if (gpu?.available == true) gpu.limitNote
      else "; a longer sequence needs the GPU files ${Graphs.PREFIX}_decide_L1024_fp16fc.tflite and up (README.md)"

  override fun bucketFor(tokens: Int): Int = if (tokens <= npu.maxLength) npu.bucketFor(tokens) else gpuFor(tokens).bucketFor(tokens)

  override fun prepare(bucket: Int) {
    if (bucket <= npu.maxLength) {
      npu.prepare(bucket)
      lastLoadMs = npu.lastLoadMs
    } else {
      val g = gpuFor(bucket)
      g.prepare(bucket)
      lastLoadMs = g.lastLoadMs
    }
  }

  private fun gpuFor(tokens: Int): LiteRtDecide =
    gpu?.takeIf { it.available } ?: throw IllegalArgumentException("a sequence of $tokens tokens needs the GPU decision files")

  override fun decide(embeds: FloatArray, tokens: Int, stateLength: Int, question: IntArray): FloatArray {
    if (tokens <= npu.maxLength) {
      gpu?.closeModel()
      val out = npu.decide(embeds, tokens, stateLength, question)
      lastLoadMs = npu.lastLoadMs
      lastBucket = npu.bucketFor(tokens)
      lastAccelerator = "NPU"
      return out
    }
    val g = gpuFor(tokens)
    val npuWasLoaded = npu.loaded
    npu.closeModel()
    if (!g.loaded) {
      Memory.ensure(
        "decision graph on the GPU (${g.bucketFor(tokens)} tokens)",
        g.needMb + Memory.RESERVE_MB,
        "Close other apps and press Decide again.",
        if (npuWasLoaded) Memory.SETTLE_MS else 0,
      )
    }
    val out = g.decide(embeds, tokens, stateLength, question)
    lastLoadMs = g.lastLoadMs
    lastBucket = g.lastBucket
    lastAccelerator = "GPU"
    return out
  }

  override fun close() {
    npu.close()
    gpu?.close()
  }
}

/** The phone's decision backend from the files in [dir]: the NPU where it is usable and its files are there, else the GPU. */
object DecideBackends {
  /** The GPU decision files (every phone without a usable NPU, and the NPU's fallback). */
  fun gpu(dir: File, maxTokens: Int, acc: Accelerators): LiteRtDecide = LiteRtDecide(dir, maxTokens, acc)

  fun npu(dir: File, maxTokens: Int, acc: Accelerators): NpuThenGpuDecide? {
    if (!acc.npu) return null
    val npu = NpuDecide(dir, acc)
    if (!npu.available) return null
    val gpu = LiteRtDecide(dir, maxTokens, acc).takeIf { it.available }
    return NpuThenGpuDecide(npu, gpu)
  }
}

/** The picture tower: one file per canvas, compiled on first use; one kept at a time. */
class LiteRtTower(private val dir: File, private val acc: Accelerators) : TowerBackend, Closeable {
  private val files: Map<Pair<Int, Int>, File> =
    (dir.listFiles() ?: emptyArray())
      .mapNotNull { f -> TOWER_FILE.matchEntire(f.name)?.let { (it.groupValues[1].toInt() to it.groupValues[2].toInt()) to f } }
      .toMap()
  private var loaded: Pair<Int, Int>? = null
  private var model: CompiledModel? = null
  private var signature: Signature? = null

  init {
    Memory.keep("picture tower") { (model != null).also { close() } }
  }

  var lastCanvas: Pair<Int, Int>? = null
    private set
  override var lastLoadMs = 0.0
    private set

  val available: Boolean
    get() = files.isNotEmpty()

  override fun tower(picture: Picture): FloatArray {
    require(files.isNotEmpty()) {
      "Missing model file ${Graphs.TOWER} in ${dir.absolutePath}: a request with a photo needs it (README.md); nothing was loaded."
    }
    val canvas = Picture.canvasFor(picture, files.keys.toList())
    lastLoadMs = 0.0
    if (loaded != canvas) {
      close()
      Memory.ensure("picture tower", Memory.TOWER_NEED_MB, "Close other apps and press Decide again.")
      val start = System.nanoTime()
      val compiled = CompiledModel.create(files.getValue(canvas).absolutePath, Graphs.gpuOptions(acc), acc.env)
      model = compiled
      signature = Signature(compiled, "tower", listOf("patches", "valid"), listOf("features"))
      loaded = canvas
      lastLoadMs = (System.nanoTime() - start) / 1e6
    }
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

/**
 * The audio encoder: one file per length (504, 1,000, 2,000 and 3,000 mel frames), each clip on the shortest file
 * that holds it (mel [1, T, 128] zeros past the clip, valid [1, T, 1, 1] ones on its frames; its tokens are the first
 * ceil(frames / 8) rows of `features`). GPU at float32; the two files used last stay compiled.
 */
class LiteRtAudio(private val dir: File, private val acc: Accelerators) : AudioBackend, Closeable {
  private class Loaded(val model: CompiledModel, val signature: Signature) : Closeable {
    override fun close() {
      signature.close()
      model.close()
    }
  }

  private val loaded = LinkedHashMap<Int, Loaded>(4, 0.75f, true)

  init {
    Memory.keep("audio encoder") { loaded.isNotEmpty().also { close() } }
  }

  override val maxFrames: Int = BUCKETS.last()
  override var lastLoadMs = 0.0
    private set
  var lastBucket = 0
    private set

  fun file(bucket: Int) = File(dir, "${Graphs.PREFIX}_audio_T${bucket}_fp16fc.tflite")

  /** The files of the buckets missing from the folder. */
  val missing: List<String>
    get() = BUCKETS.map { file(it) }.filter { !it.isFile }.map { it.name }

  fun bucketFor(frames: Int): Int = BUCKETS.first { it >= frames }

  override fun audio(features: FloatArray, frames: Int): FloatArray {
    val bucket = bucketFor(frames)
    val f = file(bucket)
    require(f.isFile) {
      "Missing model file ${f.name} in ${dir.absolutePath}: clips of up to ${bucket} frames need it (README.md); nothing was loaded."
    }
    lastLoadMs = 0.0
    val graph =
      loaded[bucket] ?: run {
        if (loaded.size >= KEEP) loaded.remove(loaded.keys.first())?.close()
        Memory.ensure("audio encoder", Memory.AUDIO_NEED_MB, "Close other apps and press Decide again.")
        val start = System.nanoTime()
        val model = CompiledModel.create(f.absolutePath, Graphs.gpuOptions(acc), acc.env)
        val signature =
          try {
            Signature(model, "audio", listOf("mel", "valid"), listOf("features"))
          } catch (failure: Throwable) {
            model.close()
            throw failure
          }
        lastLoadMs = (System.nanoTime() - start) / 1e6
        Loaded(model, signature).also { loaded[bucket] = it }
      }
    val mel = FloatArray(bucket * MELS)
    for (m in 0 until MELS) for (t in 0 until frames) mel[t * MELS + m] = features[m * frames + t]
    val valid = FloatArray(bucket).also { java.util.Arrays.fill(it, 0, frames, 1f) }
    val out = graph.signature.run(mapOf("mel" to mel, "valid" to valid), "features")
    lastBucket = bucket
    val tokens = (frames + 7) / 8
    return out.copyOf(tokens * HIDDEN)
  }

  override fun close() {
    loaded.values.forEach { it.close() }
    loaded.clear()
  }

  companion object {
    val BUCKETS = listOf(504, 1000, 2000, 3000)
    const val MELS = 128
    /** Audio files kept compiled: a request's state clip and its spoken questions often need two lengths. */
    const val KEEP = 2
  }
}
