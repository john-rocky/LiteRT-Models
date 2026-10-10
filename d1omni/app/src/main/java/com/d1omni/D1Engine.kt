package com.d1omni

import android.content.Context
import java.io.Closeable
import java.io.File

/**
 * One graph compile: the bucket, where it was asked for and where it runs, its time, the GPU's
 * error after a fallback, and the memory right before and after it ([D1Device.memory]); [graph] is
 * `decide` (bucket L) or `audio` (bucket T, [residentBefore] = the decision graphs compiled then).
 */
class D1Compile(
  val bucket: Int,
  val requested: D1Backend,
  val backend: D1Backend,
  val precision: D1Precision,
  val compileMs: Double,
  val gpuFailure: String?,
  val residentBefore: List<Int>,
  val memoryBefore: Map<String, Any?>,
  val memoryAfter: Map<String, Any?>,
  val graph: String = "decide",
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "bucket" to bucket,
      "requested" to requested.wireName,
      "backend" to backend.wireName,
      "precision" to if (backend == D1Backend.GPU) precision.wireName else null,
      "compile_ms" to compileMs,
      "gpu_failure" to gpuFailure,
      "resident_before" to residentBefore,
      "memory_before" to memoryBefore,
      "memory_after" to memoryAfter,
      "graph" to graph,
    )
}

/**
 * The tokenizer, the contract, the compiled decision graphs ([D1Residency]) and the audio graph
 * ([audio], at [audioPrecision]). Loading reads `contract.json` and `tokenizer.json` from `files/`
 * (the tokenizer's sha256 and token IDs checked against the contract); graphs compile when a request
 * needs them, on [backend] at [precision], and fall back to the CPU (four threads) when the GPU
 * cannot compile or run one. Use only on [D1Runtime.dispatcher].
 */
class D1Engine
private constructor(
  private val context: Context,
  val tokenizer: D1Tokenizer,
  val contract: D1Contract,
  /** Wall time of reading and checking contract.json and tokenizer.json, in milliseconds. */
  val loadMs: Double,
  val backend: D1Backend,
  val precision: D1Precision,
  /** The GPU precision of the audio graph. */
  val audioPrecision: D1Precision,
) : Closeable {
  private val graphs = D1Residency<D1Decider>()

  /** Every compile of this engine (decision and audio graphs), in order. */
  val compiles = ArrayList<D1Compile>()

  /** The audio graph: one bucket compiled at a time, next to the decision graphs. */
  val audio = D1AudioEngine(context, contract, backend, audioPrecision, compiles) { graphs.resident }

  /** The decision buckets whose file is in `files/` with the contract's size, ascending. */
  val installed: List<Int> =
    contract.decisionFiles.filter { (_, entry) -> file(entry.name).length() == entry.bytes }.keys
      .sorted()

  /** The buckets the contract lists that are not installed. */
  val missing: List<Int>
    get() = contract.buckets.filter { it !in installed }

  /** The resident buckets, ascending. */
  val resident: List<Int>
    get() = graphs.resident

  /** Where each resident graph runs, in the order of [resident]. */
  val residentBackends: List<D1Backend>
    get() = graphs.all.map { it.backend }

  /** The rows of a text request (`D1Omni.rows`, kind text). */
  fun textRows(request: D1Request): List<D1Row> =
    D1Rows.rows(tokenizer, contract, request.state, request.questions.values.toList(), 0, D1Kind.TEXT)

  /** The rows of an audio request after its prefix (`D1Omni.rows`, kind audio: a null state is {}). */
  fun audioRows(request: D1Request, prefix: D1AudioPrefix): List<D1Row> =
    D1Rows.rows(
      tokenizer,
      contract,
      request.state,
      request.questions.values.toList(),
      prefix.info.prefixRows,
      D1Kind.AUDIO,
    )

  /** Compiles what rows of [positions] need (see [D1Residency.prepare]). */
  fun prepare(positions: List<Int>): D1Prepared =
    graphs.prepare(positions, installed, { D1Device.availableMemoryBytes(context) }, ::open)

  /** Makes [target] resident (see [D1Residency.ensure]); the rows of [positions] are assigned. */
  fun ensure(target: List<Int>, positions: List<Int>): D1Prepared {
    for (bucket in target) {
      require(bucket in installed) { "L$bucket is not installed (${fileOf(bucket)})" }
    }
    return graphs.ensure(target, positions, { D1Device.availableMemoryBytes(context) }, ::open)
  }

  /** The resident graph of [bucket]. */
  fun graph(bucket: Int): D1Decider = graphs.graph(bucket)

  /**
   * One call of the [bucket] graph. A GPU graph that fails to run is closed and compiled on the
   * CPU, and the call is made there (`GPU_FALLBACK` in logcat).
   */
  fun call(bucket: Int, inputs: D1Inputs): D1Call {
    val graph = graphs.graph(bucket)
    return try {
      graph.run(inputs)
    } catch (failure: Exception) {
      if (graph.backend != D1Backend.GPU) throw failure
      val reason = "run L$bucket: ${D1Decider.describe(failure)}"
      D1Demo.gpuFallback(reason)
      graphs.close(bucket)
      graphs.ensure(listOf(bucket), listOf(inputs.length), { Long.MAX_VALUE }) {
        compile(it, D1Backend.CPU, reason)
      }
      graphs.graph(bucket).run(inputs)
    }
  }

  /** The file of [bucket] in `files/`. */
  fun fileOf(bucket: Int): String =
    requireNotNull(contract.decisionFiles[bucket]) { "contract.json has no L$bucket" }.name

  override fun close() {
    try {
      graphs.close()
    } finally {
      audio.close()
    }
  }

  private fun file(name: String) = File(context.filesDir, name)

  private fun open(bucket: Int): D1Decider = compile(bucket, backend, null)

  private fun compile(bucket: Int, target: D1Backend, earlierFailure: String?): D1Decider {
    val before = D1Device.memory(context)
    val residentBefore = graphs.resident
    val graph =
      D1Decider.create(
        context,
        file(fileOf(bucket)),
        bucket,
        target,
        precision,
        fallback = target == D1Backend.GPU,
      )
    val failure = graph.gpuFailure ?: earlierFailure
    if (graph.gpuFailure != null) D1Demo.gpuFallback("compile L$bucket: ${graph.gpuFailure}")
    compiles.add(
      D1Compile(
        bucket,
        target,
        graph.backend,
        precision,
        graph.compileMs,
        failure,
        residentBefore,
        before,
        D1Device.memory(context),
      )
    )
    return graph
  }

  companion object {
    /**
     * Reads contract.json and tokenizer.json from `files/`, checks the tokenizer's sha256 and token
     * IDs against the contract; graphs compile later, when a request needs them.
     */
    fun load(
      context: Context,
      backend: D1Backend,
      precision: D1Precision,
      audioPrecision: D1Precision = D1AudioEngine.DEFAULT_PRECISION,
    ): D1Engine {
      val files = context.filesDir
      val start = System.nanoTime()
      val contractFile = File(files, D1Contract.FILE)
      check(contractFile.isFile) { "Missing ${D1Contract.FILE}" }
      val contract = D1Contract.read(contractFile)
      val tokenizerFile = File(files, contract.tokenizerFile)
      check(tokenizerFile.isFile) { "Missing ${contract.tokenizerFile}" }
      check(D1Contract.sha256(tokenizerFile) == contract.tokenizerSha256) {
        "${contract.tokenizerFile} is not the tokenizer of contract.json (sha256 differs)"
      }
      val tokenizer = D1Tokenizer(tokenizerFile)
      contract.checkTokenizer(tokenizer)
      val loadMs = (System.nanoTime() - start) / 1e6
      return D1Engine(
        context.applicationContext,
        tokenizer,
        contract,
        loadMs,
        backend,
        precision,
        audioPrecision,
      )
    }
  }
}
