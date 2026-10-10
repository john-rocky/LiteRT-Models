package com.d1omni

import android.content.Context
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.Closeable
import java.io.File

/** One call of a [D1Graph]: the output [values] and the wall time of each part, in nanoseconds. */
class D1GraphCall(
  val values: FloatArray,
  val writeNanos: Long,
  val runNanos: Long,
  val readNanos: Long,
) {
  /** Input writes + `run()` + output read-back, in milliseconds (as [D1Call.totalMs]). */
  val totalMs: Double
    get() = (writeNanos + runNanos + readNanos) / NANOS_PER_MS

  val writeMs: Double
    get() = writeNanos / NANOS_PER_MS

  /** `run()` alone: on the GPU it returns before the work ends; reading the output waits. */
  val runMs: Double
    get() = runNanos / NANOS_PER_MS

  val readMs: Double
    get() = readNanos / NANOS_PER_MS

  private companion object {
    const val NANOS_PER_MS = 1e6
  }
}

/**
 * The declarations of a one-signature graph and the checks on them, Android-free (the JVM tests
 * cover them): every input and the output are float32 tensors of a declared shape, the feeds of a
 * call name exactly the declared inputs and hold each shape's element count.
 */
object D1GraphIo {
  /** Values in a tensor of [shape]: every dimension >= 1, at most Int.MAX_VALUE / 4 values. */
  fun elementCount(shape: List<Int>): Int {
    require(shape.isNotEmpty()) { "an empty shape" }
    var count = 1L
    for (dimension in shape) {
      require(dimension >= 1) { "shape $shape has a dimension < 1" }
      count *= dimension
      require(count <= Int.MAX_VALUE / Float.SIZE_BYTES) { "shape $shape holds too many values" }
    }
    return count.toInt()
  }

  /** At least one input, valid shapes, an output name not among the inputs. */
  fun checkDeclaration(inputs: Map<String, List<Int>>, output: Pair<String, List<Int>>) {
    require(inputs.isNotEmpty()) { "a graph without inputs" }
    for ((name, shape) in inputs) {
      require(name.isNotEmpty()) { "an input without a name" }
      elementCount(shape)
    }
    require(output.first.isNotEmpty()) { "an output without a name" }
    require(output.first !in inputs) { "${output.first} is declared as an input and as the output" }
    elementCount(output.second)
  }

  /** Throws unless the graph's tensor [name] is float32 [declared] (its element type and dimensions). */
  fun checkTensor(name: String, declared: List<Int>, float32: Boolean, dimensions: List<Int>?) {
    require(float32) { "$name is not float32 in the graph" }
    require(dimensions == declared) { "$name: declared $declared, the graph has $dimensions" }
  }

  /** The signature has as many inputs and outputs as were declared. */
  fun checkComplete(declaredInputs: Int, graphInputs: Int, graphOutputs: Int) {
    require(declaredInputs == graphInputs) { "$declaredInputs inputs declared, the signature has $graphInputs" }
    require(graphOutputs == 1) { "the signature has $graphOutputs outputs; one is read" }
  }

  /** Throws unless [feeds] name exactly the declared [inputs], each with its shape's element count. */
  fun checkFeeds(inputs: Map<String, List<Int>>, feeds: Map<String, FloatArray>) {
    val missing = inputs.keys - feeds.keys
    val extra = feeds.keys - inputs.keys
    require(missing.isEmpty() && extra.isEmpty()) {
      "feeds must name ${inputs.keys}: missing $missing, not an input $extra"
    }
    for ((name, shape) in inputs) {
      val size = feeds.getValue(name).size
      require(size == elementCount(shape)) { "$name holds $size values, its shape $shape holds ${elementCount(shape)}" }
    }
  }
}

/**
 * One compiled one-signature graph on LiteRT `CompiledModel` with float32 inputs and one float32
 * output (the audio graph `audio_<T>`, the vision tower, the projector): the declared inputs
 * ([inputShapes], in the graph's order) and output checked against the graph's tensor types at
 * compile, one buffer per tensor created once and reused for every call. Create, run and close only
 * on [D1Runtime.dispatcher].
 */
class D1Graph
private constructor(
  /** Where the graph was compiled (the CPU after a GPU failure). */
  val backend: D1Backend,
  /** The GPU precision it was compiled with (the CPU ignores it). */
  val precision: D1Precision,
  val file: File,
  val signature: String,
  /** Wall time of `CompiledModel.create` (load and compile), in milliseconds. */
  val compileMs: Double,
  /** The GPU's error when the GPU was asked for and the graph runs on the CPU instead. */
  val gpuFailure: String?,
  /** The declared inputs and their shapes, in the order they are written. */
  val inputShapes: Map<String, List<Int>>,
  val outputName: String,
  val outputShape: List<Int>,
  private val model: CompiledModel,
  private val inputs: Map<String, TensorBuffer>,
  private val output: TensorBuffer,
) : Closeable {
  private val outputs = mapOf(outputName to output)
  private val outputCount = D1GraphIo.elementCount(outputShape)
  private var closed = false

  val isClosed: Boolean
    get() = closed

  /** Writes every input of [feeds] (in [inputShapes] order), runs the graph, reads the output back. */
  fun run(feeds: Map<String, FloatArray>): D1GraphCall {
    check(!closed) { "${file.name} is closed" }
    D1GraphIo.checkFeeds(inputShapes, feeds)
    val start = System.nanoTime()
    for (name in inputShapes.keys) inputs.getValue(name).writeFloat(feeds.getValue(name))
    val written = System.nanoTime()
    model.run(inputs, outputs, signature)
    val ran = System.nanoTime()
    // run() only enqueues the work on the GPU; reading the output waits for it.
    val values = output.readFloat()
    val read = System.nanoTime()
    check(values.size == outputCount) { "$outputName has ${values.size} values, its shape $outputShape holds $outputCount" }
    return D1GraphCall(values, written - start, ran - written, read - ran)
  }

  override fun close() {
    if (closed) return
    closed = true
    try {
      (inputs.values + output).forEach { it.close() }
    } finally {
      model.close()
    }
  }

  companion object {
    /**
     * Compiles [file]'s [signature] on [backend] at [precision] (options as [D1Decider.options]) and
     * checks every declared input and the [output] (name to shape) against the graph: float32, the
     * declared dimensions, and no undeclared tensor in the signature. With [fallback], a GPU failure
     * is kept as [gpuFailure] and the graph compiled on the CPU (four threads) instead.
     */
    fun create(
      context: Context,
      file: File,
      signature: String,
      inputs: Map<String, List<Int>>,
      output: Pair<String, List<Int>>,
      backend: D1Backend,
      precision: D1Precision,
      fallback: Boolean,
    ): D1Graph {
      check(file.isFile) { "Missing ${file.name}" }
      D1GraphIo.checkDeclaration(inputs, output)
      if (backend == D1Backend.CPU || !fallback) {
        return compile(context, file, signature, inputs, output, backend, precision, null)
      }
      return try {
        compile(context, file, signature, inputs, output, D1Backend.GPU, precision, null)
      } catch (failure: Exception) {
        compile(context, file, signature, inputs, output, D1Backend.CPU, precision, D1Decider.describe(failure))
      } catch (failure: LinkageError) {
        compile(context, file, signature, inputs, output, D1Backend.CPU, precision, D1Decider.describe(failure))
      }
    }

    private fun checkType(type: TensorType, name: String, shape: List<Int>) =
      D1GraphIo.checkTensor(
        name,
        shape,
        type.elementType == TensorType.ElementType.FLOAT,
        type.layout?.dimensions?.toList(),
      )

    private fun compile(
      context: Context,
      file: File,
      signature: String,
      inputs: Map<String, List<Int>>,
      output: Pair<String, List<Int>>,
      backend: D1Backend,
      precision: D1Precision,
      gpuFailure: String?,
    ): D1Graph {
      val start = System.nanoTime()
      val model =
        CompiledModel.create(
          file.absolutePath,
          D1Decider.options(backend, precision),
          D1Runtime.environment(context),
        )
      val compileMs = (System.nanoTime() - start) / 1e6
      val buffers = ArrayList<TensorBuffer>()
      try {
        for ((name, shape) in inputs) checkType(model.getInputTensorType(name, signature), name, shape)
        checkType(model.getOutputTensorType(output.first, signature), output.first, output.second)
        // Every tensor of the signature is declared: the names were checked one by one above, the
        // counts come from the signature's own buffer lists (the Kotlin API has no list of names).
        val graphInputs = model.createInputBuffers(signature).let { all -> all.forEach { it.close() }; all.size }
        val graphOutputs = model.createOutputBuffers(signature).let { all -> all.forEach { it.close() }; all.size }
        D1GraphIo.checkComplete(inputs.size, graphInputs, graphOutputs)
        val inputBuffers = LinkedHashMap<String, TensorBuffer>()
        for (name in inputs.keys) {
          inputBuffers[name] = model.createInputBuffer(name, signature).also { buffers.add(it) }
        }
        val outputBuffer = model.createOutputBuffer(output.first, signature).also { buffers.add(it) }
        return D1Graph(
          backend,
          precision,
          file,
          signature,
          compileMs,
          gpuFailure,
          LinkedHashMap(inputs),
          output.first,
          output.second,
          model,
          inputBuffers,
          outputBuffer,
        )
      } catch (failure: Throwable) {
        buffers.forEach { it.close() }
        model.close()
        throw failure
      }
    }
  }
}
