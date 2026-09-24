package com.nemotron3diar

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * The two Nemotron-3-Diarization graphs on the LiteRT CompiledModel GPU (no CPU fallback: a graph that does not
 * compile on the GPU throws).
 *
 * graph A ([FRONTEND]) always runs with GpuOptions precision FP32: it is 0.2 ms, and its rows go into the speaker
 * cache and FIFO, where they are attended to for minutes. graph B ([ENCODER], or [ENCODER_OFFLINE] for the offline
 * pass) runs with the chosen [Precision]. One [Environment] serves both. Every call is write -> run -> readFloat (run() is
 * asynchronous; the readback is the sync point).
 */
class LiteRtEngine(
  private val env: Environment,
  private val modelsDir: File,
  private val frontendFile: String = FRONTEND,
  private val encoderFile: String = ENCODER,
  precision: Precision = Precision.FP32,
) : Engine {

  /** graph B GPU precision: FP32 (matches the fp32 reference), FP16 (the delegate default), FP16 with FP32 accumulation. */
  enum class Precision(val label: String) {
    FP32("fp32"),
    DEFAULT("default"),
    ACCUM("accum");

    companion object {
      fun parse(s: String): Precision = entries.first { it.label == s || it.name.equals(s, ignoreCase = true) }
    }
  }

  private class Graph(val model: CompiledModel, inputs: List<String>, outputs: List<String>) : AutoCloseable {
    val ins: Map<String, TensorBuffer> = inputs.associateWith { model.createInputBuffer(it) }
    val outs: Map<String, TensorBuffer> = outputs.associateWith { model.createOutputBuffer(it) }

    override fun close() {
      ins.values.forEach { it.close() }
      outs.values.forEach { it.close() }
      model.close()
    }
  }

  private var a: Graph
  private var b: Graph
  private var ropeWritten: FloatArray? = null

  var precision: Precision = precision
    private set

  /** Load + compile ms of the last compile of each graph. */
  var frontendCompileMs = 0.0
    private set

  var encoderCompileMs = 0.0
    private set

  init {
    val t0 = System.nanoTime()
    a = compile(frontendFile, Precision.FP32, listOf("mel"), listOf("chunk_embeds"))
    frontendCompileMs = (System.nanoTime() - t0) / 1e6
    b = compileEncoder(precision)
  }

  private fun options(p: Precision): CompiledModel.Options {
    val options = CompiledModel.Options(Accelerator.GPU)
    when (p) {
      Precision.FP32 ->
        options.gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
      Precision.DEFAULT -> {}
      Precision.ACCUM ->
        options.gpuOptions =
          CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM)
    }
    return options
  }

  private fun compile(file: String, p: Precision, inputs: List<String>, outputs: List<String>): Graph {
    val path = File(modelsDir, file)
    check(path.exists()) { "model not found: ${path.absolutePath} (run scripts/install_to_device.sh)" }
    return Graph(CompiledModel.create(path.absolutePath, options(p), env), inputs, outputs)
  }

  private fun compileEncoder(p: Precision): Graph {
    val t0 = System.nanoTime()
    val g = compile(encoderFile, p, listOf("packed_embeds", "attn_bias", "rope_cos", "rope_sin"), listOf("logits"))
    encoderCompileMs = (System.nanoTime() - t0) / 1e6
    return g
  }

  /** Recompiles graph B with another precision (graph A stays FP32). */
  fun setPrecision(p: Precision) {
    if (p == precision) return
    b.close()
    ropeWritten = null
    b = compileEncoder(p)
    precision = p
  }

  override fun frontend(mel: FloatArray): FloatArray {
    a.ins.getValue("mel").writeFloat(mel)
    a.model.run(a.ins, a.outs)
    return a.outs.getValue("chunk_embeds").readFloat()
  }

  override fun encoder(packed: FloatArray, bias: FloatArray, cos: FloatArray, sin: FloatArray): FloatArray {
    b.ins.getValue("packed_embeds").writeFloat(packed)
    b.ins.getValue("attn_bias").writeFloat(bias)
    if (ropeWritten !== cos) { // the RoPE tables are constant: written once per compiled graph
      b.ins.getValue("rope_cos").writeFloat(cos)
      b.ins.getValue("rope_sin").writeFloat(sin)
      ropeWritten = cos
    }
    b.model.run(b.ins, b.outs)
    return b.outs.getValue("logits").readFloat()
  }

  override fun close() {
    a.close()
    b.close()
  }

  companion object {
    const val FRONTEND = "nemotron3_diar_frontend.tflite"
    const val ENCODER = "nemotron3_diar_encoder_low_latency_fp16.tflite"

    /** graph B of the offline pass (StreamConfig.OFFLINE, T = 684). */
    const val ENCODER_OFFLINE = "nemotron3_diar_encoder_offline_fp16.tflite"
  }
}
