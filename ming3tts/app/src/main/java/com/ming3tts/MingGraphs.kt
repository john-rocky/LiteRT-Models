package com.ming3tts

import android.content.Context
import android.os.Build
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * The four Ming-3.0-tts graphs placed as measured on a Galaxy S26: the LM and the aggregator on the
 * GPU at FP32, the flow on the Qualcomm NPU (on the GPU when the phone or the APK has no NPU), the
 * decoder on the GPU. [run] calls one signature: inputs args_0, args_1, ... in export order, outputs
 * output_0, output_1, ... (names and shapes in contract.json).
 */
class MingGraphs(context: Context, private val dir: File) : AutoCloseable {
    private val libs = context.applicationInfo.nativeLibraryDir
    private val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER.lowercase() else ""
    private val env = if (("qti" in soc || "qualcomm" in soc) && File(libs, "libLiteRtDispatch_Qualcomm.so").exists()) {
        Environment.create(
            context,
            mapOf(
                Environment.Option.DispatchLibraryDir to libs,
                Environment.Option.CompilerPluginLibraryDir to libs,
            ),
        )
    } else {
        Environment.create()
    }
    val npu = Accelerator.NPU in env.getAvailableAccelerators()

    private fun load(file: String, accelerator: Accelerator, fp32: Boolean = false): CompiledModel {
        val path = File(dir, file)
        require(path.isFile) { "$path is missing: copy the model files with scripts/install_to_device.sh" }
        val options = CompiledModel.Options(accelerator)
        if (accelerator == Accelerator.GPU) {
            options.gpuOptions = CompiledModel.GpuOptions(
                precision = if (fp32) CompiledModel.GpuOptions.Precision.FP32 else null,
                constantTensorSharing = true, // the LM's prefill and step signatures share one copy of the weights
            )
        }
        if (npu) {
            options.qualcommOptions = CompiledModel.QualcommOptions(
                htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST,
            )
        }
        return CompiledModel.create(path.absolutePath, options, env)
    }

    // The NPU graph loads first: its first load compiles it for this phone (about a minute, cached after).
    val flow = load("flow_r4b_s10_wfp16.tflite", if (npu) Accelerator.NPU else Accelerator.GPU)
    val lm = load("lm2_wfp16.tflite", Accelerator.GPU, fp32 = true)
    val agg = load("agg_r4b_wfp16.tflite", Accelerator.GPU, fp32 = true)
    val dec = load("dec_c16_wfp16.tflite", Accelerator.GPU)

    private class Io(val ins: Map<String, TensorBuffer>, val outs: Map<String, TensorBuffer>)

    private val io = ConcurrentHashMap<Pair<CompiledModel, String>, Io>() // the decoder may run on its own thread

    fun run(model: CompiledModel, signature: String, vararg inputs: FloatArray): List<FloatArray> {
        val b = io.getOrPut(model to signature) {
            Io(
                inputs.indices.associate { "args_$it" to model.createInputBuffer("args_$it", signature) },
                (0 until OUTPUTS.getValue(signature)).associate { "output_$it" to model.createOutputBuffer("output_$it", signature) },
            )
        }
        inputs.forEachIndexed { i, x -> b.ins.getValue("args_$i").writeFloat(x) }
        model.run(b.ins, b.outs, signature)
        return List(b.outs.size) { b.outs.getValue("output_$it").readFloat() }
    }

    override fun close() {
        io.values.forEach { b -> (b.ins.values + b.outs.values).forEach { it.close() } }
        listOf(flow, lm, agg, dec).forEach { it.close() }
        env.close()
    }

    companion object {
        val OUTPUTS = mapOf("prefill" to 3, "step" to 3, "flow" to 1, "agg" to 1, "dec" to 3)
    }
}
