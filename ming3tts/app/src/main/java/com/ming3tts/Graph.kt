package com.ming3tts

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
 * Where a graph runs. GPU = the delegate's default precision (fp16 on Mali and Adreno).
 * NPU = the Qualcomm HTP through LiteRT's JIT compile (vendor libraries inside the APK).
 */
enum class Backend(val label: String) {
    CPU("CPU"),
    GPU("GPU"),
    GPU32("GPU FP32"),
    GPU16A32("GPU FP16+FP32acc"),
    NPU("NPU");

    companion object {
        fun parse(s: String): Backend = when (s.lowercase()) {
            "cpu" -> CPU
            "gpu" -> GPU
            "gpu32" -> GPU32
            "gpu16a32" -> GPU16A32
            "npu" -> NPU
            else -> error("unknown backend '$s' (cpu, gpu, gpu32, gpu16a32, npu)")
        }
    }
}

/**
 * The accelerators of this process: one shared LiteRT Environment, and whether the NPU is usable.
 *
 * The Qualcomm NPU needs its vendor libraries in the APK (copied into src/main/jniLibs before the
 * build, never committed) and a phone with a supported Hexagon HTP. The Environment points LiteRT at them, and
 * the context overload puts the NPU's JIT cache under cacheDir, so only the first load of an NPU
 * graph compiles it (minutes); later loads take about a second.
 */
class Accelerators private constructor(val env: Environment, val npu: Boolean, val npuNote: String) : Closeable {

    override fun close() = env.close()

    companion object {
        private val VENDOR_LIBS = listOf(
            "libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so", "libQnnHtp.so",
            "libQnnSystem.so", "libQnnHtpPrepare.so",
        )

        /** The Qualcomm libraries are only loaded on a Qualcomm SoC (a Pixel never opens them). */
        private fun qualcommSoc(): Boolean {
            val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else ""
            return soc.contains("qti", ignoreCase = true) || soc.contains("qualcomm", ignoreCase = true) ||
                (Build.VERSION.SDK_INT < 31 && Build.HARDWARE.contains("qcom", ignoreCase = true))
        }

        fun create(context: Context): Accelerators {
            val libDir = context.applicationInfo.nativeLibraryDir
            val missing = VENDOR_LIBS.filter { !File(libDir, it).exists() }
            val qualcomm = qualcommSoc()
            val env = if (missing.isEmpty() && qualcomm) {
                Environment.create(
                    context,
                    mapOf(
                        Environment.Option.DispatchLibraryDir to libDir,
                        Environment.Option.CompilerPluginLibraryDir to libDir,
                    ),
                )
            } else {
                Environment.create()
            }
            val available: Set<Accelerator> = runCatching { env.getAvailableAccelerators() }.getOrDefault(emptySet())
            val npu = missing.isEmpty() && qualcomm && Accelerator.NPU in available
            val note = when {
                missing.isNotEmpty() -> "this app was built without the NPU libraries"
                !qualcomm -> "this phone has no Qualcomm NPU"
                !npu -> "LiteRT could not register this phone's NPU"
                else -> "NPU ready"
            }
            Log.i(Graph.TAG, "accelerators: $available, npu=$npu ($note), missing libraries $missing, soc " +
                (if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE))
            return Accelerators(env, npu, note)
        }
    }
}

/** A graph the engine can call: the LiteRT [Graph], or a stand-in in the JVM replay test. */
interface GraphCall : Closeable {
    val compileMs: Double
    val lastWriteMs: Double

    fun call(sig: String, inputs: Array<FloatArray>, read: IntArray? = null): Array<FloatArray?>
}

/**
 * One .tflite with the signatures its role uses, in one CompiledModel. Inputs are bound by name
 * (`args_0`, `args_1`, ... in export order), outputs as `output_0`, ... The GPU and the NPU run
 * accelerator-only: a graph that does not fully delegate fails to compile instead of falling back
 * to the CPU. When the NPU is usable every graph carries the Qualcomm options, because the first
 * model a process loads fixes the NPU's dispatch options for all later ones.
 */
class Graph(
    acc: Accelerators,
    val file: File,
    val backend: Backend,
    signatures: Map<String, Pair<Int, Int>>,
    shareConstants: Boolean = false,
) : GraphCall {
    val model: CompiledModel
    override val compileMs: Double
    private val ins = HashMap<String, Array<TensorBuffer>>()
    private val outs = HashMap<String, Array<TensorBuffer>>()
    private val inMaps = HashMap<String, Map<String, TensorBuffer>>()
    private val outMaps = HashMap<String, Map<String, TensorBuffer>>()

    /** Wall times of the last call: inputs written, run() returned, outputs read (ms). */
    override var lastWriteMs = 0.0
    var lastRunMs = 0.0
    var lastReadMs = 0.0

    init {
        require(backend != Backend.NPU || acc.npu) { "${file.name}: NPU requested but ${acc.npuNote}" }
        val opts = when (backend) {
            Backend.CPU -> CompiledModel.Options(Accelerator.CPU).also {
                it.cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
            }
            Backend.NPU -> CompiledModel.Options(Accelerator.NPU)
            else -> CompiledModel.Options(Accelerator.GPU).also {
                val precision = when (backend) {
                    Backend.GPU32 -> CompiledModel.GpuOptions.Precision.FP32
                    Backend.GPU16A32 -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                    else -> null
                }
                if (precision != null || shareConstants) {
                    it.gpuOptions = CompiledModel.GpuOptions(
                        precision = precision,
                        constantTensorSharing = if (shareConstants) true else null,
                    )
                }
            }
        }
        if (acc.npu) {
            // Without BURST the HTP runs at its default clock, several times slower.
            opts.qualcommOptions = CompiledModel.QualcommOptions(
                htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST,
            )
        }
        val t0 = System.nanoTime()
        model = CompiledModel.create(file.absolutePath, opts, acc.env)
        for ((sig, io) in signatures) {
            val i = Array(io.first) { model.createInputBuffer("args_$it", sig) }
            val o = Array(io.second) { model.createOutputBuffer("output_$it", sig) }
            ins[sig] = i
            outs[sig] = o
            inMaps[sig] = i.withIndex().associate { "args_${it.index}" to it.value }
            outMaps[sig] = o.withIndex().associate { "output_${it.index}" to it.value }
        }
        compileMs = (System.nanoTime() - t0) / 1e6
        Log.i(TAG, "compiled ${file.name} on ${backend.label} in ${"%.0f".format(compileMs)} ms " +
            "(signatures ${signatures.keys}, shareConstants=$shareConstants, qualcomm=${acc.npu})")
    }

    /**
     * Run [sig] with [inputs] (one FloatArray per `args_i`) and return the outputs listed in
     * [read] (null = all). run() is asynchronous on the GPU: the first read waits for the result.
     */
    override fun call(sig: String, inputs: Array<FloatArray>, read: IntArray?): Array<FloatArray?> {
        val i = ins.getValue(sig)
        require(inputs.size == i.size) { "$sig takes ${i.size} inputs, got ${inputs.size}" }
        val t0 = System.nanoTime()
        for (k in i.indices) i[k].writeFloat(inputs[k])
        val t1 = System.nanoTime()
        model.run(inMaps.getValue(sig), outMaps.getValue(sig), sig)
        val t2 = System.nanoTime()
        val o = outs.getValue(sig)
        val res = arrayOfNulls<FloatArray>(o.size)
        for (k in read ?: IntArray(o.size) { it }) res[k] = o[k].readFloat()
        val t3 = System.nanoTime()
        lastWriteMs = (t1 - t0) / 1e6
        lastRunMs = (t2 - t1) / 1e6
        lastReadMs = (t3 - t2) / 1e6
        return res
    }

    override fun close() {
        ins.values.forEach { a -> a.forEach { it.close() } }
        outs.values.forEach { a -> a.forEach { it.close() } }
        ins.clear()
        outs.clear()
        model.close()
    }

    companion object {
        const val TAG = "Ming3TTS"

        // signature -> (inputs, outputs), from contract.json. The flow role binds only the
        // 10-step solve: the one-evaluation `velocity` signature of the first flow export is not used.
        val LM = mapOf("prefill" to (6 to 3), "step" to (6 to 3))
        val PREFILL = mapOf("prefill" to (6 to 3))
        val STEP = mapOf("step" to (6 to 3))
        val FRAME = mapOf("prefill" to (6 to 3), "frame" to (9 to 1))
        val FLOW = mapOf("flow" to (4 to 1))
        val AGG = mapOf("agg" to (1 to 1))
        val DEC = mapOf("dec" to (6 to 3))

        fun signaturesOf(role: String) = when (role) {
            "frame" -> FRAME
            "lm" -> LM
            "prefill" -> PREFILL
            "step" -> STEP
            "flow" -> FLOW
            "agg" -> AGG
            else -> DEC
        }
    }
}
