// Copyright 2026 Daisuke Majima. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Bonsai Image 4B on-device pipeline for Android (port of the iOS app's
// BonsaiPipeline.swift). Three fixed-shape .tflite graphs over the LiteRT
// CompiledModel API on CPU/XNNPACK, loaded sequentially and closed between
// stages so peak memory stays ~DiT-sized rather than the ~4 GiB sum — the
// difference between running and being LMK-killed on 8 GB devices.
//
// One DiT + VAE decoder pair per output size (pipeline_meta.json: `files` is
// 512x512, `variants["256"]` is 256x256); the text encoder is shared. Every
// stage is timed with System.nanoTime() and reported as it finishes.

package com.bonsai.imagegen

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.File

class BonsaiPipeline(private val modelsDir: File, private val meta: JSONObject) {

    class MissingModels(val files: List<String>) : Exception("missing: $files")
    class Cancelled : Exception()

    private val textencFile = meta.getJSONObject("files").getString("textenc")
    private val bnScale = meta.getJSONArray("latent_bn_scale").let { a ->
        FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }
    private val bnShift = meta.getJSONArray("latent_bn_shift").let { a ->
        FloatArray(a.length()) { a.getDouble(it).toFloat() }
    }

    /** Set from any thread to stop the run at the next stage boundary (a DiT step is never interrupted). The
     *  caller clears it before each run, not [generate]: a Cancel pressed before the worker reaches
     *  [generate] must still count. */
    @Volatile var cancelled = false

    companion object {
        val THREADS = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)

        /** Output sizes this pipeline_meta.json has graphs for: 512 (`files`)
         *  and every `variants` key. */
        fun sizes(meta: JSONObject): List<Int> {
            val out = mutableListOf(512)
            meta.optJSONObject("variants")?.keys()?.forEach { k -> k.toIntOrNull()?.let { out.add(it) } }
            return out.distinct().sorted()
        }

        /** DiT and VAE decoder file names for one output size (generate.py
         *  graph_files): 512 = `files`, any other size = `variants["<size>"]`. */
        fun graphFiles(meta: JSONObject, size: Int): Pair<String, String> {
            val entry = if (size == 512) meta.getJSONObject("files")
            else meta.optJSONObject("variants")?.optJSONObject(size.toString())
                ?: throw IllegalArgumentException("no ${size}x$size graphs in pipeline_meta.json")
            return entry.getString("dit") to entry.getString("vae")
        }

        /** The published file name, or its `_fixed` sibling (the zero-scale-
         *  patched DiT used during device verification). */
        fun resolveModel(name: String, dir: File): File? =
            listOf(name, name.replace(".tflite", "_fixed.tflite"))
                .map { File(dir, it) }.firstOrNull { it.exists() }

        /** Every graph the meta names that is not in [modelsDir]: the shared
         *  text encoder and each size's DiT and VAE decoder (five files with
         *  the 256 variant). */
        fun missingFiles(modelsDir: File, meta: JSONObject): List<String> {
            val names = listOf(meta.getJSONObject("files").getString("textenc")) +
                sizes(meta).flatMap { graphFiles(meta, it).toList() }
            return names.distinct().filter { resolveModel(it, modelsDir) == null }
        }

        /** The output sizes whose three graphs are all in [modelsDir]. */
        fun readySizes(modelsDir: File, meta: JSONObject): List<Int> {
            val missing = missingFiles(modelsDir, meta).toSet()
            if (meta.getJSONObject("files").getString("textenc") in missing) return emptyList()
            return sizes(meta).filter { s -> graphFiles(meta, s).toList().none { it in missing } }
        }

        private fun msSince(t0: Long) = (System.nanoTime() - t0) / 1e6
    }

    /** Wall times of one generation in ms, filled stage by stage. */
    class Timings {
        var tokenizeMs = 0.0
        var textencLoadMs = 0.0
        var textencRunMs = 0.0
        var textencCloseMs = 0.0
        var noiseMs = 0.0
        var ditLoadMs = 0.0
        val stepMs = ArrayList<Double>()
        var ditCloseMs = 0.0
        var unpatchifyMs = 0.0
        var vaeLoadMs = 0.0
        var vaeRunMs = 0.0
        var vaeCloseMs = 0.0
        var rgbMs = 0.0

        /** The three graph loads (file open + XNNPACK setup + buffer creation). */
        val modelLoadMs get() = textencLoadMs + ditLoadMs + vaeLoadMs
    }

    /** Stage events, delivered on the generating thread as they happen. */
    sealed interface Progress {
        data object TextEncoder : Progress
        data class TextEncoderDone(val loadMs: Double, val runMs: Double) : Progress
        data class DitLoaded(val loadMs: Double) : Progress
        data class Step(val k: Int, val n: Int) : Progress              // k is 1-based
        data class StepDone(val k: Int, val n: Int, val ms: Double) : Progress
        data object Decode : Progress
        data class DecodeDone(val loadMs: Double, val runMs: Double) : Progress
    }

    class Result(
        val rgb: ByteArray,            // size*size*3, row-major RGB
        val size: Int,
        val sigmas: FloatArray,
        val attentionTokens: Int,      // real tokens the text encoder sees (template included)
        val files: List<String>,       // text encoder, DiT, VAE decoder actually opened
        val timings: Timings,
    )

    // MARK: single fixed-shape graph, CPU/XNNPACK

    /** One graph over the CompiledModel CPU path. The classic Interpreter path
     *  ran XNNPACK on ONE thread whatever numThreads said (Galaxy S26: 103 %
     *  process CPU, the same step time at 6 and 8 threads); CpuOptions sets the
     *  XNNPACK thread count, as generate.py's CpuOptions(num_threads) does. */
    private class Graph(file: File, threads: Int, argCount: Int) : AutoCloseable {
        private val model: CompiledModel
        // inputs by signature name args_<k> (k = argument position), NEVER by
        // shape/index: at 256x256 the DiT's img_ids and txt_ids are both (256, 4)
        private val inputs: Map<String, TensorBuffer>
        private val inBytes: IntArray
        private val outputs: Map<String, TensorBuffer>
        private val outBytes: Int
        val loadMs: Double

        init {
            val t = System.nanoTime()
            // created from the file path, never from a Java buffer: a heap copy
            // of the 2.11 GiB DiT would double the footprint
            model = CompiledModel.create(
                file.path,
                CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(threads, null, null)
                }
            )
            val created = ArrayList<TensorBuffer>()
            try {
                // (name, signature) order — named, because both are Strings
                inputs = (0 until argCount).associate { k ->
                    val name = "args_$k"
                    name to model.createInputBuffer(inputName = name, signature = SIGNATURE).also { created.add(it) }
                }
                inBytes = IntArray(argCount) { k ->
                    model.getInputBufferRequirements(inputName = "args_$k", signature = SIGNATURE).bufferSize
                }
                outputs = mapOf(
                    OUTPUT to model.createOutputBuffer(outputName = OUTPUT, signature = SIGNATURE).also { created.add(it) }
                )
                outBytes = model.getOutputBufferRequirements(outputName = OUTPUT, signature = SIGNATURE).bufferSize
            } catch (e: Exception) {
                created.forEach { it.close() }
                model.close()
                throw e
            }
            loadMs = msSince(t)
        }

        /** Arguments in ARGUMENT order (FloatArray or IntArray, as the graph
         *  declares); returns output_0 as floats. */
        fun run(args: List<Any>, outCount: Int): FloatArray {
            require(args.size == inputs.size) { "inputCount ${args.size} != ${inputs.size}" }
            // an output of another size would be read partly or past the end
            require(outBytes == outCount * 4) { "byteSize output_0: graph $outBytes != host ${outCount * 4}" }
            for ((k, a) in args.withIndex()) {
                val buf = inputs.getValue("args_$k")
                val n = when (a) {
                    is FloatArray -> a.size
                    is IntArray -> a.size
                    else -> throw IllegalArgumentException("args_$k: ${a.javaClass.simpleName}")
                }
                require(inBytes[k] == n * 4) { "byteSize args_$k: graph ${inBytes[k]} != host ${n * 4}" }
                if (a is FloatArray) buf.writeFloat(a) else buf.writeInt(a as IntArray)
            }
            model.run(inputs, outputs, signature = SIGNATURE)
            val y = outputs.getValue(OUTPUT).readFloat()
            require(y.size == outCount) { "output_0: ${y.size} floats != host $outCount" }
            return y
        }

        override fun close() {
            inputs.values.forEach { it.close() }
            outputs.values.forEach { it.close() }
            model.close()
        }

        companion object {
            const val SIGNATURE = "serving_default"
            const val OUTPUT = "output_0"
        }
    }

    /** Runs [block] on this graph, then closes it and reports the close time. */
    private inline fun <R> Graph.runThenClose(onClosed: (Double) -> Unit, block: (Graph) -> R): R {
        try {
            return block(this)
        } finally {
            val t = System.nanoTime()
            close()
            onClosed(msSince(t))
        }
    }

    private fun checkCancel() {
        if (cancelled) throw Cancelled()
    }

    // MARK: generation

    fun generate(
        tokenizer: QwenTokenizer,
        prompt: String,
        seed: Long,
        steps: Int,
        size: Int = 512,
        threads: Int = THREADS,
        status: (String) -> Unit = {},
        progress: (Progress) -> Unit = {},
        gpuDit: Boolean = false,
    ): Result {
        require(steps >= 1) { "steps must be >= 1: $steps" }
        require(!gpuDit || size == 512) { "the GPU DiT export is 512x512 only" }
        val grid = BonsaiMath.gridFor(size)
        val tokens = grid * grid
        val (ditName, vaeName) = graphFiles(meta, size)
        // resolve the graphs up front: a missing VAE should not cost a DiT run
        val names = listOf(textencFile, ditName, vaeName)
        val files = names.map { resolveModel(it, modelsDir) }
        val needed = if (gpuDit) listOf(0, 2) else listOf(0, 1, 2)   // the GPU DiT is its own file
        needed.filter { files[it] == null }.takeIf { it.isNotEmpty() }?.let { absent ->
            throw MissingModels(absent.map { names[it] })
        }
        val (textencPath, ditPath, vaePath) = files
        val tm = Timings()

        // ---- stage 1: tokenize + text encoder ------------------------------
        progress(Progress.TextEncoder)
        var t = System.nanoTime()
        val enc = tokenizer.encodePrompt(prompt)
        tm.tokenizeMs = msSince(t)
        val attentionTokens = enc.mask.sum()
        status("tokens: ${enc.promptTokenCount} for \"user\\n\" + prompt, $attentionTokens of ${BonsaiMath.SEQ} attended")
        checkCancel()
        val embeds = Graph(textencPath!!, threads, 2).runThenClose({ tm.textencCloseMs = it }) { te ->
            tm.textencLoadMs = te.loadMs
            checkCancel()
            val tr = System.nanoTime()
            val e = te.run(listOf(enc.ids, enc.mask), BonsaiMath.SEQ * 7680)
            tm.textencRunMs = msSince(tr)
            e
        }
        status("text encoder %.1fs (load %.1fs)".format(tm.textencRunMs / 1e3, tm.textencLoadMs / 1e3))
        progress(Progress.TextEncoderDone(tm.textencLoadMs, tm.textencRunMs))
        checkCancel()

        // ---- stage 2: DiT Euler loop ----------------------------------------
        val sigmas = BonsaiMath.sigmas(steps, tokens)
        t = System.nanoTime()
        val lat = BonsaiMath.noise(seed, tokens)
        tm.noiseMs = msSince(t)
        if (gpuDit) {
            // GPU-shaped export (dit_gpu_int4b32.tflite) over the CompiledModel
            // GPU path. FP32 precision is REQUIRED: default fp16 execution
            // corrupts this model's activations (measured on the Metal backend).
            val gpuFile = File(modelsDir, "dit_gpu_int4b32.tflite")
            if (!gpuFile.exists()) throw MissingModels(listOf(gpuFile.name))
            val opts = CompiledModel.Options(Accelerator.GPU)
            opts.gpuOptions = CompiledModel.GpuOptions(
                null, null, null, CompiledModel.GpuOptions.Precision.FP32,
                null, null, null, null, null, null, null, null, null, null, null
            )
            val t0 = System.nanoTime()
            val dit = CompiledModel.create(gpuFile.path, opts)
            val bi = dit.createInputBuffers()
            tm.ditLoadMs = msSince(t0)
            status("DiT GPU compiled %.1fs".format(tm.ditLoadMs / 1e3))
            progress(Progress.DitLoaded(tm.ditLoadMs))
            try {
                val imgIdsArr = BonsaiMath.imgIds()
                val txtIdsArr = BonsaiMath.txtIds()
                for (k in 0 until steps) {
                    checkCancel()
                    progress(Progress.Step(k + 1, steps))
                    val ts = System.nanoTime()
                    bi[0].writeFloat(lat)
                    bi[1].writeFloat(embeds)
                    bi[2].writeFloat(floatArrayOf(sigmas[k]))
                    bi[3].writeFloat(imgIdsArr)
                    bi[4].writeFloat(txtIdsArr)
                    val out = dit.run(bi)
                    val v = out[0].readFloat()
                    out.forEach { it.close() }
                    val ds = sigmas[k + 1] - sigmas[k]
                    for (i in lat.indices) lat[i] += ds * v[i]
                    val ms = msSince(ts)
                    tm.stepMs.add(ms)
                    status("step ${k + 1}/$steps  sigma %.3f  %.1fs (GPU)".format(sigmas[k], ms / 1e3))
                    progress(Progress.StepDone(k + 1, steps, ms))
                }
            } finally {
                val tc = System.nanoTime()
                bi.forEach { it.close() }
                dit.close()
                tm.ditCloseMs = msSince(tc)
            }
        } else {
            val imgIds = BonsaiMath.imgIds(grid)
            val txtIds = BonsaiMath.txtIds()
            Graph(ditPath!!, threads, 5).runThenClose({ tm.ditCloseMs = it }) { dit ->
                tm.ditLoadMs = dit.loadMs
                status("DiT loaded %.1fs (%s)".format(dit.loadMs / 1e3, ditPath.name))
                progress(Progress.DitLoaded(dit.loadMs))
                for (k in 0 until steps) {
                    checkCancel()
                    progress(Progress.Step(k + 1, steps))
                    val ts = System.nanoTime()
                    val v = dit.run(
                        listOf(lat, embeds, floatArrayOf(sigmas[k]), imgIds, txtIds),
                        tokens * BonsaiMath.PACKED_CHANNELS
                    )
                    val ds = sigmas[k + 1] - sigmas[k]
                    for (i in lat.indices) lat[i] += ds * v[i]
                    val ms = msSince(ts)
                    tm.stepMs.add(ms)
                    status("step ${k + 1}/$steps  sigma %.3f  %.1fs".format(sigmas[k], ms / 1e3))
                    progress(Progress.StepDone(k + 1, steps, ms))
                }
            }
        }
        checkCancel()

        // ---- stage 3: unpatchify + VAE decode -------------------------------
        progress(Progress.Decode)
        t = System.nanoTime()
        val z = BonsaiMath.unpatchify(lat, bnScale, bnShift, grid)
        tm.unpatchifyMs = msSince(t)
        val plane = size * size
        val rgb = ByteArray(plane * 3)
        Graph(vaePath!!, threads, 1).runThenClose({ tm.vaeCloseMs = it }) { vae ->
            tm.vaeLoadMs = vae.loadMs
            val tv = System.nanoTime()
            val y = vae.run(listOf(z), 3 * plane)
            tm.vaeRunMs = msSince(tv)
            val tr = System.nanoTime()
            for (c in 0 until 3) for (p in 0 until plane) {
                val v = ((y[c * plane + p] / 2f + 0.5f) * 255f).coerceIn(0f, 255f)
                // round half-up == Swift .rounded() for non-negative values
                rgb[p * 3 + c] = Math.round(v).toByte()
            }
            tm.rgbMs = msSince(tr)
        }
        status("VAE decode %.1fs (load %.1fs, %s)".format(tm.vaeRunMs / 1e3, tm.vaeLoadMs / 1e3, vaePath.name))
        progress(Progress.DecodeDone(tm.vaeLoadMs, tm.vaeRunMs))

        val opened = listOf(
            textencPath.name,
            if (gpuDit) "dit_gpu_int4b32.tflite" else ditPath!!.name,
            vaePath.name,
        )
        return Result(rgb, size, sigmas, attentionTokens, opened, tm)
    }
}
