package com.emalightning

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * EMA Lightning (Turkish TTS, 8.6M) on LiteRT CompiledModel: four static-shape graphs, every bucket a signature of
 * its file, the integer bookkeeping on the host (this file), exactly as the Hub repo's ema_host.py does it.
 *
 *   letters --text(CPU)--> h [L,224], frames per letter
 *           --host: word counts, frame timeline, letter / frame positions, aligner window-->
 *           --align(CPU)--> cond [T,224]
 *           --host Euler x4: step(CPU)(x, cond, step table) -> v; x1 = x + (1-t) v; x = (1-t') n' + t' x1-->
 *           latents [T,64] --decoder(GPU fp16) per window of 25 / 100 frames + 8 context--> 48 kHz audio
 *
 * Models: litert-community/EMA-Lightning-LiteRT, in filesDir (downloaded by MainActivity or pushed by
 * scripts/install_to_device.sh). config.json and the step table are bundled assets.
 */
class EmaSynthesizer(private val context: Context, private val cpuThreads: Int = 4) : Closeable {

    companion object {
        const val SAMPLE_RATE = 48000
        const val HOP = 1920
        const val LATENT = 64
        const val D = 224
        const val NEG = -1e4f
        const val FIRST_WINDOW = 25
        const val WINDOW = 100
        const val CONTEXT = 8
        const val MAX_WORD_FRAMES = 250
        const val MAX_FRAMES = 3000
        val FILES = listOf("ema_text_multi_fp32.tflite", "ema_align_multi_fp32.tflite", "ema_step_multi_fp32.tflite", "ema_decoder_multi_fp16.tflite")
        const val HUB = "https://huggingface.co/litert-community/EMA-Lightning-LiteRT/resolve/main/"
        fun missing(context: Context): List<String> = FILES.filter { !File(context.filesDir, it).exists() }
    }

    val config: JSONObject = JSONObject(context.assets.open("config.json").readBytes().decodeToString())
    val vocab: List<String> = config.getJSONArray("vocab").let { a -> List(a.length()) { a.getString(it) } }
    private val times: FloatArray = config.getJSONArray("times").let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
    private val textBuckets: IntArray = config.getJSONArray("text_buckets").let { a -> IntArray(a.length()) { a.getInt(it) } }
    private val soundBuckets: List<Pair<Int, Int>> = config.getJSONArray("sound_buckets").let { a ->
        List(a.length()) { a.getJSONArray(it).let { p -> p.getInt(0) to p.getInt(1) } }
    }
    private val decoderBuckets: IntArray = config.getJSONArray("decoder_buckets").let { a -> IntArray(a.length()) { a.getInt(it) } }
    private val stageLengths: Map<Int, IntArray> = config.getJSONObject("decoder_stage_lengths").let { o ->
        o.keys().asSequence().associate { k -> k.toInt() to o.getJSONArray(k).let { a -> IntArray(a.length()) { a.getInt(it) } } }
    }
    private val lookback = config.getJSONObject("aligner").getInt("lookback")
    private val lookahead = config.getJSONObject("aligner").getInt("lookahead")

    // the four steps' modulation vectors: times [4], bc [4,6,224], sg [4,2,224] as little-endian float32 (ema_step_table.bin)
    private val stepBc: Array<FloatArray>
    private val stepSg: Array<FloatArray>
    init {
        val raw = context.assets.open("ema_step_table.bin").readBytes()
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        val all = FloatArray(raw.size / 4) { bb.float }
        val n = times.size
        check(all.size == n + n * 6 * D + n * 2 * D) { "step table size ${all.size}" }
        stepBc = Array(n) { k -> all.copyOfRange(n + k * 6 * D, n + (k + 1) * 6 * D) }
        stepSg = Array(n) { k -> all.copyOfRange(n + n * 6 * D + k * 2 * D, n + n * 6 * D + (k + 1) * 2 * D) }
    }

    private fun load(name: String, acc: Accelerator): CompiledModel {
        val f = File(context.filesDir, name)
        check(f.exists()) { "Model not found: $name (download it first)" }
        val opts = CompiledModel.Options(acc)
        if (acc == Accelerator.CPU) opts.cpuOptions = CompiledModel.CpuOptions(numThreads = cpuThreads)
        return CompiledModel.create(f.absolutePath, opts, null)
    }

    private val text = load(FILES[0], Accelerator.CPU)
    private val align = load(FILES[1], Accelerator.CPU)
    private val step = load(FILES[2], Accelerator.CPU)
    private val decoder = load(FILES[3], Accelerator.GPU)  // the GPU's default precision (fp16): 40.8 dB vs fp32, 3x faster than the CPU

    private class Bound(val model: CompiledModel, val signature: String, val inputs: Map<String, TensorBuffer>, val outputs: Map<String, TensorBuffer>) {
        fun run() = model.run(inputs, outputs, signature)
    }
    private val bound = HashMap<String, Bound>()
    private fun bind(model: CompiledModel, signature: String, ins: List<String>, outs: List<String>): Bound =
        bound.getOrPut(signature) {
            Bound(model, signature, ins.associateWith { model.createInputBuffer(it, signature) }, outs.associateWith { model.createOutputBuffer(it, signature) })
        }

    class Timing { var textMs = 0.0; var alignMs = 0.0; var stepsMs = 0.0; val decodeMs = ArrayList<Double>(); var firstAudioMs = 0.0; var totalMs = 0.0 }
    class Result(val audio: FloatArray, val letters: Int, val frames: Int, val lb: Int, val tb: Int, val timing: Timing)

    private fun textBucket(l: Int) = textBuckets.first { l <= it }
    private fun bucketFor(l: Int, t: Int) = soundBuckets.first { l <= it.first && t <= it.second }
    private fun pBucket(n: Int) = decoderBuckets.first { n <= it }

    /** One piece of alphabet text (≤ 250 letters) -> float32 audio at 48 kHz. seed picks the take. */
    fun synthesize(piece: EmaFrontend.Piece, seed: Long, speed: Double = 1.0): Result {
        val t0 = System.nanoTime()
        val timing = Timing()
        val l = piece.ids.size
        val lb = textBucket(l)

        // ---- text graph (CPU): ids + mask -> h [lb,224], dur [lb] ----
        val tb0 = bind(text, "ema_text_L$lb", listOf("ids", "mask"), listOf("output_0", "output_1"))
        tb0.inputs.getValue("ids").writeInt(IntArray(lb) { if (it < l) piece.ids[it] else 0 })
        tb0.inputs.getValue("mask").writeFloat(FloatArray(lb) { if (it < l) 1f else 0f })
        tb0.run()
        val hAll = tb0.outputs.getValue("output_0").readFloat()
        val durAll = tb0.outputs.getValue("output_1").readFloat()
        timing.textMs = ms(t0)

        // ---- host timeline (engine.plan + sound_stage + the aligner's letter positions) ----
        val cw = piece.cw; val wstart = piece.wstart
        val nWords = cw[l - 1] + 1
        val dur = DoubleArray(l) { durAll[it].toDouble() / speed }
        val counts = DoubleArray(nWords)
        for (i in 0 until l) counts[cw[i]] += dur[i]
        val count = IntArray(nWords) { min(max(Math.rint(counts[it]).toInt(), 1), MAX_WORD_FRAMES) }  // rint = torch.round (half to even)
        val frames = min(count.sum(), MAX_FRAMES)
        val fw = IntArray(frames); val fp = FloatArray(frames)
        run {
            var f = 0
            for (w in 0 until nWords) for (j in 0 until count[w]) { if (f < frames) { fw[f] = w; fp[f] = j.toFloat() / count[w]; f++ } }
        }
        val c = FloatArray(l) { max(dur[it], 1e-4).toFloat() }
        val done = FloatArray(l); var acc = 0f
        for (i in 0 until l) { acc += c[i]; done[i] = acc }
        val total = FloatArray(nWords)
        for (i in 0 until l) total[cw[i]] += c[i]
        val cp = FloatArray(l) { i -> ((done[i] - (done[wstart[i]] - c[wstart[i]]) - 0.5f * c[i]) / max(total[cw[i]], 1e-8f)).coerceIn(0f, 1f) }
        val wlen = FloatArray(nWords)
        for (i in 0 until l) wlen[cw[i]] += 1f
        for (w in 0 until nWords) wlen[w] = max(wlen[w], 1f)
        val woff = FloatArray(nWords); acc = 0f
        for (w in 0 until nWords) { woff[w] = acc; acc += wlen[w] }
        val cg = FloatArray(l) { woff[cw[it]] + cp[it] * wlen[cw[it]] }
        val fg = FloatArray(frames) { woff[fw[it]] + fp[it] * wlen[fw[it]] }
        val (lb2, tb) = bucketFor(l, frames)

        // ---- aligner inputs: padded h / cg / fg, the window (allow) and the frame key bias ----
        val h = FloatArray(lb2 * D); System.arraycopy(hAll, 0, h, 0, l * D)  // the text bucket lb <= lb2 (same L family)
        val cgP = FloatArray(lb2); System.arraycopy(cg, 0, cgP, 0, l)
        val fgP = FloatArray(tb); System.arraycopy(fg, 0, fgP, 0, frames)
        val allow = FloatArray(tb * lb2)
        for (t in 0 until frames) for (i in 0 until l) {
            val rel = cw[i] - fw[t]
            if (rel >= -lookback && rel <= lookahead) allow[t * lb2 + i] = 1f
        }
        val fbias = FloatArray(tb) { if (it < frames) 0f else NEG }
        val t1 = System.nanoTime()
        val ab = bind(align, "ema_align_L${lb2}_T$tb", listOf("h", "cg", "fg", "allow"), listOf("output_0"))
        ab.inputs.getValue("h").writeFloat(h); ab.inputs.getValue("cg").writeFloat(cgP)
        ab.inputs.getValue("fg").writeFloat(fgP); ab.inputs.getValue("allow").writeFloat(allow)
        ab.run()
        val cond = ab.outputs.getValue("output_0").readFloat()
        timing.alignMs = ms(t1)

        // ---- the four distilled steps (CPU): Euler on the host with seeded Gaussian noise ----
        val t2 = System.nanoTime()
        val rnd = Random(seed)
        val noise = Array(times.size) { FloatArray(tb * LATENT).also { n -> for (i in 0 until frames * LATENT) n[i] = rnd.nextGaussian().toFloat() } }
        val sb = bind(step, "ema_step_T$tb", listOf("x", "cond", "bc", "sg", "fbias"), listOf("output_0"))
        sb.inputs.getValue("cond").writeFloat(cond); sb.inputs.getValue("fbias").writeFloat(fbias)
        var x = noise[0].copyOf()
        var x1 = x
        for (k in times.indices) {
            val t = times[k]
            sb.inputs.getValue("x").writeFloat(x); sb.inputs.getValue("bc").writeFloat(stepBc[k]); sb.inputs.getValue("sg").writeFloat(stepSg[k])
            sb.run()
            val v = sb.outputs.getValue("output_0").readFloat()
            x1 = FloatArray(x.size) { x[it] + (1f - t) * v[it] }
            if (k + 1 < times.size) { val tn = times[k + 1]; val nn = noise[k + 1]; x = FloatArray(x.size) { (1f - tn) * nn[it] + tn * x1[it] } }
        }
        timing.stepsMs = ms(t2)
        val lat = x1  // [tb, 64], rows < frames are real

        // ---- decoder (GPU) per window: 25 frames first, then 100, with 8 frames of context cut away ----
        val out = ArrayList<FloatArray>()
        var s = 0
        while (s < frames) {
            val e = min(frames, s + if (s == 0) FIRST_WINDOW else WINDOW)
            val a = max(0, s - CONTEXT); val b = min(frames, e + CONTEXT)
            val n = b - a
            val p = pBucket(n)
            val t3 = System.nanoTime()
            val db = bind(decoder, "ema_decoder_P$p", listOf("z") + (0 until 7).map { "m$it" }, listOf("output_0"))
            val z = FloatArray(LATENT * p)
            for (i in 0 until n) for (d in 0 until LATENT) z[d * p + i] = lat[(a + i) * LATENT + d]
            db.inputs.getValue("z").writeFloat(z)
            val full = stageLengths.getValue(p); val real = stageLengthsOf(n)
            for (i in 0 until 7) db.inputs.getValue("m$i").writeFloat(FloatArray(full[i]) { j -> if (j < real[i]) 1f else 0f })
            db.run()
            val y = db.outputs.getValue("output_0").readFloat()
            out.add(y.copyOfRange((s - a) * HOP, (e - a) * HOP))
            timing.decodeMs.add(ms(t3))
            if (out.size == 1) timing.firstAudioMs = ms(t0)
            s = e
        }
        timing.totalMs = ms(t0)
        val audio = FloatArray(out.sumOf { it.size })
        var o = 0
        for (w in out) { System.arraycopy(w, 0, audio, o, w.size); o += w.size }
        for (i in audio.indices) audio[i] = audio[i].coerceIn(-1f, 1f)
        return Result(audio, l, frames, lb2, tb, timing)
    }

    /** decoder.py's length recurrence: rates (8,6,5,2,2,2), kernels (16,12,10,4,4,4), padding (k - r) / 2. */
    private fun stageLengthsOf(n: Int): IntArray {
        val rates = intArrayOf(8, 6, 5, 2, 2, 2); val kernels = intArrayOf(16, 12, 10, 4, 4, 4)
        val out = IntArray(7); out[0] = n
        var len = n
        for (i in 0 until 6) { len = (len - 1) * rates[i] - 2 * ((kernels[i] - rates[i]) / 2) + kernels[i]; out[i + 1] = len }
        return out
    }

    private fun ms(since: Long) = (System.nanoTime() - since) / 1e6

    override fun close() {
        bound.values.forEach { b -> b.inputs.values.forEach { it.close() }; b.outputs.values.forEach { it.close() } }
        text.close(); align.close(); step.close(); decoder.close()
    }
}
