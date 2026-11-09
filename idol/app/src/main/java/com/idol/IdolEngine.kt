package com.idol

import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.sqrt

/** The stages the Run screen lists, in order. */
enum class Stage(val label: String) {
    MATTING("Removing background"),
    DINO("Image features (DINOv3)"),
    VAE("Image latent (VAE)"),
    DENOISER("Generating the 3D shape"),
    OCTREE("Placing points"),
    GAUSSIANS("Decoding Gaussians"),
    PLY("Saving the .ply"),
}

/** What the engine tells the screen while it works (called on the model thread). */
interface Progress {
    fun stage(stage: Stage, detail: String?, seconds: Double?, done: Boolean) {}
    fun status(text: String) {}
    fun cutout(composite: Image8) {}
}

class CancelledException : Exception("cancelled")

data class EngineOptions(
    /** Compile the graphs before the first run and keep them (true), or per stage, closed after it (false). */
    val preload: Boolean = true,
    /**
     * MemAvailable the memory guard keeps free. MemAvailable counts the page cache of the mapped graph files, and with
     * 1 GB kept a compile burst outran its reclaim: lmkd killed the app on the Galaxy S26 at about 1.9 GB available.
     */
    val reserveMb: Long = 2500,
    /** Fail a run with a message when a graph does not fit even after closing all others (instead of trying). */
    val memoryCheck: Boolean = true,
    /** Pass a graph's output buffer straight to the next graph instead of copying it through the host. */
    val directHandoff: Boolean = false,
    /** "bir" (BiRefNet on the CPU, the author's matting) or "dis" (IS-Net DIS on the GPU). */
    val matting: String = "bir",
    val cpuThreads: Int = 6,
    val gpuPriority: CompiledModel.GpuOptions.Priority? = null,
    val disFloor: Int = Matting.DIS_ALPHA_FLOOR,
)

class RunResult(
    val ply: File,
    val gaussians: Int,
    /** Seconds: matting, preprocess (matting included, as idol_runner times it), dinov3, vae, denoiser, octree,
     * gaussian_decoder, ply, generate (preprocessed pixels ready -> .ply written), all. */
    val times: LinkedHashMap<String, Double>,
    val pre: Preprocessed,
    val compiles: List<GraphPool.CompileRecord>,
    val events: List<JSONObject>,
    val noiseSource: String,
    val handoff: String,
)

/**
 * The phone pipeline, step for step scripts/idol_runner.py: matting -> preprocessing -> DINOv3 (4 graphs) + VAE
 * (6 graphs) -> 5 Euler steps (den_ctx once per LoRA phase, den_in, 24 joint graphs) -> octree (8 levels, row
 * buckets) -> Gaussian decoder (8 graphs) -> .ply. Every graph runs on its own GPU call (the 2 s rule of F0.md
 * finding 2), every call on the one model thread. Create, use and close it on that thread.
 */
class IdolEngine(val modelDir: File, val manifest: Manifest, val options: EngineOptions) : AutoCloseable {
    private val env: Environment = Environment.create()
    val pool = GraphPool(
        env, modelDir, manifest, options.reserveMb * 1024, options.memoryCheck, options.cpuThreads, options.gpuPriority,
    )

    private val posEmb = side("den_pos_emb.f32")
    private val ropeCos = side("dino_rope_cos.f32")
    private val ropeSin = side("dino_rope_sin.f32")
    private val perturbation = side("gs_perturbation.f32")
    private val bnMean: FloatArray
    private val bnStd: FloatArray
    private val rep: GaussianRep
    private val gateSlow = Glue.loraGate(true)
    private val gateFast = Glue.loraGate(false)
    private val timeGrid = Glue.timeGrid()

    /** Set from the screen thread; the model thread stops at the next graph. */
    @Volatile var cancel = false

    /** Direct handoff, until the first failure (then host copies, and the reason is kept). */
    var direct = options.directHandoff
        private set
    var directFailure: String? = null
        private set

    private var progress: Progress = object : Progress {}

    init {
        val bn = JSONObject(File(modelDir, "vae_bn.json").readText())
        val mean = bn.getJSONArray("bn_mean")
        val variance = bn.getJSONArray("bn_var")
        val eps = bn.getDouble("bn_eps").toFloat()
        bnMean = FloatArray(mean.length()) { mean.getDouble(it).toFloat() }
        bnStd = FloatArray(variance.length()) { sqrt((variance.getDouble(it).toFloat() + eps).toDouble()).toFloat() }
        rep = GaussianRep.parse(JSONObject(File(modelDir, "gs_rep.json").readText()))
    }

    private fun side(name: String): FloatArray {
        val s = requireNotNull(manifest.sideFiles[name]) { "manifest.json has no side file $name" }
        return Fixture.readF32(File(modelDir, name), s.elements.toInt())
    }

    override fun close() {
        pool.close()
        runCatching { env.close() }
    }

    // ------------------------------------------------------------------------------------------------ schedule
    // Each chain's graphs in manifest order (manifest.json lists them in pipeline order: dino_0..3, vae_0..3, vae_attn,
    // vae_4, den_joint_00..23, gs_0..7), so a stage split into more or fewer graphs needs a new manifest, not new code.
    private fun chain(prefix: String): List<String> = manifest.graphs.keys.filter { it.startsWith(prefix) }.also {
        require(it.isNotEmpty()) { "manifest.json has no $prefix graphs" }
    }
    private val dinoNames = chain("dino_")
    private val vaeNames = chain("vae_")
    private val jointNames = chain("den_joint_")
    private val gsNames = chain("gs_")

    /** The order a run asks for graphs in. */
    fun runSchedule(): List<String> {
        val s = ArrayList<String>()
        s.add(options.matting)
        s.addAll(dinoNames)
        s.addAll(vaeNames)
        var phase: Boolean? = null
        for (i in 0 until timeGrid.size - 1) {
            val slow = timeGrid[i] >= Glue.PHASE_THRESHOLD
            if (slow != phase) s.add("den_ctx")
            phase = slow
            s.add("den_in")
            s.addAll(jointNames)
        }
        repeat(Glue.MAX_LEVEL) { s.add("oct") }
        s.addAll(gsNames)
        return s
    }

    /** The GPU graphs of a run in first-use order (the CPU matting graph is built for each matting and closed). */
    fun preloadOrder(): List<String> = runSchedule().distinct().filter { !manifest.graph(it).onCpu }

    /**
     * preload = true: compiles the graphs before the first run, in first-use order, as many as fit above the reserve.
     * Returns the load record (seconds per graph and the total).
     */
    fun preload(progress: Progress): JSONObject {
        cancel = false
        val names = preloadOrder()
        pool.phase = "preload"
        pool.setSchedule(runSchedule())
        // a run starts with the matting graph: on the CPU its call needs most of its memory, so leave that free and
        // the graphs compiled here stay resident through it
        val matting = manifest.graph(options.matting)
        pool.preloadHeadroomKb = if (matting.onCpu) pool.estimateKb(matting.name) else 0L
        val first = pool.compiles.size
        val t0 = SystemClock.elapsedRealtimeNanos()
        var loaded = 0
        var stoppedAt: String? = null
        try {
            for (n in names) {
                if (cancel) throw CancelledException()
                progress.status("Loading models ${loaded + 1}/${names.size}")
                if (!pool.preload(n)) {
                    stoppedAt = n
                    break
                }
                loaded++
            }
        } finally {
            pool.phase = "run"
        }
        val total = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9
        progress.status(
            if (stoppedAt == null) "Models loaded ($loaded/${names.size})"
            else "Loaded $loaded/${names.size} models; the rest load during the run (memory)",
        )
        val graphs = JSONArray()
        for (r in pool.compiles.subList(first, pool.compiles.size)) graphs.put(r.json())
        return JSONObject().put("graphs", graphs).put("loaded", loaded).put("of", names.size)
            .put("stopped_at", stoppedAt ?: JSONObject.NULL).put("total_s", total)
            .put("compile_s", pool.compiles.subList(first, pool.compiles.size).sumOf { it.seconds })
            .put("resident", JSONArray(pool.residentNames)).put("headroom_kb", pool.preloadHeadroomKb)
    }

    // ------------------------------------------------------------------------------------------------ the run
    /** A picture (or a fixture's raw image) -> .ply at [outPly]. */
    fun generate(image: Image8, mode: String, noise: Noise, outPly: File, progress: Progress): RunResult {
        cancel = false
        this.progress = progress
        val firstCompile = pool.compiles.size
        val firstEvent = pool.events.size
        pool.phase = "run"
        pool.setSchedule(runSchedule())
        val times = LinkedHashMap<String, Double>()
        val tAll = now()

        progress.stage(Stage.MATTING, null, null, false)
        var mattingS = 0.0
        val tPre = now()
        val pre = Preprocess.run(image, mode) { rgb ->
            val t = now()
            val a = alpha(rgb)
            mattingS = since(t)
            a
        }
        times["matting"] = mattingS
        times["preprocess"] = since(tPre)
        pool.release(options.matting)
        progress.cutout(pre.composite)
        progress.stage(Stage.MATTING, null, times["preprocess"], true)
        val pixels = Preprocess.pixels(pre.composite)

        val tGen = now()
        progress.stage(Stage.DINO, null, null, false)
        var t = now()
        val f1 = dinov3(pixels)
        times["dinov3"] = since(t)
        progress.stage(Stage.DINO, null, times["dinov3"], true)
        if (!options.preload) pool.releaseIf { it.startsWith("dino_") }

        progress.stage(Stage.VAE, null, null, false)
        t = now()
        val moments = vaeMoments(pixels)
        times["vae"] = since(t)
        val f2 = cond2(Glue.vaeTokens(moments, noise.vae, bnMean, bnStd))
        progress.stage(Stage.VAE, null, times["vae"], true)
        if (!options.preload) pool.releaseIf { it.startsWith("vae_") }

        val tDen = now()
        val latent = sample(noise, f1, f2, times)
        progress.stage(Stage.DENOISER, null, since(tDen), true)
        if (!options.preload) pool.releaseIf { it.startsWith("den_") }

        progress.stage(Stage.OCTREE, null, null, false)
        t = now()
        val latentTag = nextTag()
        val anchors = Glue.octreeDescend({ cells, rows, level ->
            progress.stage(Stage.OCTREE, "level $level of ${Glue.MAX_LEVEL}", null, false)
            Glue.softmax8(octreeLogits(cells, rows, level, latent, latentTag), rows)
        }, noise.pool, noise.leaf)
        times["octree"] = since(t)
        progress.stage(Stage.OCTREE, null, times["octree"], true)
        if (!options.preload) pool.release("oct")

        progress.stage(Stage.GAUSSIANS, null, null, false)
        t = now()
        val features = gaussianFeatures(anchors, latent, latentTag)
        times["gaussian_decoder"] = since(t)
        if (!options.preload) pool.releaseIf { it.startsWith("gs_") }
        val g = Glue.toGaussians(anchors, features, rep, perturbation)
        progress.stage(Stage.GAUSSIANS, null, since(t), true)

        progress.stage(Stage.PLY, null, null, false)
        t = now()
        Glue.writePly(Glue.plyRows(g), outPly)
        times["ply"] = since(t)
        progress.stage(Stage.PLY, null, times["ply"], true)
        times["generate"] = since(tGen)
        times["all"] = since(tAll)
        Log.i(TAG, "run done: " + times.entries.joinToString { "${it.key} ${"%.2f".format(it.value)}" })
        return RunResult(
            outPly, g.count, times, pre,
            pool.compiles.subList(firstCompile, pool.compiles.size).toList(),
            pool.events.subList(firstEvent, pool.events.size).toList(),
            noise.source, handoffName(),
        )
    }

    fun handoffName(): String = when {
        direct -> "direct"
        options.directHandoff -> "host (direct failed: $directFailure)"
        else -> "host"
    }

    /** feature2 = 5 zero tokens + the 4,096 VAE tokens: [1, 4101, 128]. */
    private fun cond2(tokens: FloatArray): FloatArray {
        val f2 = FloatArray(Glue.COND_TOKENS * 128)
        System.arraycopy(tokens, 0, f2, 5 * 128, tokens.size)
        return f2
    }

    // ------------------------------------------------------------------------------------------------ stages
    private fun graph(name: String): Graph {
        if (cancel) throw CancelledException()
        if (!pool.isResident(name)) progress.status("Loading $name")
        val g = pool.get(name)
        progress.status("")
        return g
    }

    /**
     * The matting graph's uint8 alpha for the resized RGB picture; the graph is closed right after. The pool makes
     * room for the call's peak before the compile (BiRefNet on the CPU takes most of its memory during the call) and
     * keeps the peak this call measured for the next check.
     */
    private fun alpha(rgb: Image8): ByteArray {
        val m = options.matting
        val input = if (m == "bir") Matting.birefnetInput(rgb) else Matting.disInput(rgb)
        try {
            if (cancel) throw CancelledException()
            if (!pool.isResident(m)) progress.status("Loading $m")
            val map = pool.runWatched(m) { g ->
                progress.status("")
                g.write(0, input)
                g.run()
                g.read(0)
            }
            pool.release(m)
            return if (m == "bir") Matting.birefnetAlpha(map, rgb.width, rgb.height)
            else Matting.disAlpha(map, rgb.width, rgb.height, options.disFloor)
        } finally {
            pool.release(m)
        }
    }

    private class Link(val from: Graph, val fromOut: Int, val to: Graph, val toIn: Int, val direct: Boolean)

    /**
     * The next graph [toName] of a chain, fed [from]'s output [fromOut] at input [toIn]: copied through the host now
     * (the read waits for [from]'s GPU work), or handed over as the buffer itself at [run] with direct handoff.
     */
    private fun link(from: Graph, fromOut: Int, toName: String, toIn: Int = 0): Link {
        if (direct) {
            pool.pin(from.name)
            try {
                return Link(from, fromOut, graph(toName), toIn, true)
            } catch (e: Throwable) {
                pool.unpin(from.name)
                throw e
            }
        }
        val data = from.read(fromOut)
        val to = graph(toName)
        to.write(toIn, data)
        return Link(from, fromOut, to, toIn, false)
    }

    private fun run(l: Link) {
        if (!l.direct) {
            l.to.run()
            return
        }
        try {
            l.to.run(overrides = mapOf(l.toIn to l.from.outputBuffer(l.fromOut)))
            l.to.untag(l.toIn)
        } catch (e: LiteRtException) {
            Log.w(TAG, "direct handoff ${l.from.name} -> ${l.to.name} failed, host copies from now on", e)
            direct = false
            directFailure = "${l.from.name} -> ${l.to.name}: ${e.message}"
            l.to.write(l.toIn, l.from.read(l.fromOut))
            l.to.run()
        } finally {
            pool.unpin(l.from.name)
        }
    }

    private fun dinov3(pixels: FloatArray): FloatArray {
        var g = graph(dinoNames[0])
        g.write(0, pixels)
        g.write(1, ropeCos, "rope_cos")
        g.write(2, ropeSin, "rope_sin")
        g.run()
        for (name in dinoNames.drop(1)) {
            val l = link(g, 0, name)
            l.to.write(1, ropeCos, "rope_cos")
            l.to.write(2, ropeSin, "rope_sin")
            run(l)
            g = l.to
        }
        return g.read(0)
    }

    private fun vaeMoments(pixels: FloatArray): FloatArray {
        var g = graph(vaeNames[0])
        g.write(0, pixels)
        g.run()
        for (name in vaeNames.drop(1)) {
            val l = link(g, 0, name)
            run(l)
            g = l.to
        }
        return g.read(0)
    }

    /** The conditioning of one LoRA phase (den_ctx), kept for the steps of that phase. */
    private class Ctx(val h: FloatArray, val tag: Long)

    /** sample: 5 Euler steps from the shared noise -> the final latent [1, 8192, 16]. */
    private fun sample(noise: Noise, f1: FloatArray, f2: FloatArray, times: MutableMap<String, Double>): FloatArray {
        val xLat = noise.latent.copyOf()
        val xCam = noise.camera.copyOf()
        val ctx = HashMap<Boolean, Ctx>()
        val condTag = nextTag()
        var den = 0.0
        val steps = timeGrid.size - 1
        for (i in 0 until steps) {
            progress.stage(Stage.DENOISER, "step ${i + 1} of $steps", null, false)
            val t0 = now()
            val (v, vc) = velocity(xLat, xCam, timeGrid[i], f1, f2, condTag, ctx)
            den += since(t0)
            val dt = (timeGrid[i] - timeGrid[i + 1]).toFloat()
            for (k in xLat.indices) xLat[k] = xLat[k] - v[k] * dt
            for (k in xCam.indices) xCam[k] = xCam[k] - vc[k] * dt
        }
        times["denoiser"] = den
        return xLat
    }

    /** velocity: one denoiser call at time t -> (latent velocity [1, 8192, 16], camera velocity [1, 1, 5]). */
    private fun velocity(
        xLat: FloatArray,
        xCam: FloatArray,
        t: Double,
        f1: FloatArray,
        f2: FloatArray,
        condTag: Long,
        ctx: HashMap<Boolean, Ctx>,
    ): Pair<FloatArray, FloatArray> {
        val slow = t >= Glue.PHASE_THRESHOLD
        val gate = if (slow) gateSlow else gateFast
        val gateTag = if (slow) "gate_slow" else "gate_fast"
        val c = ctx.getOrPut(slow) {
            val g = graph("den_ctx")
            g.write(0, f1, "f1_$condTag")
            g.write(1, f2, "f2_$condTag")
            g.write(2, gate, gateTag)
            g.run()
            Ctx(g.read(0), nextTag())
        }
        val din = graph("den_in")
        din.write(0, xLat)
        din.write(1, xCam)
        din.write(2, Glue.sinusoid(1000 * t))
        din.write(3, posEmb, "pos_emb")
        din.write(4, gate, gateTag)
        din.run()
        val hCam = din.read(1)
        val tMod = din.read(2)
        val tEmb = din.read(3)
        val tModTag = nextTag()
        var prev = din
        for ((k, name) in jointNames.withIndex()) {
            val l = link(prev, 0, name)
            val j = l.to
            when (k) {
                0 -> {
                    j.write(1, c.h, c.tag)
                    j.write(2, hCam)
                    j.write(3, tMod, tModTag)
                    j.write(4, gate, gateTag)
                }
                jointNames.size - 1 -> {
                    j.write(1, tMod, tModTag)
                    j.write(2, tEmb)
                    j.write(3, gate, gateTag)
                }
                else -> {
                    j.write(1, tMod, tModTag)
                    j.write(2, gate, gateTag)
                }
            }
            run(l)
            prev = j
        }
        return prev.read(0) to prev.read(1)
    }

    /** octree_logits: the decoder's child logits for [n] cells at [level], rows padded to the bucket -> [n, 8]. */
    private fun octreeLogits(cells: FloatArray, n: Int, level: Int, latent: FloatArray, latentTag: Any?): FloatArray {
        val b = Glue.OCT_BUCKETS.first { n <= it }
        val key = "L$b"
        val c = FloatArray(b * 3)
        System.arraycopy(cells, 0, c, 0, n * 3)
        val pe = Glue.pointPe(c, b)
        val lf = Glue.sinusoid((1 shl level).toDouble(), maxPeriod = 1024.0, twoPi = true)
        val g = graph("oct")
        g.write(0, c, key = key)
        g.write(1, pe, key = key)
        g.write(2, lf, key = key)
        g.write(3, latent, latentTag, key)
        g.run(key)
        return g.read(0, key).copyOf(n * 8)
    }

    /** The Gaussian decoder's 8 graphs: anchors [8192, 3] + latent -> features [8192, 480]. */
    private fun gaussianFeatures(anchors: FloatArray, latent: FloatArray, latentTag: Any?): FloatArray {
        val pe = Glue.pointPe(anchors, Glue.NUM_POINTS)
        var g = graph(gsNames[0])
        g.write(0, anchors)
        g.write(1, pe)
        g.write(2, latent, latentTag)
        g.run()
        for (name in gsNames.drop(1)) {
            val l = link(g, 0, name)
            l.to.write(1, latent, latentTag)
            run(l)
            g = l.to
        }
        return g.read(0)
    }

    // ------------------------------------------------------------------------------------------------ parity
    /**
     * CONTRACT.md line (a) on the phone: each part gets the fixture's reference input for it (no chaining) and its
     * output is written as `<id>.<name>.f32` (little-endian float32) in [out]; the matting alpha as `<id>.alpha.u8`.
     */
    fun parity(fx: Fixture, out: File, id: String, progress: Progress): JSONObject {
        cancel = false
        this.progress = progress
        pool.phase = "parity"
        pool.setSchedule(emptyList())
        val arrays = JSONObject()
        val seconds = JSONObject()
        fun dump(name: String, data: FloatArray, shape: IntArray) {
            Fixture.writeF32(File(out, "$id.$name.f32"), data)
            arrays.put(name, JSONObject().put("file", "$id.$name.f32").put("shape", JSONArray(shape.toList())))
        }
        val result = JSONObject().put("id", id).put("fixture", fx.dir.name).put("matting", options.matting)

        progress.stage(Stage.MATTING, null, null, false)
        var t = now()
        val photo = Photo.fromFile(fx.image)
        val pre = Preprocess.run(photo.image, fx.mode) { alpha(it) }
        File(out, "$id.alpha.u8").writeBytes(pre.alphaRaw)
        result.put("alpha", JSONObject().put("file", "$id.alpha.u8").put("width", pre.alphaWidth)
            .put("height", pre.alphaHeight).put("box", JSONArray(pre.box.toList())))
        seconds.put("matting", since(t))
        progress.stage(Stage.MATTING, null, since(t), true)

        val pixels = fx.floats("pixels")
        progress.stage(Stage.DINO, null, null, false)
        t = now()
        dump("feature1", dinov3(pixels), intArrayOf(1, Glue.COND_TOKENS, 1280))
        seconds.put("dinov3", since(t))
        pool.releaseIf { it.startsWith("dino_") }
        progress.stage(Stage.DINO, null, since(t), true)

        progress.stage(Stage.VAE, null, null, false)
        t = now()
        val moments = vaeMoments(pixels)
        dump("vae_moments", moments, intArrayOf(1, 64, 128, 128))
        dump("vae_tokens", Glue.vaeTokens(moments, fx.floats("vae_noise"), bnMean, bnStd), intArrayOf(1, 4096, 128))
        seconds.put("vae", since(t))
        pool.releaseIf { it.startsWith("vae_") }
        progress.stage(Stage.VAE, null, since(t), true)

        t = now()
        val f1 = fx.floats("feature1")
        val f2 = fx.floats("feature2")
        // t comes from the engine's float64 grid, as the reference passes float(ts[i]) (1000 t rounds differently from
        // the fixture's float32 copy); the fixture's grid only confirms it is the same schedule
        val tg = fx.floats("t_grid")
        result.put("t_grid_matches_fixture", tg.size == timeGrid.size && tg.indices.all { tg[it] == timeGrid[it].toFloat() })
        val ctx = HashMap<Boolean, Ctx>()
        val condTag = nextTag()
        for (i in 0 until timeGrid.size - 1) {
            progress.stage(Stage.DENOISER, "step ${i + 1} of ${timeGrid.size - 1}", null, false)
            val (v, vc) = velocity(fx.floats("x_latent_$i"), fx.floats("x_camera_$i"), timeGrid[i], f1, f2, condTag, ctx)
            dump("v_latent_$i", v, intArrayOf(1, Glue.NUM_POINTS, Glue.IN_CH))
            dump("v_camera_$i", vc, intArrayOf(1, 1, Glue.CAM_CH))
        }
        seconds.put("denoiser", since(t))
        pool.releaseIf { it.startsWith("den_") }
        progress.stage(Stage.DENOISER, null, since(t), true)

        t = now()
        val latent = fx.floats("latent")
        val latentTag = nextTag()
        for (level in 1..Glue.MAX_LEVEL) {
            progress.stage(Stage.OCTREE, "level $level of ${Glue.MAX_LEVEL}", null, false)
            val cells = fx.floats("oct_cells_$level")
            val n = cells.size / 3
            dump("oct_logits_$level", octreeLogits(cells, n, level, latent, latentTag), intArrayOf(1, n, 8))
        }
        seconds.put("octree", since(t))
        pool.release("oct")
        progress.stage(Stage.OCTREE, null, since(t), true)

        progress.stage(Stage.GAUSSIANS, null, null, false)
        t = now()
        val anchors = fx.floats("anchors")
        val feats = gaussianFeatures(anchors, latent, latentTag)
        seconds.put("gaussian_decoder", since(t))
        pool.releaseIf { it.startsWith("gs_") }
        dump("gs_features", feats, intArrayOf(1, Glue.NUM_POINTS, 480))
        val g = Glue.toGaussians(anchors, feats, rep, perturbation)
        dump("g_xyz", g.xyz, intArrayOf(g.count, 3))
        dump("g_features_dc", g.featuresDc, intArrayOf(g.count, 1, 3))
        dump("g_scaling", g.scaling, intArrayOf(g.count, 3))
        dump("g_rotation", g.rotation, intArrayOf(g.count, 4))
        dump("g_opacity", g.opacity, intArrayOf(g.count, 1))
        progress.stage(Stage.GAUSSIANS, null, since(t), true)
        pool.phase = "run"
        return result.put("arrays", arrays).put("seconds", seconds).put("handoff", handoffName())
    }

    companion object {
        const val TAG = "IdolEngine"
        private val tags = AtomicLong(1)

        fun nextTag(): Long = tags.getAndIncrement()

        private fun now() = SystemClock.elapsedRealtimeNanos()

        private fun since(t0: Long) = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9
    }
}
