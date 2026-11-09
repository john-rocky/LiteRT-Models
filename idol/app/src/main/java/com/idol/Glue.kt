package com.idol

import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

/** The five arrays of shared randomness (CONTRACT.md "Shared randomness"), flat in C order. */
class Noise(
    /** [1, 32, 128, 128] normal: the VAE posterior noise. */
    val vae: FloatArray,
    /** [1, 8192, 16] normal: x at t = 1 (the denoiser's latent tokens). */
    val latent: FloatArray,
    /** [1, 1, 5] normal: the camera at t = 1. */
    val camera: FloatArray,
    /** [8, 8192] uniform: the octree sampler's per-row offsets. */
    val pool: FloatArray,
    /** [1, 8192, 3] uniform: the leaf positions (a decoder with fewer anchors uses the first rows). */
    val leaf: FloatArray,
    /** "seed <n>" or the fixture it came from. */
    val source: String,
) {
    init {
        require(vae.size == 32 * 128 * 128 && latent.size == Glue.LATENT_TOKENS * Glue.IN_CH && camera.size == Glue.CAM_CH)
        require(pool.size == Glue.MAX_LEVEL * Glue.LATENT_TOKENS && leaf.size == Glue.LATENT_TOKENS * 3)
    }

    companion object {
        /**
         * The app's own draws for a run from a picture: same distributions as the reference (normal, uniform [0, 1)),
         * from java.util.Random, not torch's generator; the fixtures carry the reference's own numbers.
         */
        fun random(seed: Long): Noise {
            val r = java.util.Random(seed)
            fun normal(n: Int) = FloatArray(n) { r.nextGaussian().toFloat() }
            fun uniform(n: Int) = FloatArray(n) { r.nextFloat() }
            val vae = normal(32 * 128 * 128)
            val latent = normal(Glue.LATENT_TOKENS * Glue.IN_CH)
            val camera = normal(Glue.CAM_CH)
            val pool = uniform(Glue.MAX_LEVEL * Glue.LATENT_TOKENS)
            val leaf = uniform(Glue.LATENT_TOKENS * 3)
            return Noise(vae, latent, camera, pool, leaf, "seed $seed")
        }
    }
}

/** Activated Gaussians (z-up, as GaussianFixedlenDecoder.to_representation returns them), N = 8192 x 32. */
class Gaussians(
    val count: Int,
    val xyz: FloatArray,
    val featuresDc: FloatArray,
    val scaling: FloatArray,
    /** (w, x, y, z) */
    val rotation: FloatArray,
    val opacity: FloatArray,
)

/** gs_rep.json: the Gaussian decoder's representation constants. */
class GaussianRep(
    val baseOffsetScale: Float,
    val lrXyz: Float,
    val lrFeaturesDc: Float,
    val lrOpacity: Float,
    val lrScaling: Float,
    val lrRotation: Float,
    val perturbeSize: Float,
    val filterKernelSize: Double,
    val scalingBias: Float,
    val opacityBias: Float,
) {
    companion object {
        fun parse(o: JSONObject): GaussianRep {
            val rep = o.getJSONObject("rep")
            val lr = rep.getJSONObject("lr")
            return GaussianRep(
                baseOffsetScale = o.getDouble("base_offset_scale").toFloat(),
                lrXyz = lr.getDouble("_xyz").toFloat(),
                lrFeaturesDc = lr.getDouble("_features_dc").toFloat(),
                lrOpacity = lr.getDouble("_opacity").toFloat(),
                lrScaling = lr.getDouble("_scaling").toFloat(),
                lrRotation = lr.getDouble("_rotation").toFloat(),
                perturbeSize = rep.getDouble("perturbe_size").toFloat(),
                filterKernelSize = rep.getDouble("3d_filter_kernel_size"),
                scalingBias = rep.getDouble("scaling_bias").toFloat(),
                opacityBias = rep.getDouble("opacity_bias").toFloat(),
            )
        }
    }
}

/**
 * The host glue between the graphs, ported line by line from scripts/idol_glue.py, scripts/idol_graphs.py
 * (sinusoid, point_pe, lora_gate) and scripts/idol_runner.py (vae_tokens): the same order of float32 operations as
 * the torch / numpy code (each transcendental is computed in double and rounded to float32).
 */
object Glue {
    const val MAX_LEVEL = 8
    const val PER_ANCHOR = 32
    const val PHASE_THRESHOLD = 0.5
    /** The denoiser's latent tokens; also the row count of the shared octree noise ([8, 8192], [1, 8192, 3]). */
    const val LATENT_TOKENS = 8192

    /**
     * The author's anchor count (num_gaussians 262,144 / 32). The graphs set the count a run uses: the Gaussian
     * decoder's first input is [1, anchors, 3] (a form exported with fewer anchors makes fewer Gaussians).
     */
    const val NUM_POINTS = 8192
    const val IN_CH = 16
    const val CAM_CH = 5
    const val COND_TOKENS = 4101
    const val LORA_SCALE = 128.0f / 32.0f
    val OCT_BUCKETS = intArrayOf(64, 256, 1024, 4096, 8192)
    const val PLY_FLOATS = 17
    val PLY_NAMES = listOf(
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )

    private val PI_F = PI.toFloat()

    private fun expf(x: Float) = exp(x.toDouble()).toFloat()
    private fun logf(x: Float) = ln(x.toDouble()).toFloat()

    /** time_grid: t = r s / (1 + (r - 1) s), s = numpy.linspace(1, 0, steps + 1), float64. */
    fun timeGrid(steps: Int = 5, rescale: Double = 3.0): DoubleArray {
        val num = steps + 1
        val step = (0.0 - 1.0) / (num - 1)
        val s = DoubleArray(num) { i -> i * step + 1.0 }
        s[num - 1] = 0.0
        return DoubleArray(num) { rescale * s[it] / (1 + (rescale - 1) * s[it]) }
    }

    /** lora_gate: [1, 64], the slow expert's half (t >= 0.5) or the fast one's at LORA_SCALE. */
    fun loraGate(slow: Boolean): FloatArray = FloatArray(64) { i ->
        if ((i < 32) == slow) LORA_SCALE else 0f
    }

    /** sinusoid: SinusoidalEmbedder's frequency features of one scalar -> [1, dim]. */
    fun sinusoid(value: Double, dim: Int = 256, maxPeriod: Double = 10000.0, twoPi: Boolean = false): FloatArray {
        val half = dim / 2
        val c = (-ln(maxPeriod)).toFloat()
        val v = value.toFloat()
        val out = FloatArray(dim)
        for (k in 0 until half) {
            val f = expf((c * k.toFloat()) / half.toFloat())
            var a = v * f
            if (twoPi) {
                a *= 2f
                a *= PI_F
            }
            out[k] = cos(a.toDouble()).toFloat()
            out[half + k] = sin(a.toDouble()).toFloat()
        }
        return out
    }

    /**
     * torch.linspace(start, end, steps) in float32 as the reference runs' torch computes it: start + step * i below
     * halfway, end - step * (steps - 1 - i) from there, each one fused multiply-add (one rounding). Only called with
     * (0, 10, 170): there the product and the sum are exact in double, so the double expression rounds once too.
     */
    private fun linspaceF(start: Float, end: Float, steps: Int): FloatArray {
        val step = (end - start) / (steps - 1).toFloat()
        val halfway = steps / 2
        return FloatArray(steps) { i ->
            if (i < halfway) (step.toDouble() * i + start).toFloat() else (end - step.toDouble() * (steps - i - 1)).toFloat()
        }
    }

    /**
     * The frequencies where torch's float32 pow(2, x) is one ulp off the correctly rounded value, read from the Mac
     * reference run's point embedding (device/fx/g1_gs_0): with them, all 170 frequency columns of that embedding agree
     * to one ulp of sin / cos (without them six columns are off by up to 1.1e-3).
     */
    private val TORCH_POW_ULP = mapOf(43 to -1, 72 to -1, 95 to -1, 113 to 1, 138 to -1, 159 to 1)

    /** torch.pow(2.0, torch.linspace(0.0, 10.0, 170)), float32. */
    private val PE_FREQS: FloatArray by lazy {
        val lin = linspaceF(0f, 10f, 1024 / 3 / 2)
        FloatArray(lin.size) {
            val f = 2.0.pow(lin[it].toDouble()).toFloat()
            Float.fromBits(f.toRawBits() + (TORCH_POW_ULP[it] ?: 0))
        }
    }

    /** point_pe: PcdAbsolutePositionEmbedderv2 of [n] points (x, y, z) -> [n, 1024]. */
    fun pointPe(points: FloatArray, n: Int, channels: Int = 1024): FloatArray {
        val freqs = PE_FREQS
        val fd = freqs.size
        val out = FloatArray(n * channels)
        for (p in 0 until n) {
            for (c in 0 until 3) {
                val v = points[p * 3 + c]
                val base = p * channels + c * 2 * fd
                for (j in 0 until fd) {
                    val ang = (v * freqs[j]) * PI_F
                    out[base + j] = sin(ang.toDouble()).toFloat()
                    out[base + fd + j] = cos(ang.toDouble()).toFloat()
                }
            }
        }
        return out
    }

    /**
     * vae_tokens: posterior sample with the shared noise (logvar clipped to [-30, 20]), 2 x 2 patchify, bn ->
     * [1, 4096, 128]. moments [1, 64, 128, 128] (mean, logvar), noise [1, 32, 128, 128].
     */
    fun vaeTokens(moments: FloatArray, noise: FloatArray, bnMean: FloatArray, bnStd: FloatArray): FloatArray {
        val hw = 128
        val plane = hw * hw
        val out = FloatArray(4096 * 128)
        for (c in 0 until 32) for (y in 0 until hw) for (x in 0 until hw) {
            val i = c * plane + y * hw + x
            var lv = moments[32 * plane + i]
            lv = if (lv < -30f) -30f else if (lv > 20f) 20f else lv
            val lat = moments[i] + expf(0.5f * lv) * noise[i]
            val k = c * 4 + (y % 2) * 2 + (x % 2)
            val t = (y / 2) * 64 + (x / 2)
            out[t * 128 + k] = (lat - bnMean[k]) / bnStd[k]
        }
        return out
    }

    /** torch.softmax over the last dimension (8) of [rows, 8] logits: exp(x - max) times 1 / sum. */
    fun softmax8(logits: FloatArray, rows: Int): FloatArray {
        val out = FloatArray(rows * 8)
        for (r in 0 until rows) {
            val b = r * 8
            var mx = logits[b]
            for (k in 1 until 8) if (logits[b + k] > mx) mx = logits[b + k]
            var sum = 0f
            for (k in 0 until 8) {
                val e = expf(logits[b + k] - mx)
                out[b + k] = e
                sum += e
            }
            val inv = 1f / sum
            for (k in 0 until 8) out[b + k] = out[b + k] * inv
        }
        return out
    }

    /**
     * systematic_rows: idol's systematic_sample for one batch, row i drawing counts[i] samples from its 8 child
     * probabilities with the offset u[i] / counts[i] -> [rows, 8] sample counts. The cumulative sum accumulates in
     * double as torch's CPU cumsum does for float32.
     */
    fun systematicRows(probs: FloatArray, counts: IntArray, u: FloatArray, rows: Int): IntArray {
        val P = 8
        val out = IntArray(rows * P)
        val p = FloatArray(P)
        val cdf = FloatArray(P)
        for (r in 0 until rows) {
            val n = counts[r]
            if (n == 0) continue
            var s = 0f
            for (k in 0 until P) {
                val v = probs[r * P + k]
                p[k] = if (v < 0f) 0f else v
                s += p[k]
            }
            if (s == 0f) {
                for (k in 0 until P) p[k] = 1.0f / P
            } else {
                val d = if (s < 1f) 1f else s
                for (k in 0 until P) p[k] = p[k] / d
            }
            var acc = 0.0
            for (k in 0 until P) {
                acc += p[k]
                val c = acc.toFloat()
                cdf[k] = if (c > 1f) 1f else c
            }
            val nf = n.toFloat()
            val u0 = u[r] / nf
            for (j in 0 until n) {
                var us = u0 + j.toFloat() / nf
                if (us > 1f) us = 1f
                var idx = 0
                while (idx < P && cdf[idx] < us) idx++
                if (idx > P - 1) idx = P - 1
                out[r * P + idx]++
            }
        }
        return out
    }

    /**
     * octree_descend: idol OctreeProbabilityFixedlenDecoder.sample for one batch. [probsFn] (cells [rows, 3], rows,
     * level) -> child probabilities [rows, 8]. pool [8, >= rows], leaf [>= numPoints, 3] -> anchors [numPoints, 3]
     * in [0, 1]^3 (each level reads the first rows of its pool row, the leaves are the first numPoints rows).
     * [cellsOut], when given, receives each level's cells (the parity record).
     */
    fun octreeDescend(
        probsFn: (FloatArray, Int, Int) -> FloatArray,
        pool: FloatArray,
        leaf: FloatArray,
        numPoints: Int = NUM_POINTS,
        levels: Int = MAX_LEVEL,
    ): FloatArray {
        val poolRow = pool.size / levels
        require(poolRow >= numPoints && leaf.size >= numPoints * 3) { "octree noise is smaller than $numPoints points" }
        var prev = IntArray(3)
        var rows = 1
        var counts = intArrayOf(numPoints)
        for (l in 1..levels) {
            val res = (1 shl (l - 1)).toFloat()
            val cells = FloatArray(rows * 3) { (prev[it].toFloat() + 0.5f) / res }
            val probs = probsFn(cells, rows, l)
            val u = FloatArray(rows) { pool[(l - 1) * poolRow + it] }
            val sampled = systematicRows(probs, counts, u, rows)
            var m = 0
            for (v in sampled) if (v > 0) m++
            val next = IntArray(m * 3)
            val nextCounts = IntArray(m)
            var o = 0
            for (r in 0 until rows) for (c in 0 until 8) {
                val s = sampled[r * 8 + c]
                if (s <= 0) continue
                // CHILD_OFFSET[c] = (i, j, k) with c = k * 4 + j * 2 + i
                next[o * 3] = prev[r * 3] * 2 + (c and 1)
                next[o * 3 + 1] = prev[r * 3 + 1] * 2 + ((c shr 1) and 1)
                next[o * 3 + 2] = prev[r * 3 + 2] * 2 + ((c shr 2) and 1)
                nextCounts[o] = s
                o++
            }
            prev = next
            counts = nextCounts
            rows = m
        }
        val out = FloatArray(numPoints * 3)
        val scale = (1 shl levels).toFloat()
        var p = 0
        for (r in 0 until rows) for (k in 0 until counts[r]) {
            check(p < numPoints) { "octree sampling produced more than $numPoints points" }
            for (d in 0 until 3) out[p * 3 + d] = (prev[r * 3 + d].toFloat() + leaf[p * 3 + d]) / scale
            p++
        }
        check(p == numPoints) { "octree sampling produced $p points, not $numPoints" }
        return out
    }

    private fun softplus(x: Float): Float = if (x > 20f) x else ln1p(expf(x).toDouble()).toFloat()

    private fun sigmoid(x: Float): Float = 1f / (1f + expf(-x))

    /**
     * to_gaussians: GaussianFixedlenDecoder.to_representation + Gaussian.__init__ for one batch. anchors [L, 3],
     * features [L, 480] (per anchor: xyz 32x3, f_dc 32x3, scaling 32x3, rotation 32x4, opacity 32, offset scale 32),
     * perturbation [32, 3].
     */
    fun toGaussians(anchors: FloatArray, features: FloatArray, rep: GaussianRep, perturbation: FloatArray): Gaussians {
        val L = anchors.size / 3
        val n = L * PER_ANCHOR
        val width = PER_ANCHOR * (3 + 3 + 3 + 4 + 1 + 1)
        require(features.size == L * width) { "features hold ${features.size} values, not $L x $width" }
        val xyz = FloatArray(n * 3)
        val fdc = FloatArray(n * 3)
        val scaling = FloatArray(n * 3)
        val rotation = FloatArray(n * 4)
        val opacity = FloatArray(n)
        // inverse_softplus(scaling_bias), inverse_sigmoid(opacity_bias): float32 tensors in the reference
        val sb = rep.scalingBias
        val scalingOffset = sb + logf(-expm1(-sb.toDouble()).toFloat())
        val ob = rep.opacityBias
        val opacityOffset = logf(ob / (1f - ob))
        val kernel2 = (rep.filterKernelSize * rep.filterKernelSize).toFloat()
        val xyzAt = 0
        val fdcAt = PER_ANCHOR * 3
        val scaleAt = PER_ANCHOR * 6
        val rotAt = PER_ANCHOR * 9
        val opaAt = PER_ANCHOR * 13
        val oscAt = PER_ANCHOR * 14
        for (a in 0 until L) {
            val f = a * width
            for (j in 0 until PER_ANCHOR) {
                val g = a * PER_ANCHOR + j
                val offsetScale = softplus(features[f + oscAt + j] + rep.baseOffsetScale)
                for (c in 0 until 3) {
                    var off = features[f + xyzAt + j * 3 + c] * rep.lrXyz + perturbation[j * 3 + c]
                    off = tanh(off.toDouble()).toFloat() * 0.5f * rep.perturbeSize * offsetScale
                    val raw = off + anchors[a * 3 + c]
                    xyz[g * 3 + c] = raw * 1.0f - 0.5f
                    fdc[g * 3 + c] = features[f + fdcAt + j * 3 + c] * rep.lrFeaturesDc
                    val s = softplus(features[f + scaleAt + j * 3 + c] * rep.lrScaling + scalingOffset)
                    scaling[g * 3 + c] = sqrt((s * s + kernel2).toDouble()).toFloat()
                }
                val r0 = features[f + rotAt + j * 4] * rep.lrRotation + 1.0f
                val r1 = features[f + rotAt + j * 4 + 1] * rep.lrRotation + 0.0f
                val r2 = features[f + rotAt + j * 4 + 2] * rep.lrRotation + 0.0f
                val r3 = features[f + rotAt + j * 4 + 3] * rep.lrRotation + 0.0f
                var ss = r0 * r0
                ss += r1 * r1
                ss += r2 * r2
                ss += r3 * r3
                val norm = maxOf(sqrt(ss.toDouble()).toFloat(), 1e-12f)
                rotation[g * 4] = r0 / norm
                rotation[g * 4 + 1] = r1 / norm
                rotation[g * 4 + 2] = r2 / norm
                rotation[g * 4 + 3] = r3 / norm
                opacity[g] = sigmoid(features[f + opaAt + j] * rep.lrOpacity + opacityOffset)
            }
        }
        return Gaussians(n, xyz, fdc, scaling, rotation, opacity)
    }

    /** ply_rows: idol Gaussian.save_ply's 17 attributes per Gaussian (z-up turned to y-up), float32. */
    fun plyRows(g: Gaussians): FloatArray {
        val n = g.count
        val rows = FloatArray(n * PLY_FLOATS)
        val eps = 1e-6f
        val hi = (1.0 - 1e-6).toFloat()
        // quaternion_multiply([sqrt(.5), sqrt(.5), 0, 0], rotation) in float64 (numpy promotes), rounded at the end
        val aw = sqrt(0.5)
        val ax = sqrt(0.5)
        val ay = 0.0
        val az = 0.0
        for (i in 0 until n) {
            val o = i * PLY_FLOATS
            rows[o] = g.xyz[i * 3]
            rows[o + 1] = -g.xyz[i * 3 + 2]
            rows[o + 2] = g.xyz[i * 3 + 1]
            rows[o + 6] = g.featuresDc[i * 3]
            rows[o + 7] = g.featuresDc[i * 3 + 1]
            rows[o + 8] = g.featuresDc[i * 3 + 2]
            var op = g.opacity[i]
            op = if (op < eps) eps else if (op > hi) hi else op
            rows[o + 9] = logf(op / (1f - op))
            for (c in 0 until 3) rows[o + 10 + c] = logf(maxOf(g.scaling[i * 3 + c], eps))
            val bw = g.rotation[i * 4].toDouble()
            val bx = g.rotation[i * 4 + 1].toDouble()
            val by = g.rotation[i * 4 + 2].toDouble()
            val bz = g.rotation[i * 4 + 3].toDouble()
            rows[o + 13] = (aw * bw - ax * bx - ay * by - az * bz).toFloat()
            rows[o + 14] = (aw * bx + ax * bw + ay * bz - az * by).toFloat()
            rows[o + 15] = (aw * by - ax * bz + ay * bw + az * bx).toFloat()
            rows[o + 16] = (aw * bz + ax * by - ay * bx + az * bw).toFloat()
        }
        return rows
    }

    /** write_ply: binary little-endian PLY with the 17 float properties (the bytes plyfile writes). */
    fun writePly(rows: FloatArray, file: File) {
        val n = rows.size / PLY_FLOATS
        val header = buildString {
            append("ply\nformat binary_little_endian 1.0\nelement vertex ").append(n).append('\n')
            for (name in PLY_NAMES) append("property float ").append(name).append('\n')
            append("end_header\n")
        }
        val tmp = File(file.parentFile, file.name + ".tmp")
        BufferedOutputStream(FileOutputStream(tmp), 1 shl 20).use { out ->
            out.write(header.toByteArray(Charsets.US_ASCII))
            val chunk = 4096 * PLY_FLOATS
            val buf = ByteBuffer.allocate(chunk * 4).order(ByteOrder.LITTLE_ENDIAN)
            var i = 0
            while (i < rows.size) {
                val m = minOf(chunk, rows.size - i)
                buf.clear()
                buf.asFloatBuffer().put(rows, i, m)
                out.write(buf.array(), 0, m * 4)
                i += m
            }
        }
        if (!tmp.renameTo(file)) {
            tmp.copyTo(file, overwrite = true)
            tmp.delete()
        }
    }
}
