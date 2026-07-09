package com.zimage

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * Full on-device Z-Image-Turbo generation with the chunked S3-DiT.
 *
 * Runs the flow-matching denoising loop entirely on the Mali GPU: the DiT is
 * split into int8 graphs that each compile fully on the ML Drift OpenCL delegate
 * (FP32 compute — the adaLN/attention path overflows fp16), loaded one at a time
 * so the peak footprint is a single graph. The text encoder, tokenizer, RoPE and
 * scheduler are precomputed on the host and pushed as .bin (gen_bins/); the loop,
 * pad-token mask, x/c concat, classifier-free guidance and unpatchify run on the
 * host, and the VAE decodes the final latent to an image.
 *
 * Per step (Z-Image CFG): pos = DiT(cond), neg = DiT(uncond),
 * noise_pred = -(pos + guidance*(pos - neg)), latent += dsigma * noise_pred.
 * The image branch (embx -> refx) is shared by cond/uncond; the context branch
 * (embc -> refc) is step-independent and computed once per prompt.
 */
object ZImageGen {

    private const val STEPS = 8
    private const val NX = 256          // image tokens
    private const val NC = 32           // context tokens
    private const val DIM = 3840
    private const val LAT = 16 * 32 * 32
    private const val GUIDANCE = 1.0f

    fun run(context: Context, log: (String) -> Unit): Bitmap =
        Environment.create().use { env -> generate(env, context, log) }

    private fun generate(env: Environment, context: Context, log: (String) -> Unit): Bitmap {
        val dir = context.getExternalFilesDir(null)!!
        // Program cache disabled: GpuOptions(serializeProgramCache) aborts the ML Drift
        // OpenCL delegate on this runtime (both internal and external dirs), so each
        // graph recompiles every step (~200 s/step). null = recompile path.
        val cache: File? = null
        fun bin(name: String) = readFloats(File(dir, "gen_bins/$name.bin"))
        fun ibin(name: String) = readInts(File(dir, "gen_bins/$name.bin"))

        val capCond = bin("cap_b0"); val capUnc = bin("cap_b1")
        val adaln = bin("adaln")            // [8*256]
        val xc = bin("xc"); val xs = bin("xs"); val cc = bin("cc"); val cs = bin("cs")
        val uc = bin("uc"); val us = bin("us")
        val xpad = bin("xpad"); val cpad = bin("cpad"); val xpt = bin("xpt"); val cpt = bin("cpt")
        val dsigma = bin("dsigma")
        val patchPerm = ibin("patch_perm"); val unpatchPerm = ibin("unpatch_perm")
        var latent = bin("steps_0")         // initial noise [16*32*32]

        val runner = ChunkRunner
        val t0 = System.currentTimeMillis()

        // Context branch is step-independent: compute the refined cond/uncond context once.
        fun contextRef(cap: FloatArray): FloatArray {
            val cRaw = runner.gpu(env, "z_embc.tflite", dir, listOf(cap), log, cache)
            val c = padMask(cRaw, cpad, cpt, NC)
            return runner.gpu(env, "z_refc.tflite", dir, listOf(c, cc, cs), log, cache)
        }
        log("precomputing context (cond + uncond)…")
        val cRefCond = contextRef(capCond)
        val cRefUnc = contextRef(capUnc)

        for (s in 0 until STEPS) {
            val ad = adaln.copyOfRange(s * 256, s * 256 + 256)
            // image branch (shared by cond + uncond)
            val xt = gather(latent, patchPerm)                       // patchify -> [256*64]
            val xRaw = runner.gpu(env, "z_embx.tflite", dir, listOf(xt), log, cache)
            val x = padMask(xRaw, xpad, xpt, NX)
            val xRef = runner.gpu(env, "z_refx.tflite", dir, listOf(x, xc, xs, ad), log, cache)

            val pos = branch(env, runner, dir, xRef, cRefCond, uc, us, ad, unpatchPerm, log, cache)
            val neg = branch(env, runner, dir, xRef, cRefUnc, uc, us, ad, unpatchPerm, log, cache)

            // Z-Image CFG + flow-matching Euler update (host)
            for (i in 0 until LAT) {
                val np = -(pos[i] + GUIDANCE * (pos[i] - neg[i]))
                latent[i] += dsigma[s] * np
            }
            log("step ${s + 1}/$STEPS done (${(System.currentTimeMillis() - t0) / 1000f}s)")
        }

        val image = runner.gpu(env, "zvae_int8_256.tflite", dir, listOf(latent), log, cache)
        log("VAE decoded (${(System.currentTimeMillis() - t0) / 1000f}s total)")
        return toBitmap(image)
    }

    /** Main-layer stack + final for one branch: cat(x_ref, c_ref) -> mains -> final -> unpatch. */
    private fun branch(env: Environment, runner: ChunkRunner, dir: File, xRef: FloatArray,
                       cRef: FloatArray, uc: FloatArray, us: FloatArray, ad: FloatArray,
                       unpatchPerm: IntArray, log: (String) -> Unit, cache: File?): FloatArray {
        var hidden = xRef + cRef                                     // concat along seq -> [288*3840]
        for (i in 0 until 6) {
            hidden = runner.gpu(env, "zc_main$i.tflite", dir, listOf(hidden, uc, us, ad), log, cache)
        }
        val out = runner.gpu(env, "zc_final.tflite", dir, listOf(hidden, ad), log, cache)  // [288*64]
        return gather(out, unpatchPerm)                             // unpatchify -> [16*32*32]
    }

    /** out[i] = src[perm[i]] (patchify / unpatchify via a precomputed index map). */
    private fun gather(src: FloatArray, perm: IntArray): FloatArray {
        val out = FloatArray(perm.size)
        for (i in perm.indices) {
            out[i] = src[perm[i]]
        }
        return out
    }

    /** raw[t] where pad[t]==0, pad_token where pad[t]==1 (host, avoids the FC->MUL Mali wall). */
    private fun padMask(raw: FloatArray, pad: FloatArray, padToken: FloatArray, n: Int): FloatArray {
        val out = FloatArray(n * DIM)
        for (t in 0 until n) {
            val p = pad[t]; val keep = 1f - p; val base = t * DIM
            for (ch in 0 until DIM) {
                out[base + ch] = raw[base + ch] * keep + padToken[ch] * p
            }
        }
        return out
    }

    /** [1,3,256,256] in [-1,1] planar RGB -> ARGB bitmap. */
    private fun toBitmap(img: FloatArray): Bitmap {
        val hw = 256 * 256
        val pixels = IntArray(hw)
        for (p in 0 until hw) {
            val r = ((img[p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            val g = ((img[hw + p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            val b = ((img[2 * hw + p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            pixels[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, 256, 256, Bitmap.Config.ARGB_8888)
    }

    private fun readFloats(f: File): FloatArray {
        val bytes = f.readBytes()
        val out = FloatArray(bytes.size / 4)
        var j = 0
        for (i in out.indices) {
            out[i] = Float.fromBits((bytes[j].toInt() and 0xff) or ((bytes[j + 1].toInt() and 0xff) shl 8) or
                ((bytes[j + 2].toInt() and 0xff) shl 16) or ((bytes[j + 3].toInt() and 0xff) shl 24))
            j += 4
        }
        return out
    }

    private fun readInts(f: File): IntArray {
        val bytes = f.readBytes()
        val out = IntArray(bytes.size / 4)
        var j = 0
        for (i in out.indices) {
            out[i] = (bytes[j].toInt() and 0xff) or ((bytes[j + 1].toInt() and 0xff) shl 8) or
                ((bytes[j + 2].toInt() and 0xff) shl 16) or ((bytes[j + 3].toInt() and 0xff) shl 24)
            j += 4
        }
        return out
    }
}
