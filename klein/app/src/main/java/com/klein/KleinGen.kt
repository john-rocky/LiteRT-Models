package com.klein

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * Full on-device FLUX.2-klein-4B text-to-image generation and image editing.
 *
 * The 4B DiT and the Qwen3-4B text encoder are split into twelve int8 LiteRT
 * graphs, each under the ~1 GB size where the ML Drift OpenCL delegate compiles
 * reliably, and executed one at a time so peak memory is a single graph rather
 * than the 6.2 GB total. Everything the GPU cannot do — tokenization, rotary
 * tables, the causal/padding mask, the scheduler and the two tail permutations —
 * is precomputed on the host and pushed as .bin files (see scripts/gen_prep_klein.py).
 *
 * klein is step-wise distilled, so the loop is unusually plain: four steps, no
 * classifier-free guidance, no sign flip, a straight flow-matching Euler update
 *     latents += dsigma[step] * noise_pred
 * and a per-channel batch-norm denormalization before the VAE.
 *
 * Passing a reference bitmap switches to editing. `Flux2KleinPipeline` appends
 * the reference's VAE latent tokens to the noise tokens every step
 *     latent_model_input = cat([latents, image_latents], dim=1)
 * which doubles the image sequence (256 -> 512 tokens, joint 768 -> 1024) and
 * needs the `kce_*` graphs re-exported at that shape. The weights are identical,
 * so only the activations grow. Only the first [N_IMG] output tokens are the
 * noise prediction; the reference half is discarded, exactly as the pipeline's
 * `noise_pred[:, :latents.size(1)]` does.
 */
object KleinGen {

    private const val STEPS = 4
    private const val N_TXT = 512               // text tokens
    private const val N_IMG = 256               // packed image tokens (16x16)
    private const val ENC_CHUNKS = 3            // encoder layers 1-9 / 10-18 / 19-27
    private const val ENC_DIM = 2560            // Qwen3-4B hidden size
    private const val DOUBLE_CHUNKS = 2
    private const val SINGLE_CHUNKS = 4
    private const val PACKED_CH = 128           // channels of one packed latent token
    private const val PACKED_HW = 16
    private const val LATENT_CH = 32
    private const val LATENT_HW = 32
    private const val IMAGE_SIZE = 256

    /**
     * Generates one image; the shared [Environment] is closed on the way out.
     *
     * @param reference when non-null, the picked image is edited rather than a
     *     new one generated. It is squared and resized to [IMAGE_SIZE] first.
     * @param prompt when non-null, it is tokenized and embedded on device instead
     *     of the prompt baked into the staged .bin files being used.
     */
    fun run(context: Context, reference: Bitmap? = null, prompt: String? = null,
            log: (String) -> Unit): Bitmap =
        Environment.create().use { env -> generate(env, context, reference, prompt, log) }

    /** True when the tokenizer and the embedding table have been staged. */
    fun isPromptEditable(context: Context): Boolean =
        PromptEncoder.isStaged(context.getExternalFilesDir(null)!!)

    private fun generate(env: Environment, context: Context, reference: Bitmap?,
                         prompt: String?, log: (String) -> Unit): Bitmap {
        val editing = reference != null
        val prefix = if (editing) "kce" else "kc"
        val binsDir = if (editing) "klein_bins_edit" else "klein_bins"
        val dir = context.getExternalFilesDir(null)!!
        fun bin(name: String) = readFloats(File(dir, "$binsDir/$name.bin"))
        fun indexBin(name: String) = readInts(File(dir, "$binsDir/$name.bin"))

        // The prompt is the only thing the graphs cannot derive: the rotary tables,
        // the schedule and the permutations all depend on positions, not on words.
        var inputsEmbeds = bin("inputs_embeds")
        var encMask = bin("enc_mask")
        val encCos = bin("enc_cos")
        val encSin = bin("enc_sin")
        val cos = bin("cos")
        val sin = bin("sin")
        val temb = bin("temb")                  // [STEPS, 3072]
        val dsigma = bin("dsigma")
        val bnMean = bin("bn_mean")
        val bnStd = bin("bn_std")
        val unpackPerm = indexBin("unpack_perm")
        val unpatchPerm = indexBin("unpatch_perm")
        var latents = bin("latents0")           // [N_IMG * PACKED_CH]

        val startMillis = System.currentTimeMillis()
        fun elapsed() = (System.currentTimeMillis() - startMillis) / 1000f

        if (prompt != null) {
            log("tokenizing prompt…")
            PromptEncoder(dir).use { encoder ->
                val tokens = encoder.tokenize(prompt)
                inputsEmbeds = encoder.embed(tokens)
                encMask = encoder.attentionMask(tokens)
                log("prompt: ${tokens.realCount} tokens (${elapsed()}s)")
            }
        }

        val imageLatents = reference?.let {
            log("encoding reference image…")
            val tokens = encodeReference(env, dir, it, indexBin("patch_perm"), bnMean, bnStd)
            log("reference encoded (${elapsed()}s)")
            tokens
        }

        log("encoding prompt…")
        val promptEmbeds = encodeText(env, dir, inputsEmbeds, encMask, encCos, encSin)
        log("prompt encoded (${elapsed()}s)")

        val tembSize = temb.size / STEPS
        for (step in 0 until STEPS) {
            val stepTemb = temb.copyOfRange(step * tembSize, (step + 1) * tembSize)
            val tokens = if (imageLatents == null) latents else latents + imageLatents
            val noisePred = denoiseStep(env, dir, prefix, tokens, promptEmbeds,
                stepTemb, cos, sin)
            for (i in latents.indices) {
                latents[i] += dsigma[step] * noisePred[i]
            }
            log("step ${step + 1}/$STEPS done (${elapsed()}s)")
        }

        val latent = toLatentImage(latents, unpackPerm, unpatchPerm, bnMean, bnStd)
        val image = ChunkRunner.gpu(env, "kv_vae.tflite", dir, listOf(latent))[0]
        log("VAE decoded (${elapsed()}s total)")
        return toBitmap(image)
    }

    /**
     * VAE-encodes the reference image into the DiT's packed latent tokens.
     *
     * Mirrors `Flux2KleinPipeline._encode_vae_image`: encode to the latent mean,
     * patchify 2x2 (a pure permutation, precomputed), batch-norm normalize per
     * packed channel, then pack [PACKED_CH, 16, 16] into [256, PACKED_CH] tokens.
     */
    private fun encodeReference(env: Environment, dir: File, reference: Bitmap,
                                patchPerm: IntArray, bnMean: FloatArray,
                                bnStd: FloatArray): FloatArray {
        val latent = ChunkRunner.gpu(env, "kv_vae_enc.tflite", dir,
            listOf(toPlanarRgb(reference)))[0]          // [LATENT_CH, 32, 32]
        val packed = gather(latent, patchPerm)          // [PACKED_CH, 16, 16]
        val plane = PACKED_HW * PACKED_HW
        for (channel in 0 until PACKED_CH) {
            val base = channel * plane
            for (i in 0 until plane) {
                packed[base + i] = (packed[base + i] - bnMean[channel]) / bnStd[channel]
            }
        }
        val tokens = FloatArray(plane * PACKED_CH)      // [256, PACKED_CH]
        for (channel in 0 until PACKED_CH) {
            for (i in 0 until plane) {
                tokens[i * PACKED_CH + channel] = packed[channel * plane + i]
            }
        }
        return tokens
    }

    /** Center-crops, resizes and converts to planar RGB in [-1,1]. */
    private fun toPlanarRgb(source: Bitmap): FloatArray {
        val side = minOf(source.width, source.height)
        val square = Bitmap.createBitmap(source, (source.width - side) / 2,
            (source.height - side) / 2, side, side)
        val scaled = Bitmap.createScaledBitmap(square, IMAGE_SIZE, IMAGE_SIZE, true)
        val plane = IMAGE_SIZE * IMAGE_SIZE
        val pixels = IntArray(plane)
        scaled.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        val out = FloatArray(3 * plane)
        for (p in 0 until plane) {
            val pixel = pixels[p]
            out[p] = ((pixel shr 16 and 0xFF) / 127.5f) - 1f
            out[plane + p] = ((pixel shr 8 and 0xFF) / 127.5f) - 1f
            out[2 * plane + p] = ((pixel and 0xFF) / 127.5f) - 1f
        }
        return out
    }

    /**
     * Runs the three encoder chunks and interleaves their outputs.
     *
     * klein conditions on Qwen3 hidden states from layers 9 / 18 / 27, stacked
     * channel-wise into 3 x 2560 = 7680 channels per token. The tap positions are
     * exactly the chunk boundaries, so each chunk's output is both the next
     * chunk's input and one third of the conditioning.
     */
    private fun encodeText(env: Environment, dir: File, inputsEmbeds: FloatArray,
                           mask: FloatArray, cos: FloatArray, sin: FloatArray): FloatArray {
        val taps = ArrayList<FloatArray>(ENC_CHUNKS)
        var hidden = inputsEmbeds
        for (i in 0 until ENC_CHUNKS) {
            hidden = ChunkRunner.gpu(env, "ke_enc$i.tflite", dir,
                listOf(hidden, mask, cos, sin))[0]
            taps.add(hidden)
        }
        val out = FloatArray(N_TXT * ENC_CHUNKS * ENC_DIM)
        for (token in 0 until N_TXT) {
            for (tap in 0 until ENC_CHUNKS) {
                System.arraycopy(taps[tap], token * ENC_DIM, out,
                    token * ENC_CHUNKS * ENC_DIM + tap * ENC_DIM, ENC_DIM)
            }
        }
        return out
    }

    /**
     * One DiT step: prep -> double blocks -> host concat -> single blocks -> final.
     *
     * [tokens] is the noise tokens for text-to-image, or the noise tokens followed
     * by the reference tokens when editing. The returned prediction is trimmed to
     * the noise half by the caller's loop bounds.
     */
    private fun denoiseStep(env: Environment, dir: File, prefix: String,
                            tokens: FloatArray, promptEmbeds: FloatArray,
                            temb: FloatArray, cos: FloatArray,
                            sin: FloatArray): FloatArray {
        val prep = ChunkRunner.gpu(env, "${prefix}_prep.tflite", dir,
            listOf(tokens, promptEmbeds, temb))
        var hidden = prep[0]
        var encoder = prep[1]
        val modImage = prep[2]
        val modText = prep[3]
        val modSingle = prep[4]

        for (i in 0 until DOUBLE_CHUNKS) {
            val out = ChunkRunner.gpu(env, "${prefix}_double$i.tflite", dir,
                listOf(hidden, encoder, cos, sin, modImage, modText))
            hidden = out[0]
            encoder = out[1]
        }
        // The single-stream blocks attend over one joint sequence: text then image.
        var joint = encoder + hidden
        for (i in 0 until SINGLE_CHUNKS) {
            joint = ChunkRunner.gpu(env, "${prefix}_single$i.tflite", dir,
                listOf(joint, cos, sin, modSingle))[0]
        }
        return ChunkRunner.gpu(env, "${prefix}_final.tflite", dir, listOf(joint, temb))[0]
    }

    /**
     * Packed tokens -> VAE latent: scatter by position id, denormalize, unpatchify.
     *
     * Both reorderings are pure permutations of the flat buffer, so they are
     * precomputed on the host as gather index maps rather than re-derived here.
     */
    private fun toLatentImage(latents: FloatArray, unpackPerm: IntArray,
                              unpatchPerm: IntArray, bnMean: FloatArray,
                              bnStd: FloatArray): FloatArray {
        val unpacked = gather(latents, unpackPerm)      // [PACKED_CH, 16, 16]
        val plane = PACKED_HW * PACKED_HW
        for (channel in 0 until PACKED_CH) {
            val base = channel * plane
            for (i in 0 until plane) {
                unpacked[base + i] = unpacked[base + i] * bnStd[channel] + bnMean[channel]
            }
        }
        return gather(unpacked, unpatchPerm)            // [LATENT_CH, 32, 32]
    }

    /** out[i] = source[perm[i]] — patchify / unpatchify via a precomputed index map. */
    private fun gather(source: FloatArray, perm: IntArray): FloatArray {
        val out = FloatArray(perm.size)
        for (i in perm.indices) {
            out[i] = source[perm[i]]
        }
        return out
    }

    /** [1,3,256,256] planar RGB in [-1,1] -> ARGB bitmap. */
    private fun toBitmap(image: FloatArray): Bitmap {
        val plane = IMAGE_SIZE * IMAGE_SIZE
        val pixels = IntArray(plane)
        for (p in 0 until plane) {
            val r = ((image[p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            val g = ((image[plane + p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            val b = ((image[2 * plane + p].coerceIn(-1f, 1f) + 1f) * 127.5f).toInt()
            pixels[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
    }

    /** Reads a little-endian float32 .bin written by scripts/gen_prep_klein.py. */
    private fun readFloats(file: File): FloatArray {
        val bytes = file.readBytes()
        val out = FloatArray(bytes.size / 4)
        var offset = 0
        for (i in out.indices) {
            out[i] = Float.fromBits(readLittleEndianInt(bytes, offset))
            offset += 4
        }
        return out
    }

    /** Reads a little-endian int32 .bin (the gather index maps). */
    private fun readInts(file: File): IntArray {
        val bytes = file.readBytes()
        val out = IntArray(bytes.size / 4)
        var offset = 0
        for (i in out.indices) {
            out[i] = readLittleEndianInt(bytes, offset)
            offset += 4
        }
        return out
    }

    private fun readLittleEndianInt(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or
            ((bytes[offset + 1].toInt() and 0xff) shl 8) or
            ((bytes[offset + 2].toInt() and 0xff) shl 16) or
            ((bytes[offset + 3].toInt() and 0xff) shl 24)
}
