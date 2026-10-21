package com.iris3b

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Environment
import java.io.Closeable
import java.io.File
import java.util.Random

/**
 * The sample's model session: the files `install_to_device.sh` put in the app's files dir, one LiteRT
 * [Environment] for the process (a null Environment leaks an OpenCL context per compile), the
 * prompt tokenizer, and one [IrisPipeline] run per image.
 *
 * The files are what the engine's [RunFiles] reads: [CONFIG_FILE] (graph files and precision, the
 * residency policy, host tensors, tokenizer, embedding table) and every file it names. The app writes
 * its own rows file ([ROWS_FILE], with no rows) so that [RunFiles] can open the config.
 */
class ImageSession private constructor(
    private val files: RunFiles,
    private val config: EngineConfig,
    private val env: Environment,
    private val tokens: PromptTokens,
    /** The most caption tokens the text encoder reads: its 300-row window minus the template suffix. */
    val maxCaptionTokens: Int,
) : Closeable {

    val teGraphs: Int = config.textEncoder.size
    val steps: Int = files.host["sigmas"].size

    /** Graph runs in one image: the text encoder, the adapter, then every step's graphs. */
    val totalRuns: Int = teGraphs + 1 + steps * (config.dual.size + config.single.size + 1)
    val residency: String = config.residency.name

    /** Caption tokens of [prompt] before the cut at [maxCaptionTokens]. */
    fun countTokens(prompt: String): Int = tokens.encode(prompt).captionTokens

    /** The verify entry's noise: the upstream reference run's `torch.randn`, patch-major, from the host pack. */
    fun referenceNoise(): FloatArray = files.host[REFERENCE_NOISE]

    /** Makes one image (blocking, minutes). [stop] is read before every graph; [log] gets the timeline. */
    fun generate(prompt: String, noise: FloatArray, stop: () -> Boolean, log: StageLog): PipelineResult {
        val pipeline = IrisPipeline(
            LiteRtExec(env, files.dir),
            config,
            files.host,
            tokens,
            availableBytes = { MemoryWatch.availableBytes() },
            stop = stop,
        )
        return pipeline.generate(prompt, noise, log)
    }

    override fun close() {
        tokens.close()
        env.close()
    }

    companion object {
        /** The config `install_to_device.sh` installs (the ship form: 20 graphs on the GPU at FP32). */
        const val CONFIG_FILE = "iris3b_config.json"
        const val ROWS_FILE = "iris3b_rows.json"
        const val REFERENCE_NOISE = "noise_pix"
        private const val TOKENIZER_DIR = "tok"
        private const val SUFFIX_IDS = "prompt_suffix_ids.txt"

        /**
         * The model files still missing from [dir], by name; empty when the model is installed. Without
         * the config only the config is listed (the other names are in it).
         */
        fun missingFiles(dir: File): List<String> {
            if (!File(dir, CONFIG_FILE).isFile) return listOf(CONFIG_FILE)
            writeRows(dir)
            return requiredFiles(RunFiles(dir, ROWS_FILE)).filter { !File(dir, it).isFile }
        }

        /** Opens the installed model (seconds: the tokenizer tables are read here). */
        fun open(context: Context, dir: File): ImageSession {
            writeRows(dir)
            val files = RunFiles(dir, ROWS_FILE)
            val config = files.engineConfig()
            files.materializeTokenizer()
            val tokenizerDir = File(dir, TOKENIZER_DIR)
            val suffix = File(tokenizerDir, SUFFIX_IDS).readText().trim().split(',').size
            val tokens = PromptTokens(tokenizerDir)
            val env = try {
                Environment.create(context)
            } catch (failure: Throwable) {
                tokens.close()
                throw failure
            }
            return ImageSession(files, config, env, tokens, PromptTokens.WINDOW - suffix)
        }

        /**
         * Standard normal starting noise from [seed]. java.util.Random is not torch's generator, so a
         * seed does not reproduce the upstream image of the same seed.
         */
        fun seedNoise(seed: Long): FloatArray {
            val random = Random(seed)
            return FloatArray(HostOps.IMAGE_FLOATS) { random.nextGaussian().toFloat() }
        }

        /** The clamped CHW image in [-1, 1] as an opaque ARGB bitmap. */
        fun toBitmap(chw: FloatArray): Bitmap {
            val size = HostOps.IMAGE_SIZE
            val plane = size * size
            val pixels = IntArray(plane) { i ->
                (0xFF shl 24) or (to8Bit(chw[i]) shl 16) or (to8Bit(chw[plane + i]) shl 8) or to8Bit(chw[2 * plane + i])
            }
            return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
        }

        private fun to8Bit(v: Float): Int = ((v + 1f) * 127.5f + 0.5f).toInt().coerceIn(0, 255)

        /** The graphs, tokenizer, embedding table and host pack the config names. */
        private fun requiredFiles(files: RunFiles): List<String> {
            val c = files.config
            val host = c["host"] as? Map<*, *>
            val named = listOfNotNull(c["tokenizer"] as? String, c["embed"] as? String, host?.get("file") as? String)
            return (files.graphFiles() + named).distinct()
        }

        private fun writeRows(dir: File) {
            val rows = mapOf("kind" to "generic", "config" to CONFIG_FILE, "inputs" to emptyList<Any>(), "rows" to emptyList<Any>())
            File(dir, ROWS_FILE).writeText(Json.write(rows))
        }
    }
}
