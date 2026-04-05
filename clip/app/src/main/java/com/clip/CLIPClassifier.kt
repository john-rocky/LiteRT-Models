package com.clip

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * CLIP zero-shot classifier: image encoder (TFLite GPU) + pre-computed text embeddings.
 *
 * Usage:
 *   1. Call classify(bitmap) — returns sorted label scores
 *   2. Call classify(bitmap, customLabels) — uses ONNX text encoder for custom labels
 */
class CLIPClassifier(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "CLIP"
        private const val IMAGE_ENCODER_MODEL = "clip_image_encoder.tflite"
        private const val TEXT_ENCODER_MODEL = "clip_text_encoder.onnx"
        private const val LABELS_FILE = "labels.txt"
        private const val EMBEDDINGS_FILE = "text_embeddings.bin"
        private const val IMAGE_SIZE = 224
        private const val EMBED_DIM = 512

        // CLIP normalization (ImageNet stats, pixel values 0-255)
        private val MEAN = floatArrayOf(0.48145466f * 255f, 0.4578275f * 255f, 0.40821073f * 255f)
        private val STD = floatArrayOf(0.26862954f * 255f, 0.26130258f * 255f, 0.27577711f * 255f)
    }

    data class ClassificationResult(
        val label: String,
        val score: Float
    )

    // Image encoder: CompiledModel GPU
    private val compiledModel: CompiledModel
    private val encoderInputBuffers: List<TensorBuffer>

    // Pre-computed text embeddings
    private val labels: List<String>
    private val textEmbeddings: FloatArray  // [numLabels * EMBED_DIM]
    private val numLabels: Int
    private val embedDim: Int

    // Pre-allocated buffers
    private val inputFloats = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
    private val resizedBitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
    private val inputPixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
    private val scaleMatrix = Matrix()
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    var lastInferenceMs = 0L; private set
    var acceleratorName = ""; private set

    // Optional: ONNX text encoder for custom labels
    private var textEncoderSession: ai.onnxruntime.OrtSession? = null
    private var ortEnv: ai.onnxruntime.OrtEnvironment? = null

    init {
        // Load image encoder from files dir (too large for APK assets)
        val modelFile = java.io.File(context.filesDir, IMAGE_ENCODER_MODEL)
        if (!modelFile.exists()) {
            // Also check assets as fallback (for smaller models)
            try {
                context.assets.open(IMAGE_ENCODER_MODEL).use { input ->
                    modelFile.outputStream().use { output -> input.copyTo(output) }
                }
                Log.i(TAG, "Copied model from assets to filesDir")
            } catch (_: Exception) {
                throw IllegalStateException(
                    "Model not found. Push via adb:\n" +
                    "adb push clip_image_encoder.tflite /data/local/tmp/ && " +
                    "adb shell run-as com.clip cp /data/local/tmp/$IMAGE_ENCODER_MODEL " +
                    "/data/data/com.clip/files/"
                )
            }
        }
        val modelPath = modelFile.absolutePath
        Log.i(TAG, "Loading image encoder: $modelPath (${modelFile.length() / 1_000_000} MB)")

        compiledModel = try {
            val gpuOpts = CompiledModel.Options(Accelerator.GPU)
            try {
                gpuOpts.gpuOptions = CompiledModel.GpuOptions(
                    null, null, null,
                    CompiledModel.GpuOptions.Precision.FP32,
                    null, null, null, null, null, null, null, null, null, null, null
                )
            } catch (_: Exception) {}
            val m = CompiledModel.create(modelPath, gpuOpts, null)
            acceleratorName = "GPU"
            Log.i(TAG, "Image encoder GPU ready")
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, falling back to CPU")
            val cpuOpts = CompiledModel.Options(Accelerator.CPU)
            val m = CompiledModel.create(modelPath, cpuOpts, null)
            acceleratorName = "CPU"
            Log.i(TAG, "Image encoder CPU ready")
            m
        }
        encoderInputBuffers = compiledModel.createInputBuffers()

        // Load pre-computed text embeddings
        labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
            .filter { it.isNotBlank() }
        Log.i(TAG, "Loaded ${labels.size} labels")

        val embStream = DataInputStream(context.assets.open(EMBEDDINGS_FILE))
        val buf = ByteBuffer.wrap(embStream.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        numLabels = buf.int
        embedDim = buf.int
        textEmbeddings = FloatArray(numLabels * embedDim)
        buf.asFloatBuffer().get(textEmbeddings)
        embStream.close()
        Log.i(TAG, "Loaded embeddings: $numLabels labels x $embedDim dim")

        // Try loading optional ONNX text encoder
        try {
            val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
            val onnxPath = copyAssetToCache(context, TEXT_ENCODER_MODEL)
            textEncoderSession = env.createSession(onnxPath)
            ortEnv = env
            Log.i(TAG, "Text encoder loaded (custom labels enabled)")
        } catch (e: Exception) {
            Log.i(TAG, "Text encoder not available (pre-computed labels only)")
        }
    }

    /**
     * Classify image using pre-computed label embeddings.
     * Returns all labels sorted by descending score.
     */
    fun classify(bitmap: Bitmap): List<ClassificationResult> {
        val t = System.nanoTime()

        val imageEmbedding = encodeImage(bitmap)
        val scores = computeSimilarity(imageEmbedding, textEmbeddings, numLabels, embedDim)

        // Softmax with temperature (CLIP uses logit_scale ≈ 100)
        val temperature = 100f
        val scaledScores = FloatArray(scores.size) { scores[it] * temperature }
        val softmaxScores = softmax(scaledScores)

        lastInferenceMs = (System.nanoTime() - t) / 1_000_000

        return labels.zip(softmaxScores.toList())
            .map { ClassificationResult(it.first, it.second) }
            .sortedByDescending { it.score }
    }

    /** Get the default label list. */
    fun getLabels(): List<String> = labels

    /** Check if custom label encoding is available. */
    fun hasTextEncoder(): Boolean = textEncoderSession != null

    private fun encodeImage(bitmap: Bitmap): FloatArray {
        // Resize to 224x224 with center crop
        val canvas = Canvas(resizedBitmap)
        val srcSize = minOf(bitmap.width, bitmap.height)
        val srcX = (bitmap.width - srcSize) / 2
        val srcY = (bitmap.height - srcSize) / 2

        scaleMatrix.setRectToRect(
            android.graphics.RectF(srcX.toFloat(), srcY.toFloat(),
                (srcX + srcSize).toFloat(), (srcY + srcSize).toFloat()),
            android.graphics.RectF(0f, 0f, IMAGE_SIZE.toFloat(), IMAGE_SIZE.toFloat()),
            Matrix.ScaleToFit.FILL
        )
        canvas.drawBitmap(bitmap, scaleMatrix, scalePaint)
        resizedBitmap.getPixels(inputPixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // Normalize: (pixel - mean) / std, NCHW layout [1, 3, 224, 224]
        val planeSize = IMAGE_SIZE * IMAGE_SIZE
        for (i in inputPixels.indices) {
            val pixel = inputPixels[i]
            inputFloats[i] = (((pixel shr 16) and 0xFF).toFloat() - MEAN[0]) / STD[0]
            inputFloats[planeSize + i] = (((pixel shr 8) and 0xFF).toFloat() - MEAN[1]) / STD[1]
            inputFloats[2 * planeSize + i] = ((pixel and 0xFF).toFloat() - MEAN[2]) / STD[2]
        }
        encoderInputBuffers[0].writeFloat(inputFloats)

        val resultBuffers = compiledModel.run(encoderInputBuffers)
        return resultBuffers[0].readFloat()
    }

    private fun computeSimilarity(
        imageEmb: FloatArray,
        textEmbs: FloatArray,
        numLabels: Int,
        dim: Int
    ): FloatArray {
        // Dot product between L2-normalized vectors = cosine similarity
        val scores = FloatArray(numLabels)
        for (i in 0 until numLabels) {
            var dot = 0f
            val offset = i * dim
            for (j in 0 until dim) {
                dot += imageEmb[j] * textEmbs[offset + j]
            }
            scores[i] = dot
        }
        return scores
    }

    private fun softmax(scores: FloatArray): FloatArray {
        val maxScore = scores.max()
        val expScores = FloatArray(scores.size) {
            kotlin.math.exp((scores[it] - maxScore).toDouble()).toFloat()
        }
        val sum = expScores.sum()
        return FloatArray(expScores.size) { expScores[it] / sum }
    }

    private fun copyAssetToCache(context: Context, name: String): String {
        val file = java.io.File(context.cacheDir, name)
        if (!file.exists()) {
            context.assets.open(name).use { input ->
                file.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return file.absolutePath
    }

    override fun close() {
        encoderInputBuffers.forEach { it.close() }
        compiledModel.close()
        textEncoderSession?.close()
        ortEnv?.close()
        resizedBitmap.recycle()
    }
}
