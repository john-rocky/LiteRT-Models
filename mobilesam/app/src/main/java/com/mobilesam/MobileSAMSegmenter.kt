package com.mobilesam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.nio.FloatBuffer

/**
 * MobileSAM segmenter: Encoder (TFLite GPU) + Decoder (ONNX Runtime CPU).
 *
 * Usage:
 *   1. Call encodeImage(bitmap) once per image — caches embeddings
 *   2. Call segment(x, y) for each tap — returns mask bitmap
 */
class MobileSAMSegmenter(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "MobileSAM"
        private const val ENCODER_MODEL = "mobilesam_encoder.tflite"
        private const val DECODER_MODEL = "mobilesam_decoder.onnx"
        private const val ENCODER_SIZE = 1024
        private const val EMBED_DIM = 256
        private const val EMBED_SIZE = 64
        private const val MASK_SIZE = 256

        // MobileSAM normalization (pixel values, not 0-1)
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }

    // Encoder: CompiledModel GPU
    private val compiledModel: CompiledModel
    private val encoderInputBuffers: List<TensorBuffer>

    // Decoder: ONNX Runtime CPU
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val decoderSession: OrtSession

    // Pre-allocated buffers
    private val encoderInputFloats = FloatArray(3 * ENCODER_SIZE * ENCODER_SIZE)
    private val resizedBitmap = Bitmap.createBitmap(ENCODER_SIZE, ENCODER_SIZE, Bitmap.Config.ARGB_8888)
    private val inputPixels = IntArray(ENCODER_SIZE * ENCODER_SIZE)
    private val scaleMatrix = Matrix()
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    // Cached state
    private var cachedEmbeddings: FloatArray? = null
    private var currentImageWidth = 0
    private var currentImageHeight = 0

    var lastEncodeTimeMs = 0L; private set
    var lastDecodeTimeMs = 0L; private set

    var acceleratorName = ""; private set

    init {
        Log.i(TAG, "Loading encoder: $ENCODER_MODEL")
        compiledModel = try {
            val gpuOpts = CompiledModel.Options(Accelerator.GPU)
            try {
                gpuOpts.gpuOptions = CompiledModel.GpuOptions(
                    null, null, null,
                    CompiledModel.GpuOptions.Precision.FP32,
                    null, null, null, null, null, null, null, null, null, null, null
                )
            } catch (_: Exception) {}
            val m = CompiledModel.create(context.assets, ENCODER_MODEL, gpuOpts, null)
            acceleratorName = "GPU"
            Log.i(TAG, "Encoder GPU ready")
            m
        } catch (e: Exception) {
            Log.w(TAG, "GPU failed: ${e.message}, falling back to CPU")
            val cpuOpts = CompiledModel.Options(Accelerator.CPU)
            val m = CompiledModel.create(context.assets, ENCODER_MODEL, cpuOpts, null)
            acceleratorName = "CPU"
            Log.i(TAG, "Encoder CPU ready")
            m
        }
        encoderInputBuffers = compiledModel.createInputBuffers()

        Log.i(TAG, "Loading decoder: $DECODER_MODEL")
        val decoderPath = copyAssetToCache(context, DECODER_MODEL)
        decoderSession = ortEnv.createSession(decoderPath)
        Log.i(TAG, "Decoder ready")
    }

    /**
     * Encode an image — run once per image load.
     * Caches embeddings for subsequent segment() calls.
     */
    fun encodeImage(bitmap: Bitmap) {
        currentImageWidth = bitmap.width
        currentImageHeight = bitmap.height

        val t = System.nanoTime()

        // Resize to 1024x1024
        val canvas = Canvas(resizedBitmap)
        scaleMatrix.setScale(
            ENCODER_SIZE.toFloat() / bitmap.width,
            ENCODER_SIZE.toFloat() / bitmap.height
        )
        canvas.drawBitmap(bitmap, scaleMatrix, scalePaint)
        resizedBitmap.getPixels(inputPixels, 0, ENCODER_SIZE, 0, 0, ENCODER_SIZE, ENCODER_SIZE)

        // Normalize: (pixel - mean) / std, NCHW layout [1, 3, 1024, 1024]
        val planeSize = ENCODER_SIZE * ENCODER_SIZE
        for (i in inputPixels.indices) {
            val pixel = inputPixels[i]
            encoderInputFloats[i] = (((pixel shr 16) and 0xFF).toFloat() - MEAN[0]) / STD[0]               // R plane
            encoderInputFloats[planeSize + i] = (((pixel shr 8) and 0xFF).toFloat() - MEAN[1]) / STD[1]    // G plane
            encoderInputFloats[2 * planeSize + i] = ((pixel and 0xFF).toFloat() - MEAN[2]) / STD[2]        // B plane
        }
        encoderInputBuffers[0].writeFloat(encoderInputFloats)

        // Run encoder
        val resultBuffers = compiledModel.run(encoderInputBuffers)

        // Output is NCHW [1, 256, 64, 64] — directly usable by decoder
        cachedEmbeddings = resultBuffers[0].readFloat()

        lastEncodeTimeMs = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Encode: ${lastEncodeTimeMs}ms, embeddings=${cachedEmbeddings!!.size}")
    }

    /**
     * Segment at (x, y) in original image coordinates.
     * Returns a mask bitmap (same size as original image) where white = segmented.
     */
    fun segment(x: Float, y: Float): Bitmap {
        val embeddings = cachedEmbeddings ?: throw IllegalStateException("Call encodeImage() first")

        val t = System.nanoTime()

        // Scale point coords to 1024x1024 space
        val sx = x * ENCODER_SIZE / currentImageWidth
        val sy = y * ENCODER_SIZE / currentImageHeight

        // Decoder inputs
        val embeddingsTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(embeddings),
            longArrayOf(1, EMBED_DIM.toLong(), EMBED_SIZE.toLong(), EMBED_SIZE.toLong())
        )
        val pointCoordsTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(floatArrayOf(sx, sy, 0f, 0f)),
            longArrayOf(1, 2, 2)
        )
        val pointLabelsTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(floatArrayOf(1f, -1f)),
            longArrayOf(1, 2)
        )
        val maskInputTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.allocate(MASK_SIZE * MASK_SIZE),
            longArrayOf(1, 1, MASK_SIZE.toLong(), MASK_SIZE.toLong())
        )
        val hasMaskTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(floatArrayOf(0f)),
            longArrayOf(1)
        )
        val origSizeTensor = OnnxTensor.createTensor(
            ortEnv, FloatBuffer.wrap(floatArrayOf(ENCODER_SIZE.toFloat(), ENCODER_SIZE.toFloat())),
            longArrayOf(2)
        )

        val inputs = mapOf(
            "image_embeddings" to embeddingsTensor,
            "point_coords" to pointCoordsTensor,
            "point_labels" to pointLabelsTensor,
            "mask_input" to maskInputTensor,
            "has_mask_input" to hasMaskTensor,
            "orig_im_size" to origSizeTensor,
        )

        // Run decoder
        val results = decoderSession.run(inputs)

        // Get masks output: [1, 1, 1024, 1024]
        val masksTensor = results.get("masks").get() as OnnxTensor
        val masksBuffer = masksTensor.floatBuffer
        val maskH = masksTensor.info.shape[2].toInt()
        val maskW = masksTensor.info.shape[3].toInt()
        Log.i(TAG, "Mask shape: ${masksTensor.info.shape.toList()}, range sample: ${masksBuffer.get(maskH * maskW / 2)}")
        masksBuffer.rewind()

        // Build mask bitmap
        val maskBitmap = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
        val maskPixels = IntArray(maskH * maskW)

        // Skip batch and channel dims (both size 1)
        for (i in 0 until maskH * maskW) {
            maskPixels[i] = if (masksBuffer.get() > 0f) 0x8000A0FF.toInt() else 0x00000000
        }
        maskBitmap.setPixels(maskPixels, 0, maskW, 0, 0, maskW, maskH)

        // Resize to original image dimensions
        val result = Bitmap.createScaledBitmap(maskBitmap, currentImageWidth, currentImageHeight, true)
        maskBitmap.recycle()

        // Cleanup tensors
        inputs.values.forEach { (it as OnnxTensor).close() }
        results.close()

        lastDecodeTimeMs = (System.nanoTime() - t) / 1_000_000
        Log.i(TAG, "Decode: ${lastDecodeTimeMs}ms at ($x, $y)")

        return result
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
        decoderSession.close()
        ortEnv.close()
        resizedBitmap.recycle()
    }
}
