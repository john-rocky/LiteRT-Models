package com.depthanything.ml

import android.graphics.Bitmap
import android.graphics.Color
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

object ImageUtils {

    // ImageNet normalization constants
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * Preprocesses a bitmap to a float buffer with ImageNet normalization.
     * Output format: NCHW [1, 3, H, W] for ONNX or NHWC [1, H, W, 3] for TFLite.
     */
    fun bitmapToFloatBuffer(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        nchw: Boolean = false,
        normalize: Boolean = true
    ): FloatBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        val pixels = IntArray(targetWidth * targetHeight)
        resized.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
        if (resized !== bitmap) resized.recycle()

        val buffer = FloatBuffer.allocate(1 * 3 * targetHeight * targetWidth)

        if (nchw) {
            for (c in 0..2) {
                for (i in pixels.indices) {
                    val pixel = pixels[i]
                    val value = when (c) {
                        0 -> Color.red(pixel)
                        1 -> Color.green(pixel)
                        else -> Color.blue(pixel)
                    }
                    val f = value / 255.0f
                    buffer.put(if (normalize) (f - MEAN[c]) / STD[c] else f)
                }
            }
        } else {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val r = Color.red(pixel) / 255.0f
                val g = Color.green(pixel) / 255.0f
                val b = Color.blue(pixel) / 255.0f
                buffer.put(if (normalize) (r - MEAN[0]) / STD[0] else r)
                buffer.put(if (normalize) (g - MEAN[1]) / STD[1] else g)
                buffer.put(if (normalize) (b - MEAN[2]) / STD[2] else b)
            }
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Converts a raw float depth output to a grayscale bitmap with min-max normalization.
     */
    fun depthFloatsToGrayscale(
        floats: FloatArray,
        width: Int,
        height: Int,
        invert: Boolean = true
    ): Bitmap {
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        for (v in floats) {
            if (v < min) min = v
            if (v > max) max = v
        }
        val range = (max - min).coerceAtLeast(1e-6f)

        val pixels = IntArray(width * height)
        for (i in floats.indices) {
            var normalized = ((floats[i] - min) / range * 255f).toInt().coerceIn(0, 255)
            if (invert) normalized = 255 - normalized
            pixels[i] = Color.rgb(normalized, normalized, normalized)
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    /**
     * Creates a ByteBuffer from a FloatBuffer (for TFLite Interpreter).
     */
    fun floatBufferToByteBuffer(floatBuffer: FloatBuffer): ByteBuffer {
        floatBuffer.rewind()
        val byteBuffer = ByteBuffer.allocateDirect(floatBuffer.capacity() * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        byteBuffer.asFloatBuffer().put(floatBuffer)
        byteBuffer.rewind()
        return byteBuffer
    }
}
