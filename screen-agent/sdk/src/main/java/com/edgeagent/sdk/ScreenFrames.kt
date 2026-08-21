package com.edgeagent.sdk

import android.graphics.Bitmap
import android.media.Image

/**
 * The pixel-level half of screen capture: one [Image] from an ImageReader to a
 * cropped [Bitmap]. Kept apart from [CaptureService] because this part is pure
 * and the service part is all Android lifecycle policy.
 */
object ScreenFrames {

    /**
     * ImageReader hands back rows padded to a hardware-friendly stride, so the
     * buffer is usually wider than the screen. Copy it at its padded width and
     * crop, rather than assuming rowStride == width * 4.
     */
    fun toBitmap(image: Image, width: Int, height: Int): Bitmap {
        val plane = image.planes[0]
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val paddedWidth = rowStride / pixelStride

        val padded = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888)
        padded.copyPixelsFromBuffer(plane.buffer)

        if (paddedWidth == width) return padded
        val cropped = Bitmap.createBitmap(padded, 0, 0, width, height)
        padded.recycle()
        return cropped
    }
}
