package com.edgeagent.sdk

import android.graphics.Bitmap
import android.graphics.PointF
import java.io.File
import java.io.FileOutputStream

/**
 * Screen bitmap -> what the model actually gets, plus the way back.
 *
 * This is the only component that knows the runtime's resolution behaviour.
 * LiteRT-LM resizes an image preserving aspect ratio to at most 1024 patches of
 * 16 px (snapped to multiples of 32) with no crop and no padding — a 1080x2400
 * screenshot is seen at 320x736. So:
 *
 *  - mapping back is a plain per-axis linear scale, no letterbox arithmetic;
 *  - anything above roughly 2x the model's view is wasted encode time, which is
 *    why [wholeScreen] downscales before writing the PNG.
 *
 * A tiling variant belongs here too, if denser screens than Settings turn out
 * to need it. Measurement first: the repaired 3B scores 10/10 on real targets
 * from a single whole-screen view.
 */
object Framing {

    // The runtime's own resize, mirrored here so we can hand it an image it will
    // not touch. Values are the LFM2 data-processor defaults.
    private const val MAX_PATCHES = 1024
    private const val PATCH = 16
    private const val POOL = 2

    /**
     * The size the runtime would resize (w, h) to: aspect preserved, at most
     * [MAX_PATCHES] patches, snapped down to multiples of pool*patch.
     */
    private fun runtimeTarget(width: Int, height: Int): Pair<Int, Int> {
        val factor = Math.sqrt(
            (MAX_PATCHES.toDouble() * PATCH * PATCH) / (width.toDouble() * height)
        )
        val side = POOL * PATCH
        return Math.floor(factor * width / side).toInt() * side to
            Math.floor(factor * height / side).toInt() * side
    }

    /**
     * The fixed point of [runtimeTarget] — a size the runtime maps to itself.
     *
     * Feeding this means the runtime's `MaybeResizeImageWithSameAspectRatio`
     * returns early and the pixels reach the patchifier exactly as written.
     * That removes a double resample (ours, then the runtime's) and makes the
     * image the engine sees reproducible off-device. A 1080x2400 screen lands
     * on 320x768 after two iterations.
     */
    private fun engineViewSize(width: Int, height: Int): Pair<Int, Int> {
        var size = width to height
        repeat(8) {
            val next = runtimeTarget(size.first, size.second)
            if (next.first <= 0 || next.second <= 0) return size
            if (next == size) return size
            size = next
        }
        return size
    }

    /**
     * One model view plus the transform from the model's normalized [0,1000]
     * coordinates back to screen pixels.
     */
    class View(val file: File, private val screenWidth: Int, private val screenHeight: Int) {
        fun toScreen(normalized: IntArray): PointF = PointF(
            normalized[0] / 1000f * screenWidth,
            normalized[1] / 1000f * screenHeight,
        )
    }

    /**
     * Writes the whole screen as a single view, at the size the runtime would
     * have resized it to anyway (see [engineViewSize]).
     */
    fun wholeScreen(screen: Bitmap, dir: File, name: String = "view.png"): View {
        val (w, h) = engineViewSize(screen.width, screen.height)
        val scaled = if (w != screen.width || h != screen.height) {
            Bitmap.createScaledBitmap(screen, w, h, true)
        } else {
            screen
        }
        val file = File(dir, name)
        FileOutputStream(file).use { out ->
            scaled.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        if (scaled !== screen) scaled.recycle()
        // The transform is expressed against the ORIGINAL screen, so the
        // downscale above stays invisible to callers.
        return View(file, screen.width, screen.height)
    }
}
