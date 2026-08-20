package com.sam3

import android.content.Context
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject

/**
 * Headless tracker verification: when filesDir/tracker/expected/manifest.json exists
 * (pushed by scripts/install_tracker_to_device.sh), run the full tracker on the
 * bundled clip and compare per frame against the Mac fp16 host-loop fixtures
 * (ids equal, |dprob|, video-res mask IoU). Results go to logcat tag SAM3TRK and
 * filesDir/tracker_result.txt.
 */
internal object TrackerAutotest {

    private const val TAG = "SAM3TRK"

    fun shouldRun(ctx: Context): Boolean =
        File(ctx.filesDir, "tracker/expected/manifest.json").exists()

    private fun readInts(f: File): IntArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
        val out = IntArray(b.remaining()); b.get(out); return out
    }

    private fun readFloats(f: File): FloatArray {
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(b.remaining()); b.get(out); return out
    }

    /** packed 1 bit/px LSB-first (np.packbits bitorder=little) -> per-object words. */
    private fun readMasks(f: File, nObj: Int, px: Int): Array<LongArray> {
        val bytes = f.readBytes()
        val bytesPerObj = (px + 7) / 8
        return Array(nObj) { o ->
            val w = LongArray(TM.packedWords(px))
            for (b in 0 until bytesPerObj) {
                val v = bytes[o * bytesPerObj + b].toLong() and 0xFF
                if (v != 0L) {
                    val bit = b * 8
                    w[bit ushr 6] = w[bit ushr 6] or (v shl (bit and 63))
                }
            }
            w
        }
    }

    fun run(ctx: Context, trackerDir: File, statusCb: (String) -> Unit): String {
        val manifest = JSONObject(File(trackerDir, "expected/manifest.json").readText())
        val frames = manifest.getInt("frames")
        val h = manifest.getInt("height")
        val w = manifest.getInt("width")
        val prompt = manifest.getString("prompt")
        val px = h * w
        val sb = StringBuilder()

        val t0 = System.nanoTime()
        val tracker = Sam3Tracker(ctx, trackerDir, prompt)
        val compileMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "graphs compiled in ${compileMs}ms")
        sb.append("compile ${compileMs}ms\n")

        val t1 = System.nanoTime()
        // logcat's ring buffer rotates away during long runs — persist progress so a
        // killed process still leaves evidence of how far it got.
        val progressFile = File(ctx.filesDir, "tracker_progress.txt").apply { writeText("") }
        val results = tracker.track(File(trackerDir, "clip")) { line ->
            Log.i(TAG, line)
            progressFile.appendText(line + "\n")
            statusCb("tracker $line")
        }
        val trackMs = (System.nanoTime() - t1) / 1_000_000
        check(tracker.vH == h && tracker.vW == w) {
            "clip resolution ${tracker.vW}x${tracker.vH} != manifest ${w}x$h"
        }

        var allIdsAgree = true
        var minIoU = 1.0
        var maxDp = 0.0
        for (fi in 0 until frames) {
            val refIds = readInts(File(trackerDir, "expected/f${fi}_ids.bin"))
            val refProbs = readFloats(File(trackerDir, "expected/f${fi}_probs.bin"))
            val refMasks = readMasks(File(trackerDir, "expected/f${fi}_masks.bin"), refIds.size, px)
            val got = results[fi]
            val gotIds = got?.ids ?: IntArray(0)
            val same = refIds.contentEquals(gotIds)
            allIdsAgree = allIdsAgree && same
            val ious = ArrayList<Double>()
            for (j in refIds.indices) {
                val k = gotIds.indexOf(refIds[j])
                if (k < 0) continue
                val inter = TM.popcountAnd(refMasks[j], got!!.masks[k]).toDouble()
                val union = TM.popcountOr(refMasks[j], got.masks[k]).toDouble()
                val iou = inter / kotlin.math.max(union, 1.0)
                ious.add(iou)
                minIoU = kotlin.math.min(minIoU, iou)
            }
            var dp = 0.0
            if (same) {
                for (j in refIds.indices) {
                    dp = kotlin.math.max(dp,
                        kotlin.math.abs(refProbs[j] - got!!.probs[j]).toDouble())
                }
                maxDp = kotlin.math.max(maxDp, dp)
            }
            val line = "f$fi: ids ref=${refIds.toList()} got=${gotIds.toList()} same=$same " +
                "|dprob|=${"%.4f".format(dp)} " +
                "IoU=${ious.joinToString("/") { "%.3f".format(it) }}"
            Log.i(TAG, line)
            sb.append(line).append('\n')
        }
        val stats = tracker.graphStats()
        Log.i(TAG, stats)
        val verdict = "TRACKER ids-agree=$allIdsAgree minIoU=${"%.4f".format(minIoU)} " +
            "max|dprob|=${"%.4f".format(maxDp)} total=${trackMs}ms/$frames frames"
        Log.i(TAG, verdict)
        sb.append(stats).append('\n').append(verdict).append('\n')
        File(ctx.filesDir, "tracker_result.txt").writeText(sb.toString())
        tracker.close()
        return verdict
    }
}
