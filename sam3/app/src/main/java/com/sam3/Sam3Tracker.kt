package com.sam3

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import java.util.TreeSet

/**
 * SAM 3.1 Object-Multiplex video tracker: the host state machine ported 1:1 from
 * scripts/tracker_host_loop.py (the executable spec) on top of seven LiteRT
 * CompiledModel graphs. Detection + quirk-exact NMS, association, hotstart,
 * occlusion suppression, recondition every 16 frames, memory bank with temporal
 * pos-enc, object pointers, mask-as-output init, masklet confirmation — all host.
 *
 * v1 always uses the fixed N=7 memory-attention graph; the bank below zero-pads to
 * 7 slots + keep mask, so swapping in trk_memattn_n{1..6} for small banks later only
 * needs a per-N graph map at the single memattn call site in
 * memoryConditionedFeatures().
 */
internal class Sam3Tracker(ctx: Context, private val trackerDir: File, prompt: String) : Closeable {

    companion object {
        const val C = 256
        const val L = 5184                       // 72*72
        const val MASK = 288
        const val IMG = 1008
        const val INMASK = 1152
        const val NO_OBJ_SCORE = -1024f
        const val PAD = -1
        const val REMOVED = -1116

        /** "rss=... MB swap=... MB" of this process, for load-time memory tracing. */
        fun procMem(): String {
            var rss = 0L
            var swap = 0L
            java.io.File("/proc/self/status").forEachLine {
                if (it.startsWith("VmRSS:")) rss = it.split(Regex("\\s+"))[1].toLong()
                if (it.startsWith("VmSwap:")) swap = it.split(Regex("\\s+"))[1].toLong()
            }
            return "rss=${rss / 1024} MB swap=${swap / 1024} MB"
        }
        val VIS_LAYOUT = listOf(
            Triple("sam3_fpn288", 256, 288), Triple("sam3_fpn144", 256, 144),
            Triple("sam3_fpn72", 256, 72), Triple("inter_h0", 32, 288),
            Triple("inter_h1", 64, 144), Triple("inter_f2", 256, 72),
            Triple("prop_h0", 32, 288), Triple("prop_h1", 64, 144), Triple("prop_f2", 256, 72))
    }

    private inner class Graph(file: File, accel: Accelerator) {
        val name = file.name
        val model = CompiledModel.create(file.absolutePath, CompiledModel.Options(accel), null)
        val inB = model.createInputBuffers()
        val outB = model.createOutputBuffers()
        var calls = 0
        var ms = 0.0

        init {
            android.util.Log.i("SAM3TRK", "loaded $name  ${procMem()}")
        }
        fun run(input: FloatArray): FloatArray {
            val t0 = System.nanoTime()
            inB[0].writeFloat(input)
            model.run(inB, outB)
            val y = outB[0].readFloat()
            ms += (System.nanoTime() - t0) / 1e6
            calls++
            return y
        }
        fun close() { inB.forEach { it.close() }; outB.forEach { it.close() }; model.close() }
    }

    /** Image-side files live in filesDir; tracker graphs in filesDir/tracker/graphs. */
    private fun rootFile(name: String): File {
        val f = File(trackerDir.parentFile, name)
        check(f.exists()) { "Missing ${f.name}. Run scripts/install_tracker_to_device.sh." }
        return f
    }
    private fun graphFile(name: String): File {
        val g = File(trackerDir, "graphs/$name")
        return if (g.exists()) g else rootFile(name)
    }

    val consts = TrackerConsts(trackerDir)
    private val tokenizer = BpeTokenizer(ctx)
    private val tokenTable: java.nio.ByteBuffer = java.nio.ByteBuffer
        .wrap(rootFile("sam3_token_embed.bin").readBytes())
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)

    // The text encoder is XNNPACK-resident (~1.7 GB unpacked from the 606 MB fp16
    // file) and the tracker needs exactly ONE prompt encoding — run it before the
    // GPU graphs are up and release it, or the process tops 5.8 GB and lmkd kills it.
    private val textMemPad: Pair<FloatArray, FloatArray> = runText(prompt)

    private val vis = Graph(graphFile("sam3_vision_tri.tflite"), Accelerator.GPU)
    private val head = Graph(rootFile("sam3_head.tflite"), Accelerator.GPU)
    private val memattn = Graph(graphFile("trk_memattn_n7.tflite"), Accelerator.GPU)
    private val maskdec = Graph(graphFile("trk_maskdec.tflite"), Accelerator.GPU)
    private val memenc = Graph(graphFile("trk_memenc.tflite"), Accelerator.GPU)
    private val initdec = Graph(graphFile("trk_initdec.tflite"), Accelerator.GPU)
    private val graphs = listOf(vis, head, memattn, maskdec, memenc, initdec)

    fun graphStats(): String =
        graphs.joinToString("  ") { "${it.name}: ${it.calls}x/${"%.0f".format(it.ms)}ms" }
    fun graphSnapshot(): DoubleArray = DoubleArray(graphs.size) { graphs[it].ms }
    fun graphDelta(prev: DoubleArray): String =
        graphs.indices.filter { graphs[it].ms > prev[it] + 0.5 }
            .joinToString(" ") { "${graphs[it].name.removeSuffix(".tflite")}=" +
                "%.0fms".format(graphs[it].ms - prev[it]) }

    // ---------------- vision layout
    private val visOff = HashMap<String, Int>()
    private val visLen = HashMap<String, Int>()
    init {
        var o = 0
        for ((name, c, hw) in VIS_LAYOUT) { visOff[name] = o; visLen[name] = c * hw * hw; o += c * hw * hw }
    }
    private val nVisHead = 256 * (288 * 288 + 144 * 144 + 72 * 72)

    // ---------------- flags
    private val scoreThresh = consts.flagFloat("score_threshold_detection")
    private val nmsThresh = consts.flagFloat("det_nms_thresh")
    private val assocIou = consts.flagFloat("assoc_iou_thresh")
    private val trkAssocIou = consts.flagFloat("trk_assoc_iou_thresh")
    private val newDetThresh = consts.flagFloat("new_det_thresh")
    private val iomRecond = consts.flagFloat("iom_thresh_recondition")
    private val oslThresh = consts.flagFloat("object_score_logit_threshold")
    private val hotstartDelay = consts.flagInt("hotstart_delay")
    private val hotstartUnmatch = consts.flagInt("hotstart_unmatch_thresh")
    private val hotstartDup = consts.flagInt("hotstart_dup_thresh")
    private val minKeepAlive = consts.flagInt("min_trk_keep_alive")
    private val maxKeepAlive = consts.flagInt("max_trk_keep_alive")
    private val initKeepAlive = consts.flagInt("init_trk_keep_alive")
    private val occlThresh = consts.flagFloat("suppress_overlap_recent_occl_thresh")
    private val recondEvery = consts.flagInt("recondition_every_nth_frame")
    private val maxObjects = consts.flagInt("max_num_objects")
    private val maxCondFrames = consts.flagInt("max_cond_frames_in_attn")
    private val maxObjPtrsFlag = consts.flagInt("max_obj_ptrs_in_encoder")
    private val muxCount = consts.flagInt("multiplex_count")
    private val confThresh = consts.flagInt("masklet_confirmation_consecutive_det_thresh")
    private val sigScale = consts.flagFloat("sigmoid_scale_for_mem_enc")
    private val sigBias = consts.flagFloat("sigmoid_bias_for_mem_enc")
    private val condFg = consts.flagFloat("condition_fg")
    private val condBg = consts.flagFloat("condition_bg")
    private val suppressBoundary = consts.flagBool("suppress_det_close_to_boundary")
    private val nonOverlapOut = consts.flagBool("non_overlap_masks_for_output")

    // ---------------- host constants, pre-shaped
    private val pos72Flat = FloatArray(L * C)          // (5184, 256)
    private val tposEnc = Array(7) { FloatArray(C) }
    init {
        val p = consts["pos_72"]
        for (t in 0 until L) for (c in 0 until C) pos72Flat[t * C + c] = p[c * L + t]
        val tp = consts["maskmem_tpos_enc"]
        for (k in 0 until 7) for (c in 0 until C) tposEnc[k][c] = tp[k * C + c]
    }

    // ---------------- run state
    private var numFrames = 0
    var vH = 0; private set
    var vW = 0; private set
    private val states = ArrayList<TrackState>()
    private var objIdsAll = IntArray(0)
    private var maxObjId = -1
    private val objIdToScore = HashMap<Int, Float>()
    private val sam2ScoreFrame = HashMap<Int, HashMap<Int, Float>>()
    private val removedObjIds = HashSet<Int>()
    private var confStatus = IntArray(0)
    private var confCnt = IntArray(0)
    private val hot = Hotstart()
    private var visY = FloatArray(0)
    private var curImageFeatures = FloatArray(0)       // (5184, 256)

    private class Hotstart {
        var n = 0
        var firstFrame = IntArray(0)
        var unmatchCnt = IntArray(0)
        var keepAlive = IntArray(0)
        var removed = BooleanArray(0)
        var lastOccl = IntArray(0)
        var overlap = Array(0) { IntArray(0) }
    }

    private class FrameEntry {
        var nRows = 0
        var predMasks = FloatArray(0)                  // (nRows, 288*288)
        var osl = FloatArray(0)                        // (nRows)
        var objPtr = FloatArray(0)                     // mux'd (nb, 16, 256)
        var conditioning = TreeSet<Int>()
        var maskmem: FloatArray? = null                // (nb, 256, 72, 72) bf16-rounded
        var imageFeatures: FloatArray? = null          // (5184, 256)
        var predMasksVideoRes: FloatArray? = null      // (nRows, H*W)
    }

    private class TrackState {
        var mux: MultiplexState? = null
        val objIdToIdx = LinkedHashMap<Int, Int>()
        val objIds: List<Int> get() = objIdToIdx.keys.toList()
        val outputCond = HashMap<Int, FrameEntry>()
        val outputNonCond = HashMap<Int, FrameEntry>()
        val tempCond = HashMap<Int, HashMap<Int, FloatArray>>()   // objIdx -> frame -> (H*W)
        val tempNonCond = HashMap<Int, HashMap<Int, FloatArray>>()
        val framesTracked = HashSet<Int>()
        val consolidatedCond = HashSet<Int>()
        val consolidatedNonCond = HashSet<Int>()
        var curPropF2: FloatArray? = null              // (256*5184)
    }

    private class MultiplexState(
        var assignments: MutableList<IntArray>, val capacity: Int, var objectIds: MutableList<Int>) {

        var numBuckets = 0; var muxCount = 0; var totalValid = 0; var totalNonPadding = 0
        val slotOf = HashMap<Int, Pair<Int, Int>>()

        init { reinit() }

        fun reinit() {
            numBuckets = assignments.size
            muxCount = assignments[0].size
            totalValid = assignments.sumOf { b -> b.count { it >= 0 } }
            totalNonPadding = assignments.sumOf { b -> b.count { it != PAD } }
            slotOf.clear()
            for (bi in assignments.indices) for (si in assignments[bi].indices) {
                val o = assignments[bi][si]
                if (o >= 0) slotOf[o] = Pair(bi, si)
            }
        }

        val availableSlots: Int get() = numBuckets * capacity - totalNonPadding

        fun mux(x: FloatArray, item: Int): FloatArray {
            val out = FloatArray(numBuckets * muxCount * item)
            for ((o, bs) in slotOf) {
                System.arraycopy(x, o * item, out, (bs.first * muxCount + bs.second) * item, item)
            }
            return out
        }

        fun demux(x: FloatArray, item: Int): FloatArray {
            val out = FloatArray(totalValid * item)
            for ((o, bs) in slotOf) {
                System.arraycopy(x, (bs.first * muxCount + bs.second) * item, out, o * item, item)
            }
            return out
        }

        fun validMask(): FloatArray {
            val m = FloatArray(numBuckets * muxCount)
            for ((_, bs) in slotOf) m[bs.first * muxCount + bs.second] = 1f
            return m
        }

        fun addObjects(objectIndices: List<Int>, ids: List<Int>) {
            val remIdx = objectIndices.toMutableList()
            val remIds = ids.toMutableList()
            for (b in assignments) {
                for (i in 0 until capacity) {
                    if (remIdx.isEmpty()) break
                    if (b[i] == PAD) { b[i] = remIdx.removeAt(0); objectIds.add(remIds.removeAt(0)) }
                }
                if (remIdx.isEmpty()) break
            }
            while (remIdx.isNotEmpty()) {
                val nb = IntArray(muxCount) { PAD }
                for (i in 0 until capacity) {
                    if (remIdx.isEmpty()) break
                    nb[i] = remIdx.removeAt(0); objectIds.add(remIds.removeAt(0))
                }
                assignments.add(nb)
            }
            reinit()
        }

        fun removeObjects(objectIndices: List<Int>) {
            val rem = objectIndices.toMutableList()
            for (b in assignments) for (si in b.indices) {
                if (rem.remove(b[si])) b[si] = REMOVED
            }
            assignments = assignments.filter { b -> !b.all { it == PAD || it == REMOVED } }
                .toMutableList()
            if (assignments.isEmpty()) { objectIds = mutableListOf(); return }
            val pos = assignments.flatMap { it.toList() }.filter { it >= 0 }.distinct().sorted()
            val remap = pos.withIndex().associate { (new, old) -> old to new }
            for (b in assignments) for (i in b.indices) if (b[i] >= 0) b[i] = remap[b[i]]!!
            val newIds = MutableList(pos.size) { 0 }
            for ((old, new) in remap) newIds[new] = objectIds[old]
            objectIds = newIds
            reinit()
        }
    }

    // ================================================================ frame loading
    /**
     * JPEG -> RGB float 0..255 -> triangle-filter resize to 1008 (PIL BILINEAR
     * shape; rounded back to uint8 like PIL) -> /255 -> fp16 -> normalize in fp16.
     */
    fun loadFrame(file: File): ShortArray {
        val bm = android.graphics.BitmapFactory.decodeFile(file.absolutePath)
        val w = bm.width; val h = bm.height
        vH = h; vW = w
        val px = IntArray(w * h)
        bm.getPixels(px, 0, w, 0, 0, w, h)
        val chan = FloatArray(3 * h * w)
        for (i in 0 until w * h) {
            val p = px[i]
            chan[i] = ((p shr 16) and 0xFF).toFloat()
            chan[h * w + i] = ((p shr 8) and 0xFF).toFloat()
            chan[2 * h * w + i] = (p and 0xFF).toFloat()
        }
        val rs = TM.interpBilinearAA(chan, 3, h, w, IMG, IMG)
        val out = ShortArray(3 * IMG * IMG)
        for (i in rs.indices) {
            var v = kotlin.math.floor(rs[i] + 0.5f).toInt()
            if (v < 0) v = 0
            if (v > 255) v = 255
            val h16 = TM.halfBitsToFloat(TM.toHalfBits(v / 255f))
            val sub = TM.halfBitsToFloat(TM.toHalfBits(h16 - 0.5f))
            out[i] = TM.toHalfBits(sub * 2f)               // /0.5 is exact in fp16
        }
        return out
    }

    // ================================================================ graph fronts
    private fun runVision(frame: ShortArray) {
        val input = FloatArray(frame.size)
        for (i in frame.indices) input[i] = TM.halfBitsToFloat(frame[i])
        visY = vis.run(input)
        val f2 = visOff["prop_f2"]!!
        curImageFeatures = FloatArray(L * C)
        for (t in 0 until L) for (c in 0 until C) curImageFeatures[t * C + c] = visY[f2 + c * L + t]
    }

    private fun visSlice(name: String): FloatArray {
        val o = visOff[name]!!
        return visY.copyOfRange(o, o + visLen[name]!!)
    }

    fun runText(prompt: String): Pair<FloatArray, FloatArray> {
        val ids = tokenizer.encode(prompt)
        val emb = FloatArray(32 * 1024)
        for (t in 0 until 32) {
            val base = ids[t] * 1024 * 2
            for (d in 0 until 1024) {
                emb[t * 1024 + d] = TM.halfToFloat(tokenTable.getShort(base + d * 2).toInt() and 0xFFFF)
            }
        }
        val text = Graph(rootFile("sam3_text.tflite"), Accelerator.CPU)
        val mem = try {
            text.run(emb)
        } finally {
            text.close()
        }
        android.util.Log.i("SAM3TRK", "text encoded + released  ${procMem()}")
        val pad = FloatArray(32) { if (ids[it] == 0) 1f else 0f }
        return Pair(mem, pad)
    }

    private class DetOut(
        val scores: FloatArray,        // (200), sorted per the deterministic perm
        val bboxXyxy: FloatArray,      // (200*4)
        val keep: BooleanArray,        // (200)
        val headY: FloatArray,         // full head output (mask rows live here)
        val perm: IntArray) {
        fun maskOffset(row: Int) = 1001 + perm[row] * (MASK * MASK)
    }

    private fun runDetection(textMem: FloatArray, pad: FloatArray): DetOut {
        val headIn = FloatArray(nVisHead + 32 * 256 + 32)
        System.arraycopy(visY, 0, headIn, 0, nVisHead)
        System.arraycopy(textMem, 0, headIn, nVisHead, 32 * 256)
        System.arraycopy(pad, 0, headIn, nVisHead + 32 * 256, 32)
        val y = head.run(headIn)

        val probs0 = FloatArray(200) { TM.sigmoid(y[it]) }
        val isValid = BooleanArray(200) { probs0[it] > scoreThresh }
        val packed = Array(200) { q -> TM.pack(y, 1001 + q * MASK * MASK, MASK * MASK, 0f) }
        val area = FloatArray(200) { TM.popcount(packed[it]).toFloat() }

        // NMS with the perflib row-area-IoM quirk; suppressed rows still suppress.
        val order = (0 until 200).sortedByDescending { probs0[it] }
        val keepS = BooleanArray(200) { isValid[order[it]] }
        for (i in 0 until 200) {
            val qi = order[i]
            for (j in i + 1 until 200) {
                if (!keepS[j]) continue
                val qj = order[j]
                if (TM.popcountAnd(packed[qi], packed[qj]) / (area[qi] + 1e-8f) > nmsThresh) {
                    keepS[j] = false
                }
            }
        }
        val keepQ = BooleanArray(200)
        for (i in 0 until 200) keepQ[order[i]] = keepS[i]

        val probs = FloatArray(200) { TM.sigmoid(y[it] - if (keepQ[it]) 0f else 1e4f) }
        val pos = BooleanArray(200)
        for (q in 0 until 200) {
            var p = probs[q] > scoreThresh
            if (p && suppressBoundary) {
                val cx = y[200 + q * 4]; val cy = y[200 + q * 4 + 1]
                val bw = y[200 + q * 4 + 2]; val bh = y[200 + q * 4 + 3]
                val xc = ((cx - bw / 2f) + (cx + bw / 2f)) / 2f
                val yc = ((cy - bh / 2f) + (cy + bh / 2f)) / 2f
                p = xc > 0.025f && xc < 0.975f && yc > 0.025f && yc < 0.975f
            }
            pos[q] = p
        }
        // deterministic stand-in for torch's unstable bool argsort (see the spec doc)
        val t = (0 until 200).filter { pos[it] }
        val permL = ArrayList<Int>(200)
        if (t.isNotEmpty()) { permL.add(t[0]); permL.addAll(t.drop(1).reversed()) }
        permL.addAll((0 until 200).filter { !pos[it] })
        val perm = permL.toIntArray()

        val scores = FloatArray(200) { probs[perm[it]] }
        val bbox = FloatArray(800)
        for (r in 0 until 200) {
            val q = perm[r]
            val cx = y[200 + q * 4]; val cy = y[200 + q * 4 + 1]
            val bw = y[200 + q * 4 + 2]; val bh = y[200 + q * 4 + 3]
            bbox[r * 4] = cx - bw / 2f; bbox[r * 4 + 1] = cy - bh / 2f
            bbox[r * 4 + 2] = cx + bw / 2f; bbox[r * 4 + 3] = cy + bh / 2f
        }
        val keep = BooleanArray(200) { pos[perm[it]] }
        return DetOut(scores, bbox, keep, y, perm)
    }

    /** det mask rows (float logits) -> packed binary at the given size. */
    private fun detMasksAt(det: DetOut, rows: IntArray, size: Int): Array<LongArray> {
        return Array(rows.size) { i ->
            val src = det.headY
            val o = det.maskOffset(rows[i])
            if (size == MASK) {
                TM.pack(src, o, MASK * MASK, 0f)
            } else {
                val up = TM.interpBilinear(src.copyOfRange(o, o + MASK * MASK), 1, MASK, MASK, size, size)
                TM.pack(up, 0, size * size, 0f)
            }
        }
    }

    // ================================================================ interactive init
    private fun denseEmbed(maskF: FloatArray, n: Int): FloatArray {
        var x = TM.convBlock(maskF, n, 1, INMASK, INMASK,
            consts["interactive_mask_downsample.w"], 1, 4, consts["interactive_mask_downsample.b"])
        x = TM.convBlock(x, n, 1, MASK, MASK,
            consts["mask_downscaling.0.w"], 4, 2, consts["mask_downscaling.0.b"])
        TM.layerNorm2dInPlace(x, n, 4, 144, 144, consts["mask_downscaling.1.w"], consts["mask_downscaling.1.b"])
        TM.geluInPlace(x)
        x = TM.convBlock(x, n, 4, 144, 144,
            consts["mask_downscaling.3.w"], 16, 2, consts["mask_downscaling.3.b"])
        TM.layerNorm2dInPlace(x, n, 16, 72, 72, consts["mask_downscaling.4.w"], consts["mask_downscaling.4.b"])
        TM.geluInPlace(x)
        val w6 = consts["mask_downscaling.6.w"]; val b6 = consts["mask_downscaling.6.b"]
        val out = FloatArray(n * C * L)
        for (b in 0 until n) for (o in 0 until C) {
            val ob = (b * C + o) * L
            for (p in 0 until L) {
                var acc = 0f
                for (c in 0 until 16) acc += x[(b * 16 + c) * L + p] * w6[o * 16 + c]
                out[ob + p] = acc + b6[o]
            }
        }
        return out
    }

    private class MaskAsOutput(val lowRes: FloatArray, val osl: FloatArray, val objPtr: FloatArray)

    private fun useMaskAsOutput(masks1152: Array<LongArray>): MaskAsOutput {
        val n = masks1152.size
        val px = INMASK * INMASK
        val maskF = FloatArray(n * px)
        for (i in 0 until n) for (p in 0 until px) {
            if (TM.testBit(masks1152[i], p)) maskF[i * px + p] = 1f
        }
        val highRes = FloatArray(n * px) { maskF[it] * 20f - 10f }
        val lowRes = TM.interpBilinearAA(highRes, n, INMASK, INMASK, MASK, MASK)
        val interF2 = visSlice("inter_f2")
        val noMem = consts["interactivity_no_mem_embed"]
        val dense = denseEmbed(maskF, n)
        val h0 = visSlice("inter_h0"); val h1 = visSlice("inter_h1")
        val input = FloatArray(C * L + h0.size + h1.size + 512 + C * L)
        for (c in 0 until C) {
            val add = noMem[c]
            for (p in 0 until L) input[c * L + p] = interF2[c * L + p] + add
        }
        System.arraycopy(h0, 0, input, C * L, h0.size)
        System.arraycopy(h1, 0, input, C * L + h0.size, h1.size)
        System.arraycopy(consts["sparse_const"], 0, input, C * L + h0.size + h1.size, 512)
        val denseOff = C * L + h0.size + h1.size + 512
        val tokens = FloatArray(n * 256)
        val oslG = FloatArray(n)
        for (i in 0 until n) {
            System.arraycopy(dense, i * C * L, input, denseOff, C * L)
            val y = initdec.run(input)
            val o = MASK * MASK
            System.arraycopy(y, o + 1, tokens, i * 256, 256)
            oslG[i] = y[o + 257]
        }
        var ptr = consts.mlp3(tokens, n, "interactive_obj_ptr_proj")
        ptr = consts.noObjPtrBlend(ptr, n, FloatArray(n) { if (oslG[it] > oslThresh) 1f else 0f })
        val lamM = FloatArray(n) { if (TM.popcount(masks1152[it]) > 0) 1f else 0f }
        val osl = FloatArray(n) { 20f * lamM[it] - 10f }
        ptr = consts.noObjPtrBlend(ptr, n, lamM)
        return MaskAsOutput(lowRes, osl, ptr)
    }

    // ================================================================ propagation
    private fun selectCond(cond: Map<Int, FrameEntry>, frameIdx: Int):
        Pair<LinkedHashMap<Int, FrameEntry>, HashMap<Int, FrameEntry>> {
        val selected = LinkedHashMap<Int, FrameEntry>()
        if (maxCondFrames == -1 || cond.size <= maxCondFrames) {
            for (t in cond.keys.sorted()) selected[t] = cond[t]!!
            return Pair(selected, HashMap())
        }
        val before = cond.keys.filter { it < frameIdx }.maxOrNull()
        if (before != null) selected[before] = cond[before]!!
        val after = cond.keys.filter { it >= frameIdx }.minOrNull()
        if (after != null) selected[after] = cond[after]!!
        for (t in cond.keys.filter { it !in selected }.sortedBy { kotlin.math.abs(it - frameIdx) }
            .take(maxCondFrames - selected.size)) selected[t] = cond[t]!!
        val unselected = HashMap<Int, FrameEntry>()
        for ((t, v) in cond) if (t !in selected) unselected[t] = v
        return Pair(selected, unselected)
    }

    /** Conditioned pix features per bucket: (nb, 256*5184). */
    private fun memoryConditionedFeatures(frameIdx: Int, st: TrackState): FloatArray {
        val (selected, unselected) = selectCond(st.outputCond, frameIdx)
        val slotTpos = ArrayList<Int>()
        val slotEntry = ArrayList<FrameEntry>()
        for ((t, e) in selected) { slotTpos.add(frameIdx - t); slotEntry.add(e) }
        for (tPos in 1 until 7) {
            val e = st.outputNonCond[frameIdx - (7 - tPos)] ?: unselected[frameIdx - (7 - tPos)]
            if (e != null) { slotTpos.add(tPos); slotEntry.add(e) }
        }
        val validIdx = slotEntry.indices.filter { slotEntry[it].maskmem != null }
        val mux = st.mux!!
        val nb = mux.numBuckets
        if (validIdx.isEmpty()) {
            val f2 = st.curPropF2!!
            val out = FloatArray(nb * C * L)
            for (b in 0 until nb) System.arraycopy(f2, 0, out, b * C * L, C * L)
            return out
        }

        val maxPtr = kotlin.math.min(numFrames, maxObjPtrsFlag)
        val ptrDist = ArrayList<Int>()
        val ptrEntry = ArrayList<FrameEntry>()
        for ((t, e) in selected) { ptrDist.add(frameIdx - t); ptrEntry.add(e) }
        for (tDiff in 1 until maxPtr) {
            val t = frameIdx - tDiff
            if (t < 0) break
            val e = st.outputNonCond[t] ?: unselected[t]
            if (e != null) { ptrDist.add(tDiff); ptrEntry.add(e) }
        }
        val p = ptrDist.size
        val p16 = p * 16

        val n = validIdx.size
        val nTok = n * L
        // transient: ~117 MB, freed after the frame (heap headroom on device)
        val memattnIn = FloatArray(L * C + 3 * 7 * L * C + 2 * 256 * C + (7 * L + 256))
        val miOff = L * C
        val mipOff = miOff + 7 * L * C
        val mmOff = mipOff + 7 * L * C
        val ptrOff = mmOff + 7 * L * C
        val ptrPosOff = ptrOff + 256 * C
        val keepOff = ptrPosOff + 256 * C

        System.arraycopy(curImageFeatures, 0, memattnIn, 0, L * C)   // pix = cur_prop_f2 (5184,256)
        for ((si, vi) in validIdx.withIndex()) {
            val tPos = slotTpos[vi]
            val te = if (tPos <= 0 || tPos >= 7) tposEnc[6] else tposEnc[7 - tPos - 1]
            val imgF = slotEntry[vi].imageFeatures!!
            System.arraycopy(imgF, 0, memattnIn, miOff + si * L * C, L * C)
            val b1 = mipOff + si * L * C
            for (t in 0 until L) for (c in 0 until C) {
                memattnIn[b1 + t * C + c] = pos72Flat[t * C + c] + te[c]
            }
        }
        for (i in 0 until nTok) memattnIn[keepOff + i] = 1f
        for (i in 0 until p16) memattnIn[keepOff + 7 * L + i] = 1f
        if (p > 0) {
            var objPos = TM.sine1dPe(FloatArray(p) { ptrDist[it].toFloat() / (maxPtr - 1) }, C)
            objPos = TM.linear(objPos, p, C, consts["obj_ptr_tpos_proj.w"], C, consts["obj_ptr_tpos_proj.b"])
            for (i in 0 until p) for (rep in 0 until 16) {
                System.arraycopy(objPos, i * C, memattnIn, ptrPosOff + (i * 16 + rep) * C, C)
            }
        }

        val out = FloatArray(nb * C * L)
        for (b in 0 until nb) {
            for ((si, vi) in validIdx.withIndex()) {
                // maskmem features (nb,256,72,72) -> tokens (5184,256) of bucket b
                val mm = slotEntry[vi].maskmem!!
                val base = mmOff + si * L * C
                val mb = b * C * L
                for (t in 0 until L) for (c in 0 until C) {
                    memattnIn[base + t * C + c] = mm[mb + c * L + t]
                }
            }
            for ((pi, e) in ptrEntry.withIndex()) {
                // obj_ptr mux'd (nb,16,256): bucket b's 16 slots
                System.arraycopy(e.objPtr, b * 16 * C, memattnIn, ptrOff + pi * 16 * C, 16 * C)
            }
            val y = memattn.run(memattnIn)                 // (5184, 256)
            for (t in 0 until L) for (c in 0 until C) out[b * C * L + c * L + t] = y[t * C + c]
        }
        return out
    }

    private class SamHeadsOut(val low: FloatArray, val osl: FloatArray, val objPtr: FloatArray)

    /** pixWithMem (nb, 256*5184) -> per-object best masks (nV,288*288), osl, obj_ptr. */
    private fun forwardSamHeadsProp(pixWithMem: FloatArray, st: TrackState): SamHeadsOut {
        val mux = st.mux!!
        val valid = mux.validMask()
        val validE = consts["output_valid_embed"]; val invalidE = consts["output_invalid_embed"]
        val h0 = visSlice("prop_h0"); val h1 = visSlice("prop_h1")
        val input = FloatArray(C * L + h0.size + h1.size + 16 * 256)
        System.arraycopy(h0, 0, input, C * L, h0.size)
        System.arraycopy(h1, 0, input, C * L + h0.size, h1.size)
        val mergedOff = C * L + h0.size + h1.size
        val nb = mux.numBuckets
        val masksB = FloatArray(nb * 16 * 3 * MASK * MASK)
        val iousB = FloatArray(nb * 16 * 3)
        val oslB = FloatArray(nb * 16)
        val tokB = FloatArray(nb * 16 * 3 * 256)
        for (b in 0 until nb) {
            System.arraycopy(pixWithMem, b * C * L, input, 0, C * L)
            for (s in 0 until 16) {
                val v = valid[b * 16 + s]
                for (c in 0 until 256) {
                    input[mergedOff + s * 256 + c] = v * validE[s * 256 + c] + (1f - v) * invalidE[s * 256 + c]
                }
            }
            val y = maskdec.run(input)
            val o = 16 * 3 * MASK * MASK
            System.arraycopy(y, 0, masksB, b * o, o)
            System.arraycopy(y, o, iousB, b * 48, 48)
            System.arraycopy(y, o + 48, oslB, b * 16, 16)
            System.arraycopy(y, o + 64, tokB, b * 16 * 3 * 256, 16 * 3 * 256)
        }
        val nV = mux.totalValid
        val lowMulti = mux.demux(masksB, 3 * MASK * MASK)
        val ious = mux.demux(iousB, 3)
        val osl = mux.demux(oslB, 1)
        val tokens = mux.demux(tokB, 3 * 256)
        val low = FloatArray(nV * MASK * MASK)
        val token = FloatArray(nV * 256)
        for (r in 0 until nV) {
            val isObj = osl[r] > oslThresh
            var best = 0
            for (k in 1 until 3) if (ious[r * 3 + k] > ious[r * 3 + best]) best = k
            val src = (r * 3 + best) * MASK * MASK
            if (isObj) {
                System.arraycopy(lowMulti, src, low, r * MASK * MASK, MASK * MASK)
            } else {
                java.util.Arrays.fill(low, r * MASK * MASK, (r + 1) * MASK * MASK, NO_OBJ_SCORE)
            }
            System.arraycopy(tokens, (r * 3 + best) * 256, token, r * 256, 256)
        }
        var ptr = consts.mlp3(token, nV, "obj_ptr_proj")
        ptr = consts.noObjPtrBlend(ptr, nV, FloatArray(nV) { if (osl[it] > oslThresh) 1f else 0f })
        return SamHeadsOut(low, osl, ptr)
    }

    /**
     * masksHigh (n, hs*hs) float logits at hs=1008 or 1152; returns maskmem
     * (nb, 256*5184). Resizes each mux slot channel directly into the graph input.
     */
    private fun encodeNewMemory(propF2: FloatArray, masksHigh: FloatArray, hs: Int,
                                osl: FloatArray, condObjs: Set<Int>, mux: MultiplexState): FloatArray {
        val n = osl.size
        val px = hs * hs
        val maskForMem = FloatArray(n * px)
        for (i in 0 until n * px) maskForMem[i] = TM.sigmoid(masksHigh[i]) * sigScale + sigBias
        val condVals = FloatArray(n) { condBg }
        for (o in condObjs) if (o < n) condVals[o] = condFg
        val nb = mux.numBuckets
        val out = FloatArray(nb * C * L)
        val memencIn = FloatArray(C * L + 32 * IMG * IMG)   // transient, ~135 MB
        for (b in 0 until nb) {
            java.util.Arrays.fill(memencIn, 0f)
            System.arraycopy(propF2, 0, memencIn, 0, C * L)
            for (s in 0 until 16) {
                val o = mux.assignments[b][s]
                if (o < 0) continue
                val chOff = C * L + s * IMG * IMG
                if (hs == IMG) {
                    System.arraycopy(maskForMem, o * px, memencIn, chOff, IMG * IMG)
                } else {
                    val rs = TM.interpBilinear(
                        maskForMem.copyOfRange(o * px, (o + 1) * px), 1, hs, hs, IMG, IMG)
                    System.arraycopy(rs, 0, memencIn, chOff, IMG * IMG)
                }
                // condition channel: constant map (bilinear of a constant is the constant)
                java.util.Arrays.fill(memencIn, C * L + (16 + s) * IMG * IMG,
                    C * L + (17 + s) * IMG * IMG, condVals[o])
            }
            val y = memenc.run(memencIn)                   // (256, 72, 72)
            System.arraycopy(y, 0, out, b * C * L, C * L)
        }
        // += sum over empty slots of no_obj_embed_spatial
        val noObj = consts["no_obj_embed_spatial"]         // (16, 256)
        val oslMuxFull = FloatArray(mux.totalValid)
        for (i in 0 until kotlin.math.min(n, mux.totalValid)) oslMuxFull[i] = osl[i]
        val oslMux = mux.mux(oslMuxFull, 1)
        for (b in 0 until nb) for (s in 0 until 16) {
            val isObj = if (oslMux[b * 16 + s] > oslThresh) 1f else 0f
            if (isObj == 1f) continue
            for (c in 0 until C) {
                val add = noObj[s * 256 + c]
                val base = b * C * L + c * L
                for (t in 0 until L) out[base + t] += add
            }
        }
        return out
    }

    // ================================================================ mask utilities
    /** masks (n, hw) float, per-pixel argmax keeps its object, losers min(x,-10). */
    private fun applyNonOverlapping(masks: FloatArray, n: Int, hw: Int) {
        if (n <= 1) return
        for (p in 0 until hw) {
            var arg = 0
            for (i in 1 until n) if (masks[i * hw + p] > masks[arg * hw + p]) arg = i
            for (i in 0 until n) {
                if (i != arg && masks[i * hw + p] > -10f) masks[i * hw + p] = -10f
            }
        }
    }

    /**
     * Kill (min -10) objects whose pixel-argmax non-overlapped area shrinks below
     * 0.3x; the non-overlap pass is used ONLY for the ratio — surviving objects
     * keep their ORIGINAL mask (reference quirk).
     */
    private fun suppressPwAreaShrinkage(masks: FloatArray, n: Int, hw: Int) {
        if (n <= 1) return
        val areaBefore = IntArray(n)
        for (i in 0 until n) for (p in 0 until hw) if (masks[i * hw + p] > 0f) areaBefore[i]++
        val pw = masks.copyOf()
        applyNonOverlapping(pw, n, hw)
        for (i in 0 until n) {
            var after = 0
            for (p in 0 until hw) if (pw[i * hw + p] > 0f) after++
            if (after / kotlin.math.max(areaBefore[i], 1).toFloat() < 0.3f) {
                for (p in 0 until hw) if (masks[i * hw + p] > -10f) masks[i * hw + p] = -10f
            }
        }
    }

    /** pred (n, 288*288) -> video res (n, H*W) with the output non-overlap rule. */
    private fun videoResOutput(pred: FloatArray, n: Int): FloatArray {
        val v = TM.interpBilinear(pred, n, MASK, MASK, vH, vW)
        if (nonOverlapOut) applyNonOverlapping(v, n, vH * vW)
        return v
    }

    // ================================================================ SAM2-state ops
    private fun addNewMasks(st: TrackState, frameIdx: Int, objIds: List<Int>,
                            masks1152: Array<LongArray>, reconditioning: Boolean) {
        val n = masks1152.size
        val objIdxs = ArrayList<Int>()
        for (oid in objIds) {
            val existing = st.objIdToIdx[oid]
            if (existing != null) {
                objIdxs.add(existing)
            } else {
                check(!reconditioning)
                val idx = st.objIdToIdx.size
                st.objIdToIdx[oid] = idx
                objIdxs.add(idx)
            }
        }
        // video-res binary via antialiased resize of the 1152 binary mask
        val px = INMASK * INMASK
        val maskF = FloatArray(n * px)
        for (i in 0 until n) for (p in 0 until px) if (TM.testBit(masks1152[i], p)) maskF[i * px + p] = 1f
        val videoF = TM.interpBilinearAA(maskF, n, INMASK, INMASK, vH, vW)
        val video = Array(n) { i ->
            BooleanArray(vH * vW) { p -> videoF[i * vH * vW + p] > 0.5f }
        }
        val isNewState = st.mux == null
        if (!reconditioning && isNewState) {
            val cap = muxCount
            val nb = (n + cap - 1) / cap
            val assignments = ArrayList<IntArray>()
            for (b in 0 until nb) {
                assignments.add(IntArray(cap) { i -> val v = b * cap + i; if (v < n) v else PAD })
            }
            st.mux = MultiplexState(assignments, cap, objIds.toMutableList())
        }
        val isCond = frameIdx !in st.framesTracked
        val storage = if (isCond) st.outputCond else st.outputNonCond
        val tstore = if (isCond) st.tempCond else st.tempNonCond

        val maskOut = useMaskAsOutput(masks1152)
        val current: FrameEntry
        if (reconditioning || !isNewState) {
            val existing = st.outputCond[frameIdx] ?: st.outputNonCond[frameIdx]!!
            val low = maskOut.lowRes                       // (n, 288*288), same size as stored
            if (reconditioning) {
                for (j in objIdxs.indices) {
                    val oi = objIdxs[j]
                    System.arraycopy(low, j * MASK * MASK, existing.predMasks, oi * MASK * MASK, MASK * MASK)
                    existing.osl[oi] = maskOut.osl[j]
                }
                val ptr = st.mux!!.demux(existing.objPtr, 256)
                for (j in objIdxs.indices) {
                    System.arraycopy(maskOut.objPtr, j * 256, ptr, objIdxs[j] * 256, 256)
                }
                existing.objPtr = st.mux!!.mux(ptr, 256)
                existing.conditioning.addAll(objIdxs)
            } else {
                val mux = st.mux!!
                val oldPtr = mux.demux(existing.objPtr, 256)
                val start = mux.totalValid
                mux.addObjects((start until start + n).toList(), objIds)
                existing.predMasks = existing.predMasks.copyOf((start + n) * MASK * MASK)
                System.arraycopy(low, 0, existing.predMasks, start * MASK * MASK, n * MASK * MASK)
                existing.osl = existing.osl.copyOf(start + n)
                System.arraycopy(maskOut.osl, 0, existing.osl, start, n)
                val allPtr = oldPtr.copyOf((start + n) * 256)
                System.arraycopy(maskOut.objPtr, 0, allPtr, start * 256, n * 256)
                existing.objPtr = mux.mux(allPtr, 256)
                existing.nRows = start + n
                existing.conditioning.addAll(start until start + n)
            }
            current = existing
            val vres = videoResOutput(existing.predMasks, existing.nRows)
            for (j in objIdxs.indices) {
                val oi = objIdxs[j]
                for (p in 0 until vH * vW) {
                    vres[oi * vH * vW + p] = if (video[j][p]) -NO_OBJ_SCORE else NO_OBJ_SCORE
                }
            }
            current.predMasksVideoRes = vres
        } else {
            current = FrameEntry()
            current.nRows = n
            current.predMasks = maskOut.lowRes
            current.osl = maskOut.osl
            current.objPtr = st.mux!!.mux(maskOut.objPtr, 256)
            current.conditioning.addAll(objIdxs)
            current.imageFeatures = curImageFeatures
            val vres = videoResOutput(current.predMasks, n)
            for (j in objIdxs.indices) {
                val oi = objIdxs[j]
                for (p in 0 until vH * vW) {
                    vres[oi * vH * vW + p] = if (video[j][p]) -NO_OBJ_SCORE else NO_OBJ_SCORE
                }
            }
            current.predMasksVideoRes = vres
        }

        if (isCond && frameIdx in st.outputNonCond) {
            st.outputNonCond.remove(frameIdx)
            st.consolidatedNonCond.remove(frameIdx)
        }
        storage[frameIdx] = current
        (if (isCond) st.consolidatedCond else st.consolidatedNonCond).add(frameIdx)

        // per-object temp entries (video res) with cross-suppression among the new masks
        val combined = BooleanArray(vH * vW)
        for (j in 0 until n) for (p in 0 until vH * vW) if (video[j][p]) combined[p] = true
        for (j in objIdxs.indices) {
            val m = FloatArray(vH * vW)
            for (p in 0 until vH * vW) {
                m[p] = if (video[j][p]) -NO_OBJ_SCORE else NO_OBJ_SCORE
                if (n > 1) {
                    var others = false
                    for (k in 0 until n) if (k != j && video[k][p]) { others = true; break }
                    if (others) m[p] = NO_OBJ_SCORE
                }
            }
            tstore.getOrPut(objIdxs[j]) { HashMap() }[frameIdx] = m
        }
        for ((oi2, d) in tstore) {
            if (oi2 in objIdxs || frameIdx !in d) continue
            val m = d[frameIdx]!!
            for (p in 0 until vH * vW) if (combined[p]) m[p] = NO_OBJ_SCORE
        }
    }

    private fun preflight(st: TrackState) {
        val nobj = st.mux!!.totalValid
        for (isCond in booleanArrayOf(false, true)) {
            val tstore = if (isCond) st.tempCond else st.tempNonCond
            val storage = if (isCond) st.outputCond else st.outputNonCond
            val frames = TreeSet<Int>()
            for (d in tstore.values) frames.addAll(d.keys)
            (if (isCond) st.consolidatedCond else st.consolidatedNonCond).addAll(frames)
            for (f in frames) {
                val allOut = st.outputCond[f] ?: st.outputNonCond[f]!!
                // cons rows: temp entries (aa-resized) where present, stored 288 rows
                // otherwise — the python base-resize is dead (all rows overwritten).
                val cons = allOut.predMasks.copyOf(nobj * MASK * MASK)
                for (oi in 0 until nobj) {
                    val src = tstore[oi]?.get(f) ?: continue
                    val rs = TM.interpBilinearAA(src, 1, vH, vW, MASK, MASK)
                    System.arraycopy(rs, 0, cons, oi * MASK * MASK, MASK * MASK)
                }
                val high = TM.interpBilinear(cons, nobj, MASK, MASK, IMG, IMG)
                applyNonOverlapping(high, nobj, IMG * IMG)
                val featsMem = encodeNewMemory(st.curPropF2!!, high, IMG, allOut.osl,
                    allOut.conditioning, st.mux!!)
                TM.bf16InPlace(featsMem)
                val e = FrameEntry()
                e.nRows = nobj
                e.predMasks = cons
                e.osl = allOut.osl
                e.objPtr = allOut.objPtr
                e.conditioning = TreeSet(allOut.conditioning)
                e.maskmem = featsMem
                e.imageFeatures = curImageFeatures
                storage[f] = e
            }
            for (d in tstore.values) d.clear()
        }
        for (f in st.outputCond.keys) {
            st.outputNonCond.remove(f)
            st.consolidatedNonCond.remove(f)
        }
    }

    private class PropOut(val ids: List<Int>, val masks: FloatArray, val scores: FloatArray)

    private fun propagateStateOneFrame(st: TrackState, frameIdx: Int): PropOut {
        val cur: FrameEntry
        if (frameIdx in st.consolidatedCond) {
            cur = st.outputCond[frameIdx]!!
        } else if (frameIdx in st.consolidatedNonCond) {
            cur = st.outputNonCond[frameIdx]!!
        } else {
            val pix = memoryConditionedFeatures(frameIdx, st)
            val out = forwardSamHeadsProp(pix, st)
            cur = FrameEntry()
            cur.nRows = out.osl.size
            cur.predMasks = out.low
            cur.osl = out.osl
            cur.objPtr = st.mux!!.mux(out.objPtr, 256)
            cur.imageFeatures = curImageFeatures
            st.outputNonCond[frameIdx] = cur
        }
        st.framesTracked.add(frameIdx)
        return PropOut(st.objIds, cur.predMasks.copyOf(cur.nRows * MASK * MASK), cur.osl.copyOf())
    }

    // ================================================================ planning
    private class Adt(
        val trkIsUnmatched: BooleanArray,
        val isNewDet: BooleanArray,
        val imMask: Array<BooleanArray>,          // (200, N)
        val hiConf: LinkedHashMap<Int, Int>,      // trk obj id -> det row
        val detMatched: LinkedHashMap<Int, IntArray>)  // det row -> matched trk ids

    private fun associate(det: DetOut, trkMasks: FloatArray, trkObjIds: IntArray): Adt {
        val nTrk = trkObjIds.size
        if (nTrk == 0) {
            // NOTE: the reference does NOT gate on `keep` in the empty-track branch.
            val isNew = BooleanArray(200) { det.scores[it] >= newDetThresh }
            return Adt(BooleanArray(0), isNew, Array(200) { BooleanArray(0) },
                LinkedHashMap(), LinkedHashMap())
        }
        val detPacked = Array(200) { r ->
            if (det.keep[r]) TM.pack(det.headY, det.maskOffset(r), MASK * MASK, 0f)
            else LongArray(TM.packedWords(MASK * MASK))
        }
        val detArea = FloatArray(200) { TM.popcount(detPacked[it]).toFloat() }
        val trkPacked = Array(nTrk) { TM.pack(trkMasks, it * MASK * MASK, MASK * MASK, 0f) }
        val trkArea = FloatArray(nTrk) { TM.popcount(trkPacked[it]).toFloat() }
        val metric = Array(200) { d ->
            FloatArray(nTrk) { t ->
                val inter = TM.popcountAnd(detPacked[d], trkPacked[t]).toFloat()
                inter / (kotlin.math.min(detArea[d], trkArea[t]) + 1e-8f)
            }
        }
        val trkIsMatched = BooleanArray(nTrk)
        for (d in 0 until 200) for (t in 0 until nTrk) {
            if (metric[d][t] >= trkAssocIou) trkIsMatched[t] = true
        }
        val trkIsUnmatched = BooleanArray(nTrk) { trkArea[it] > 0f && !trkIsMatched[it] }
        val isNew = BooleanArray(200) { d ->
            det.scores[d] >= newDetThresh && det.keep[d] &&
                (0 until nTrk).none { metric[d][it] >= assocIou }
        }
        val detMany = BooleanArray(200) { d -> (0 until nTrk).count { metric[d][it] >= iomRecond } > 1 }
        val trkMany = BooleanArray(nTrk) { t -> (0 until 200).count { metric[it][t] >= iomRecond } > 1 }
        val metricZ = Array(200) { d ->
            FloatArray(nTrk) { t ->
                if (trkMany[t] || detMany[d]) 0f else metric[d][t]
            }
        }
        val imMask = Array(200) { d -> BooleanArray(nTrk) { metricZ[d][it] >= assocIou } }
        val hiConf = LinkedHashMap<Int, Int>()
        val detMatched = LinkedHashMap<Int, IntArray>()
        for (d in 0 until 200) {
            if (!det.keep[d]) continue
            detMatched[d] = (0 until nTrk).filter { imMask[d][it] }.map { trkObjIds[it] }.toIntArray()
            var arg = 0
            var mx = metricZ[d][0]
            for (t in 1 until nTrk) if (metricZ[d][t] > mx) { mx = metricZ[d][t]; arg = t }
            if (det.scores[d] >= 0.8f && !isNew[d] && mx >= iomRecond) {
                hiConf[trkObjIds[arg]] = d
            }
        }
        return Adt(trkIsUnmatched, isNew, imMask, hiConf, detMatched)
    }

    private fun processHotstart(frameIdx: Int, adt: Adt): BooleanArray {
        val n = if (adt.imMask.isEmpty()) 0 else adt.imMask[0].size
        if (n == 0) return BooleanArray(0)
        check(hot.n == n) { "hotstart N mismatch: ${hot.n} vs $n" }
        val matched = BooleanArray(n)
        for (d in 0 until 200) for (t in 0 until n) if (adt.imMask[d][t]) matched[t] = true
        for (t in 0 until n) {
            hot.keepAlive[t] = (hot.keepAlive[t] + if (matched[t]) 1 else -1)
                .coerceIn(minKeepAlive, maxKeepAlive)
            if (adt.trkIsUnmatched[t]) hot.unmatchCnt[t]++
        }
        for (d in 0 until 200) {
            var cnt = 0
            for (t in 0 until n) if (adt.imMask[d][t]) cnt++
            if (cnt <= 1) continue
            for (i in 0 until n) if (adt.imMask[d][i]) {
                for (j in i + 1 until n) if (adt.imMask[d][j]) hot.overlap[i][j]++
            }
        }
        val toRemove = BooleanArray(n)
        for (t in 0 until n) {
            val within = hot.firstFrame[t] > frameIdx - hotstartDelay
            if (!within || hot.removed[t]) continue
            var maxOv = 0
            for (e in 0 until n) {
                // overlap is upper-triangular (zeros below), exactly like the reference
                if (hot.firstFrame[e] < hot.firstFrame[t]) {
                    maxOv = kotlin.math.max(maxOv, hot.overlap[e][t])
                }
            }
            if (hot.unmatchCnt[t] >= hotstartUnmatch || maxOv >= hotstartDup) toRemove[t] = true
        }
        for (t in 0 until n) if (toRemove[t]) hot.removed[t] = true
        return toRemove
    }

    private fun suppressOverlappingOccl(frameIdx: Int, trkMasks: FloatArray, n: Int,
                                        toRemove: BooleanArray) {
        val packed = Array(n) { TM.pack(trkMasks, it * MASK * MASK, MASK * MASK, 0f) }
        val last = IntArray(n) { if (toRemove.size > it && toRemove[it]) 100000 else hot.lastOccl[it] }
        val sup = BooleanArray(n)
        if (n > 1) {
            for (i in 0 until n) for (j in i + 1 until n) {
                val inter = TM.popcountAnd(packed[i], packed[j]).toFloat()
                val union = kotlin.math.max(TM.popcountOr(packed[i], packed[j]).toFloat(), 1f)
                if (inter / union >= occlThresh) {
                    if (last[i] > last[j] && last[j] > -1) sup[i] = true
                    if (last[j] > last[i] && last[i] > -1) sup[j] = true
                }
            }
        }
        for (i in 0 until n) {
            val occluded = TM.popcount(packed[i]) == 0
            hot.lastOccl[i] = if (occluded || sup[i]) frameIdx else last[i]
            if (sup[i]) {
                java.util.Arrays.fill(trkMasks, i * MASK * MASK, (i + 1) * MASK * MASK, -10f)
            }
        }
    }

    private fun updateMemories(frameIdx: Int, trkMasks: FloatArray, nTrk: Int) {
        val high = TM.interpBilinear(trkMasks, nTrk, MASK, MASK, INMASK, INMASK)
        suppressPwAreaShrinkage(high, nTrk, INMASK * INMASK)
        val osl = FloatArray(nTrk) { i ->
            var any = false
            for (p in 0 until INMASK * INMASK) if (high[i * INMASK * INMASK + p] > 0f) { any = true; break }
            if (any) 10f else -10f
        }
        // global sorted-by-id positions per state
        data class Own(val stateIdx: Int, val objId: Int)
        val owners = ArrayList<Own>()
        for ((si, st) in states.withIndex()) for (oid in st.objIds) owners.add(Own(si, oid))
        val order = owners.indices.sortedBy { owners[it].objId }
        val assign = HashMap<Int, MutableList<Int>>()
        for ((gpos, li) in order.withIndex()) {
            assign.getOrPut(owners[li].stateIdx) { ArrayList() }.add(gpos)
        }
        for ((si, st) in states.withIndex()) {
            if (st.objIds.isEmpty()) continue
            val idxs = assign[si]!!
            val entry = st.outputCond[frameIdx] ?: st.outputNonCond[frameIdx]
            val condObjs: Set<Int> = entry?.conditioning ?: emptySet()
            val subHigh = FloatArray(idxs.size * INMASK * INMASK)
            val subOsl = FloatArray(idxs.size)
            for ((k, g) in idxs.withIndex()) {
                System.arraycopy(high, g * INMASK * INMASK, subHigh, k * INMASK * INMASK, INMASK * INMASK)
                subOsl[k] = osl[g]
            }
            val featsMem = encodeNewMemory(st.curPropF2!!, subHigh, INMASK, subOsl, condObjs, st.mux!!)
            if (entry != null) {
                TM.bf16InPlace(featsMem)
                entry.maskmem = featsMem
                entry.imageFeatures = curImageFeatures
            }
        }
    }

    private fun recondition(frameIdx: Int, det: DetOut, adt: Adt,
                            trkMasks: FloatArray, trkScores: FloatArray): Set<Int> {
        val recondIds = HashSet<Int>()
        val idsAll = objIdsAll.toList()
        data class Cand(val trkId: Int, val detRow: Int, val objPos: Int)
        val cands = adt.hiConf.map { (t, d) -> Cand(t, d, idsAll.indexOf(t)) }
            .filter { TM.sigmoid(trkScores[it.objPos]) > 0.8f }
        if (cands.isEmpty()) return recondIds
        val newBin1152 = detMasksAt(det, cands.map { it.detRow }.toIntArray(), INMASK)
        for (cd in cands) {
            val o = det.maskOffset(cd.detRow)
            val tb = cd.objPos * MASK * MASK
            for (p in 0 until MASK * MASK) {
                val newV = det.headY[o + p]
                if ((newV > 0f) != (trkMasks[tb + p] > 0f)) trkMasks[tb + p] = newV
            }
        }
        for (st in states) {
            val pairIdx = cands.indices.filter { cands[it].trkId in st.objIdToIdx }
            if (pairIdx.isEmpty()) continue
            addNewMasks(st, frameIdx, pairIdx.map { cands[it].trkId },
                Array(pairIdx.size) { newBin1152[pairIdx[it]] }, reconditioning = true)
            recondIds.addAll(st.objIds)
            preflight(st)
        }
        return recondIds
    }

    private fun updateConfirmation(prevIds: IntArray, newIdsAll: IntArray, adt: Adt,
                                   newDetIds: IntArray) {
        val status = IntArray(newIdsAll.size) { 1 }
        val cnt = IntArray(newIdsAll.size)
        val pos = HashMap<Int, Int>()
        for ((i, o) in newIdsAll.withIndex()) pos[o] = i
        for ((i, o) in prevIds.withIndex()) {
            val j = pos[o] ?: continue
            status[j] = confStatus[i]
            cnt[j] = confCnt[i]
        }
        val matched = HashSet<Int>()
        newDetIds.forEach { matched.add(it) }
        for (ids in adt.detMatched.values) ids.forEach { matched.add(it) }
        for (j in newIdsAll.indices) {
            cnt[j] = if (newIdsAll[j] in matched) cnt[j] + 1 else 0
            if (cnt[j] >= confThresh) status[j] = 2
        }
        confStatus = status
        confCnt = cnt
    }

    // ================================================================ execution
    private fun addObjectsExecution(frameIdx: Int, det: DetOut, newFa: IntArray, newIds: IntArray) {
        val masks1152 = detMasksAt(det, newFa, INMASK)
        var best: TrackState? = null
        for (st in states) {
            val mux = st.mux ?: continue
            val av = mux.availableSlots
            if (av >= newFa.size && (best == null || av < best!!.mux!!.availableSlots)) best = st
        }
        if (best == null) {
            best = TrackState()
            states.add(best!!)
        }
        best!!.curPropF2 = visSlice("prop_f2")
        addNewMasks(best!!, frameIdx, newIds.toList(), masks1152, reconditioning = false)
        preflight(best!!)
    }

    private fun removeObjectsExecution(objIds: Set<Int>) {
        // NOT exercised by the verification clip; simplified port of remove_objects.
        val keep = ArrayList<TrackState>()
        for (st in states) {
            val idxs = objIds.mapNotNull { st.objIdToIdx[it] }.sorted()
            if (idxs.isNotEmpty()) {
                st.mux!!.removeObjects(idxs)
                val removeSet = idxs.toHashSet()
                val old2new = HashMap<Int, Int>()
                var neu = 0
                for (old in 0 until st.objIdToIdx.size) {
                    if (old !in removeSet) { old2new[old] = neu; neu++ }
                }
                val newMap = LinkedHashMap<Int, Int>()
                for ((oid, i) in st.objIdToIdx) if (i in old2new) newMap[oid] = old2new[i]!!
                st.objIdToIdx.clear(); st.objIdToIdx.putAll(newMap)
                val keepRows = old2new.keys.sorted()
                for (storage in listOf(st.outputCond, st.outputNonCond)) {
                    for (e in storage.values) {
                        val pm = FloatArray(keepRows.size * MASK * MASK)
                        val os = FloatArray(keepRows.size)
                        for ((k, r) in keepRows.withIndex()) {
                            if (r < e.nRows) {
                                System.arraycopy(e.predMasks, r * MASK * MASK, pm, k * MASK * MASK, MASK * MASK)
                                os[k] = e.osl[r]
                            }
                        }
                        e.predMasks = pm; e.osl = os; e.nRows = keepRows.size
                        val newCond = TreeSet<Int>()
                        for (o in e.conditioning) old2new[o]?.let { newCond.add(it) }
                        e.conditioning = newCond
                    }
                }
            }
            if (st.objIds.isNotEmpty()) keep.add(st)
        }
        states.clear(); states.addAll(keep)
    }

    // ================================================================ per-frame step
    class FrameOut(
        val objIdToMask: LinkedHashMap<Int, BooleanArray>,
        val removedNow: Set<Int>,
        val unconfirmed: List<Int>,
        val sam2Scores: Map<Int, Float>)

    private fun detTrackOneFrame(frameIdx: Int, det: DetOut): FrameOut {
        // Step 2: propagation
        val objIdsLocal = ArrayList<Int>()
        val lowList = ArrayList<FloatArray>()
        val scoreList = ArrayList<FloatArray>()
        for (st in states) {
            if (st.objIds.isEmpty()) continue
            st.curPropF2 = visSlice("prop_f2")
            val out = propagateStateOneFrame(st, frameIdx)
            objIdsLocal.addAll(out.ids)
            lowList.add(out.masks)
            scoreList.add(out.scores)
        }
        var nTrk = objIdsLocal.size
        var trkMasks = FloatArray(nTrk * MASK * MASK)
        var trkScores = FloatArray(nTrk)
        run {
            var r = 0
            for (k in lowList.indices) {
                System.arraycopy(lowList[k], 0, trkMasks, r * MASK * MASK, scoreList[k].size * MASK * MASK)
                System.arraycopy(scoreList[k], 0, trkScores, r, scoreList[k].size)
                r += scoreList[k].size
            }
        }
        if (objIdsLocal != objIdsLocal.sorted()) {
            val order = objIdsLocal.indices.sortedBy { objIdsLocal[it] }
            val m2 = FloatArray(nTrk * MASK * MASK)
            val s2 = FloatArray(nTrk)
            for ((k, o) in order.withIndex()) {
                System.arraycopy(trkMasks, o * MASK * MASK, m2, k * MASK * MASK, MASK * MASK)
                s2[k] = trkScores[o]
            }
            trkMasks = m2; trkScores = s2
            val sorted = objIdsLocal.sorted()
            objIdsLocal.clear(); objIdsLocal.addAll(sorted)
        }
        check(objIdsAll.toList() == objIdsLocal) { "obj id bookkeeping diverged" }

        // Step 3: planning
        val adt = associate(det, trkMasks, objIdsAll)
        val toRemove = processHotstart(frameIdx, adt)
        if (recondEvery > 0 && frameIdx % recondEvery == 0 && adt.hiConf.isNotEmpty()) {
            recondition(frameIdx, det, adt, trkMasks, trkScores)
        }
        if (nTrk > 0) {
            suppressOverlappingOccl(frameIdx, trkMasks, nTrk, toRemove)
            updateMemories(frameIdx, trkMasks, nTrk)
        }

        var newFa = (0 until 200).filter { adt.isNewDet[it] }.toIntArray()
        val prevN = objIdsAll.size
        if (prevN + newFa.size > maxObjects) {
            val keepN = kotlin.math.max(maxObjects - prevN, 0)
            // stable ascending then reversed, like np.argsort(kind="stable")[::-1]
            val order = newFa.indices.sortedBy { det.scores[newFa[it]] }.reversed()
            newFa = IntArray(keepN) { newFa[order[it]] }
        }
        val newIds = IntArray(newFa.size) { maxObjId + 1 + it }
        val removedNow = HashSet<Int>()
        for (t in toRemove.indices) if (toRemove[t]) removedNow.add(objIdsAll[t])

        val prevIds = objIdsAll
        objIdsAll = (prevIds.filter { it !in removedNow } + newIds.toList()).toIntArray()
        val frameScores = sam2ScoreFrame.getOrPut(frameIdx) { HashMap() }
        for (k in newIds.indices) {
            objIdToScore[newIds[k]] = det.scores[newFa[k]]
            frameScores[newIds[k]] = det.scores[newFa[k]]
        }
        if (newIds.isNotEmpty()) maxObjId = kotlin.math.max(maxObjId, newIds.max())
        for (oid in removedNow) {
            objIdToScore[oid] = -1e4f
            frameScores[oid] = -1e4f
        }
        updateConfirmation(prevIds, objIdsAll, adt, newIds)

        // hotstart array bookkeeping
        if (hot.n > 0) {
            val keepIdx = (0 until hot.n).filter { !hot.removed[it] }
            hot.firstFrame = IntArray(keepIdx.size) { hot.firstFrame[keepIdx[it]] }
            hot.unmatchCnt = IntArray(keepIdx.size) { hot.unmatchCnt[keepIdx[it]] }
            hot.keepAlive = IntArray(keepIdx.size) { hot.keepAlive[keepIdx[it]] }
            hot.lastOccl = IntArray(keepIdx.size) { hot.lastOccl[keepIdx[it]] }
            hot.overlap = Array(keepIdx.size) { i -> IntArray(keepIdx.size) { j ->
                hot.overlap[keepIdx[i]][keepIdx[j]] } }
            hot.removed = BooleanArray(keepIdx.size)
            hot.n = keepIdx.size
        }
        if (newIds.isNotEmpty()) {
            val nn = newIds.size
            val oldN = hot.n
            hot.firstFrame = hot.firstFrame.copyOf(oldN + nn).also {
                for (i in 0 until nn) it[oldN + i] = frameIdx }
            hot.unmatchCnt = hot.unmatchCnt.copyOf(oldN + nn)
            hot.keepAlive = hot.keepAlive.copyOf(oldN + nn).also {
                for (i in 0 until nn) it[oldN + i] = initKeepAlive }
            hot.removed = hot.removed.copyOf(oldN + nn)
            hot.lastOccl = hot.lastOccl.copyOf(oldN + nn).also {
                for (i in 0 until nn) it[oldN + i] = -1 }
            val ov = Array(oldN + nn) { IntArray(oldN + nn) }
            for (i in 0 until oldN) for (j in 0 until oldN) ov[i][j] = hot.overlap[i][j]
            hot.overlap = ov
            hot.n = oldN + nn
        }
        removedObjIds.addAll(removedNow)

        // Step 4: execution
        if (newFa.isNotEmpty()) addObjectsExecution(frameIdx, det, newFa, newIds)
        if (removedNow.isNotEmpty()) removeObjectsExecution(removedNow)

        for (i in prevIds.indices) frameScores[prevIds[i]] = TM.sigmoid(trkScores[i])

        // Step 5: outputs
        val objIdToMask = LinkedHashMap<Int, BooleanArray>()
        if (nTrk > 0) {
            val vid = TM.interpBilinear(trkMasks, nTrk, MASK, MASK, vH, vW)
            for (i in prevIds.indices) {
                objIdToMask[prevIds[i]] = BooleanArray(vH * vW) { vid[i * vH * vW + it] > 0f }
            }
        }
        if (newFa.isNotEmpty()) {
            for (k in newFa.indices) {
                val o = det.maskOffset(newFa[k])
                val up = TM.interpBilinear(det.headY.copyOfRange(o, o + MASK * MASK),
                    1, MASK, MASK, vH, vW)
                objIdToMask[newIds[k]] = BooleanArray(vH * vW) { up[it] > 0f }
            }
        }
        val unconfirmed = objIdsAll.indices.filter { confStatus[it] == 1 }.map { objIdsAll[it] }
        return FrameOut(objIdToMask, removedNow, unconfirmed, HashMap(frameScores))
    }

    // ================================================================ full-clip run
    class FrameResult(val ids: IntArray, val probs: FloatArray, val masks: Array<LongArray>)

    /**
     * Track `prompt` through the numbered jpgs in clipDir. Returns per-frame results
     * (ids ascending, probs = first detection score, packed video-res binary masks).
     * `progress` gets one line per processed frame.
     */
    fun track(clipDir: File, progress: (String) -> Unit): Map<Int, FrameResult> {
        val frameFiles = clipDir.listFiles { f -> f.extension.lowercase() in listOf("jpg", "jpeg", "png") }!!
            .sortedBy { it.nameWithoutExtension.toInt() }
        numFrames = frameFiles.size
        val frames = ArrayList<ShortArray>()
        for (f in frameFiles) frames.add(loadFrame(f))

        val (textMem, pad) = textMemPad

        fun runFrame(fi: Int): FrameOut {
            runVision(frames[fi])
            val det = runDetection(textMem, pad)
            return detTrackOneFrame(fi, det)
        }

        val delay = confThresh - 1
        val unconfirmedPerFrame = HashMap<Int, List<Int>>()
        val outs = HashMap<Int, FrameOut>()
        val removedSnapshotOf = HashMap<Int, Set<Int>>()
        val hotRemoved = HashSet<Int>()

        runFrame(0)                                        // add_prompt(frame 0)
        for (fi in 0 until numFrames) {                    // propagate_in_video forward
            val snap = graphSnapshot()
            val t0 = System.nanoTime()
            val out = runFrame(fi)
            outs[fi] = out
            hotRemoved.addAll(out.removedNow)
            unconfirmedPerFrame[fi] = out.unconfirmed
            if (fi == numFrames - 1) {
                for (yf in outs.keys) if (yf !in removedSnapshotOf) removedSnapshotOf[yf] = HashSet(hotRemoved)
            } else if (fi >= hotstartDelay - 1) {
                removedSnapshotOf[fi - (hotstartDelay - 1)] = HashSet(hotRemoved)
            }
            // heap relief: the memory bank only reaches cond + t-6..t-1 back, so the
            // big per-frame tensors of older non-cond entries are dead (obj_ptr is
            // still read up to 15 frames back and is kept).
            for (st in states) {
                for ((f, e) in st.outputNonCond) {
                    if (f <= fi - 6) { e.maskmem = null; e.imageFeatures = null }
                }
            }
            progress("f$fi ${(System.nanoTime() - t0) / 1_000_000}ms  ${graphDelta(snap)}")
        }

        val results = HashMap<Int, FrameResult>()
        for (fi in 0 until numFrames) {
            val out = outs[fi]!!
            val snapshot = removedSnapshotOf[fi] ?: hotRemoved
            val ids = out.objIdToMask.keys.sorted()
            if (ids.isEmpty()) {
                results[fi] = FrameResult(IntArray(0), FloatArray(0), arrayOf())
                continue
            }
            val fUnc = kotlin.math.max(0, kotlin.math.min(fi + delay, numFrames - 1))
            val hide = HashSet(snapshot)
            hide.addAll(unconfirmedPerFrame[fUnc] ?: emptyList())
            val kept = ids.filter { id ->
                val m = out.objIdToMask[id]!!
                m.any { it } && id !in hide
            }
            val masksB = kept.map { out.objIdToMask[it]!! }
            val sam2 = FloatArray(kept.size) { out.sam2Scores[kept[it]] ?: 0f }
            // object-wise non-overlap at video res, scored by the per-frame sam2 probs
            val px = vH * vW
            val packed: Array<LongArray>
            if (kept.size > 1) {
                // obj_wise_non_overlap: per pixel the highest-scoring claimant keeps it
                // (argmax ties -> first), losers lose the pixel.
                val fin = Array(kept.size) { LongArray(TM.packedWords(px)) }
                for (p in 0 until px) {
                    var arg = 0
                    var mx = if (masksB[0][p]) sam2[0] else 0f
                    for (i in 1 until kept.size) {
                        val v = if (masksB[i][p]) sam2[i] else 0f
                        if (v > mx) { mx = v; arg = i }
                    }
                    if (mx > 0f && masksB[arg][p]) {
                        fin[arg][p ushr 6] = fin[arg][p ushr 6] or (1L shl (p and 63))
                    }
                }
                packed = fin
            } else {
                packed = Array(kept.size) { i ->
                    val w = LongArray(TM.packedWords(px))
                    for (p in 0 until px) if (masksB[i][p]) w[p ushr 6] = w[p ushr 6] or (1L shl (p and 63))
                    w
                }
            }
            results[fi] = FrameResult(kept.toIntArray(),
                FloatArray(kept.size) { objIdToScore[kept[it]]!! }, packed)
        }
        return results
    }

    override fun close() {
        graphs.forEach { it.close() }
    }
}
