package com.sam3

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject

/**
 * Host-side constants and flags for the tracker loop, produced by
 * scripts/dump_tracker_device_assets.py (consts/<name>.bin raw LE float32 +
 * consts/manifest.json name->shape, flags.json).
 */
internal class TrackerConsts(trackerDir: File) {

    private val tensors = HashMap<String, FloatArray>()
    val flags: JSONObject

    init {
        val cdir = File(trackerDir, "consts")
        val manifest = JSONObject(File(cdir, "manifest.json").readText())
        for (name in manifest.keys()) {
            val entry = manifest.getJSONObject(name)
            val bytes = File(cdir, entry.getString("file")).readBytes()
            val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val arr = FloatArray(fb.remaining())
            fb.get(arr)
            tensors[name] = arr
        }
        flags = JSONObject(File(trackerDir, "flags.json").readText())
    }

    operator fun get(name: String): FloatArray =
        tensors[name] ?: throw IllegalStateException("missing const $name")

    fun flagInt(name: String): Int = flags.getInt(name)
    fun flagFloat(name: String): Float = flags.getDouble(name).toFloat()
    fun flagBool(name: String): Boolean = flags.getBoolean(name)

    /** 3-layer ReLU MLP 256->256 (obj_ptr_proj / interactive_obj_ptr_proj). */
    fun mlp3(x: FloatArray, m: Int, prefix: String): FloatArray {
        var v = x
        for (i in 0 until 3) {
            v = TM.linear(v, m, 256, get("$prefix.$i.w"), 256, get("$prefix.$i.b"))
            if (i < 2) for (j in v.indices) if (v[j] < 0f) v[j] = 0f
        }
        return v
    }

    fun noObjPtrBlend(ptr: FloatArray, m: Int, lam: FloatArray): FloatArray {
        val alt = TM.linear(ptr, m, 256, get("no_obj_ptr_linear.w"), 256, get("no_obj_ptr_linear.b"))
        val out = FloatArray(m * 256)
        for (r in 0 until m) for (c in 0 until 256) {
            out[r * 256 + c] = lam[r] * ptr[r * 256 + c] + (1f - lam[r]) * alt[r * 256 + c]
        }
        return out
    }
}
