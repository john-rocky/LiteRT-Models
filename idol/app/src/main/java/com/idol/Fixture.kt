package com.idol

import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * A fixture directory `files/fx/<tag>/` as scripts/make_fixtures.py writes it: fixture.json
 * {"tag", "mode", "seed", "image", "arrays": {name: {"file", "shape"}}}, the raw input image and little-endian
 * float32 arrays (the reference run's shared randomness and each part's own reference input).
 */
class Fixture(val dir: File) {
    private val json = JSONObject(File(dir, "fixture.json").readText())
    val tag: String = json.optString("tag", dir.name)
    val mode: String = json.getString("mode")
    val seed: Long = json.optLong("seed", 0L)
    val image: File = File(dir, json.getString("image"))
    private val arrays = json.getJSONObject("arrays")

    fun has(name: String) = arrays.has(name)

    fun shape(name: String): IntArray {
        val a = arrays.getJSONObject(name).getJSONArray("shape")
        return IntArray(a.length()) { a.getInt(it) }
    }

    fun floats(name: String): FloatArray {
        val e = arrays.getJSONObject(name)
        val shape = shape(name)
        val n = shape.fold(1) { a, b -> a * b }
        return readF32(File(dir, e.getString("file")), n)
    }

    /** The five noise arrays, in place of the app's random draws. */
    fun noise(): Noise = Noise(
        vae = floats("vae_noise"),
        latent = floats("latent_noise"),
        camera = floats("camera_noise"),
        pool = floats("octree_pool"),
        leaf = floats("leaf_pool"),
        source = "fixture ${dir.name}",
    )

    companion object {
        /** [n] little-endian float32 values from [file] (its size must be n x 4 bytes). */
        fun readF32(file: File, n: Int): FloatArray {
            require(file.length() == n * 4L) { "${file.name} is ${file.length()} bytes, expected ${n * 4L}" }
            val out = FloatArray(n)
            RandomAccessFile(file, "r").use { raf ->
                val map = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, n * 4L)
                map.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
            }
            return out
        }

        fun writeF32(file: File, data: FloatArray) {
            val buf = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buf.asFloatBuffer().put(data)
            FileOutputStream(file).use { it.write(buf.array()) }
        }
    }
}
