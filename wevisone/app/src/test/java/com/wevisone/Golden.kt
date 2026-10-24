package com.wevisone

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Where the JVM tests find their references: the committed test resources, and the conversion run's reference
 * directory (`-Pwevisone.golden=<dir>`, scripts/TEST_DATA.md) for the tests that need the model's files or
 * reference arrays. Tests that need an absent directory are skipped, not failed.
 */
object Golden {
    val dir: File? = System.getProperty("wevisone.golden")?.let(::File)?.takeIf { it.isDirectory }
    val moduleRoot: File = File(System.getProperty("wevisone.moduleRoot") ?: ".")

    fun file(name: String): File? = dir?.let { File(it, name) }?.takeIf { it.isFile }

    fun floats(file: File): FloatArray {
        val bytes = file.readBytes()
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    /** Distance in float32 ulps (0 for equal values, including -0 = 0). */
    fun ulps(a: Float, b: Float): Long {
        if (a == b) return 0
        fun ordered(x: Float): Long {
            val i = java.lang.Float.floatToRawIntBits(x).toLong()
            return if (i < 0) Int.MIN_VALUE.toLong() - i else i
        }
        return kotlin.math.abs(ordered(a) - ordered(b))
    }
}
