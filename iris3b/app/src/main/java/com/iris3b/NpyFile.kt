package com.iris3b

import java.io.File
import java.io.FileOutputStream

/**
 * Writes float32 arrays as NumPy `.npy` files (format 1.0, little-endian, C order), so an image
 * made on the phone can be scored with `np.load` on a computer. No Android API (the Mac check runs
 * this file on the JVM).
 */
object NpyFile {

    /** Writes [values] as an array of [shape] to [file], through a `.partial` file renamed at the end. */
    fun write(file: File, values: FloatArray, shape: IntArray) {
        require(shape.fold(1L) { n, d -> n * d } == values.size.toLong()) {
            "shape ${shape.toList()} does not hold ${values.size} values"
        }
        val dims = shape.joinToString(", ") + if (shape.size == 1) "," else ""
        val dict = "{'descr': '<f4', 'fortran_order': False, 'shape': ($dims), }"
        // Pad the header with spaces so the data starts at a multiple of ALIGN bytes; it ends with '\n'.
        val unpadded = PREAMBLE + dict.length + 1
        val header = dict + " ".repeat((ALIGN - unpadded % ALIGN) % ALIGN) + "\n"
        val partial = File(file.path + ".partial")
        FileOutputStream(partial).use { out ->
            out.write(MAGIC)
            out.write(byteArrayOf(1, 0, (header.length and 0xff).toByte(), (header.length shr 8).toByte()))
            out.write(header.toByteArray(Charsets.US_ASCII))
            out.write(HostOps.littleEndian(values))
        }
        check(partial.renameTo(file)) { "cannot rename $partial to $file" }
    }

    /** Magic (6 bytes), version (2) and header length (2). */
    private const val PREAMBLE = 10
    private const val ALIGN = 64
    private val MAGIC = byteArrayOf(0x93.toByte()) + "NUMPY".toByteArray(Charsets.US_ASCII)
}
