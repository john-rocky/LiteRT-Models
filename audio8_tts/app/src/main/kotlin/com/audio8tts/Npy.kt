package com.audio8tts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Minimal NumPy .npy reader and writer for the integer code tables a registered voice stores (codes.npy, '<u2' [10, N]). */
object Npy {
    class Ints(val shape: IntArray, val data: IntArray)

    fun loadInts(file: File): Ints {
        val bytes = file.readBytes()
        require(bytes.size > 10 && bytes[0] == 0x93.toByte() && String(bytes, 1, 5) == "NUMPY") { "not an .npy file: $file" }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val (headerLen, headerStart) = if (bytes[6].toInt() == 1) (bb.getShort(8).toInt() and 0xFFFF) to 10 else bb.getInt(8) to 12
        val header = String(bytes, headerStart, headerLen)
        require(!header.contains("'fortran_order': True")) { "fortran order not supported: $file" }
        val descr = Regex("'descr':\\s*'([^']+)'").find(header)?.groupValues?.get(1) ?: error("no descr in $file")
        val shape = Regex("'shape':\\s*\\(([^)]*)\\)").find(header)!!.groupValues[1]
            .split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }.toIntArray()
        val count = shape.fold(1) { a, b -> a * b }
        val off = headerStart + headerLen
        bb.position(off)
        val data = IntArray(count)
        when (descr) {
            "<u2" -> for (i in 0 until count) data[i] = bb.getShort(off + 2 * i).toInt() and 0xFFFF
            "<i2" -> for (i in 0 until count) data[i] = bb.getShort(off + 2 * i).toInt()
            "<i4" -> for (i in 0 until count) data[i] = bb.getInt(off + 4 * i)
            "<i8" -> for (i in 0 until count) data[i] = bb.getLong(off + 8 * i).toInt()
            else -> error("unsupported dtype $descr in $file")
        }
        return Ints(shape, data)
    }

    /**
     * Writes [data] (row-major, every value in 0..65535) as a version 1.0 '<u2' array of [shape], the format of the
     * bundled voices' codes.npy: magic, version, header length, then the header dict padded with spaces so that the data
     * starts at a multiple of 64 bytes.
     */
    fun saveU2(file: File, shape: IntArray, data: IntArray) {
        require(shape.fold(1) { a, b -> a * b } == data.size) { "shape ${shape.toList()} does not hold ${data.size} values" }
        val shapeText = if (shape.size == 1) "(${shape[0]},)" else shape.joinToString(", ", "(", ")")
        val dict = "{'descr': '<u2', 'fortran_order': False, 'shape': $shapeText, }"
        val header = dict + " ".repeat((64 - (10 + dict.length + 1) % 64) % 64) + "\n"
        val bb = ByteBuffer.allocate(10 + header.length + 2 * data.size).order(ByteOrder.LITTLE_ENDIAN)
        bb.put(0x93.toByte()).put("NUMPY".toByteArray(Charsets.US_ASCII)).put(1.toByte()).put(0.toByte())
        bb.putShort(header.length.toShort()).put(header.toByteArray(Charsets.US_ASCII))
        for (v in data) {
            require(v in 0..0xFFFF) { "value $v does not fit '<u2'" }
            bb.putShort(v.toShort())
        }
        file.writeBytes(bb.array())
    }
}
