package com.d1omni

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The position table as the model repository ships it (`host/vision_position_table.npy`, numpy's
 * `.npy` format 1.0: the magic `\x93NUMPY`, the version, a little-endian header length, then a
 * Python dict literal `{'descr': '<f4', 'fortran_order': False, 'shape': (16, 16, 768), }` padded
 * with spaces and a newline, then the raw values). Android-free.
 */
object D1Npy {
  /** A parsed header: the dtype string, the order flag, the shape and where the values start. */
  class Header(val descr: String, val fortranOrder: Boolean, val shape: List<Int>, val dataOffset: Int)

  private val MAGIC = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(),
    'P'.code.toByte(), 'Y'.code.toByte())
  private val DESCR = Regex("""'descr'\s*:\s*'([^']*)'""")
  private val FORTRAN = Regex("""'fortran_order'\s*:\s*(True|False)""")
  private val SHAPE = Regex("""'shape'\s*:\s*\(([^)]*)\)""")

  /** Reads the header of an `.npy` file's [bytes] (versions 1.0, 2.0 and 3.0). */
  fun header(bytes: ByteArray): Header {
    require(bytes.size >= 10 && (0 until 6).all { bytes[it] == MAGIC[it] }) { "not an .npy file" }
    val major = bytes[6].toInt()
    val (length, start) =
      when (major) {
        1 -> ((bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8)) to 10
        2, 3 -> {
          require(bytes.size >= 12) { "truncated .npy header" }
          ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int to 12
        }
        else -> throw IllegalArgumentException(".npy version $major is not 1, 2 or 3")
      }
    require(length >= 0 && start + length <= bytes.size) { "truncated .npy header" }
    val text = String(bytes, start, length, if (major == 3) Charsets.UTF_8 else Charsets.ISO_8859_1)
    require(text.endsWith("\n")) { ".npy header does not end with a newline" }
    val descr = requireNotNull(DESCR.find(text)) { "no descr in $text" }.groupValues[1]
    val fortran = requireNotNull(FORTRAN.find(text)) { "no fortran_order in $text" }.groupValues[1] == "True"
    val shapeText = requireNotNull(SHAPE.find(text)) { "no shape in $text" }.groupValues[1]
    val shape = shapeText.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toInt() }
    return Header(descr, fortran, shape, start + length)
  }

  /**
   * The float32 values of an `.npy` file whose header is `'<f4'`, C order and [shape] exactly; the
   * file must hold exactly those values after the header.
   */
  fun readFloat32(bytes: ByteArray, shape: List<Int>): FloatArray {
    val header = header(bytes)
    require(header.descr == "<f4") { "dtype ${header.descr}, expected <f4 (little-endian float32)" }
    require(!header.fortranOrder) { "fortran_order True; expected C order" }
    require(header.shape == shape) { "shape ${header.shape}, expected $shape" }
    val count = shape.fold(1L) { product, side -> product * side }
    require(bytes.size.toLong() - header.dataOffset == count * 4) {
      "${bytes.size - header.dataOffset} value bytes for $count float32 values"
    }
    val values = FloatArray(count.toInt())
    ByteBuffer.wrap(bytes, header.dataOffset, values.size * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
      .get(values)
    return values
  }

  /**
   * The vision position table from [file]: its sha256 checked against [sha256] (contract.json
   * `vision_position_table.sha256`), then float32 [16, 16, 768].
   */
  fun positionTable(file: File, sha256: String): FloatArray {
    check(file.isFile) { "Missing ${file.name}" }
    val bytes = file.readBytes()
    val digest =
      java.security.MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    check(digest == sha256) { "${file.name} is not the position table of contract.json (sha256 differs)" }
    return readFloat32(bytes, listOf(D1Vision.TABLE_SIDE, D1Vision.TABLE_SIDE, D1Vision.HIDDEN))
  }
}
