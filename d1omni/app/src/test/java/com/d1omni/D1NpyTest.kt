package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.fail
import org.junit.Test

/**
 * The position table's `.npy` reader: the repository's file (header, sha256 = contract.json, the
 * values' bytes = numpy's, a few values) and the headers it refuses.
 */
class D1NpyTest {
  private fun npy(header: String, values: Int, major: Int = 1): ByteArray {
    val text = header.toByteArray(Charsets.ISO_8859_1)
    val prefix = if (major == 1) 10 else 12
    val out = java.io.ByteArrayOutputStream()
    out.write(byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(), 'M'.code.toByte(), 'P'.code.toByte(),
      'Y'.code.toByte(), major.toByte(), 0))
    if (major == 1) {
      out.write(text.size and 0xff)
      out.write(text.size shr 8)
    } else {
      out.write(java.nio.ByteBuffer.allocate(4).order(java.nio.ByteOrder.LITTLE_ENDIAN).putInt(text.size).array())
    }
    out.write(text)
    check(out.size() == prefix + text.size)
    out.write(ByteArray(values * 4))
    return out.toByteArray()
  }

  @Test
  fun theRepositoryTable() {
    val facts = ExternalTestData.json(ExternalTestData.demoFile("fixtures/vision/table.json"))
    val file = ExternalTestData.repoFile("host/vision_position_table.npy")
    val bytes = file.readBytes()
    val header = D1Npy.header(bytes)
    assertEquals("<f4", header.descr)
    assertFalse(header.fortranOrder)
    assertEquals(listOf(16, 16, 768), header.shape)
    assertEquals((facts["data_offset"] as JsonNumber).toInt(), header.dataOffset)
    val contract = D1VisionContract.read(ExternalTestData.repoFile(D1Contract.FILE))
    assertEquals(facts["sha256"], contract.tableSha256)
    val table = D1Npy.positionTable(file, contract.tableSha256)
    assertEquals(16 * 16 * 768, table.size)
    assertEquals(facts["value_bits_sha256"], D1VisionChecks.sha256(table))
    val first = ExternalTestData.doubles(facts["first"])
    val last = ExternalTestData.doubles(facts["last"])
    for (i in first.indices) assertEquals(first[i].toFloat().toRawBits(), table[i].toRawBits())
    for (i in last.indices) assertEquals(last[i].toFloat().toRawBits(), table[table.size - last.size + i].toRawBits())
    try {
      D1Npy.positionTable(file, "0".repeat(64))
      fail("a table with another sha256 was accepted")
    } catch (expected: IllegalStateException) {
      // contract.json's sha256 decides.
    }
  }

  @Test
  fun headersItRefuses() {
    val shape = listOf(2, 3)
    val good = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }    \n"
    assertEquals(6, D1Npy.readFloat32(npy(good, 6), shape).size)
    assertEquals(6, D1Npy.readFloat32(npy(good, 6, major = 2), shape).size)
    val bad =
      listOf(
        npy("{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }    \n", 6),
        npy("{'descr': '>f4', 'fortran_order': False, 'shape': (2, 3), }    \n", 6),
        npy("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }    \n", 6),
        npy("{'descr': '<f4', 'fortran_order': False, 'shape': (3, 2), }    \n", 6),
        npy("{'descr': '<f4', 'fortran_order': False, 'shape': (6,), }    \n", 6),
        npy(good, 5),
        npy(good, 7),
        npy("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }    ", 6),
        npy(good, 6).copyOfRange(0, 9),
        "not a numpy file at all".toByteArray(),
      )
    for ((index, bytes) in bad.withIndex()) {
      try {
        D1Npy.readFloat32(bytes, shape)
        fail("case $index was accepted")
      } catch (expected: IllegalArgumentException) {
        // refused with its reason
      }
    }
  }
}
