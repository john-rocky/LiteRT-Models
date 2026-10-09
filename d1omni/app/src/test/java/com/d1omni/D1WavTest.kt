package com.d1omni

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The RIFF / WAVE reader against the Python host's `read_audio` (`soundfile.read(path,
 * dtype="int16")`): the six public clips give the int16 samples of `demo/fixtures/audio/<id>/
 * samples.i16` bit for bit; another rate, more than one channel and other sample formats are
 * refused, extra chunks are skipped, WAVE_FORMAT_EXTENSIBLE PCM is read.
 */
class D1WavTest {
  @Test
  fun publicClipsGiveThePythonSamples() {
    val doc = ExternalTestData.json(ExternalTestData.repoFile("fixtures/public_audio.json"))
    var clips = 0
    for (record in doc["records"] as List<*>) {
      val entry = record as Map<*, *>
      val id = entry["id"] as String
      val media = entry["media"] as Map<*, *>
      val wav = ExternalTestData.repoFile("fixtures/${media["file"]}")
      assertEquals(id, media["sha256"], D1Contract.sha256(wav))
      val samples = D1Wav.read(wav)
      val python = shorts(ExternalTestData.demoFile("fixtures/audio/$id/samples.i16"))
      assertArrayEquals(id, python, samples)
      clips++
    }
    println("D1_WAV clips=$clips bit-equal to read_audio")
    assertEquals(6, clips)
  }

  @Test
  fun encodeWritesWhatParseReadsAndWhatPythonReads() {
    val samples = ShortArray(16000) { ((it * 37) % 65536 - 32768).toShort() }
    val bytes = D1Wav.encode(samples)
    assertEquals(44 + 2 * samples.size, bytes.size)
    assertArrayEquals(samples, D1Wav.parse(bytes, "encoded"))
    // The header Python's wave module writes for 16 kHz mono 16-bit (sample count aside): RIFF size, fmt 16, PCM 1,
    // one channel, 16,000 Hz, 32,000 bytes/s, block 2, 16 bits, data size.
    val header = ByteBuffer.wrap(bytes, 0, 44).order(ByteOrder.LITTLE_ENDIAN)
    assertEquals(36 + 2 * samples.size, header.getInt(4))
    assertEquals(16, header.getInt(16))
    assertEquals(1, header.getShort(20).toInt())
    assertEquals(1, header.getShort(22).toInt())
    assertEquals(16000, header.getInt(24))
    assertEquals(32000, header.getInt(28))
    assertEquals(2, header.getShort(32).toInt())
    assertEquals(16, header.getShort(34).toInt())
    assertEquals(2 * samples.size, header.getInt(40))
    // The bundled sample clip (ffmpeg's file, with a LIST chunk) keeps its samples through encode and parse.
    val bundled = D1Wav.parse(File("src/main/res/raw/sample_voice_note.wav").readBytes())
    assertArrayEquals(bundled, D1Wav.parse(D1Wav.encode(bundled)))
  }

  @Test
  fun otherRatesChannelsAndFormatsAreRefused() {
    val pcm = shortArrayOf(0, 1, -1, 32767, -32768)
    assertArrayEquals(pcm, D1Wav.parse(wav(pcm)))
    refused("8000 Hz", wav(pcm, rate = 8000))
    refused("2 channels", wav(shortArrayOf(1, 2, 3, 4), channels = 2))
    refused("24-bit", wav(ByteArray(9), bits = 24, blockAlign = 3))
    refused("format 3", wav(ByteArray(8), code = 3, bits = 32, blockAlign = 4))
    refused("not a RIFF", "RIFX".toByteArray() + ByteArray(40))
    refused("no data chunk", riff(chunk("fmt ", fmt(1, 1, 16000, 16, 2))))
    refused("before the fmt chunk", riff(chunk("data", ByteArray(4)), chunk("fmt ", fmt(1, 1, 16000, 16, 2))))
  }

  @Test
  fun extraChunksExtensibleAndShortData() {
    val pcm = shortArrayOf(5, -6, 7)
    // A LIST chunk before fmt, an odd-sized chunk (with its pad byte) before data.
    val withChunks =
      riff(
        chunk("LIST", "INFOISFT".toByteArray()),
        chunk("fmt ", fmt(1, 1, 16000, 16, 2)),
        chunk("junk", byteArrayOf(1, 2, 3)),
        chunk("data", le(pcm)),
      )
    assertArrayEquals(pcm, D1Wav.parse(withChunks))
    // WAVE_FORMAT_EXTENSIBLE with the PCM sub-format GUID.
    val extensible =
      ByteBuffer.allocate(40).order(ByteOrder.LITTLE_ENDIAN).apply {
        put(fmt(0xFFFE, 1, 16000, 16, 2))
        putShort(22)
        putShort(16)
        putInt(4)
        put(byteArrayOf(1, 0, 0, 0, 0, 0, 0x10, 0, 0x80.toByte(), 0, 0, 0xAA.toByte(), 0, 0x38, 0x9B.toByte(), 0x71))
      }.array()
    assertArrayEquals(pcm, D1Wav.parse(riff(chunk("fmt ", extensible), chunk("data", le(pcm)))))
    // A data chunk that claims more than the file holds: the whole samples there are.
    val short = riff(chunk("fmt ", fmt(1, 1, 16000, 16, 2))) + "data".toByteArray() + le32(100) + le(pcm) + byteArrayOf(9)
    assertArrayEquals(pcm, D1Wav.parse(short))
  }

  private fun refused(reason: String, bytes: ByteArray) {
    try {
      D1Wav.parse(bytes, "clip.wav")
      fail("accepted: $reason")
    } catch (expected: IllegalArgumentException) {
      assertTrue("${expected.message} should say $reason", expected.message!!.contains(reason))
    }
  }

  private fun shorts(file: File): ShortArray {
    val bytes = file.readBytes()
    val out = ShortArray(bytes.size / 2)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(out)
    return out
  }

  private fun le(values: ShortArray): ByteArray {
    val buffer = ByteBuffer.allocate(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    buffer.asShortBuffer().put(values)
    return buffer.array()
  }

  private fun le32(value: Int): ByteArray = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()

  private fun fmt(code: Int, channels: Int, rate: Int, bits: Int, blockAlign: Int): ByteArray =
    ByteBuffer.allocate(16)
      .order(ByteOrder.LITTLE_ENDIAN)
      .putShort(code.toShort())
      .putShort(channels.toShort())
      .putInt(rate)
      .putInt(rate * blockAlign)
      .putShort(blockAlign.toShort())
      .putShort(bits.toShort())
      .array()

  private fun chunk(id: String, body: ByteArray): ByteArray {
    val out = ByteArrayOutputStream()
    out.write(id.toByteArray(Charsets.ISO_8859_1))
    out.write(le32(body.size))
    out.write(body)
    if (body.size % 2 == 1) out.write(0)
    return out.toByteArray()
  }

  private fun riff(vararg chunks: ByteArray): ByteArray {
    val body = ByteArrayOutputStream()
    body.write("WAVE".toByteArray())
    chunks.forEach { body.write(it) }
    return "RIFF".toByteArray() + le32(body.size()) + body.toByteArray()
  }

  private fun wav(pcm: ShortArray, rate: Int = 16000, channels: Int = 1): ByteArray =
    riff(chunk("fmt ", fmt(1, channels, rate, 16, 2 * channels)), chunk("data", le(pcm)))

  private fun wav(data: ByteArray, code: Int = 1, bits: Int, blockAlign: Int): ByteArray =
    riff(chunk("fmt ", fmt(code, 1, 16000, bits, blockAlign)), chunk("data", data))
}
