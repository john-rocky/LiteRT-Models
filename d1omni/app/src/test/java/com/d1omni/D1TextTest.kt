package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The words and numbers on the screen (the answer's rule, the ms line, the summary that adds the inputs' ms as shown)
 * and the photo step before the model's preprocessing (the shrink to 384 px, the file format).
 */
class D1TextTest {
  @Test
  fun shownRule() {
    val noul = D1Question(QuestionType.NOUL, "q", null)
    assertEquals(D1Shown("yes", "0.891"), D1Answers.shown(noul, doubleArrayOf(0.8908922076225281, 0.1091077923774719)))
    assertEquals(D1Shown("no", "0.750"), D1Answers.shown(noul, doubleArrayOf(0.25, 0.75)))
    // P(yes) exactly 0.5 shows yes
    assertEquals("yes", D1Answers.shown(noul, doubleArrayOf(0.5, 0.5)).answer)
    val choice = D1Question(QuestionType.CHOICE, "q", linkedMapOf("a" to "A", "b" to "B"))
    assertEquals(D1Shown("a", "0.500"), D1Answers.shown(choice, doubleArrayOf(0.5, 0.5)))
    val score = D1Question(QuestionType.SCORE, "q", listOf("Low", "High"))
    assertEquals(D1Shown("High", "0.938"), D1Answers.shown(score, doubleArrayOf(0.0625, 0.9375)))
    assertEquals(listOf("0", "1"), D1Answers.keys(score))
    // three decimals of the exact value, half to even: 0.9375 is a half (-> 0.938, even); the double 0.9985 lies just
    // above its half (-> 0.999)
    assertEquals("0.999", D1Answers.shown(choice, doubleArrayOf(0.9985, 0.0015)).prob)
    assertEquals("1.000", D1Answers.shown(choice, doubleArrayOf(0.99994, 0.00006)).prob)
    // int32 little-endian: [1] -> 01 00 00 00
    assertEquals(D1Answers.sha256(byteArrayOf(1, 0, 0, 0)), D1Answers.idsSha256(intArrayOf(1)))
  }

  @Test
  fun screenWords() {
    assertEquals("312 ms · L256", D1Text.msLine(312, listOf(256)))
    assertEquals("160 ms · L128", D1Text.msLine(160, listOf(128, 128)))
    assertEquals("400 ms · L128 + L256", D1Text.msLine(400, listOf(256, 128)))
    assertEquals("7.4 s", D1Text.seconds(117600))
    assertEquals(319L, D1Text.itemMs(318_600_000L))
    assertEquals(
      "3 inputs · 4 answers · 760 ms · airplane mode on",
      D1Text.summary(3, 4, 760, true),
    )
    assertEquals("1 input · 1 answer · 12 ms · airplane mode off", D1Text.summary(1, 1, 12, false))
    // the summary screen draws the same line as three: the counts, the ms, airplane mode
    assertEquals("3 inputs · 4 answers", D1Text.summaryCounts(3, 4))
    assertEquals("airplane mode on", D1Text.airplane(true))
    assertEquals("Message: yes 0.999 · billing 0.939", D1Text.summaryLine("Message", listOf(D1Shown("yes", "0.999"), D1Shown("billing", "0.939"))))
    val gpu = D1Backend.GPU
    assertEquals("GPU FP32", D1Text.accelerator(listOf(gpu to D1Precision.FP32, gpu to D1Precision.FP32)))
    assertEquals("GPU FP16 (FP32 accum)", D1Text.accelerator(listOf(gpu to D1Precision.FP16_FP32_ACCUM)))
    assertEquals("GPU", D1Text.accelerator(listOf(gpu to D1Precision.FP32, gpu to D1Precision.FP16_FP32_ACCUM)))
    assertEquals("CPU 4 threads", D1Text.accelerator(listOf(D1Backend.CPU to D1Precision.FP32)))
    assertEquals("GPU + CPU", D1Text.accelerator(listOf(gpu to D1Precision.FP32, D1Backend.CPU to D1Precision.FP32)))
    assertEquals("Galaxy S26 · LiteRT 2.2.0 · GPU · d1-omni-600M", D1Text.deviceLine("Galaxy S26", "GPU"))
    assertEquals(listOf("loading", "ready", "recording", "deciding", "done"), D1Text.PILL.keys.toList())
    assertEquals(D1Text.PILL.keys, D1Text.PILL_PALETTE.keys)
    assertEquals("Galaxy S26", D1Device.marketName("SM-S942Q"))
  }

  @Test
  fun photoShrink() {
    // the check-set picture stays as it is
    assertEquals(384 to 216, D1Photo.shrunkSize(384, 216))
    assertEquals(100 to 50, D1Photo.shrunkSize(100, 50))
    // a 12 MP photo: long side 384, short side rounded half to even (3000 * 384 / 4000 = 288)
    assertEquals(384 to 288, D1Photo.shrunkSize(4000, 3000))
    assertEquals(288 to 384, D1Photo.shrunkSize(3000, 4000))
    // 1000 x 625 -> 240.0 exactly; 1000 x 626 -> 240.384 -> 240; 1000 x 627 -> 240.768 -> 241; 1001 x 1 -> 1
    assertEquals(384 to 240, D1Photo.shrunkSize(1000, 625))
    assertEquals(384 to 241, D1Photo.shrunkSize(1000, 627))
    assertEquals(384 to 1, D1Photo.shrunkSize(1001, 1))
    // half to even: 768 x 385 -> 192.5 -> 192
    assertEquals(384 to 192, D1Photo.shrunkSize(768, 385))
    val small = D1Rgb(2, 2, ByteArray(12) { it.toByte() })
    assertSame(small, D1Photo.shrink(small))
    val big = D1Rgb(800, 400, ByteArray(800 * 400 * 3) { (it % 251).toByte() })
    val shrunk = D1Photo.shrink(big)
    assertEquals(384, shrunk.width)
    assertEquals(192, shrunk.height)
    assertTrue(shrunk.data.contentEquals(D1Vision.resizeFloat(big, 192, 384).data))
    assertEquals("png", D1Photo.format(byteArrayOf(0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)))
    assertEquals("jpeg", D1Photo.format(byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte())))
    assertEquals("webp", D1Photo.format("RIFF\u0000\u0000\u0000\u0000WEBPVP8 ".toByteArray(Charsets.ISO_8859_1)))
    assertEquals("heif", D1Photo.format("\u0000\u0000\u0000\u0018ftypheic".toByteArray(Charsets.ISO_8859_1)))
    assertEquals("unknown", D1Photo.format(byteArrayOf(1, 2, 3)))
  }
}
