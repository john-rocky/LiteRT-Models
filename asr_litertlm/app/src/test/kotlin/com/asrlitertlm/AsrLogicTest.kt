package com.asrlitertlm

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** The parts of the app that run without a phone: answer parsing, transcript matching and the wav helpers. */
class AsrLogicTest {
  @get:Rule val folder = TemporaryFolder()

  @Test
  fun taggedAnswerSplitsLanguageAndText() {
    val answer =
      ModelProfile.QWEN3_ASR_1_7B.parse(
        "language English<asr_text>Everything in the universe is made of matter."
      )
    assertEquals(Answer("English", "Everything in the universe is made of matter."), answer)
  }

  @Test
  fun taggedAnswerWithLanguageNoneHasNoLanguage() {
    assertEquals(Answer("", "hello"), ModelProfile.CONFUCIUS4_R2T2.parse("language None<asr_text>hello"))
  }

  @Test
  fun answerWithoutTagIsAllText() {
    assertEquals(Answer("", "宇宙中的一切"), ModelProfile.QWEN3_ASR_1_7B.parse("  宇宙中的一切 "))
  }

  @Test
  fun plainAnswerKeepsTheTagLikeText() {
    // Fun-ASR writes the transcript alone; nothing in it is read as a language tag.
    assertEquals(
      Answer("", "language English<asr_text>x"),
      ModelProfile.FUN_ASR_NANO_2512.parse("language English<asr_text>x"),
    )
  }

  @Test
  fun normalizationDropsPunctuationSpaceAndCase() {
    assertTrue(
      TextMatch.matches(
        "Everything in the universe is made of matter. All matter is made of tiny particles called atoms.",
        "everything in the Universe is made of matter, all matter is made of tiny particles called atoms",
      )
    )
    assertTrue(TextMatch.matches("宇宙中的一切都由物质构成，而所有的物质", "宇宙中的一切都由物质构成 而所有的物质。"))
    assertTrue(TextMatch.matches("ＡＢＣ１２３", "abc123")) // NFKC folds full-width forms
  }

  @Test
  fun cerCountsCharacterEdits() {
    // 微小粒子 -> 微小颗粒: two substitutions over 32 characters (punctuation removed).
    val reference = "宇宙中的一切都由物质构成，而所有的物质都由被称为原子的微小粒子组成。"
    val hypothesis = "宇宙中的一切都由物质构成，而所有的物质都由被称为原子的微小颗粒组成。"
    assertEquals(2.0 / 32.0, TextMatch.cer(reference, hypothesis), 1e-9)
    assertEquals(0.0, TextMatch.cer(reference, reference), 0.0)
    assertEquals(1.0, TextMatch.cer(reference, ""), 0.0)
  }

  @Test
  fun shortWavIsSentAsIs() {
    val wav = tone(seconds = 2.0)
    assertSame(wav, Wav.trimmed(wav, 30.0, folder.newFile("cut.wav")))
  }

  @Test
  fun longWavIsCutAtThirtySeconds() {
    val wav = tone(seconds = 33.6)
    assertEquals(33.6, Wav.seconds(wav), 1e-6)
    val cut = Wav.trimmed(wav, 30.0, folder.newFile("cut.wav"))
    assertEquals(30.0, Wav.seconds(cut), 1e-6)
    val original = Wav.pcm16(wav).data
    val kept = Wav.pcm16(cut).data
    assertTrue(kept.contentEquals(original.copyOf(kept.size)))
  }

  @Test
  fun silenceAndToneLevels() {
    val silence = Wav.levels(ByteArray(32_000))
    assertEquals(-120.0, silence.rmsDbfs, 0.0)
    assertTrue(silence.rmsDbfs < AsrSession.NO_SPEECH_RMS_DBFS)
    val tone = Wav.levels(Wav.pcm16(tone(seconds = 1.0)).data)
    // A sine at half scale: peak about -6 dBFS, RMS about -9 dBFS.
    assertEquals(-6.0, tone.peakDbfs, 0.1)
    assertEquals(-9.0, tone.rmsDbfs, 0.1)
    assertFalse(tone.rmsDbfs < AsrSession.NO_SPEECH_RMS_DBFS)
  }

  private fun tone(seconds: Double, rate: Int = 16_000): File {
    val samples = (seconds * rate).toInt()
    val pcm = ByteBuffer.allocate(samples * 2).order(ByteOrder.LITTLE_ENDIAN)
    for (i in 0 until samples) pcm.putShort((16384 * sin(2 * Math.PI * 440 * i / rate)).toInt().toShort())
    return folder.newFile().also { Wav.writeMono16(it, pcm.array(), rate) }
  }
}
